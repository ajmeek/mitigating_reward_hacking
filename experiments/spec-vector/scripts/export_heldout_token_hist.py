#!/usr/bin/env python
"""Per-token projections of the HELD-OUT rollouts (the 42-rollout bf16 capture, val + test-A splits)
onto BOTH u and v_cite, by judge segment class (hack / decline / other), for the merged cap-vs-floor
figure: the same held-out tokens on both axes.

Class rules and token mapping follow experiments/anti-hack-labeling/scripts/u_token_hist.py
(EXCLUDED dropped; duplicate segment text per rollout counted once per class; every token of a
kept segment inherits its class; tokens outside any judge segment are not counted), but on the
canonical bf16 capture (`paths["token_capture_bf16"]`) with the canonical bf16 u and v1_cites.

Writes artifacts/cap_floor/heldout_token_hist.csv (split, vector, cls, bin_left, count; 0.5 bins)
and heldout_token_hist.json (per split/vector/class: n, quantiles, medians).
    uv run --with torch --with safetensors python experiments/spec-vector/scripts/export_heldout_token_hist.py
"""
import collections, csv, json, sys
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import load_file

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = EXP.parents[1]
sys.path.insert(0, str(REPO / "src"))
from steering_reasoning.canonical import load as canon, paths  # noqa: E402
from steering_reasoning.activations import join_judge_labels  # noqa: E402

L = 35
DUMP = Path(paths["token_capture_bf16"])
ART = REPO / "experiments/anti-hack-labeling/artifacts"
OUT = EXP / "artifacts/cap_floor"; OUT.mkdir(parents=True, exist_ok=True)
BIN = 0.5; EDGES = np.arange(-45, 60 + BIN, BIN)
splits = json.loads((REPO / "src/steering_reasoning/data/splits.json").read_text())
split_of = {t: k for k in ("train", "val", "test") for t in splits["impossible"][k]}
JUNK = {"no letters", "code / arithmetic / list item", "prompt quotation", "empty"}
labels = json.load(open(ART / "labels_full.json"))
kinds = {(o["task_id"], o["message_index"], o["segment_idx"]): o.get("kind") for o in json.load(open(ART / "decline_unique.json"))}
DECLINE, EXCLUDED = set(), set()
for o in labels:
    k = (o["task_id"], o["message_index"], o["segment_idx"])
    if o["label"] == "EXCLUDED": EXCLUDED.add(k)
    elif o["label"] == "DECLINE" and kinds.get(k) not in JUNK: DECLINE.add(k)


def classes(r):
    seen = {"hack": set(), "decline": set(), "other": set()}; out = []
    for s in r["segment_meta"]:
        key = (r["task_id"], s["message_index"], s["segment_idx"])
        if key in EXCLUDED: out.append(None); continue
        txt = (s.get("text") or "").strip().lower()
        c = "hack" if s.get("judge_label") else ("decline" if key in DECLINE else "other")
        if txt in seen[c]: out.append(None); continue
        seen[c].add(txt); out.append(c)
    return out


def token_rows(meta, turn_ids):
    counts = collections.Counter(turn_ids.tolist()); nturns = int(turn_ids.max()) + 1
    offs = np.cumsum([0] + [counts.get(k, 0) for k in range(nturns - 1)])
    msgs = sorted({s["message_index"] for s in meta["segments"]}); turn_of_msg = {m: k for k, m in enumerate(msgs)}
    g0 = {m: min(s["token_start"] for s in meta["segments"] if s["message_index"] == m) for m in msgs}
    for s in meta["segments"]:
        m = s["message_index"]; k = turn_of_msg[m]; base = offs[k] - g0[m]
        yield np.array([base + g for g in range(s["token_start"], s["token_end"]) if 0 <= base + g - offs[k] < counts.get(k, 0)], dtype=int)


def main():
    import torch
    u = canon("u").float(); v = torch.tensor(load_file(str(EXP / "artifacts/v1_cites_L35.safetensors"))["vector"]).float()
    W = torch.stack([u, v], 1)   # (5120, 2)
    man = json.loads((DUMP / "manifest.json").read_text()); li = man["layers"].index(L)
    hist = collections.defaultdict(lambda: np.zeros(len(EDGES) - 1, dtype=np.int64)); vals = collections.defaultdict(list)
    for t in man["transcripts"]:
        tid = t["task_id"]; sp = split_of.get(tid, "?")
        if sp == "train":
            continue
        meta = json.loads((DUMP / f"{tid}.json").read_text())
        with safe_open(str(DUMP / f"{tid}.safetensors"), "pt") as f:
            sl = f.get_slice("token_acts"); N = sl.get_shape()[0]; proj = np.empty((N, 2), dtype=np.float32)
            for a in range(0, N, 8192): proj[a:a + 8192] = (sl[a:a + 8192][:, li].float() @ W).numpy()
            turn_ids = f.get_tensor("token_turn").numpy(); segm = f.get_tensor("segment_means")[:, li].float()
        row = {"task_id": tid, "segment_means": segm, "segment_meta": meta["segments"]}; join_judge_labels([row])
        cls = classes(row); n_used = 0
        for c, rows in zip(cls, token_rows(meta, turn_ids)):
            if c is None or len(rows) == 0: continue
            for j, name in enumerate(("u", "v_cite")):
                x = proj[rows, j]; hist[(sp, name, c)] += np.histogram(x, EDGES)[0]; vals[(sp, name, c)].append(x)
            n_used += len(rows)
        print(f"{tid:12s} {sp:5s} N={N:7d} tokens in segments={n_used}", flush=True)
    with open(OUT / "heldout_token_hist.csv", "w") as f:
        w = csv.writer(f); w.writerow(["split", "vector", "cls", "bin_left", "count"])
        for (sp, name, c), h in sorted(hist.items()):
            for e, n in zip(EDGES[:-1], h): w.writerow([sp, name, c, f"{e:.2f}", int(n)])
    summ = {"layer": L, "capture": str(DUMP), "vectors": {"u": "canonical bf16 u", "v_cite": "experiments/spec-vector/artifacts v1_cites"},
            "bin_width": BIN, "splits": "val + test (test = group A, the 10 labelled hacked-origin test tasks); train excluded", "groups": {}}
    for (sp, name, c), vs in sorted(vals.items()):
        x = np.concatenate(vs); summ["groups"][f"{sp}/{name}/{c}"] = {"n": int(len(x)), "median": float(np.median(x)), "p10": float(np.percentile(x, 10)), "p90": float(np.percentile(x, 90))}
    (OUT / "heldout_token_hist.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: (g["n"], round(g["median"], 1)) for k, g in summ["groups"].items()}, indent=1))


if __name__ == "__main__":
    main()
