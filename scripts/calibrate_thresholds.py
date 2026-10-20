#!/usr/bin/env python
"""Matched-dose calibration for a steering artifact against the canonical u.

The clamp's operating point is the fraction of held-out HACK segments whose projection lies
above the threshold (the clamp "binds" on them). u's grid 0/-4/-8/-12/-16 binds on a known
set of fractions; for another direction (e.g. an optimized vector at L35) the matched grid is
the five thresholds that bind on the same fractions of the same hack segments. Ordinary and
anti-hack bind rates at those thresholds are reported, not matched (one threshold fixes one
number). Where u already binds on every hack segment (the -12 and -16 arms on val), the match
falls back to the ordinary-segment bind fraction (recorded as matched_on). Calibration uses
the VAL rollouts only (test stays clean).

    uv run --with torch python scripts/calibrate_thresholds.py \
        --artifact experiments/opt-vector/artifacts/opt_27b_L35_v3c_cl.safetensors \
        [--capture data/captures/qwen35_27b_bf16_token_L31_35] [--layer 35]
--artifact may be relative to the cwd or to the repo root. Writes <artifact stem>.calibration.json
beside the artifact and prints the table. Needs torch and safetensors. See
the paper's matched-dose comparison.
"""
import argparse, json
from pathlib import Path
import numpy as np
from safetensors import safe_open

from steering_reasoning.canonical import load, paths
from steering_reasoning.activations import load_dump, join_judge_labels
from steering_reasoning.labels import load_anti_hack_labels

REPO = Path(__file__).resolve().parents[1]

U_GRID = [0, -4, -8, -12, -16]
ap = argparse.ArgumentParser()
ap.add_argument("--artifact", required=True, help=".safetensors with tensor 'vector' (proxy format) and a .json sidecar with layer_indices")
ap.add_argument("--capture", default=str(paths["token_capture_bf16"]), help="dump dir with segment_means (bf16 L31/35 capture by default)")
ap.add_argument("--layer", type=int, default=35)
ap.add_argument("--split", default="val", choices=["val", "test_A"], help="rollouts used for calibration (val by default)")
A = ap.parse_args()
art = Path(A.artifact)
if not art.is_absolute() and not art.exists(): art = REPO / art   # relative to the cwd if it exists there, else to the repo root
art_s = str(art)

with safe_open(art_s, "pt") as f: v = f.get_tensor("vector").float().reshape(-1, 5120)
side = json.load(open(art_s.replace(".safetensors", ".json"))); li_art = side["layer_indices"].index(A.layer); v = v[li_art]; v = v / v.norm()
u = load("u")
DEC, EXC = load_anti_hack_labels(paths["decline_labels"], paths["decline_kinds"])
sp = json.load(open(paths["splits"])); ids = set(sp["impossible"]["val"]) if A.split == "val" else set(sp["test_groups"]["A_labeled_hacked"])
man, rows = load_dump(Path(A.capture), layers=[A.layer]); join_judge_labels(rows); li = man["layers"].index(A.layer)
proj = {"u": {"hack": [], "other": [], "decline": []}, "v": {"hack": [], "other": [], "decline": []}}
for r in rows:
    if r["task_id"] not in ids or r.get("segment_means") is None: continue
    Z = r["segment_means"][:, li].float(); pu, pv = Z @ u, Z @ v; seen = {c: set() for c in proj["u"]}
    for i, sg in enumerate(r["segment_meta"]):
        key = (r["task_id"], sg["message_index"], sg["segment_idx"])
        if key in EXC: continue
        t = (sg.get("text") or "").strip().lower(); c = "hack" if sg.get("judge_label") else ("decline" if key in DEC else "other")
        if t in seen[c]: continue
        seen[c].add(t); proj["u"][c].append(float(pu[i])); proj["v"][c].append(float(pv[i]))
P = {k: {c: np.array(x) for c, x in d.items()} for k, d in proj.items()}
bind = lambda x, thr: float((x > thr).mean())
out = {"artifact": str(Path(A.artifact).relative_to(REPO)) if Path(A.artifact).is_absolute() else A.artifact, "layer": A.layer, "capture": A.capture, "calibration_split": A.split,
       "n_segments": {c: int(len(P["v"][c])) for c in P["v"]}, "cos_with_u": float(v @ u), "rows": []}
print(f"{'u thr':>6} {'hack bind':>9} {'match on':>8} | {'matched thr':>11} {'hack':>5} {'ordinary':>8} {'anti-hack':>9}   (u: ordinary {'':>3} anti-hack)")
for thr in U_GRID:
    fh = bind(P["u"]["hack"], thr)                                      # u's operating point
    if 0 < fh < 1:
        m, on = float(np.quantile(P["v"]["hack"], 1 - fh)), "hack"
    else:   # hack bind saturated (all hack segments already above thr): match the ordinary-segment bind fraction instead
        fo = bind(P["u"]["other"], thr); m, on = float(np.quantile(P["v"]["other"], 1 - fo)), "ordinary"
    row = {"u_threshold": thr, "hack_bind_fraction": round(fh, 4), "matched_on": on, "matched_threshold": round(m, 3),
           "v_bind": {c: round(bind(P["v"][c], m), 4) for c in ("hack", "other", "decline")}, "u_bind": {c: round(bind(P["u"][c], thr), 4) for c in ("hack", "other", "decline")}}
    out["rows"].append(row)
    print(f"{thr:6d} {fh:9.2f} {on:>8} | {m:11.2f} {row['v_bind']['hack']:5.2f} {row['v_bind']['other']:8.2f} {row['v_bind']['decline']:9.2f}   (u: {row['u_bind']['other']:.2f} {row['u_bind']['decline']:9.2f})")
dst = art_s.replace(".safetensors", ".calibration.json"); json.dump(out, open(dst, "w"), indent=1); print("wrote", dst)
