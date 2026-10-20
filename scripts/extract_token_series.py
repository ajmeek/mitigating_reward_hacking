"""Extract per-token projections onto u (L35) with segment labels for all 42 hacked rollouts.

Input: the layer-35 per-token capture at data/captures/qwen35_27b_bf16_token_L35 (scripts/dump_activations.py, then
scripts/pull_bf16_token_L35.py --src <that dump>; the two-layer dump itself also works) and the canonical u. Output: data/derived/token_series.parquet with one row per
reasoning token: task_id, split, tok (global token order within rollout, turn-major), turn (0-based turn
index), message_index, pos (within-turn), seg (global segment index in the rollout's meta order, -1 if
uncovered), label (0 other / 1 hack / 2 decline), proj.

    uv run --with torch python scripts/extract_token_series.py
"""
import json, collections, statistics as st
from pathlib import Path
import numpy as np, pandas as pd
from safetensors import safe_open

from steering_reasoning.canonical import paths, load as canon
from steering_reasoning.labels import load_anti_hack_labels

ROOT = str(Path(__file__).resolve().parents[1])
D = f"{ROOT}/data/captures/qwen35_27b_bf16_token_L35"
OUT = f"{ROOT}/data/derived/token_series.parquet"; Path(OUT).parent.mkdir(parents=True, exist_ok=True)
LI = json.load(open(f"{D}/manifest.json"))["layers"].index(35)   # works on the layer-35 slice or the two-layer dump
splits = json.load(open(f"{ROOT}/src/steering_reasoning/data/splits.json"))
split_of = {t: s for s in ("train", "val", "test") for t in splits[s]}
u = canon("u").float(); u = u / u.norm()
DEC, _EXCLUDED = load_anti_hack_labels(paths["decline_labels"], paths["decline_kinds"])   # EXCLUDED unused here (as before)

frames = []
for tid, split in sorted(split_of.items(), key=lambda kv: int(kv[0].split("_")[1])):
    meta = json.load(open(f"{D}/{tid}.json"))
    with safe_open(f"{D}/{tid}.safetensors", "pt") as f:
        N = f.get_slice("token_acts").get_shape()[0]
        proj = np.empty(N, dtype=np.float32)
        sl = f.get_slice("token_acts")
        for a in range(0, N, 4096):
            proj[a:a + 4096] = (sl[a:a + 4096, LI].float() @ u).numpy()
        turn_ids = f.get_tensor("token_turn").numpy()
        pos = f.get_tensor("token_pos").numpy()
        seg_proj = (f.get_tensor("segment_means")[:, LI].float() @ u).numpy()
    counts = collections.Counter(turn_ids.tolist())
    nturns = int(turn_ids.max()) + 1
    offs = np.zeros(nturns, dtype=int); acc = 0
    for k in range(nturns):
        offs[k] = acc; acc += counts.get(k, 0)
    msgs_sorted = sorted({s["message_index"] for s in meta["segments"]})
    turn_of_msg = {m: k for k, m in enumerate(msgs_sorted)}
    g0 = {m: min(s["token_start"] for s in meta["segments"] if s["message_index"] == m) for m in msgs_sorted}
    seg = np.full(N, -1, dtype=np.int32); lab = np.zeros(N, dtype=np.int8)
    msg_of_turn = np.full(nturns, -1, dtype=np.int32)
    for m, k in turn_of_msg.items(): msg_of_turn[k] = m
    errs = []
    for si, s in enumerate(meta["segments"]):
        m = s["message_index"]; k = turn_of_msg[m]; base = offs[k] - g0[m]
        rows = np.array([base + g for g in range(s["token_start"], s["token_end"])
                         if 0 <= base + g - offs[k] < counts.get(k, 0)], dtype=int)
        if len(rows) == 0: continue
        l = 1 if s.get("judge_label") else (2 if (tid, m, s["segment_idx"]) in DEC else 0)
        seg[rows] = si; lab[rows] = l
        errs.append(abs(proj[rows].mean() - seg_proj[si]))
    df = pd.DataFrame({"task_id": tid, "split": split, "tok": np.arange(N), "turn": turn_ids.astype(np.int16),
                       "message_index": msg_of_turn[turn_ids].astype(np.int16), "pos": pos.astype(np.int32),
                       "seg": seg, "label": lab, "proj": proj})
    frames.append(df)
    print(f"{tid:12s} {split:5s} N={N:6d} turns={nturns:3d} segs={len(meta['segments']):5d} covered={(seg>=0).mean():.3f} "
          f"hack_tok={(lab==1).sum():5d} decl_tok={(lab==2).sum():5d} |tokmean-segmean| mean={st.mean(errs):.3f} p95={sorted(errs)[int(.95*len(errs))]:.3f}", flush=True)
all_df = pd.concat(frames, ignore_index=True)
all_df.to_parquet(OUT, index=False)
print("rows", len(all_df), "->", OUT)
