#!/usr/bin/env python
"""Rebuild the per-token judge labels of the L35 capture with the right index arithmetic.

    uv run --with numpy python experiments/intervention-magnitude/scripts/fix_labels.py

The local capture carries the per-token dump layout: `token_acts` holds only reasoning
tokens, concatenated turn by turn, while each segment's `token_start` / `token_end` are
**prompt-absolute** (experiments/full-token-capture/DATASET.md, "segments[]: legacy schema").
A segment therefore lands at

    row = sum(n_tokens of earlier turns) + (seg.token_start - turn.reasoning_start)

Using `token_start` as a row index directly, as a first pass here did, silently drops about two
thirds of the labelled hack tokens and leaves the rest misaligned. This script writes
labels_L35.npz (token-aligned with stats_L35.npz) and prints the hack / ordinary projection
means, which is the check that the mapping is right: hack tokens must sit well above ordinary
ones on u, not on top of them.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
CAP = REPO / "data/captures/qwen35_27b_bf16_token_L35"

S = np.load(EXP / "artifacts/stats_L35.npz")
proj, tasks = S["proj"].astype(np.float64), list(S["tasks"])
label = np.full(len(proj), -1, np.int8)
gold = np.full(len(proj), -1, np.int8)

off = 0
n_seg = n_placed = n_hack_tok = 0
for t in tasks:
    meta = json.load(open(CAP / f"{t}.json"))
    turns = meta["turns"]
    row_of_turn, c = {}, 0
    for tr in turns:
        row_of_turn[tr["message_index"]] = (c, tr["reasoning_start"], tr["n_tokens"])
        c += tr["n_tokens"]
    for sg in meta["segments"]:
        n_seg += 1
        mi = sg["message_index"]
        if mi not in row_of_turn:
            continue
        base, rstart, nt = row_of_turn[mi]
        a = base + sg["token_start"] - rstart
        b = a + sg["n_tokens"]
        if a < base or b > base + nt:
            continue
        label[off + a:off + b] = 1 if sg.get("judge_label") else 0
        gold[off + a:off + b] = 1 if sg.get("gold_label") else 0
        n_placed += 1
        n_hack_tok += sg["n_tokens"] if sg.get("judge_label") else 0
    off += c

np.savez(EXP / "artifacts/labels_L35.npz", label=label, gold=gold)
print(f"placed {n_placed}/{n_seg} segments; {n_hack_tok} tokens in judge-hack segments")
for k, name in ((1, "hack"), (0, "ordinary"), (-1, "unlabelled")):
    m = label == k
    if m.any():
        print(f"{name:11s} n={m.sum():8d}  h.u mean {proj[m].mean():7.2f}  sd {proj[m].std():5.2f}  "
              f"p50 {np.median(proj[m]):7.2f}  p95 {np.quantile(proj[m], 0.95):7.2f}")
mg = gold == 1
if mg.any():
    print(f"{'gold hack':11s} n={mg.sum():8d}  h.u mean {proj[mg].mean():7.2f}")
