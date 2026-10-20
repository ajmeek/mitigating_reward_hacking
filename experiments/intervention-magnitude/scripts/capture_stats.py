#!/usr/bin/env python
"""Pass 1: streaming mean, covariance and per-token u projection of the bf16 L35 capture.

    uv run --with safetensors --with numpy python experiments/intervention-magnitude/scripts/capture_stats.py \
        [--capture data/captures/qwen35_27b_bf16_token_L35] [--out artifacts/stats_L35.npz]

Walks every task's `token_acts` (the model's own reasoning tokens, which is exactly the population
the clamp acts on) and accumulates, in float64: n, sum(h) and the Gram matrix sum(h hᵀ), from which
the mean mu and the covariance Sigma follow. Also records per token its projection h·u, its turn,
its position within the turn, its task, and the judge label of the segment it falls in (-1 outside
any labelled segment). One pass over ~18 GB; the Gram matmul dominates (~9e13 flops).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from safetensors import safe_open

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from steering_reasoning.canonical import VECTORS   # noqa: E402  (the file canonical.load reads;
# opened here through safetensors' numpy framework so this CPU-only analysis needs no torch)

ap = argparse.ArgumentParser()
ap.add_argument("--capture", default="data/captures/qwen35_27b_bf16_token_L35")
ap.add_argument("--out", default="experiments/intervention-magnitude/artifacts/stats_L35.npz")
ap.add_argument("--chunk", type=int, default=8192)
A = ap.parse_args()
CAP = REPO / A.capture
D = 5120

with safe_open(str(VECTORS), "np") as f:
    u = f.get_tensor("u").astype(np.float64).ravel()
u /= np.linalg.norm(u)
files = sorted(CAP.glob("*.safetensors"))
print(f"{len(files)} tasks", flush=True)

n = 0
s1 = np.zeros(D, np.float64)
G = np.zeros((D, D), np.float64)
proj, turn, pos, task_i, lab, tasks = [], [], [], [], [], []
t0 = time.time()
for fi, f in enumerate(files):
    meta = json.load(open(f.with_suffix(".json")))
    tasks.append(meta["task_id"])
    with safe_open(str(f), "np") as h:
        N = h.get_slice("token_acts").get_shape()[0]
        tt, tp = h.get_tensor("token_turn"), h.get_tensor("token_pos")
        # label each token by the segment it falls in (judge_label; -1 if in none)
        lv = np.full(N, -1, np.int8)
        for sg in meta["segments"]:
            a, b = sg["token_start"], sg["token_end"]
            if 0 <= a < b <= N:
                lv[a:b] = 1 if sg.get("judge_label") else 0
        acts = h.get_slice("token_acts")
        for a in range(0, N, A.chunk):
            b = min(a + A.chunk, N)
            X = np.asarray(acts[a:b]).reshape(b - a, D).astype(np.float32)
            s1 += X.sum(0, dtype=np.float64)
            G += (X.T @ X).astype(np.float64)
            proj.append((X.astype(np.float64) @ u).astype(np.float32))
            n += b - a
    turn.append(tt)
    pos.append(tp)
    lab.append(lv)
    task_i.append(np.full(N, fi, np.int16))
    print(f"[{fi + 1}/{len(files)}] {meta['task_id']:14s} N={N:7d} total={n:9d} {time.time() - t0:6.0f}s", flush=True)

mu = s1 / n
Sigma = G / n - np.outer(mu, mu)
out = REPO / A.out
out.parent.mkdir(parents=True, exist_ok=True)
np.savez(out, mu=mu, Sigma=Sigma, n=n, u=u,
         proj=np.concatenate(proj), turn=np.concatenate(turn), pos=np.concatenate(pos),
         label=np.concatenate(lab), task=np.concatenate(task_i), tasks=np.array(tasks))
print("wrote", out, f"n={n:,} in {time.time() - t0:.0f}s", flush=True)
