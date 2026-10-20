#!/usr/bin/env python
"""Per-token projections of the bf16 L35 capture onto the optimized vectors (same population and
token order as capture_stats.py, so the arrays align with stats_L35.npz['proj']).

    uv run python experiments/intervention-magnitude/scripts/capture_proj_opt.py

Writes artifacts/proj_opt_L35.npz with proj_optc (unit v3c_cl), proj_opta (unit v3c) and proj_u
(recomputed, as a check against stats_L35.npz). One streaming pass, no Gram matrix: minutes.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from safetensors import safe_open

REPO = Path(__file__).resolve().parents[3]
ap = argparse.ArgumentParser()
ap.add_argument("--capture", default="data/captures/qwen35_27b_bf16_token_L35")
ap.add_argument("--out", default="experiments/intervention-magnitude/data/proj_opt_L35.npz")
ap.add_argument("--chunk", type=int, default=4096)
A = ap.parse_args()
CAP = REPO / A.capture; D = 5120
def unit(p, key="vector"):
    with safe_open(str(REPO / p), "np") as f: v = f.get_tensor(key).astype(np.float64).ravel()
    return v / np.linalg.norm(v)
with safe_open(str(REPO / "src/steering_reasoning/canonical/vectors_L35.safetensors"), "np") as f:
    u = f.get_tensor("u").astype(np.float64).ravel(); u /= np.linalg.norm(u)
V = np.stack([unit("experiments/opt-vector/artifacts/opt_27b_L35_v3c_cl.safetensors"),
              unit("experiments/opt-vector/artifacts/opt_27b_L35_v3c.safetensors"), u], 1)   # (D, 3)
files = sorted(CAP.glob("*.safetensors")); out = []; tasks = []; t0 = time.time(); n = 0
for fi, f in enumerate(files):
    tasks.append(json.load(open(f.with_suffix(".json")))["task_id"])
    with safe_open(str(f), "np") as h:
        acts = h.get_slice("token_acts"); N = acts.get_shape()[0]
        for a in range(0, N, A.chunk):
            b = min(a + A.chunk, N)
            X = np.asarray(acts[a:b]).reshape(b - a, D).astype(np.float64)
            out.append((X @ V).astype(np.float32)); n += b - a
    print(f"[{fi + 1}/{len(files)}] {tasks[-1]:14s} N={N:7d} total={n:9d} {time.time() - t0:5.0f}s", flush=True)
P = np.concatenate(out)
np.savez(REPO / A.out, proj_optc=P[:, 0], proj_opta=P[:, 1], proj_u=P[:, 2], tasks=np.array(tasks))
print("wrote", A.out, P.shape)
