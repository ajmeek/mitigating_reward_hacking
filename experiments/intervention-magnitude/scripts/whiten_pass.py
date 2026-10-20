#!/usr/bin/env python
"""Pass 2: the whitened u coordinate per token, per-class means, and a token subsample.

    uv run --with safetensors --with numpy python experiments/intervention-magnitude/scripts/whiten_pass.py \
        [--stats artifacts/stats_L35.npz] [--alpha 0.05] [--subsample 200000]

Pass 1 gave mu and Sigma. This pass needs one matrix solve and then only matrix-vector work:

    w = Sigma_a^-1 u,    kappa = sqrt(uᵀ Sigma_a^-1 u) = 1 / sigma_{u|rest}

with Sigma_a = (1-a) Sigma + a (tr Sigma / D) I, the shrinkage that keeps the inverse stable
(1.73M correlated tokens for 5120 dimensions is enough for Sigma but not for a trusted raw
inverse). Because every intervention displaces h only along u, the Mahalanobis length of the
displacement is m * kappa, and

    D^2(h + d) - D^2(h) = m^2 kappa^2 - 2 m (w . (h - mu))          (clamp: d = -m u, m >= 0)

so the only per-token quantity still missing is w . h. That is what this pass records, together
with per-class sums (hack / ordinary / unlabelled, from the judge labels) and a random subsample
of raw token vectors, which every later step (full D^2, shrinkage sensitivity, within-class
covariance, position conditioning) uses instead of re-reading 18 GB. Class labels come from
labels_L35.npz (fix_labels.py), not from the raw segment offsets.
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

ap = argparse.ArgumentParser()
ap.add_argument("--capture", default="data/captures/qwen35_27b_bf16_token_L35")
ap.add_argument("--stats", default="experiments/intervention-magnitude/artifacts/stats_L35.npz")
ap.add_argument("--labels", default="experiments/intervention-magnitude/artifacts/labels_L35.npz")
ap.add_argument("--out", default="experiments/intervention-magnitude/artifacts/whitened_L35.npz")
ap.add_argument("--alpha", type=float, default=0.05, help="shrinkage weight toward the scaled identity")
ap.add_argument("--subsample", type=int, default=200_000)
ap.add_argument("--chunk", type=int, default=8192)
ap.add_argument("--seed", type=int, default=0)
A = ap.parse_args()
D = 5120

S = np.load(REPO / A.stats)
mu, Sigma, u, n_tot = S["mu"], S["Sigma"], S["u"], int(S["n"])
Sig_a = (1 - A.alpha) * Sigma + A.alpha * (np.trace(Sigma) / D) * np.eye(D)
t0 = time.time()
w = np.linalg.solve(Sig_a, u)
kappa = float(np.sqrt(u @ w))
print(f"alpha={A.alpha}  kappa={kappa:.5f}  sigma_u|rest={1 / kappa:.4f}  solve {time.time() - t0:.1f}s", flush=True)

rng = np.random.default_rng(A.seed)
keep_p = min(1.0, A.subsample / n_tot)
files = sorted((REPO / A.capture).glob("*.safetensors"))
wh, sub, sub_idx = [], [], []
cls_sum = {k: np.zeros(D, np.float64) for k in (-1, 0, 1)}
cls_n = {k: 0 for k in (-1, 0, 1)}
off = 0
label_all = np.load(REPO / A.labels)["label"]
for fi, f in enumerate(files):
    meta = json.load(open(f.with_suffix(".json")))
    with safe_open(str(f), "np") as h:
        acts = h.get_slice("token_acts")
        N = acts.get_shape()[0]
        lv = label_all[off:off + N]
        for a in range(0, N, A.chunk):
            b = min(a + A.chunk, N)
            X = np.asarray(acts[a:b]).reshape(b - a, D).astype(np.float32)
            wh.append((X.astype(np.float64) @ w).astype(np.float32))
            lc = lv[a:b]
            for k in (-1, 0, 1):
                msk = lc == k
                if msk.any():
                    cls_sum[k] += X[msk].sum(0, dtype=np.float64)
                    cls_n[k] += int(msk.sum())
            take = rng.random(b - a) < keep_p
            if take.any():
                sub.append(X[take].astype(np.float16))
                sub_idx.append(np.nonzero(take)[0] + off + a)
    off += N
    print(f"[{fi + 1}/{len(files)}] {meta['task_id']:14s} {time.time() - t0:6.0f}s", flush=True)

out = REPO / A.out
np.savez(out, w=w, kappa=kappa, alpha=A.alpha,
         wh=np.concatenate(wh), sub=np.concatenate(sub), sub_idx=np.concatenate(sub_idx),
         cls_sum=np.stack([cls_sum[k] for k in (-1, 0, 1)]),
         cls_n=np.array([cls_n[k] for k in (-1, 0, 1)]), cls_keys=np.array([-1, 0, 1]))
print("wrote", out, "classes n =", cls_n, f"{time.time() - t0:.0f}s", flush=True)
