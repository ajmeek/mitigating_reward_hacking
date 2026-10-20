#!/usr/bin/env python
"""Per-token projections of the bf16 L35 capture onto Austin's spec-citation direction v1_cites
(experiments/spec-vector, branch worktree-model-spec-expm), aligned with stats_L35.npz['proj'].
    uv run python experiments/intervention-magnitude/scripts/capture_proj_spec.py
"""
import json, sys, time
from pathlib import Path
import numpy as np
from safetensors import safe_open
REPO = Path(__file__).resolve().parents[3]
CAP = REPO / "data/captures/qwen35_27b_bf16_token_L35"; D = 5120
with safe_open(str(REPO / "experiments/spec-vector/artifacts/v1_cites_L35.safetensors"), "np") as f:   # the spec vector of record
    v = f.get_tensor("vector").astype(np.float64).ravel()
v /= np.linalg.norm(v)
out = []; n = 0; t0 = time.time()
for fi, f in enumerate(sorted(CAP.glob("*.safetensors"))):
    with safe_open(str(f), "np") as h:
        acts = h.get_slice("token_acts"); N = acts.get_shape()[0]
        for a in range(0, N, 4096):
            b = min(a + 4096, N)
            out.append((np.asarray(acts[a:b]).reshape(b - a, D).astype(np.float64) @ v).astype(np.float32)); n += b - a
    print(f"[{fi + 1}/42] total={n:9d} {time.time() - t0:5.0f}s", flush=True)
P = np.concatenate(out)
S = np.load(REPO / "experiments/intervention-magnitude/artifacts/stats_L35.npz"); W = np.load(REPO / "experiments/intervention-magnitude/artifacts/whitened_L35.npz")
alpha = float(W["alpha"]); Sigma = S["Sigma"]
Sig_a = (1 - alpha) * Sigma + alpha * (np.trace(Sigma) / D) * np.eye(D)
w = np.linalg.solve(Sig_a, v); kappa = float(np.sqrt(v @ w))
np.savez(REPO / "experiments/intervention-magnitude/data/proj_spec_L35.npz", proj_spec=P, v_spec=v, w_spec=w, kappa_spec=kappa)
print(f"kappa_v1_cites={kappa:.3f}  proj: mean {P.mean():.2f} p10 {np.percentile(P,10):.2f} p90 {np.percentile(P,90):.2f}  cos(u,v)={float(S['u']/np.linalg.norm(S['u'])@v):.3f}")
