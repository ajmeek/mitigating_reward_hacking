"""Centring mean mu at L35 from the local bf16 per-token capture.

For each of the 42 hacked rollouts take the mean over all its judge segments of segment_means[:, L35]
(no dedup, no EXCLUDED filter), then average the per-rollout means with equal weight. Reads
data/captures/qwen35_27b_bf16_token_L35 (scripts/pull_bf16_token_L35.py) and writes
experiments/probe-refit-bf16/artifacts/mu_bf16_L35.safetensors: mu (f32 [5120]) and rollout_segmeans_L35
(f32 [42, 5120], rows = metadata rollout_ids). Used for centring by hack-action-anchor/scripts/segment_projections_eu.py
and turn_paths_bf16.py.

    uv run --with torch python scripts/compute_mu.py
"""
import datetime, json
from pathlib import Path
import numpy as np, torch
from safetensors import safe_open
from safetensors.torch import save_file

REPO = Path(__file__).resolve().parents[1]; EXP = REPO/"experiments"
CAP = REPO/"data/captures/qwen35_27b_bf16_token_L35"; OUT = EXP/"probe-refit-bf16/artifacts/mu_bf16_L35.safetensors"
L = 35

man = json.loads((CAP/"manifest.json").read_text()); li = man["layers"].index(L)
ids = [t["task_id"] for t in man["transcripts"]]; assert len(ids) == 42, len(ids)
R, nseg = [], 0
for tid in ids:
    with safe_open(str(CAP/f"{tid}.safetensors"), "pt") as f: sm = f.get_tensor("segment_means")[:, li].double()
    R.append(sm.mean(0)); nseg += sm.shape[0]
R = torch.stack(R); mu = R.mean(0)
print(f"42 rollouts, {nseg} segments; |mu| = {float(mu.norm()):.3f}")
meta = {"recipe": "per rollout: mean over all judge segments of segment_means[:, L35] (no dedup, no EXCLUDED filter); "
                  "then equal-weight mean over the 42 hacked rollouts.",
        "capture": "data/captures/qwen35_27b_bf16_token_L35 (bf16 serving; segment_means F16)", "layer": str(L),
        "script": "scripts/compute_mu.py", "date": datetime.date.today().isoformat(),
        "rollout_ids": json.dumps(ids), "n_segments": str(nseg),
        "norm": f"{float(mu.norm()):.4f}"}
save_file({"mu": mu.float().contiguous(), "rollout_segmeans_L35": R.float().contiguous()}, str(OUT), metadata=meta)
print("wrote", OUT)
