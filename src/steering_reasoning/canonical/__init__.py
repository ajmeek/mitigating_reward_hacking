"""Canonical vectors and pointers for the steering setup. See INDEX.md in this folder.

    from steering_reasoning.canonical import load, a_b_hat, COS_AB, paths
    u = load("u")          # the steering vector (unit, L35)
    e = load("e")          # approach-to-action axis (unit, L35)
    s = load("s")          # shared hack/anti-hack axis (unit, L35)
    a_hat, b_hat = a_b_hat()

torch/safetensors are imported lazily so the package imports from the eval env.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
LAYER = 35
VECTORS = HERE / "vectors_L35.safetensors"
META = json.loads((HERE / "vectors_L35.meta.json").read_text())
COS_AB = float(META["cos_ab"])

# Large or many-file objects are not copied here; these are their canonical locations.
paths = {
    "splits": REPO / "src/steering_reasoning/data/splits.json",   # impossible.{train,val,test}, original.test, test_groups
    "token_capture_bf16": REPO / "data/captures/qwen35_27b_bf16_token_L31_35",   # pulled from S3; see INDEX.md
    "decline_labels": REPO / "experiments/anti-hack-labeling/artifacts/labels_full.json",
    "decline_kinds": REPO / "experiments/anti-hack-labeling/artifacts/decline_unique.json",
    "action_anchor_table": REPO / "experiments/hack-action-anchor/artifacts/action_anchor_table.csv",
    "token_series": REPO / "data/derived/token_series.parquet",
    "e_variants": REPO / "experiments/wrapup-axis-paperstyle/artifacts/e_paper_bf16_L35.safetensors",
    "directions_all_layers_bf16": REPO / "experiments/probe-refit-bf16/artifacts/directions_bf16_all_layers.safetensors",
}


def load(name: str):
    """Return the canonical vector `name` in {'u','s','e'} as a float32 torch tensor (5120,).

    Without torch on the path (the eval / figure environment) the same vector is returned as a float32
    numpy array, so the figure scripts, which only take dot products, run from a plain `uv sync`.
    """
    from safetensors import safe_open
    try:
        import torch  # noqa: F401
        fw, cast = "pt", lambda t: t.float().flatten()
    except ModuleNotFoundError:
        import numpy as np
        fw, cast = "np", lambda t: t.astype(np.float32).reshape(-1)
    with safe_open(str(VECTORS), fw) as f:
        if name not in f.keys():
            raise KeyError(f"{name!r} not in {VECTORS.name}; have {list(f.keys())}")
        return cast(f.get_tensor(name))


def a_b_hat():
    """Reconstruct the unit hack pole a_hat and anti-hack pole b_hat from s, u and cos(a, b)."""
    s, u = load("s"), load("u")
    ns, nd = (2 + 2 * COS_AB) ** 0.5, (2 - 2 * COS_AB) ** 0.5
    return (s * ns + u * nd) / 2, (s * ns - u * nd) / 2


def u_perp_e():
    """u with its e component projected out (the y-axis of the paper's action-plane figures), unit."""
    u, e = load("u"), load("e")
    v = u - float(u @ e) * e
    return v / float(v @ v) ** 0.5
