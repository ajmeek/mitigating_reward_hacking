"""DiffMean steering-vector computation and artifact I/O.

Artifact format: a .safetensors file holding tensor "vector" of shape
(n_layers, d_model) float32, with a JSON metadata string under key "meta" in the
safetensors header (also written as a .json sidecar for humans). Layer indices live in the
metadata; the tensor row order matches them.

Loading does NOT require vllm-lens (the eval env can inspect artifacts); converting to a
vllm_lens.SteeringVector imports it lazily.
"""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


class RunningMean:
    """Streaming mean of (n_layers, d_model) activations over tokens."""

    def __init__(self):
        self.total: torch.Tensor | None = None
        self.count: int = 0

    def update(self, acts: torch.Tensor) -> None:
        """acts: (n_layers, n_tokens, d_model) — accumulates over the token dim."""
        contrib = acts.float().sum(dim=1)
        self.total = contrib if self.total is None else self.total + contrib
        self.count += acts.shape[1]

    def mean(self) -> torch.Tensor:
        if self.total is None or self.count == 0:
            raise ValueError("no activations accumulated")
        return self.total / self.count


def diffmean(pos_means: list[torch.Tensor], neg_means: list[torch.Tensor]) -> torch.Tensor:
    """Mean-of-means difference: mean over positive samples minus mean over negatives.

    Each element is (n_layers, d_model) — typically one per transcript, so every
    transcript gets equal weight regardless of its token count.
    """
    if not pos_means or not neg_means:
        raise ValueError(f"need both classes: {len(pos_means)} pos, {len(neg_means)} neg")
    pos = torch.stack(pos_means).mean(dim=0)
    neg = torch.stack(neg_means).mean(dim=0)
    return pos - neg


def save_artifact(path: str | Path, vector: torch.Tensor, layer_indices: list[int], meta: dict) -> None:
    if vector.shape[0] != len(layer_indices):
        raise ValueError(
            f"vector has {vector.shape[0]} layer rows but {len(layer_indices)} layer_indices "
            "— a requested capture layer probably exceeds the model's layer count"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    full_meta = {
        **meta,
        "layer_indices": layer_indices,
        "shape": list(vector.shape),
        "layer_norms": {str(li): round(float(vector[i].norm()), 4) for i, li in enumerate(layer_indices)},
    }
    save_file(
        {"vector": vector.float().contiguous()},
        str(path),
        metadata={"meta": json.dumps(full_meta)},
    )
    path.with_suffix(".json").write_text(json.dumps(full_meta, indent=2) + "\n")


def load_artifact(path: str | Path) -> tuple[torch.Tensor, list[int], dict]:
    path = Path(path)
    tensors = load_file(str(path))
    meta = json.loads(path.with_suffix(".json").read_text())
    return tensors["vector"], meta["layer_indices"], meta


def to_steering_vector(
    path: str | Path,
    *,
    layers: list[int] | None = None,
    scale: float = 1.0,
    norm_match: bool = False,
):
    """Load an artifact as a vllm_lens.SteeringVector (optionally a subset of layers)."""
    from vllm_lens import SteeringVector

    vector, layer_indices, _ = load_artifact(path)
    if layers is not None:
        rows = [layer_indices.index(li) for li in layers]
        vector = vector[rows]
        layer_indices = layers
    return SteeringVector(
        activations=vector,
        layer_indices=layer_indices,
        scale=scale,
        norm_match=norm_match,
    )
