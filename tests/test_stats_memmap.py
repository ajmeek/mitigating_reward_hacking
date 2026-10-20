"""Tests for stats.wilson / auroc_ranks and activations.memmap_tensor."""

import math

import numpy as np
import pytest

from steering_reasoning.activations import memmap_tensor, safetensors_header
from steering_reasoning.stats import Z90, auroc_ranks, wilson


def test_wilson_matches_closed_form_and_clamps():
    lo, hi = wilson(3, 10)
    p, z, n = 0.3, Z90, 10
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    assert lo == pytest.approx(c - h) and hi == pytest.approx(c + h)
    assert wilson(0, 10)[0] == 0.0 and wilson(10, 10)[1] == 1.0
    assert all(math.isnan(v) for v in wilson(0, 0))


def test_auroc_ranks_matches_pairwise_count():
    rng = np.random.default_rng(0)
    pos = np.round(rng.normal(1, 1, 60), 1)   # rounding makes ties
    neg = np.round(rng.normal(0, 1, 80), 1)
    pairwise = ((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (60 * 80)
    assert auroc_ranks(pos, neg) == pytest.approx(pairwise)
    assert math.isnan(auroc_ranks([], [1.0]))


def test_memmap_tensor_reads_safetensors(tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    x = torch.arange(24, dtype=torch.float16).reshape(4, 2, 3)
    y = torch.arange(4, dtype=torch.int32)
    p = tmp_path / "t.safetensors"
    save_file({"x": x, "y": y}, str(p))
    hdr, base = safetensors_header(p)
    mx = memmap_tensor(p, "x", hdr, base, dtype_expect="F16")
    assert mx.shape == (4, 2, 3) and np.array_equal(np.asarray(mx), x.numpy())
    assert np.array_equal(np.asarray(memmap_tensor(p, "y")), y.numpy())
    with pytest.raises(AssertionError):
        memmap_tensor(p, "x", dtype_expect="F32")
