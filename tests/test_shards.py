"""Tests for the per-token dump writers (steering/shards.py) against both readers.

`create_acts_shard` hand-writes a safetensors header and hands back a memmap over the bf16
region; the reference reader (safetensors.torch.load_file) and our zero-copy reader
(token_dump.tensors_from_bytes) must both see exactly what was written. Odd N matters: the
i8 kind array precedes the i32 arrays, so the i32 offsets are then unaligned and the
zero-copy reader has to copy them. d_model is a parameter, not the 27B's 5120.
"""

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from safetensors.torch import load_file  # noqa: E402

from steering_reasoning.steering.shards import (  # noqa: E402
    create_acts_shard,
    index_tensors,
    write_meta,
    write_segmeans,
    write_tokens,
)
from steering_reasoning.steering.token_dump import tensors_from_bytes  # noqa: E402


def _tree(n: int) -> dict:
    return {
        "kinds": [i % 8 for i in range(n)],
        "msgs": [i // 3 for i in range(n)],
        "turn_of": [i // 5 for i in range(n)],
        "ctxpos": list(range(n)),
    }


@pytest.mark.parametrize("n,d", [(7, 16), (8, 24), (1, 8)])
def test_acts_shard_round_trips_through_both_readers(tmp_path: Path, n: int, d: int) -> None:
    idx = index_tensors(_tree(n))
    path = tmp_path / "L03" / "r.safetensors"
    mm = create_acts_shard(path, n, d, idx)
    assert mm.shape == (n, d) and mm.dtype == np.uint16

    acts = torch.randn(n, d).bfloat16()
    mm[:] = acts.view(torch.int16).numpy().view(np.uint16)
    mm.flush()
    del mm

    ref = load_file(str(path))
    assert torch.equal(ref["token_acts"], acts) and ref["token_acts"].dtype == torch.bfloat16
    for k, t in idx.items():
        assert torch.equal(ref[k], t) and ref[k].dtype == t.dtype

    ours = tensors_from_bytes(bytearray(path.read_bytes()))
    assert set(ours) == set(ref)
    for k in ref:
        assert torch.equal(ours[k], ref[k]) and ours[k].dtype == ref[k].dtype


def test_acts_shard_data_region_is_8_byte_aligned(tmp_path: Path) -> None:
    path = tmp_path / "r.safetensors"
    create_acts_shard(path, 3, 8, index_tensors(_tree(3)))
    raw = path.read_bytes()
    header_len = int.from_bytes(raw[:8], "little")
    assert (8 + header_len) % 8 == 0
    header = json.loads(raw[8 : 8 + header_len])
    assert header["token_acts"]["dtype"] == "BF16" and header["token_acts"]["shape"] == [3, 8]
    assert header["token_acts"]["data_offsets"][0] == 0  # acts lead the data block


def test_small_writers_round_trip(tmp_path: Path) -> None:
    n, L, d = 5, 2, 8
    idx = index_tensors(_tree(n))
    p = write_tokens(tmp_path, "rid", list(range(100, 100 + n)), idx)
    t = load_file(str(p))
    assert t["token_ids"].tolist() == list(range(100, 100 + n)) and t["token_ids"].dtype == torch.int32
    assert torch.equal(t["token_kind"], idx["token_kind"])

    tm = torch.randn(2, L, d)
    sm = torch.randn(3, L, d) * 1e5  # exceeds f16 range → must be clamped, not inf
    p = write_segmeans(tmp_path, "rid", tm, [10, 20], sm, [1, 2, 3])
    s = load_file(str(p))
    assert torch.equal(s["turn_means"], tm) and s["turn_means"].dtype == torch.float32
    assert s["segment_means"].dtype == torch.float16 and torch.isfinite(s["segment_means"]).all()
    assert s["turn_tokens"].tolist() == [10, 20] and s["segment_tokens"].tolist() == [1, 2, 3]

    p = write_meta(tmp_path, "rid", {"task_id": "rid", "label": True}, {"turns": [{"turn_row": 0}]})
    m = json.loads(p.read_text())
    assert m["task_id"] == "rid" and m["label"] is True and m["turns"] == [{"turn_row": 0}]
    assert p == tmp_path / "meta" / "rid.json"
