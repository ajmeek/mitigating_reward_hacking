"""Tests for the steering payload and the sidecar join keys (steering/clamp.py)."""

import base64
import json
from pathlib import Path

import numpy as np
import pytest

from steering_reasoning.steering.clamp import build_clamp_payload, build_steer_payload, ids_sha, proxy_view, request_sha


def test_request_sha_is_canonical():
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "x", "reasoning": "r"}]
    tools = [{"type": "function", "function": {"name": "bash", "strict": True}}]
    a = request_sha(msgs, tools)
    # key order and whitespace do not matter; content does
    b = request_sha([{"content": "hi", "role": "user"}, {"reasoning": "r", "content": "x", "role": "assistant"}],
                    [{"function": {"strict": True, "name": "bash"}, "type": "function"}])
    assert a == b and len(a) == 24
    assert request_sha(msgs, None) == request_sha(msgs, [])
    assert request_sha(msgs + [{"role": "tool", "content": "o", "tool_call_id": "1"}], tools) != a
    assert ids_sha([1, 2, 3]) != ids_sha([1, 2]) and len(ids_sha([1])) == 24


def test_proxy_view_matches_what_the_proxy_hashes():
    # the proxy hashes messages AFTER strip_think; the exporter must hash proxy_view(pre-proxy)
    msgs = [{"role": "assistant", "content": "x", "reasoning": "<think>\nr\n</think>"}]
    assert request_sha(proxy_view(msgs), None) == request_sha([{"role": "assistant", "content": "x", "reasoning": "r"}], None)


def _artifact(tmp_path: Path, vec: np.ndarray, layer: int) -> str:
    save_file = pytest.importorskip("safetensors.numpy").save_file
    p = tmp_path / "v.safetensors"
    save_file({"vector": vec}, str(p))
    p.with_suffix(".json").write_text(json.dumps({"layer_indices": [layer]}))
    return str(p)


def test_build_clamp_payload(tmp_path: Path):
    zstd = pytest.importorskip("zstandard")
    vec = np.arange(8, dtype=np.float32).reshape(1, 8)
    art = _artifact(tmp_path, vec, 35)

    (sv,) = json.loads(build_clamp_payload(art, -8.0))
    assert sv["mode"] == "clamp" and sv["threshold"] == -8.0 and sv["layer_indices"] == [35]
    assert sv["scale"] == 1.0 and sv["position_indices"] is None and sv["steer_positions"] is None
    raw = zstd.ZstdDecompressor().decompress(base64.b64decode(sv["activations"]["data"]))
    assert np.array_equal(np.frombuffer(raw, np.float32).reshape(sv["activations"]["shape"]), vec)
    (sv,) = json.loads(build_steer_payload(art, threshold=-5.0, steer_positions=[[3, 9], [12, 40]]))
    assert sv["steer_positions"] == [[3, 9], [12, 40]]
