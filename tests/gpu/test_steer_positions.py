"""GPU test: the positions-aware gate reproduces model-tokens steering inside one prefill.

Runs on a small Qwen3.5 model in the serving env with vllm-lens patched by
serving/patch_vllm_lens.sh (modes + the positions-aware gate); the gate env var is read by the
workers at spawn:

    cd serving && VLLM_LENS_STEER_DECODE_ONLY=1 uv run pytest ../tests/gpu/test_steer_positions.py -x -q

One engine, one prompt of ids = context + "earlier reply" span + branch, captures at the
steering layer:
  A  unsteered
  B  clamp with steer_positions = [[a, b], [gs, n]] → rows in the spans satisfy the clamp
     invariant and moved; every other row bit-identical to A (prompt/tool rows untouched)
  C  clamp with steer_positions = [[a, b]] only (what a generation PREFILL of this turn
     steered) → the earlier reply's rows clamped, the branch untouched
And the offline guard refuses the payload forms that would silently capture unsteered rows
under this gate (no positions; empty positions).
"""

import json
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("vllm_lens")

from steering_reasoning.steering.activations import ActivationCapturer, apply_no_nvcc_workarounds  # noqa: E402
from steering_reasoning.steering.clamp import build_clamp_payload, build_steer_payload  # noqa: E402
from steering_reasoning.steering.replay import CLAMP_TOL, SteeringSpec, clamp_check, span_markers, steered_positions  # noqa: E402
from steering_reasoning.steering.shards import worker_patch_markers  # noqa: E402

MODEL = os.environ.get("REPLAY_TEST_MODEL", "Qwen/Qwen3.5-2B")
CHUNK = int(os.environ.get("REPLAY_TEST_CHUNK", "0") or 0)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.fixture(scope="module")
def capturer():
    markers = worker_patch_markers()
    if not (markers["modes"] and markers["seq_lens"] and markers["positions_gate"]):
        pytest.skip(f"vllm-lens not patched (modes/seq_lens/positions_gate): {markers}")
    if os.environ.get("VLLM_LENS_STEER_DECODE_ONLY") != "1":
        pytest.skip("set VLLM_LENS_STEER_DECODE_ONLY=1 for the steer_positions tests")
    apply_no_nvcc_workarounds()
    kw = dict(enable_prefix_caching=False, enable_chunked_prefill=bool(CHUNK), max_num_batched_tokens=CHUNK or 2048)
    return ActivationCapturer(MODEL, max_model_len=2048, gpu_memory_utilization=0.55, **kw)


@pytest.fixture(scope="module")
def prompt(capturer):
    """ids of a two-turn chat rendered by the real template + a branch; the earlier reply's
    span comes from the same marker logic the proxy and the replay use."""
    tok = capturer.llm.get_tokenizer()
    msgs = [{"role": "system", "content": "You are a careful engineer. " * 8},
            {"role": "user", "content": "Count perfect powers up to n. " * 4},
            {"role": "assistant", "content": "I will iterate over exponents b >= 2 and bases. " * 3},
            {"role": "user", "content": "Now handle duplicates such as 16 = 2^4 = 4^2. " * 3}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    branch = tok("Let me think about a^b with b >= 2 and a set to dedupe. " * 4, add_special_tokens=False)["input_ids"]
    spans = steered_positions(ids, len(branch), span_markers(tok), recorded=None)
    assert len(spans) == 2 and spans[1] == [len(ids), len(ids) + len(branch)], spans
    if CHUNK:
        assert len(ids) > 2 * CHUNK and len(branch) > CHUNK
    return ids + branch, spans


@pytest.fixture(scope="module")
def artifact(tmp_path_factory, capturer):
    from safetensors.torch import save_file

    layer = capturer.n_layers // 2
    g = torch.Generator().manual_seed(0)
    p = tmp_path_factory.mktemp("art") / "u.safetensors"
    save_file({"vector": torch.randn(1, capturer.d_model, generator=g)}, str(p))
    p.with_suffix(".json").write_text(json.dumps({"layer_indices": [layer]}))
    return str(p), layer, SteeringSpec.from_artifact(str(p), 0.0).unit()


def _mask(n, spans):
    m = torch.zeros(n, dtype=torch.bool)
    for a, b in spans:
        m[a:b] = True
    return m


def test_steer_positions_mask(capturer, prompt, artifact):
    ids, spans = prompt
    art, layer, u = artifact
    n = len(ids)
    inside = _mask(n, spans)
    A = capturer.capture([ids], [layer], token_ids=True)[0][0].float()
    proj_a = A @ u
    thr = float(proj_a[inside].min()) - 2.0  # every steered row binds
    spec = SteeringSpec.from_artifact(art, thr)
    B = capturer.capture([ids], [layer], token_ids=True, steering=spec.payload(spans))[0][0].float()
    C = capturer.capture([ids], [layer], token_ids=True, steering=spec.payload(spans[:1]))[0][0].float()

    assert torch.equal(A[~inside], B[~inside]), "rows outside steer_positions changed"
    cs = clamp_check(B[inside], u, thr)
    assert cs["n_violations"] == 0, cs
    rel = float(((B[inside] @ u) - thr).clamp(min=0).max() / B[inside].norm(dim=1).max())
    print(f"\nclamp excess max = {rel:.2e} × ‖h‖ (CLAMP_TOL={CLAMP_TOL:.2e}); chunk={CHUNK}; spans={spans}")
    assert rel < CLAMP_TOL / 4
    assert (proj_a[inside] > thr).all() and not torch.equal(A[inside], B[inside])
    delta = A[inside] - B[inside]
    along = delta @ u
    ortho = delta - along[:, None] * u[None, :]
    assert (ortho.norm(dim=1) <= CLAMP_TOL * A[inside].norm(dim=1) * 4).all()
    # the earlier reply's span alone (what a generation prefill steered) also binds on its own
    a, b = spans[0]
    assert clamp_check(B[a:b], u, thr)["n_violations"] == 0 and not torch.equal(A[a:b], B[a:b])
    # only the earlier reply's span (a generation prefill): those rows clamped, the branch untouched
    only = _mask(n, spans[:1])
    assert torch.equal(A[~only], C[~only]) and torch.equal(B[only], C[only])


def test_guard_refuses_unsteerable_payloads_under_the_gate(capturer, artifact):
    art, _, _ = artifact
    with pytest.raises(RuntimeError, match="without steer_positions"):
        capturer.steering_objects(build_clamp_payload(art, 0.0))
    with pytest.raises(RuntimeError, match="without steer_positions"):
        capturer.steering_objects(build_steer_payload(art, threshold=0.0, steer_positions=[]))
    assert len(capturer.steering_objects(build_steer_payload(art, threshold=0.0, steer_positions=[[0, 2]]))) == 1
