"""CPU tests for steering/replay.py.

The renderer tests use a FakeLLM whose `_preprocess_chat_one` mimics vLLM's chat
preprocessing (reasoning → reasoning_content, tool-call argument JSON parsed) over the real
Qwen3.5-27B tokenizer/template, and run `build_turns` over a real n5 record when the
sandbox export is present: every turn's rendered prompt length must equal the token count
the arms server logged — the strongest renderer-fidelity check available without a GPU.
"""

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from steering_reasoning.steering import replay  # noqa: E402

SANDBOX = Path(__file__).resolve().parents[1] / "data" / "scratch" / "n5logs"
RECORDS = SANDBOX / "records_cn8_s42.jsonl"


def test_proxy_view_strips_think_tags_only_on_assistant():
    msgs = [{"role": "user", "content": "<think>keep</think>"},
            {"role": "assistant", "content": "x", "reasoning": "<think>\na\n</think>"},
            {"role": "assistant", "content": "y", "reasoning_content": "<think>b</think>"}]
    out = replay.proxy_view(msgs)
    assert out[0]["content"] == "<think>keep</think>"
    assert "<think>" not in out[1]["reasoning"] and "a" in out[1]["reasoning"]
    assert "<think>" not in out[2]["reasoning_content"]
    assert msgs[1]["reasoning"].startswith("<think>")  # input untouched


def test_clamp_check_flags_violations():
    u = torch.zeros(8); u[0] = 1.0
    rows = torch.zeros(4, 8); rows[:, 0] = torch.tensor([-9.0, -8.0, -7.9, -3.0])
    cs = replay.clamp_check(rows, u, threshold=-8.0)
    assert cs["n_rows"] == 4 and cs["n_violations"] == 2 and cs["max_proj"] == -3.0
    ok = replay.clamp_check(rows[:2], u, threshold=-8.0)
    assert ok["n_violations"] == 0 and ok["max_excess"] <= 0


def test_steering_spec(tmp_path: Path):
    save_file = pytest.importorskip("safetensors.torch").save_file
    art = tmp_path / "u.safetensors"
    save_file({"vector": torch.tensor([[3.0, 4.0]])}, str(art))
    art.with_suffix(".json").write_text(json.dumps({"layer_indices": [35]}))
    spec = replay.SteeringSpec.from_artifact(str(art), -8)
    assert spec.layer == 35 and spec.threshold == -8.0 and len(spec.sha) == 16
    # worker-identical: normalised after the bf16 cast, so equal only up to bf16 rounding
    assert torch.allclose(spec.unit(), torch.tensor([0.6, 0.8]), atol=4e-3)
    assert spec.unit().dtype == torch.float32
    pytest.importorskip("zstandard")
    (sv,) = json.loads(spec.payload([[0, 12], [30, 41]]))
    assert sv["steer_positions"] == [[0, 12], [30, 41]] and sv["threshold"] == -8.0 and sv["mode"] == "clamp"


def test_layer_chunk_for_respects_payload_cap():
    pytest.importorskip("psutil")
    # 81k-token prompt at d=5120: 3 GiB / (81k*5120*2) = 3.88 → 3 layers per pass
    assert replay.layer_chunk_for(81_000, 5120, 64, host_margin_gib=0) == 3
    assert replay.layer_chunk_for(1_000, 5120, 64, host_margin_gib=0) == 64
    assert replay.layer_chunk_for(1_000, 5120, 8, host_margin_gib=0) == 8


# ------------------------------------------------------------------ renderer (fake vLLM)

class FakeLLM:
    """vLLM's offline chat preprocessing, minus the engine: the same HF template applied to
    messages after chat_utils' normalisation (reasoning → reasoning_content; tool-call
    `arguments` JSON strings parsed to mappings)."""

    def __init__(self, tokenizer):
        self.tok = tokenizer

    def get_tokenizer(self):
        return self.tok

    @staticmethod
    def _normalise(messages):
        out = []
        for m in messages:
            m = dict(m)
            if m.get("role") == "assistant":
                if m.get("reasoning") and not m.get("reasoning_content"):
                    m["reasoning_content"] = m.pop("reasoning")
                if m.get("tool_calls"):
                    tcs = []
                    for tc in m["tool_calls"]:
                        tc = json.loads(json.dumps(tc))
                        a = tc["function"].get("arguments")
                        if isinstance(a, str):
                            tc["function"]["arguments"] = json.loads(a) if a else {}
                        tcs.append(tc)
                    m["tool_calls"] = tcs
            out.append(m)
        return out

    def _preprocess_chat_one(self, messages, tools=None, add_generation_prompt=True):
        text = self.tok.apply_chat_template(self._normalise(messages), tools=tools, tokenize=False,
                                            add_generation_prompt=add_generation_prompt)
        return {"type": "token", "prompt_token_ids": self.tok(text, add_special_tokens=False)["input_ids"],
                "prompt": text}


@pytest.fixture(scope="module")
def llm27():
    pytest.importorskip("jinja2")
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B", local_files_only=True)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Qwen/Qwen3.5-27B tokenizer not cached: {e}")
    return FakeLLM(tok)


@pytest.fixture(scope="module")
def n5_record():
    if not RECORDS.exists():
        pytest.skip(f"{RECORDS} not present (run scripts/export_eval_records.py)")
    for line in open(RECORDS):
        r = json.loads(line)
        if r["task_id"] == "lcbhard_13":
            return r
    pytest.skip("lcbhard_13 not in the sandbox records")


def test_build_turns_matches_logged_prompt_lengths(llm27, n5_record):
    turns, skipped = replay.build_turns(n5_record, llm27, llm27.get_tokenizer())
    # build_turns asserted len(prompt_ids) == server-logged prompt_tokens on every turn.
    # The branch was re-rendered (no token_ids in n5 logs): turns whose re-render does not
    # reproduce the logged completion length are left out, never written as exact.
    assert len(turns) + len(skipped) == len(n5_record["turns"])
    assert all(s["reason"] == "branch_delta" and s["delta"] != 0 for s in skipped)
    assert len(skipped) <= 2, skipped  # 1 of 57 on this rollout (a str_replace tool call)
    assert all(t.branch_delta == 0 and t.token_ids_source == "rendered" for t in turns)
    assert all(t.prompt_text.endswith(replay.THINK_OPEN) for t in turns)
    tok = llm27.get_tokenizer()
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    assert all(t.branch_ids[-1] == im_end for t in turns)
    with pytest.raises(RuntimeError, match="not the generated stream"):
        replay.build_turns(n5_record, llm27, llm27.get_tokenizer(), on_branch_mismatch="fail")
    with pytest.raises(RuntimeError, match="max_model_len"):
        replay.build_turns(n5_record, llm27, llm27.get_tokenizer(), max_model_len=2000)


def test_build_turns_rejects_render_affecting_request_keys(llm27, n5_record):
    rec = json.loads(json.dumps(n5_record))
    rec["turns"][0]["request"]["chat_template_kwargs"] = {"enable_thinking": False}
    with pytest.raises(RuntimeError, match="render-affecting"):
        replay.build_turns(rec, llm27, llm27.get_tokenizer())


def test_build_turns_rejects_length_mismatch(llm27, n5_record):
    rec = json.loads(json.dumps(n5_record))
    rec["turns"][0]["usage"]["prompt_tokens"] += 1
    with pytest.raises(RuntimeError, match="server logged"):
        replay.build_turns(rec, llm27, llm27.get_tokenizer())


def test_build_turns_uses_sidecar_ids_and_checks_prompt_hash(llm27, n5_record):
    from steering_reasoning.steering.clamp import ids_sha

    rec = json.loads(json.dumps(n5_record))
    tok = llm27.get_tokenizer()
    t0 = rec["turns"][0]
    msgs = replay.proxy_view(t0["request"]["messages"])
    ids, text = replay.render_prompt_ids(llm27, msgs, t0["request"]["tools"])
    # "the sidecar recorded what the model generated": here the template re-render of the reply,
    # which on this turn reproduces the logged completion length exactly
    branch, _ = replay.render_branch_ids(llm27, tok, msgs, t0["response"], t0["request"]["tools"], ids, text)
    assert len(branch) == t0["usage"]["completion_tokens"]
    t0["token_ids"] = list(branch)
    t0["prompt_ids_sha256"], t0["n_prompt_ids"] = ids_sha(ids), len(ids)
    t0["ids_join"] = "sidecar"
    turns, _ = replay.build_turns(rec, llm27, tok)
    assert turns[0].token_ids_source == "logged" and turns[0].branch_ids == branch
    assert turns[0].branch_delta == 0 and turns[0].ids_join == "sidecar"
    assert turns[1].token_ids_source == "rendered" and turns[1].ids_join == "none"
    # sidecar ids that do not decode to the logged reply (a join collision) are fatal
    t0["token_ids"] = [branch[0]] * len(branch)   # same length, another rollout's (garbage) reply
    with pytest.raises(RuntimeError, match="do not decode to the logged"):
        replay.build_turns(rec, llm27, tok)
    # a sidecar branch whose length disagrees with the server's count is fatal
    t0["token_ids"] = list(branch) + [11]
    with pytest.raises(RuntimeError, match="sidecar has"):
        replay.build_turns(rec, llm27, tok)
    t0["token_ids"] = list(branch)
    # a hash that does not match the render is fatal (byte identity failed)
    t0["prompt_ids_sha256"] = "deadbeef"
    with pytest.raises(RuntimeError, match="differ from the server"):
        replay.build_turns(rec, llm27, tok)


def test_build_turns_skips_synthetic(llm27, n5_record):
    rec = json.loads(json.dumps(n5_record))
    rec["turns"][3]["synthetic_length"] = True
    turns, skipped = replay.build_turns(rec, llm27, llm27.get_tokenizer())
    synth = [s for s in skipped if s["reason"] == "synthetic_length"]
    assert synth == [{"message_index": rec["turns"][3]["message_index"], "reason": "synthetic_length"}]
    assert all(s["reason"] in ("synthetic_length", "branch_delta") for s in skipped)
    assert len(turns) + len(skipped) == len(rec["turns"])


# --- steering modes and the model-tokens span mask ---

def test_steering_spec_modes(tmp_path: Path):
    save_file = pytest.importorskip("safetensors.torch").save_file
    pytest.importorskip("zstandard")
    art = tmp_path / "u.safetensors"
    save_file({"vector": torch.tensor([[3.0, 4.0]])}, str(art))
    art.with_suffix(".json").write_text(json.dumps({"layer_indices": [35]}))
    add = replay.SteeringSpec.from_artifact(str(art), -4, mode="add")
    assert add.mode == "add" and add.dose == -4.0 and add.threshold is None
    (sv,) = json.loads(add.payload([[2, 5], [9, 12]]))
    assert sv["mode"] == "add" and sv["scale"] == -4.0 and sv["steer_positions"] == [[2, 5], [9, 12]]
    clamp = replay.SteeringSpec.from_artifact(str(art), -8)
    (sv,) = json.loads(clamp.payload([[0, 3]]))
    assert sv["threshold"] == -8.0 and sv["scale"] == 1.0 and sv["steer_positions"] == [[0, 3]]
    with pytest.raises(ValueError):
        clamp.payload([])
    with pytest.raises(ValueError):
        replay.SteeringSpec.from_artifact(str(art), None, mode="add")
    abl = replay.SteeringSpec.from_artifact(str(art), None, mode="ablate")
    assert abl.dose is None and abl.threshold is None
    assert clamp.provenance()["mode"] == "clamp" and clamp.provenance()["dose"] == -8.0


def test_steered_positions_composes_spans_and_checks_recorded():
    # header = [10, 11], think = [12, 13], end = 99 — a synthetic token stream
    markers = ([10, 11], [12, 13], 99)
    prompt = [1, 2, 10, 11, 12, 13, 5, 6, 99, 3, 10, 11, 12, 13]   # one completed reply [6:8), then the generation prompt
    spans = replay.steered_positions(prompt, 4, markers, recorded=None)
    assert spans == [[6, 8], [14, 18]]
    assert replay.steered_positions(prompt, 4, markers, recorded=[[6, 8]]) == [[6, 8], [14, 18]]
    with pytest.raises(RuntimeError, match="differ from the spans the proxy sent"):
        replay.steered_positions(prompt, 4, markers, recorded=[[6, 9]])
    # a turn with no generated tokens has nothing to steer (an empty span list would skip the prefill)
    with pytest.raises(RuntimeError, match="no generated tokens"):
        replay.steered_positions(prompt, 0, markers, recorded=[[6, 8]])
    # first turn: no earlier reply, only the branch
    assert replay.steered_positions([1, 2, 10, 11, 12, 13], 3, markers, recorded=[]) == [[6, 9]]
    # the proxy sent no positions on this turn (generation's prefill unsteered): branch only
    assert replay.steered_positions(prompt, 4, markers, recorded=None, prefill_steered=False) == [[14, 18]]
    with pytest.raises(RuntimeError, match="recorded spans"):
        replay.steered_positions(prompt, 4, markers, recorded=[[6, 8]], prefill_steered=False)


def test_span_markers_match_the_proxy(llm27):
    """The replay recomputes assistant spans on the rendered prompt with the same markers the
    proxy tokenizes through /tokenize; on a real multi-turn render both agree with the
    template's structure (one span per completed assistant message, reasoning included
    only for turns after the last user message)."""
    tok = llm27.get_tokenizer()
    hdr, think, end = replay.span_markers(tok)
    assert tok.decode(hdr) == "<|im_start|>assistant\n" and tok.decode([end]) == "<|im_end|>"
    msgs = [{"role": "user", "content": "a"},
            {"role": "assistant", "content": "b", "reasoning": "think-b"},
            {"role": "user", "content": "c"},
            {"role": "assistant", "content": "d", "reasoning": "think-d"},
            {"role": "user", "content": "e"}]
    ids, text = replay.render_prompt_ids(llm27, msgs, None)
    spans = replay.steered_positions(ids, 3, (hdr, think, end), recorded=None)
    assert len(spans) == 3 and spans[-1] == [len(ids), len(ids) + 3]
    for a, b in spans[:-1]:
        piece = tok.decode(ids[a:b])
        assert "<|im_start|>" not in piece and "<|im_end|>" not in piece and not piece.startswith("<think>\n")
    assert tok.decode(ids[spans[0][0]:spans[0][1]]).strip() == "b"        # reasoning dropped before the last user turn
    assert "d" in tok.decode(ids[spans[1][0]:spans[1][1]])


def test_build_turns_carries_recorded_spans(llm27, n5_record):
    rec = json.loads(json.dumps(n5_record))
    rec["turns"][0]["steer_positions"] = []
    rec["turns"][1]["steer_positions"] = [[5, 9]]
    turns, _ = replay.build_turns(rec, llm27, llm27.get_tokenizer())
    assert turns[0].steer_positions == [] and turns[1].steer_positions == [[5, 9]]
    assert turns[2].steer_positions is None
