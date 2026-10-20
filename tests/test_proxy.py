"""Unit tests for the capping proxy's pure logic.

parse_overflow is the load-bearing test: it regex-parses vLLM's 400 wording, which has
changed across versions and once silently broke (aborted 7/8 Phase-3 arms, PAPERCUTS.md
2026-08-25). When bumping vLLM, capture the new overflow message and add it here.
"""

import json

import pytest

from steering_reasoning.evals.config import MAX_TOKENS
from steering_reasoning.proxy import CTX_MARGIN, normalize_tool_args, parse_overflow, skip_history_spans, stamp_max_tokens
from steering_reasoning.steering import assistant_spans, history_boundary


def err(msg: str) -> bytes:
    return json.dumps({"error": {"message": msg, "type": "BadRequestError", "code": 400}}).encode()


# --- parse_overflow: one case per phrasing vLLM has been observed to use ---

def test_overflow_vllm_024_wording():
    # Captured verbatim from PAPERCUTS.md:70 (vLLM 0.24, the wording whose silent
    # regex mismatch aborted 7/8 Phase-3 arms on 2026-08-25).
    body = err("This model's maximum context length is 131072 tokens. However, you requested "
               "16000 output tokens and your prompt contains at least 120000 input tokens.")
    assert parse_overflow(body) == 131072 - 120000 - CTX_MARGIN


def test_overflow_tokens_in_the_messages_phrasing():
    body = err("This model's maximum context length is 131072 tokens. However, you requested "
               "136000 tokens (120000 tokens in the messages, 16000 in the completion).")
    assert parse_overflow(body) == 131072 - 120000 - CTX_MARGIN


def test_overflow_parenthesized_messages_phrasing():
    body = err("maximum context length is 131072 tokens. However, you requested 136000 tokens "
               "(120000 in the messages, 16000 in the completion).")
    assert parse_overflow(body) == 131072 - 120000 - CTX_MARGIN


def test_overflow_vllm_024_minimal_bound_wording():
    # Captured live 2026-08-27 (32k window): vLLM reports "at least 16385" — the
    # overflow threshold + 1, NOT the true prompt size (21421). parse_overflow must
    # still DETECT this as an overflow; the proxy must never use the number to
    # recompute max_tokens (that retry 400s again — the bug this run caught).
    body = err("This model's maximum context length is 32768 tokens. However, you requested "
               "16384 output tokens and your prompt contains at least 16385 input tokens, "
               "for a total of at least 32769 tokens. Please reduce the length of the input "
               "prompt or the number of requested output tokens. "
               "(parameter=input_tokens, value=16385)")
    assert parse_overflow(body) is not None


def test_overflow_window_exhausted_clamps_to_zero():
    body = err("maximum context length is 131072 tokens. However, your request "
               "prompt contains at least 131050 input tokens.")
    assert parse_overflow(body) == 0


def test_overflow_unrecognized_wording_returns_none():
    # The 2026-08-25 failure mode: a new phrasing must surface as None (no retry),
    # never as a bogus number.
    assert parse_overflow(err("input length exceeds the model context window")) is None
    assert parse_overflow(b"not json at all") is None


def test_overflow_openrouter_wording_returns_none():
    # Real OpenRouter phrasing (seen in the MisTrace deepseek_r1 records): no
    # phrase the parser knows, so it must decline to retry rather than misparse.
    body = err("This endpoint's maximum context length is 64000 tokens. However, you "
               "requested about 65315 tokens (64526 of text input, 789 of tool input).")
    assert parse_overflow(body) is None


def test_overflow_non_json_body_with_known_wording():
    body = (b"maximum context length is 131072 tokens; prompt contains at least "
            b"100000 input tokens")
    assert parse_overflow(body) == 131072 - 100000 - CTX_MARGIN


# --- stamp_max_tokens ---

def test_stamp_caps_unset():
    data = {}
    stamp_max_tokens(data)
    assert data["max_tokens"] == MAX_TOKENS


def test_stamp_caps_oversized_and_normalizes_completion_tokens():
    data = {"max_completion_tokens": 100000}
    stamp_max_tokens(data)
    assert data["max_tokens"] == MAX_TOKENS
    assert "max_completion_tokens" not in data


def test_stamp_keeps_smaller_client_value():
    data = {"max_tokens": 512}
    stamp_max_tokens(data)
    assert data["max_tokens"] == 512


def test_parse_qwen_tool_calls_keeps_optional_anyof_params():
    """The qwen3_coder XML the model writes, parsed by the proxy's own parser: optional
    parameters typed anyOf[string,null] survive (vLLM 0.24's strict grammar dropped new_str), integers are cast, strings keep inner newlines."""
    from steering_reasoning.proxy import parse_qwen_tool_calls
    tools = [{"type": "function", "function": {"name": "text_editor", "parameters": {"type": "object", "properties": {
        "command": {"type": "string"}, "path": {"type": "string"},
        "new_str": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "old_str": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "insert_line": {"anyOf": [{"type": "integer"}, {"type": "null"}]}}}}},
             {"type": "function", "function": {"name": "bash", "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}}}}]
    text = ("</think>\n\n<tool_call>\n<function=text_editor>\n<parameter=command>\nstr_replace\n</parameter>\n"
            "<parameter=new_str>\nx = 1\n    return x\n</parameter>\n<parameter=old_str>\nraise NotImplementedError\n</parameter>\n"
            "<parameter=path>\n/func.py\n</parameter>\n<parameter=insert_line>\n3\n</parameter>\n</function>\n</tool_call>\n"
            "<tool_call>\n<function=bash>\n<parameter=cmd>\npython test.py\n</parameter>\n</function>\n</tool_call>")
    calls = parse_qwen_tool_calls(text, tools)
    assert [c["function"]["name"] for c in calls] == ["text_editor", "bash"]
    args = json.loads(calls[0]["function"]["arguments"])
    assert args == {"command": "str_replace", "new_str": "x = 1\n    return x", "old_str": "raise NotImplementedError",
                    "path": "/func.py", "insert_line": 3}
    assert json.loads(calls[1]["function"]["arguments"]) == {"cmd": "python test.py"}


TOOLS = [{"type": "function", "function": {"name": "bash", "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}}},
         {"type": "function", "function": {"name": "text_editor", "parameters": {"type": "object", "properties": {"command": {}, "path": {}, "old_str": {}, "new_str": {}}, "required": ["command", "path"]}}}]


def _resp(name, args):
    return {"choices": [{"message": {"tool_calls": [{"function": {"name": name, "arguments": json.dumps(args)}}]}}]}


def test_normalize_renames_bash_command_to_cmd():
    r = _resp("bash", {"command": "ls"})
    assert normalize_tool_args(r, TOOLS) == 1
    assert json.loads(r["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]) == {"cmd": "ls"}


def test_normalize_leaves_correct_and_ambiguous_calls_alone():
    r = _resp("bash", {"cmd": "ls"}); assert normalize_tool_args(r, TOOLS) == 0
    r = _resp("bash", {"command": "ls", "path": "/"}); assert normalize_tool_args(r, TOOLS) == 0     # two extras: ambiguous
    r = _resp("text_editor", {"command": "view", "path": "/f.py"}); assert normalize_tool_args(r, TOOLS) == 0
    r = _resp("text_editor", {"cmd": "view", "path": "/f.py"}); assert normalize_tool_args(r, TOOLS) == 1
    assert json.loads(r["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])["command"] == "view"
    assert normalize_tool_args(_resp("unknown_tool", {"x": 1}), TOOLS) == 0
    assert normalize_tool_args(_resp("bash", {"command": "ls"}), None) == 0


# --- history table: the recorded history is never steered (experiments/prefill-action-clamp, turn-start arms)


def test_skip_history_first_request_leaves_no_prefill_positions():
    # first request of a resumed rollout: every assistant span lies inside the rendered history
    kept, dropped = skip_history_spans([[10, 20], [30, 40]], prefix_len=45)
    assert kept == [] and dropped == 2


def test_skip_history_keeps_exactly_the_post_cut_replies():
    kept, dropped = skip_history_spans([[10, 20], [30, 40], [50, 60], [70, 80]], prefix_len=45)
    assert kept == [[50, 60], [70, 80]] and dropped == 2


def test_skip_history_is_unaffected_by_an_empty_history_reply():
    # an assistant message with an empty body renders no span at all; position-based skipping does not care
    kept, dropped = skip_history_spans([[10, 20], [50, 60]], prefix_len=45)
    assert kept == [[50, 60]] and dropped == 1


def test_skip_history_refuses_a_span_straddling_the_boundary():
    import pytest
    with pytest.raises(ValueError):
        skip_history_spans([[40, 50]], prefix_len=45)


HDR, THINK, END = [1, 2], [3, 4], 9


def _render(bodies):
    """[body tokens per assistant message] -> prompt ids with a user filler between turns and a generation prompt."""
    ids = [100]
    for b in bodies:
        ids += HDR + THINK + b + [END] + [60]
    return ids + HDR + THINK


def test_history_boundary_first_request_is_the_generation_prompt():
    ids = _render([[50, 51], [52]])
    assert history_boundary(ids, HDR, 2) == len(ids) - len(HDR) - len(THINK)


def test_history_boundary_counts_headers_not_spans():
    # the second history reply renders empty (no span), the third message is generated after the cut
    ids = _render([[50, 51], [], [70, 71]])
    b = history_boundary(ids, HDR, 2)
    spans = assistant_spans(ids, HDR, THINK, END)
    assert len(spans) == 2                       # the empty body yields no span
    kept, dropped = skip_history_spans(spans, b)
    assert kept == [[b + len(HDR) + len(THINK), b + len(HDR) + len(THINK) + 2]] and dropped == 1


def test_history_boundary_refuses_too_few_headers():
    import pytest
    with pytest.raises(ValueError):
        history_boundary(_render([[50]]), HDR, 5)

# --- ids sidecar (--ids-log): exact token ids per turn, for later activation replay ---

from steering_reasoning.proxy import assemble_stream, ids_sidecar_line  # noqa: E402
from steering_reasoning.steering.clamp import ids_sha  # noqa: E402


def _chunks(prompt_ids, gen_ids, text="hi"):
    """SSE chunks the way vLLM emits them with return_token_ids: prompt ids on the first chunk,
    per-chunk delta ids on the choice, usage on the last."""
    return [
        {"id": "c1", "created": 5, "model": "m", "prompt_token_ids": prompt_ids,
         "choices": [{"index": 0, "delta": {"role": "assistant", "content": text[:1]}, "token_ids": gen_ids[:1]}]},
        {"id": "c1", "choices": [{"index": 0, "delta": {"content": text[1:]}, "token_ids": gen_ids[1:], "finish_reason": "stop"}]},
        {"id": "c1", "choices": [], "usage": {"prompt_tokens": len(prompt_ids), "completion_tokens": len(gen_ids), "total_tokens": 0}},
    ]


def test_assemble_stream_carries_token_ids():
    out = assemble_stream(_chunks([1, 2, 3], [7, 8, 9]), "m")
    assert out["prompt_token_ids"] == [1, 2, 3]
    assert out["choices"][0]["token_ids"] == [7, 8, 9]
    assert out["choices"][0]["message"]["content"] == "hi"
    # without the flag nothing is added
    plain = assemble_stream([{"id": "c", "choices": [{"index": 0, "delta": {"content": "x"}, "finish_reason": "stop"}]}], "m")
    assert "prompt_token_ids" not in plain and "token_ids" not in plain["choices"][0]


def test_ids_sidecar_line_pops_ids_and_records_the_turn():
    resp = assemble_stream(_chunks([1, 2, 3], [7, 8, 9]), "m")
    line = ids_sidecar_line(resp, "key", [[4, 9]], n_tool_args_renamed=1, seed=42)
    assert line["request_sha"] == "key" and line["seed"] == 42
    assert line["token_ids"] == [7, 8, 9] and line["completion_tokens"] == 3 and line["finish_reason"] == "stop"
    assert line["n_prompt_ids"] == 3 and line["prompt_ids_sha256"] == ids_sha([1, 2, 3])
    assert line["steer_positions"] == [[4, 9]] and line["n_tool_args_renamed"] == 1
    # the ids are gone from what the client will see
    assert "prompt_token_ids" not in resp and "token_ids" not in resp["choices"][0]
    # a reply without ids (synthetic, or a server ignoring the flag) yields no line
    assert ids_sidecar_line({"choices": [{"message": {"content": ""}}], "usage": {}}, "k", None) is None
    assert ids_sidecar_line({"choices": [], "usage": {}}, "k", None) is None


def test_proxy_end_to_end_writes_sidecar_and_strips_ids(tmp_path):
    """A fake vLLM: /tokenize renders nothing fancy, /v1/chat/completions streams chunks with
    token ids. Through the proxy: the upstream request carries return_token_ids, the sidecar
    line joins by request_sha of the forwarded messages, and Inspect's reply has no ids."""
    import asyncio

    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    from steering_reasoning.proxy import make_app
    from steering_reasoning.steering.clamp import request_sha

    seen = {}

    async def chat(request):
        data = await request.json()
        seen["request"] = data
        body = "".join(f"data: {json.dumps(c)}\n\n" for c in _chunks([1, 2, 3], [7, 8, 9])) + "data: [DONE]\n\n"
        return web.Response(body=body.encode(), content_type="text/event-stream")

    async def main():
        up = web.Application(); up.router.add_post("/v1/chat/completions", chat)
        async with TestServer(up) as upstream:
            ids_log = tmp_path / "ids.jsonl"
            app = make_app(str(upstream.make_url("")).rstrip("/"), ids_log=str(ids_log))
            async with TestClient(TestServer(app)) as client:
                req = {"model": "m", "seed": 42, "max_tokens": 10, "tools": [{"type": "function", "function": {"name": "bash"}}],
                       "messages": [{"role": "user", "content": "hi"},
                                    {"role": "assistant", "content": "x", "reasoning": "<think>\nr\n</think>"},
                                    {"role": "user", "content": "go"}]}
                r = await client.post("/v1/chat/completions", json=req)
                assert r.status == 200
                reply = await r.json()
            return ids_log, reply

    ids_log, reply = asyncio.run(main())
    assert seen["request"]["return_token_ids"] is True and seen["request"]["stream"] is True
    assert reply["choices"][0]["message"]["content"] == "hi"
    assert "prompt_token_ids" not in reply and "token_ids" not in reply["choices"][0]
    (line,) = [json.loads(l) for l in ids_log.read_text().splitlines()]
    # join key = hash of the messages as forwarded (think tags stripped) + tools: what the
    # exporter recomputes from the Inspect log via proxy_view
    fwd = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "x", "reasoning": "r"},
           {"role": "user", "content": "go"}]
    assert line["request_sha"] == request_sha(fwd, seen["request"]["tools"])
    assert line["token_ids"] == [7, 8, 9] and line["seed"] == 42 and line["steer_positions"] is None
    assert line["steered"] is False   # an unsteered proxy: null positions mean "nothing to steer"
    assert line["n_prompt_ids"] == 3 and line["prompt_ids_sha256"] == ids_sha([1, 2, 3])


def test_proxy_refuses_ids_log_with_all_completions(tmp_path):
    from steering_reasoning.proxy import make_app

    with pytest.raises(ValueError, match="all-completions"):
        make_app("http://127.0.0.1:1", ids_log=str(tmp_path / "ids.jsonl"), all_completions=True)


def test_proxy_key_ignores_replay_reasoning_and_mutations(tmp_path):
    """The join key is taken after strip_think but before --replay-reasoning and mutate_request,
    which the exporter cannot reproduce from the Inspect log."""
    import asyncio

    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    from steering_reasoning.proxy import make_app
    from steering_reasoning.steering.clamp import request_sha

    async def chat(request):
        body = "".join(f"data: {json.dumps(c)}\n\n" for c in _chunks([1], [2])) + "data: [DONE]\n\n"
        return web.Response(body=body.encode(), content_type="text/event-stream")

    def mutate(data):
        data["messages"][0]["content"] = "MUTATED"

    async def main():
        up = web.Application(); up.router.add_post("/v1/chat/completions", chat)
        async with TestServer(up) as upstream:
            ids_log = tmp_path / "ids.jsonl"
            app = make_app(str(upstream.make_url("")).rstrip("/"), ids_log=str(ids_log),
                           replay_reasoning=True, mutate_request=mutate)
            async with TestClient(TestServer(app)) as client:
                msgs = [{"role": "user", "content": "hi"},
                        {"role": "assistant", "content": "x", "reasoning": "<think>r</think>"}]
                r = await client.post("/v1/chat/completions", json={"model": "m", "messages": msgs})
                assert r.status == 200
            return ids_log

    ids_log = asyncio.run(main())
    (line,) = [json.loads(l) for l in ids_log.read_text().splitlines()]
    expected = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "x", "reasoning": "r"}]
    assert line["request_sha"] == request_sha(expected, None)


def test_proxy_span_failures_are_bounded_and_render_mismatch_fails_the_sample(tmp_path):
    """A steered request whose /tokenize fails is refused with a retryable 502; after three in a row
    for one rollout it becomes a 400 (Inspect errors the sample instead of retrying forever). And
    when /tokenize renders a different length than the server prefilled, the reply is refused."""
    import asyncio

    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    from steering_reasoning.proxy import make_app

    state = {"tokenize_ok": False, "prompt_ids": [1, 2, 3]}

    async def tokenize(request):
        if not state["tokenize_ok"]:
            return web.Response(status=500, text="boom")
        data = await request.json()
        toks = {"<|im_start|>assistant\n": [10, 11], "<think>\n": [12, 13], "<|im_end|>": [99]}
        if "prompt" in data:
            return web.json_response({"tokens": toks[data["prompt"]]})
        return web.json_response({"tokens": [1, 2, 3, 4]})   # 4 ids: one more than the server will prefill

    async def chat(request):
        body = "".join(f"data: {json.dumps(c)}\n\n" for c in _chunks(state["prompt_ids"], [7, 8])) + "data: [DONE]\n\n"
        return web.Response(body=body.encode(), content_type="text/event-stream")

    async def main():
        up = web.Application(); up.router.add_post("/tokenize", tokenize); up.router.add_post("/v1/chat/completions", chat)
        async with TestServer(up) as upstream:
            app = make_app(str(upstream.make_url("")).rstrip("/"), steer_payload="[{}]", ids_log=str(tmp_path / "ids.jsonl"))
            async with TestClient(TestServer(app)) as client:
                req = {"model": "m", "seed": 1, "messages": [{"role": "user", "content": "hi"}]}
                codes = [(await client.post("/v1/chat/completions", json=req)).status for _ in range(4)]
                state["tokenize_ok"] = True
                mismatch = (await client.post("/v1/chat/completions", json=req)).status
                state["prompt_ids"] = [1, 2, 3, 4]
                ok = (await client.post("/v1/chat/completions", json=req)).status
            return codes, mismatch, ok

    codes, mismatch, ok = asyncio.run(main())
    assert codes == [502, 502, 400, 400]
    assert mismatch == 400 and ok == 200
