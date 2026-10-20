"""steering_reasoning.judge: the one judge client (no network; the HTTP layer is stubbed)."""
import json

import pytest

from steering_reasoning import judge


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for v in ("JUDGE_API_KEY", "JUDGE_PROVIDER", "JUDGE_BASE_URL", "JUDGE_MODEL", "JUDGE_REGION"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(judge, "load_env", lambda path=None: None)      # never read the developer's .env here
    monkeypatch.setattr(judge, "have_aws_credentials", lambda: False)


def test_load_env_parses_exports_quotes_and_comments(tmp_path, monkeypatch):
    monkeypatch.undo()
    monkeypatch.delenv("JUDGE_MODEL", raising=False)
    monkeypatch.setenv("JUDGE_API_KEY", "already-set")
    f = tmp_path / ".env"
    monkeypatch.setenv("JUDGE_REGION", "")                      # exported empty: the file value should win
    f.write_text("# comment\nexport JUDGE_API_KEY='from-file'\nJUDGE_MODEL = \"m/1\"   \nJUDGE_PROVIDER=openai  # inline comment\nJUDGE_REGION=eu-west-1\n\nBROKEN LINE\n")
    judge.load_env(f)
    import os
    assert os.environ["JUDGE_API_KEY"] == "already-set" and os.environ["JUDGE_MODEL"] == "m/1"
    assert os.environ["JUDGE_PROVIDER"] == "openai" and os.environ["JUDGE_REGION"] == "eu-west-1"


def test_settings_requires_key_and_model(monkeypatch):
    with pytest.raises(SystemExit, match="JUDGE_MODEL"):
        judge.settings()
    with pytest.raises(SystemExit, match="JUDGE_API_KEY"):
        judge.settings(model="m")
    monkeypatch.setenv("JUDGE_API_KEY", "k")
    s = judge.settings(model="m")
    assert (s.provider, s.base_url, s.model, s.region) == ("openai", judge.DEFAULT_BASE_URL, "m", judge.DEFAULT_REGION)
    monkeypatch.setenv("JUDGE_PROVIDER", "nope")
    with pytest.raises(SystemExit, match="JUDGE_PROVIDER"):
        judge.settings(model="m")
    monkeypatch.setenv("JUDGE_PROVIDER", "bedrock")
    monkeypatch.setenv("JUDGE_BASE_URL", "https://example.test/v1/")
    s = judge.settings(model="m")
    assert s.provider == "bedrock" and s.base_url == "https://example.test/v1"


def test_complete_openai_builds_a_forced_tool_call_and_reads_it_back(monkeypatch):
    monkeypatch.setenv("JUDGE_API_KEY", "k")
    monkeypatch.setenv("JUDGE_BASE_URL", "https://router.test/v1")
    seen = {}

    def fake_post(url, body, headers, timeout):
        seen.update(url=url, body=body, headers=headers)
        return {"choices": [{"finish_reason": "tool_calls", "message": {"content": None, "tool_calls": [
                    {"function": {"name": "grade", "arguments": json.dumps({"answer": "lie"})}}]}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3, "prompt_tokens_details": {"cached_tokens": 4}}}
    monkeypatch.setattr(judge, "_post", fake_post)
    r = judge.complete("sys", "usr", model="m", max_tokens=50, temperature=0, tool=("grade", "d", {"type": "object"}))
    assert seen["url"] == "https://router.test/v1/chat/completions" and seen["headers"]["Authorization"] == "Bearer k"
    b = seen["body"]
    assert b["model"] == "m" and b["max_completion_tokens"] == 50 and b["temperature"] == 0
    assert b["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}]
    assert b["tool_choice"] == {"type": "function", "function": {"name": "grade"}} and b["tools"][0]["function"]["parameters"] == {"type": "object"}
    assert r.tool_input == {"answer": "lie"} and r.text == "" and r.stop_reason == "tool_calls"
    assert r.usage == dict(input_tokens=10, output_tokens=3, cache_read_tokens=4)


def test_complete_openai_falls_back_on_parameter_names(monkeypatch):
    monkeypatch.setenv("JUDGE_API_KEY", "k")
    calls = []

    def fake_post(url, body, headers, timeout):
        calls.append(dict(body))
        if "max_completion_tokens" in body:
            raise judge.JudgeHTTPError(400, "Unsupported parameter: max_completion_tokens")
        if "temperature" in body:
            raise judge.JudgeHTTPError(400, "temperature is not supported")
        return {"choices": [{"finish_reason": "stop", "message": {"content": "LABEL: NOT"}}], "usage": {}}
    monkeypatch.setattr(judge, "_post", fake_post)
    r = judge.complete("s", "u", model="m", max_tokens=20)
    assert r.text == "LABEL: NOT" and len(calls) == 3 and calls[-1]["max_tokens"] == 20 and "temperature" not in calls[-1]


def test_complete_bedrock_with_key_uses_converse(monkeypatch):
    monkeypatch.setenv("JUDGE_API_KEY", "bk")
    monkeypatch.setenv("JUDGE_PROVIDER", "bedrock")
    monkeypatch.setenv("JUDGE_REGION", "eu-west-1")
    seen = {}

    def fake_post(url, body, headers, timeout):
        seen.update(url=url, body=body, headers=headers)
        return {"stopReason": "tool_use", "usage": {"inputTokens": 7, "outputTokens": 2},
                "output": {"message": {"content": [{"toolUse": {"name": "grade", "input": {"answer": "truth"}}}]}}}
    monkeypatch.setattr(judge, "_post", fake_post)
    r = judge.complete("sys", "usr", model="zai.glm-5", max_tokens=9, temperature=0, tool=("grade", "d", {"type": "object"}),
                       extra={"additionalModelRequestFields": {"x": 1}})
    assert seen["url"] == "https://bedrock-runtime.eu-west-1.amazonaws.com/model/zai.glm-5/converse"
    b = seen["body"]
    assert b["system"] == [{"text": "sys"}] and b["inferenceConfig"] == {"maxTokens": 9, "temperature": 0}
    assert b["toolConfig"]["toolChoice"] == {"tool": {"name": "grade"}} and b["additionalModelRequestFields"] == {"x": 1}
    assert r.tool_input == {"answer": "truth"} and r.usage["input_tokens"] == 7


def test_post_error_bodies_and_tool_argument_shapes(monkeypatch):
    import io, urllib.error
    def raising(body, code=400):
        def opener(req, timeout):
            raise urllib.error.HTTPError(req.full_url, code, "bad", {}, io.BytesIO(body.encode()))
        return opener
    monkeypatch.setattr(judge.urllib.request, "urlopen", raising('{"error": {"message": "too long", "code": "context_length_exceeded"}}'))
    with pytest.raises(judge.JudgeHTTPError) as ei:
        judge._post("https://x/v1/chat/completions", {}, {}, 5)
    assert ei.value.message == "too long" and ei.value.code == "context_length_exceeded" and "context_length_exceeded" in ei.value.body
    for body in ('"plain string"', "[1, 2]", "not json", '{"error": {}}'):
        monkeypatch.setattr(judge.urllib.request, "urlopen", raising(body))
        with pytest.raises(judge.JudgeHTTPError) as ei:
            judge._post("https://x", {}, {}, 5)
        assert ei.value.status == 400 and ei.value.message        # never None, never a crash

    class R(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False
    monkeypatch.setattr(judge.urllib.request, "urlopen", lambda req, timeout: R(b"<html>proxy</html>"))
    with pytest.raises(judge.JudgeHTTPError) as ei:
        judge._post("https://x", {}, {}, 5)
    assert ei.value.retryable and ei.value.status is None

    monkeypatch.setenv("JUDGE_API_KEY", "k")
    def tool_reply(args):
        return lambda url, body, headers, timeout: {"choices": [{"finish_reason": "tool_calls", "message": {"content": "{\"answer\": \"text\"}",
                "tool_calls": [{"function": {"name": "grade", "arguments": args}}]}}], "usage": {}}
    monkeypatch.setattr(judge, "_post", tool_reply({"answer": "dict"}))
    assert judge.complete("s", "u", model="m", tool=("grade", "d", {})).tool_input == {"answer": "dict"}
    monkeypatch.setattr(judge, "_post", tool_reply(None))
    r = judge.complete("s", "u", model="m", tool=("grade", "d", {}))
    assert r.tool_input is None and judge.json_from_text(r.text) == {"answer": "text"}


def test_keyless_self_hosted_endpoint(monkeypatch):
    monkeypatch.setenv("JUDGE_BASE_URL", "http://localhost:8000/v1")
    assert judge.settings(model="m").key == "none"
    monkeypatch.delenv("JUDGE_BASE_URL")
    with pytest.raises(SystemExit):
        judge.settings(model="m")


def test_http_error_classification_and_json_from_text():
    assert judge.JudgeHTTPError(429, "slow down").retryable and not judge.JudgeHTTPError(429, "x").fatal
    assert judge.JudgeHTTPError(401, "no").fatal and not judge.JudgeHTTPError(401, "no").retryable
    assert judge.JudgeHTTPError(None, "timeout").retryable
    assert not judge.JudgeHTTPError(400, "bad").retryable and not judge.JudgeHTTPError(400, "bad").fatal
    assert judge.json_from_text("Sure: ```json\n{\"a\": 1}\n``` done") == {"a": 1}
    assert judge.json_from_text("no json") is None
