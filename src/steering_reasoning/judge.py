"""One judge client for every grader in the repo: the School of Reward Hacks grader, the story grader,
the decline-segment, hack-edit and spec-citation judges, and the honest-submission judge.

Configuration is read from the environment, else from the repo-root .env (loaded here; variables that
are already set win):

    JUDGE_API_KEY    the key for your judge service
    JUDGE_PROVIDER   openai (default) | bedrock
                     openai  = any OpenAI-compatible chat-completions endpoint (OpenAI, OpenRouter,
                               Together, Fireworks, a local vLLM server, ...)
                     bedrock = Amazon Bedrock's Converse API, the service the paper's grades came from
    JUDGE_BASE_URL   openai provider only: the endpoint, default https://api.openai.com/v1
    JUDGE_MODEL      the model id; every grader's --model overrides it
    JUDGE_REGION     bedrock provider only, default us-east-1

The paper's grades were made with provider=bedrock, model=zai.glm-5 (GLM-5), region us-east-1
(the honest-submission judge used gpt-6-luna through OpenAI).

=====================================================================================
JUDGE PROVIDER. >>> ADD JUDGE TOOLING FOR YOUR PROVIDER HERE. <<<
If your service is neither OpenAI-compatible nor Bedrock, add a branch to complete() below
that builds the request from (system, user, max_tokens, temperature, tool) and returns a
Reply, and add its name to PROVIDERS. Every grader goes through complete() except the
School of Reward Hacks grader (evals/sorh/grade.py), which keeps its own async OpenAI and
Bedrock clients and takes only the key, provider, endpoint and model from here; a new
provider needs a branch there too (its header says where).
=====================================================================================

A local or self-hosted OpenAI-compatible server that needs no key: set JUDGE_BASE_URL and leave
JUDGE_API_KEY empty; a placeholder bearer token is sent.
"""
from __future__ import annotations

import http.client
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PROVIDERS = ("openai", "bedrock")
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_REGION = "us-east-1"
RETRY_STATUSES = (408, 409, 429, 500, 502, 503, 504)
FATAL_STATUSES = (401, 403, 404)

_env_loaded = False


def load_env(path: Path | None = None) -> None:
    """Read KEY=VALUE lines (optionally `export`-prefixed, quotes stripped) from the repo-root .env into
    os.environ, without overriding variables that are already set. Idempotent."""
    global _env_loaded
    if _env_loaded and path is None:
        return
    p = path or REPO / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            m = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$", line)
            if m and not line.lstrip().startswith("#"):
                v = m.group(1), m.group(2)
                val = v[1]
                if val[:1] in ("'", '"'):
                    val = val[1:].split(val[0], 1)[0]          # quoted: up to the closing quote
                else:
                    val = val.split(" #", 1)[0].split("\t#", 1)[0].strip()   # unquoted: drop an inline comment
                if not os.environ.get(v[0]):                    # unset or exported empty: the file value wins
                    os.environ[v[0]] = val
    if path is None:
        _env_loaded = True


class JudgeConfigError(SystemExit):
    """Missing key / model / provider: exits with the message (code 2), like the graders always did."""

    def __init__(self, msg: str):
        super().__init__(f"judge: {msg} (see .env.example)")


@dataclass
class Settings:
    provider: str
    key: str | None
    base_url: str
    model: str | None
    region: str


def settings(model: str | None = None, provider: str | None = None, *, need_key: bool = True,
             need_model: bool = True) -> Settings:
    """Resolve the judge configuration. `model` / `provider` come from a grader's command line and win
    over the environment. Raises JudgeConfigError when a required piece is missing."""
    load_env()
    provider = (provider or os.environ.get("JUDGE_PROVIDER") or "openai").lower()
    if provider not in PROVIDERS:
        raise JudgeConfigError(f"JUDGE_PROVIDER must be one of {PROVIDERS}, got {provider!r}; add your own in judge.complete()")
    key = os.environ.get("JUDGE_API_KEY") or None
    model = model or os.environ.get("JUDGE_MODEL") or None
    s = Settings(provider=provider, key=key, base_url=(os.environ.get("JUDGE_BASE_URL") or DEFAULT_BASE_URL).rstrip("/"),
                 model=model, region=os.environ.get("JUDGE_REGION") or DEFAULT_REGION)
    if need_model and not s.model:
        raise JudgeConfigError("set JUDGE_MODEL in .env or pass --model")
    if need_key and not s.key:
        if provider == "openai" and s.base_url != DEFAULT_BASE_URL:
            s.key = "none"                                      # a self-hosted endpoint without auth
        elif not (provider == "bedrock" and have_aws_credentials()):
            raise JudgeConfigError("set JUDGE_API_KEY in .env" + (" (or hold AWS credentials for SigV4)" if provider == "bedrock" else ""))
    return s


def have_aws_credentials() -> bool:
    """Bedrock without a bearer key: the usual AWS credential chain through boto3 (optional dependency)."""
    try:
        import boto3
        return boto3.Session().get_credentials() is not None
    except Exception:
        return False


class JudgeHTTPError(Exception):
    """A failed call. `retryable` for throttling / 5xx / transport errors, `fatal` for auth and unknown
    model, else a per-request problem (bad request, too long). `message` is the service's text."""

    def __init__(self, status: int | None, message: str, *, retryable: bool | None = None, fatal: bool | None = None,
                 code: str | None = None, body: str = ""):
        super().__init__(f"judge HTTP {status}: {message[:300]}")
        self.status, self.message = status, message
        self.code, self.body = code, body or message           # the service's error code and raw body, for callers that match on them
        self.retryable = retryable if retryable is not None else (status is None or status in RETRY_STATUSES)
        self.fatal = fatal if fatal is not None else (status in FATAL_STATUSES)


@dataclass
class Reply:
    text: str                                   # the assistant text ("" when the reply was a tool call only)
    tool_input: dict | None                     # the forced tool's arguments, when `tool` was given
    stop_reason: str | None
    usage: dict = field(default_factory=dict)   # input_tokens, output_tokens, cache_read_tokens
    raw: dict = field(default_factory=dict)     # the service's JSON, for anything provider-specific


def json_from_text(text: str) -> dict | None:
    """The first JSON object in a text reply (fenced or inline), or None."""
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[4:] if t.lower().startswith("json") else t
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j < i:
        return None
    try:
        return json.loads(t[i:j + 1])
    except json.JSONDecodeError:
        return None


def _post(url: str, body: dict, headers: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        msg, code = body, None
        try:
            j = json.loads(body)
        except ValueError:
            j = None
        if isinstance(j, dict):                                 # OpenAI: {"error": {"message", "code"}}; Bedrock: {"message"}
            err = j.get("error")
            if isinstance(err, dict):
                msg = err.get("message") or body; code = err.get("code") or err.get("type")
            elif isinstance(err, str):
                msg = err
            else:
                msg = j.get("message") or body
        raise JudgeHTTPError(e.code, str(msg), code=str(code) if code else None, body=body) from None
    except (urllib.error.URLError, http.client.HTTPException, TimeoutError, ConnectionError, OSError) as e:
        raise JudgeHTTPError(None, f"{type(e).__name__}: {e}", retryable=True, fatal=False) from None
    try:
        return json.loads(raw)
    except ValueError:
        raise JudgeHTTPError(None, f"non-JSON 200 reply: {raw[:200]!r}", retryable=True, fatal=False) from None


def complete(system: str, user: str, *, model: str | None = None, max_tokens: int = 400, temperature: float | None = 0.0,
             tool: tuple[str, str, dict] | None = None, extra: dict | None = None, timeout: float = 300.0,
             provider: str | None = None, cfg: Settings | None = None) -> Reply:
    """One judge call: a system prompt, one user message, and optionally a forced tool call
    `tool = (name, description, json_schema)` whose arguments come back as Reply.tool_input (the way the
    graders get schema-conforming labels). `extra` is merged into the request body for provider-specific
    fields. Raises JudgeHTTPError; retries are the caller's."""
    s = cfg or settings(model, provider)
    model = model or s.model
    if s.provider == "openai":
        return _complete_openai(s, model, system, user, max_tokens, temperature, tool, extra, timeout)
    if s.provider == "bedrock":
        return _complete_bedrock(s, model, system, user, max_tokens, temperature, tool, extra, timeout)
    # >>> ADD JUDGE TOOLING FOR YOUR PROVIDER HERE: elif s.provider == "yours": return _complete_yours(...)
    raise JudgeConfigError(f"no client for provider {s.provider!r}")


# ------------------------------------------------------------------ OpenAI-compatible chat completions
def _complete_openai(s: Settings, model: str, system: str, user: str, max_tokens: int, temperature, tool, extra, timeout) -> Reply:
    body = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_completion_tokens": max_tokens}
    if temperature is not None:
        body["temperature"] = temperature
    if tool:
        name, desc, schema = tool
        body["tools"] = [{"type": "function", "function": {"name": name, "description": desc, "parameters": schema}}]
        body["tool_choice"] = {"type": "function", "function": {"name": name}}
    if extra:
        body.update(extra)
    headers = {"Authorization": f"Bearer {s.key}"}
    url = f"{s.base_url}/chat/completions"
    unsupported = re.compile(r"unsupported|unrecognized|unknown|not supported|does not support|extra inputs|invalid_request_error", re.I)
    for attempt in range(3):   # small compatibility loop: servers differ on these two parameters
        try:
            resp = _post(url, body, headers, timeout)
            break
        except JudgeHTTPError as e:
            if e.status == 400 and "max_completion_tokens" in e.message and "max_completion_tokens" in body and unsupported.search(e.message) and "too large" not in e.message:
                body["max_tokens"] = body.pop("max_completion_tokens"); continue
            if e.status == 400 and "temperature" in e.message and "temperature" in body and unsupported.search(e.message):
                body.pop("temperature"); continue
            raise
    ch = (resp.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    tool_input = None
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        if tool and fn.get("name") == tool[0]:
            a = fn.get("arguments")
            try:
                tool_input = a if isinstance(a, dict) else (json.loads(a) if a else None)
            except (TypeError, ValueError):
                tool_input = None
            if not isinstance(tool_input, dict) or not tool_input:
                tool_input = None                               # empty or malformed: let the caller fall back to the text
            break
    u = resp.get("usage") or {}
    usage = dict(input_tokens=int(u.get("prompt_tokens") or 0), output_tokens=int(u.get("completion_tokens") or 0),
                 cache_read_tokens=int(((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0))
    return Reply(text=msg.get("content") or "", tool_input=tool_input, stop_reason=ch.get("finish_reason"), usage=usage, raw=resp)


# ------------------------------------------------------------------ Amazon Bedrock Converse (the authors' service)
def _bedrock_body(system: str, user: str, max_tokens: int, temperature, tool, extra) -> dict:
    body = {"system": [{"text": system}], "messages": [{"role": "user", "content": [{"text": user}]}],
            "inferenceConfig": {"maxTokens": max_tokens}}
    if temperature is not None:
        body["inferenceConfig"]["temperature"] = temperature
    if tool:
        name, desc, schema = tool
        body["toolConfig"] = {"tools": [{"toolSpec": {"name": name, "description": desc, "inputSchema": {"json": schema}}}],
                              "toolChoice": {"tool": {"name": name}}}
    if extra:
        body.update(extra)
    return body


def _bedrock_reply(resp: dict, tool) -> Reply:
    content = ((resp.get("output") or {}).get("message") or {}).get("content") or []
    tool_input = None
    for block in content:
        tu = block.get("toolUse")
        if tu and tool and tu.get("name") == tool[0]:
            tool_input = tu.get("input"); break
    text = "\n".join(b["text"] for b in content if b.get("text"))
    u = resp.get("usage") or {}
    usage = dict(input_tokens=int(u.get("inputTokens") or 0), output_tokens=int(u.get("outputTokens") or 0),
                 cache_read_tokens=int(u.get("cacheReadInputTokens") or 0))
    return Reply(text=text, tool_input=tool_input, stop_reason=resp.get("stopReason"), usage=usage, raw=resp)


def _complete_bedrock(s: Settings, model: str, system: str, user: str, max_tokens: int, temperature, tool, extra, timeout) -> Reply:
    body = _bedrock_body(system, user, max_tokens, temperature, tool, extra)
    if s.key:   # Bedrock API key (bearer token), plain HTTPS
        url = f"https://bedrock-runtime.{s.region}.amazonaws.com/model/{urllib.parse.quote(model, safe='')}/converse"
        return _bedrock_reply(_post(url, body, {"Authorization": f"Bearer {s.key}"}, timeout), tool)
    # no key: sign with the AWS credential chain through boto3
    import botocore.exceptions
    client = _sigv4_client(s, timeout)
    try:
        return _bedrock_reply(client.converse(modelId=model, **body), tool)
    except botocore.exceptions.ClientError as e:
        r = e.response or {}
        code = (r.get("Error") or {}).get("Code") or type(e).__name__
        status = (r.get("ResponseMetadata") or {}).get("HTTPStatusCode")
        msg = (r.get("Error") or {}).get("Message") or str(e)
        raise JudgeHTTPError(status, f"{code}: {msg}",
                             retryable=code in ("ThrottlingException", "ServiceUnavailableException", "ModelNotReadyException", "InternalServerException") or status in RETRY_STATUSES,
                             fatal=code in ("UnrecognizedClientException", "ExpiredTokenException", "AccessDeniedException", "ResourceNotFoundException", "InvalidSignatureException") or status in FATAL_STATUSES) from None
    except botocore.exceptions.BotoCoreError as e:
        raise JudgeHTTPError(None, f"{type(e).__name__}: {e}", retryable=True, fatal=False) from None


_clients: dict = {}


def _sigv4_client(s: Settings, timeout: float):
    key = (s.region, timeout)
    if key not in _clients:
        import boto3
        from botocore.config import Config
        _clients[key] = boto3.client("bedrock-runtime", region_name=s.region,
                                     config=Config(read_timeout=timeout, connect_timeout=30, retries={"max_attempts": 0}, max_pool_connections=64))
    return _clients[key]


__all__ = ["settings", "Settings", "complete", "Reply", "JudgeHTTPError", "JudgeConfigError", "json_from_text",
           "load_env", "have_aws_credentials", "PROVIDERS", "DEFAULT_BASE_URL", "DEFAULT_REGION"]
