"""The OpenAI-compatible proxy between Inspect and vLLM: per-turn cap, clamp steering, request
dumps, and the prefill / raw-completion modes. One module, one process per arm.

    uv run python -m steering_reasoning.proxy --port 8100 --upstream http://host:8000            # cap only
    uv run python -m steering_reasoning.proxy --port 8104 --upstream http://host:8000 \\
        --artifact src/steering_reasoning/canonical/u_L35_steering.safetensors \\
        --threshold -8 --dump-dir ~/dumps/cn8                                                    # clamp arm

Why a proxy at all: Inspect's ``max_tokens`` is static while vLLM hard-400s any request where
prompt + max_tokens exceeds the window, and an unset cap permits degenerate 80k-token turns.
The proxy stamps ``max_tokens = min(cap, what the client asked)`` and repairs overflows:
retry with the window vLLM reports, then with the cap removed, then synthesize an empty
``finish_reason: "length"`` reply (what uncapped vLLM produced for the MisTrace originals; under
Inspect's agent it maps to stop_reason max_tokens). vLLM's "prompt contains at least N" is a
lower bound, so ``parse_overflow`` is unit-tested against every wording seen (tests/test_proxy.py).

What else it does, all on the chat-completions path:

- **Steering** (``--artifact`` + ``--mode clamp --threshold T`` | ``--mode add --scale S`` |
  ``--mode ablate``): injects the vllm-lens payload built by
  ``steering_reasoning.steering.build_steer_payload`` into every request. The server applies it
  to generated tokens, and during prefill to the spans the model wrote in earlier turns: the proxy
  tokenizes each request (``/tokenize``, same chat template) and sends ``steer_positions`` with the
  payload (bootstrap patch 2), so prompts and tool outputs are never steered and cached vs fresh KV
  agree. ``--no-steer`` is the explicit baseline. Default mode is clamp.
- **Cache salt** (``--cache-salt``): stamps vLLM's ``cache_salt`` on every request so this proxy's
  arm has a private prefix cache. Steered requests read the prefix cache (bootstrap patch 3),
  and generated tokens' KV carries the arm's steering, so arms sharing one server must not share
  cache blocks; run_arm.sh passes the arm name.
- **Token-id sidecar** (``--ids-log FILE``): asks vLLM for ``return_token_ids`` and appends one JSON
  line per completed chat turn — the exact generated ``token_ids`` (pre-collapse, pre-rename), the
  length and sha of the prompt ids the server prefilled, the ``steer_positions`` sent, and a hash of
  the forwarded request as the join key (``ids_sidecar_line``). The ids are stripped from the reply
  Inspect sees, so the eval log is unchanged; a later replay (steering/replay.py) joins the sidecar
  by that key and reproduces every turn token-for-token without re-rendering anything. Steered
  requests whose span computation fails are refused (never sent with an unsteered prefill), so a
  sidecar line's ``steer_positions`` is always what the server applied; with the sidecar on, the
  length of the render the spans came from is checked against the prompt the server prefilled
  (a mismatch fails the sample: ``RENDER_MISMATCH``). Not available with ``--all-completions``;
  ``mutate_request`` must leave ``messages`` alone when the sidecar is on (the key is taken
  before it runs).
- **Tool-argument hygiene**: a tool call whose single unexpected argument key stands in for the
  single missing required one is renamed (``normalize_tool_args``: bash ``command`` -> ``cmd``);
  the proxy log counts renames per rollout.
- **Reasoning hygiene**: literal <think> tags the reasoning parser leaves in replayed reasoning
  are stripped on the way in and out (``strip_think``); ``--replay-reasoning`` renames
  ``reasoning`` to ``reasoning_content`` (diagnostic only).
- **Streaming upstream**: vLLM is asked to stream and the SSE chunks are reassembled into the
  non-streaming reply Inspect expects, because some providers' port mappings (RunPod's, 2026-09-09) drop connections that
  carry no bytes for ~10 minutes. TCP keepalive on upstream sockets for the same
  reason. ``--upstream`` accepts a URL, a comma list (round-robin) or a file of URLs re-read
  per request.
- **Tool-call collapse**: at most ``MAX_TOOL_CALLS`` distinct calls per reply (the served model
  sometimes emits hundreds of identical calls in one capped turn; originals never exceed 2).
- **Dumps** (``--dump-dir``): every request's message history and reply, keyed by a hash of the
  first four messages, so a killed rollout can be resumed.
- **Prefill** (``--prefill-table``): a request whose fingerprint (message count + last message
  text) matches is served from ``/v1/completions`` with prompt = ``/tokenize``(messages + tools,
  generation prompt) + ``/tokenize``(reasoning prefix), so the assistant turn resumes inside its
  own <think>; the completion is parsed back into a chat reply. Only a sample's first request
  can match. Used by experiments/prefill-action-clamp.
- **Raw completions** (``--all-completions``): every request goes through that same path with
  an empty prefix, bypassing vLLM's chat-side tool machinery. Kept as a diagnostic; the strict
  structural-tag grammar that made it necessary is now disabled in the serving bootstrap.
- **Reasoning budget** (``--reasoning-budget N``): stop a rollout after N generated reasoning
  tokens (counted per rollout via ``/tokenize``); further requests get a no-op reply.
- **History table** (``--history-table``): for rollouts resumed from a recorded history (the
  turn-start arms of experiments/prefill-action-clamp), ``{fingerprint: {task_id, n_history}}``
  keyed like the prefill table. On the matching first request the proxy remembers ``n_history``
  for that rollout and, on every request of it, drops the ``steer_positions`` spans before the
  history's end in the rendered prompt (the start of the (k+1)-th assistant header, k = assistant
  messages in the history; robust to the template dropping earlier reasoning): the history was
  written by the unsteered original model and counts as prompt, so it is never steered; everything
  generated after the cut is steered when written and again when re-read, as in the ordinary arms.
  With a table set, a chat request from a rollout the table never matched is refused (502) rather
  than steered wholesale; on restart the proxy rebuilds its rollout table from the request dumps.
  ``--steer-history`` keeps the table (and the guard) but steers the history's model-written spans
  as well, as if the whole transcript had been generated under steering.
- **``--require-positions``**: a ``/tokenize`` failure aborts the request (502) instead of falling
  back to unsteered prefill for that turn.

Streaming to the *client* is not supported (Inspect does not ask for it).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import itertools
import json
import os
import re
import socket
import time
import uuid
from collections.abc import Callable

import aiohttp
from aiohttp import web

from steering_reasoning.evals.config import MAX_TOKENS
from steering_reasoning.steering import assistant_spans, build_steer_payload, history_boundary, strip_think, with_positions
from steering_reasoning.steering.clamp import ids_sha, request_sha

CTX_MARGIN = 256
UPSTREAM_TIMEOUT_S = 7200
SOCK_READ_TIMEOUT_S = 900
MAX_TOOL_CALLS = 2   # originals: 1 call/turn in 99.6% of turns, never more than 2

_SESSION = web.AppKey("session", aiohttp.ClientSession)

CALL_RE = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>", re.S)
PARAM_RE = re.compile(r"<parameter=([^>\n]+)>(.*?)</parameter>", re.S)


# ----------------------------------------------------------------------------- pure helpers

def stamp_max_tokens(data: dict, cap: int = MAX_TOKENS) -> None:
    """Cap the request's max_tokens in place (normalizing max_completion_tokens)."""
    mt = data.get("max_tokens") or data.get("max_completion_tokens")
    if mt is None or mt > cap:
        data["max_tokens"] = cap
        data.pop("max_completion_tokens", None)


def parse_overflow(body: bytes) -> int | None:
    """Return a reduced max_tokens from a vLLM context-overflow 400, else None.

    0 means the window is exhausted (caller should synthesize a length finish).
    """
    try:
        msg = json.loads(body).get("error", {}).get("message", "") or body.decode("utf-8", "replace")
    except Exception:
        msg = body.decode("utf-8", "replace")
    m = re.search(r"maximum context length is (\d+) tokens", msg)
    # vLLM has used several phrasings for the prompt length; accept all of them
    p = (re.search(r"prompt contains at least (\d+) input tokens", msg)
         or re.search(r"(\d+) (?:tokens )?in the messages", msg)
         or re.search(r"\((\d+) in the messages", msg))
    if not (m and p):
        return None
    allowed = int(m.group(1)) - int(p.group(1)) - CTX_MARGIN
    return max(allowed, 0)


def synthetic_length_response(model: str) -> dict:
    """Empty completion with finish_reason=length, for an exhausted context window."""
    return {
        "id": "chatcmpl-ctxfull", "object": "chat.completion", "created": 0, "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": ""},
                     "finish_reason": "length"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def msg_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return "" if content is None else str(content)


def fingerprint(messages: list[dict]) -> str:
    """Prefill-table key: message count + role and text of the last message."""
    last = messages[-1] if messages else {}
    return hashlib.sha1(f"{len(messages)}|{last.get('role')}|{msg_text(last.get('content')).strip()}".encode()).hexdigest()[:16]


def rollout_key(messages: list[dict]) -> str:
    """Dump / budget key. The first user message is the same generic instruction for every LCB
    task, so key on the first four messages (system, user, first assistant turn, its tool
    result): stable for the rest of the rollout, distinct across tasks."""
    return hashlib.sha1(json.dumps(messages[:4], sort_keys=True).encode()).hexdigest()[:12]


def skip_history_spans(spans: list[list[int]], prefix_len: int) -> tuple[list[list[int]], int]:
    """Keep only the spans that start at or after ``prefix_len``, the token index where the recorded
    history ends in the rendered prompt (``history_boundary``), so the history's own replies are
    never steered whatever they render as (an assistant message with an empty body yields no span
    at all, which is why spans are not counted per message). Returns (spans kept, spans dropped).
    A span straddling the boundary is a render mismatch and raises."""
    kept, dropped = [], 0
    for a, b in spans:
        if b <= prefix_len: dropped += 1
        elif a >= prefix_len: kept.append([a, b])
        else: raise ValueError(f"assistant span [{a}, {b}) straddles the history boundary at {prefix_len}")
    return kept, dropped


def parse_qwen_tool_calls(text: str, tools: list[dict] | None) -> list[dict]:
    """qwen3_coder XML tool calls -> OpenAI tool_calls. Values are cast by the schema's ``type``
    or its first non-null ``anyOf`` type; one leading and one trailing newline are stripped from
    every value, as vLLM's parser does."""
    schema = {}
    for t in tools or []:
        fn = t.get("function") or {}
        schema[fn.get("name")] = (fn.get("parameters") or {}).get("properties") or {}
    out = []
    for name, body in CALL_RE.findall(text):
        name = name.strip(); props = schema.get(name, {}); args = {}
        for k, v in PARAM_RE.findall(body):
            k = k.strip()
            if v.startswith("\n"): v = v[1:]
            if v.endswith("\n"): v = v[:-1]
            spec = props.get(k) or {}
            typ = spec.get("type") or next((o.get("type") for o in spec.get("anyOf", []) if o.get("type") not in (None, "null")), None)
            try:
                if typ == "integer": v = int(v.strip())
                elif typ == "number": v = float(v.strip())
                elif typ == "boolean": v = v.strip().lower() == "true"
                elif typ in ("array", "object"): v = json.loads(v)
            except Exception:
                pass
            args[k] = v
        out.append({"id": f"chatcmpl-tool-{uuid.uuid4().hex[:16]}", "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
    return out


def assemble_stream(chunks: list[dict], model: str) -> dict:
    """SSE chat chunks (dicts) -> one chat.completion object."""
    out = {"id": None, "object": "chat.completion", "created": 0, "model": model, "choices": [], "usage": None}
    acc_by_index: dict[int, dict] = {}
    for c in chunks:
        out["id"] = out["id"] or c.get("id"); out["created"] = out["created"] or c.get("created", 0)
        out["model"] = c.get("model") or out["model"]
        if c.get("usage"): out["usage"] = c["usage"]
        if c.get("prompt_token_ids") is not None:   # return_token_ids: on the first chunk only
            out["prompt_token_ids"] = c["prompt_token_ids"]
        for choice in c.get("choices", []):
            i = choice.get("index", 0)
            acc = acc_by_index.setdefault(i, {"content": [], "reasoning": [], "tool_calls": {}, "finish_reason": None, "role": "assistant", "token_ids": []})
            if choice.get("token_ids"):                 # return_token_ids: per-chunk delta ids on the choice
                acc["token_ids"].extend(choice["token_ids"])
            d = choice.get("delta") or {}
            if d.get("role"): acc["role"] = d["role"]
            if d.get("content"): acc["content"].append(d["content"])
            for k in ("reasoning_content", "reasoning"):
                if d.get(k): acc["reasoning"].append(d[k])
            for tc in d.get("tool_calls") or []:
                j = tc.get("index", 0)
                t = acc["tool_calls"].setdefault(j, {"id": None, "type": "function", "function": {"name": "", "arguments": ""}})
                if tc.get("id"): t["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"): t["function"]["name"] += fn["name"]
                if fn.get("arguments"): t["function"]["arguments"] += fn["arguments"]
            if choice.get("finish_reason"): acc["finish_reason"] = choice["finish_reason"]
    for i in sorted(acc_by_index):
        acc = acc_by_index[i]; msg = {"role": acc["role"], "content": "".join(acc["content"]) or None}
        if acc["reasoning"]: msg["reasoning_content"] = "".join(acc["reasoning"])
        if acc["tool_calls"]:
            msg["tool_calls"] = [acc["tool_calls"][j] for j in sorted(acc["tool_calls"])]
            for j, t in enumerate(msg["tool_calls"]): t["id"] = t["id"] or f"call_{i}_{j}"
        choice_out = {"index": i, "message": msg, "finish_reason": acc["finish_reason"] or "stop", "logprobs": None}
        if acc["token_ids"]:
            choice_out["token_ids"] = acc["token_ids"]
        out["choices"].append(choice_out)
    if out["usage"] is None: out["usage"] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return out


def ids_sidecar_line(resp: dict, request_key: str, steer_positions: list[list[int]] | None,
                     n_tool_args_renamed: int = 0, seed: int | None = None) -> dict | None:
    """Pop the token ids vLLM returned (``return_token_ids``) out of a chat completion and build
    the sidecar line for it; None when the response carries none (a synthetic reply, or a server
    that ignored the flag). The line is what steering/replay.py needs to reproduce the turn:
    ``token_ids`` are the generated ids exactly as sampled — before tool-call collapse and the
    argument rename, which edit the message the client sees, never these — and the prompt is
    recorded as length + sha (the replay re-renders it and must reproduce that hash byte-for-byte).
    ``request_key`` is ``request_sha`` of the forwarded messages + tools, the join key
    ``evals/eval_records.py`` recomputes from the Inspect log; ``seed`` (the request's) separates
    rollouts of different seeds whose first requests are identical."""
    pids = resp.pop("prompt_token_ids", None)
    choices = resp.get("choices") or []
    gen = [ch.pop("token_ids", None) for ch in choices]
    if not choices or gen[0] is None:
        return None
    usage = resp.get("usage") or {}
    return {"ts": time.time(), "id": resp.get("id"), "request_sha": request_key, "seed": seed,
            "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
            "n_prompt_ids": len(pids) if pids is not None else None,
            "prompt_ids_sha256": ids_sha(pids) if pids is not None else None,
            "token_ids": gen[0], "finish_reason": choices[0].get("finish_reason"),
            "steer_positions": steer_positions, "n_tool_args_renamed": n_tool_args_renamed}


def normalize_tool_args(resp: dict, tools: list[dict] | None) -> int:
    """Rename a mis-keyed tool argument to the schema's name when the mapping is unambiguous.

    With vLLM's strict tool-calling grammar off (serving/bootstrap.sh; it dropped new_str from
    str_replace calls) nothing forces schema argument names, and Qwen writes bash {"command": ...}
    for Inspect's bash tool, whose only parameter is ``cmd`` (~9% of bash calls in the 2026-09-14
    sweep; each miss cost two of the rollout's messages and a 'the bash tool is having issues'
    detour). Rule: if exactly one required parameter is missing and exactly one supplied key is
    not in the schema, rename it. Anything else is left for Inspect to reject as before. Returns
    the number of calls changed."""
    schemas = {}
    for tdef in tools or []:
        fn = tdef.get("function") or {}; params = fn.get("parameters") or {}
        schemas[fn.get("name")] = (set(params.get("properties") or {}), set(params.get("required") or []))
    changed = 0
    for ch in resp.get("choices", []):
        for tc in (ch.get("message") or {}).get("tool_calls") or []:
            fn = tc.get("function") or {}
            if fn.get("name") not in schemas: continue
            props, required = schemas[fn["name"]]
            try: args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError: continue
            if not isinstance(args, dict): continue
            missing = [k for k in required if k not in args]; extra = [k for k in args if k not in props]
            if len(missing) == 1 and len(extra) == 1:
                args[missing[0]] = args.pop(extra[0]); fn["arguments"] = json.dumps(args); changed += 1
    return changed


def collapse_tool_calls(resp: dict) -> bool:
    """Keep at most MAX_TOOL_CALLS distinct calls per choice and strip stray <think> tags.
    Returns True when the response was changed."""
    changed = False
    for ch in resp.get("choices", []):
        msg = ch.get("message") or {}
        for k in ("reasoning", "reasoning_content"):
            if msg.get(k) and ("<think>" in msg[k] or "</think>" in msg[k]):
                msg[k] = strip_think(msg[k]); changed = True
        tcs = msg.get("tool_calls") or []
        if len(tcs) > 1:
            seen, kept = set(), []
            for tc in tcs:
                key = json.dumps(tc.get("function"), sort_keys=True)
                if key in seen: continue
                seen.add(key); kept.append(tc)
                if len(kept) >= MAX_TOOL_CALLS: break
            if len(kept) != len(tcs):
                msg["tool_calls"] = kept; changed = True
                print(f"collapsed {len(tcs)} tool calls -> {len(kept)}", flush=True)
    return changed


class KeepaliveConnector(aiohttp.TCPConnector):
    """TCP keepalive on upstream sockets (some providers' port mappings drop idle connections)."""

    async def _wrap_create_connection(self, *a, **k):
        tr, pr = await super()._wrap_create_connection(*a, **k)
        sock = tr.get_extra_info("socket")
        if sock is not None:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            for opt, val in (("TCP_KEEPIDLE", 30), ("TCP_KEEPINTVL", 15), ("TCP_KEEPCNT", 8)):
                if hasattr(socket, opt):
                    sock.setsockopt(socket.IPPROTO_TCP, getattr(socket, opt), val)
        return tr, pr


# ----------------------------------------------------------------------------- the app


async def iter_sse_lines(content):
    """Yield the decoded, stripped lines of an SSE body without aiohttp's per-line cap.

    ``async for raw in resp.content`` goes through StreamReader.readline, which raises LineTooLong
    above 512 KB regardless of the ClientSession's max_line_size (that only bounds headers). With
    return_token_ids the first chunk carries every prompt token id, so a 100k-token prompt is a
    ~600 KB line and the request fails forever (the retry sees the same line). Read raw pieces and
    split on newlines ourselves; a trailing partial line is yielded at EOF.
    """
    buf = b""
    async for piece in content.iter_any():
        buf += piece
        while True:
            i = buf.find(b"\n")
            if i < 0:
                break
            raw, buf = buf[:i], buf[i + 1:]
            yield raw.decode("utf-8", "replace").strip()
    if buf:
        yield buf.decode("utf-8", "replace").strip()

def make_app(
    upstream: str,
    *,
    mutate_request: Callable[[dict], None] | None = None,
    max_tokens: int = MAX_TOKENS,
    steer_payload: str | None = None,
    replay_reasoning: bool = False,
    dump_dir: str | None = None,
    stream_upstream: bool = True,
    prefill_table: dict | None = None,
    all_completions: bool = False,
    reasoning_budget: int | None = None,
    cache_salt: str | None = None,
    history_table: dict | None = None,
    steer_history: bool = False,
    ids_log: str | None = None,
) -> web.Application:
    """Build the proxy app. ``mutate_request`` runs on every capped chat request after the cap
    and the steering payload are stamped (a seam for experiment-specific request edits).
    ``ids_log`` appends the token-id sidecar line of every completed chat turn to that file."""
    prefill_table = prefill_table or {}
    budget_used: dict[str, int] = {}
    history_n: dict[str, int] = {}     # rollout_key -> messages that are recorded history (never steered)
    if history_table is not None and dump_dir and os.path.isdir(dump_dir):   # survive a proxy restart mid-rollout
        for fn in os.listdir(dump_dir):
            if fn.endswith(".req.json"):
                try:
                    n = json.load(open(os.path.join(dump_dir, fn))).get("n_history")
                    if n: history_n[fn[:-len(".req.json")]] = int(n)
                except Exception:  # noqa: BLE001
                    pass
        if history_n: print(f"history table: restored {len(history_n)} rollouts from dumps", flush=True)
    _rr = itertools.count()
    ids_fh = None
    if ids_log:
        if all_completions:
            raise ValueError("--ids-log cannot record the raw-completions path (--all-completions)")
        os.makedirs(os.path.dirname(os.path.abspath(ids_log)), exist_ok=True)
        ids_fh = open(ids_log, "a")   # fail at startup, not per request; one handle for the process

    def pick_upstream() -> str:
        if os.path.isfile(upstream):
            urls = [l.strip() for l in open(upstream) if l.strip() and not l.startswith("#")]
        else:
            urls = [u.strip() for u in upstream.split(",") if u.strip()]
        return urls[next(_rr) % len(urls)]

    markers: dict[str, tuple] = {}   # upstream -> (header ids, think ids, im_end id) for assistant_spans

    MAX_POSITION_FAILURES = 3   # consecutive per rollout: then the turn errors instead of retrying forever
    position_failures: dict[str, int] = {}

    async def steer_positions_for(session: aiohttp.ClientSession, up: str, data: dict) -> tuple[list[list[int]], list[int], list[int]]:
        """(absolute token spans of the model's own earlier replies in this request's rendered prompt,
        the rendered prompt ids, the assistant header ids). len(ids) is cross-checked against the
        prompt the server actually prefilled when the ids sidecar is on."""
        async def tok(obj):
            async with session.post(up + "/tokenize", json=obj, timeout=aiohttp.ClientTimeout(total=120)) as r:
                if r.status != 200:
                    raise RuntimeError(f"/tokenize {r.status}: {(await r.read())[:200]!r}")
                return (await r.json())["tokens"]
        if up not in markers:
            hdr = await tok({"model": data.get("model"), "prompt": "<|im_start|>assistant\n", "add_special_tokens": False})
            think = await tok({"model": data.get("model"), "prompt": "<think>\n", "add_special_tokens": False})
            end = await tok({"model": data.get("model"), "prompt": "<|im_end|>", "add_special_tokens": False})
            assert len(end) == 1, end
            markers[up] = (hdr, think, end[0])
        hdr, think, end_id = markers[up]
        ids = await tok({"model": data.get("model"), "messages": _with_reasoning(data["messages"]), "tools": data.get("tools"), "add_generation_prompt": True})
        return assistant_spans(ids, hdr, think, end_id), ids, hdr

    def _with_reasoning(messages: list[dict]) -> list[dict]:
        # /tokenize renders assistant reasoning only from vLLM's own `reasoning` field, while
        # /chat/completions maps `reasoning_content` onto it (verified 2026-09-14: with `reasoning` set
        # the two renders are token-identical). Tokenize a copy with `reasoning` filled in.
        msgs = []
        for m in messages:
            if m.get("role") == "assistant" and (m.get("reasoning_content") or m.get("reasoning")):
                m = dict(m); m["reasoning"] = m.get("reasoning") or m.get("reasoning_content")
            msgs.append(m)
        return msgs

    # One SSE `data:` line can be very large: with return_token_ids on (the --ids-log sidecar)
    # vLLM puts the whole prompt_token_ids array in the first chunk, so the line grows with the
    # context. aiohttp's stream reader raises LineTooLong above read_bufsize * 2, and the 256 KiB
    # default (512 KiB limit) is exceeded by ~40k-token prompts: 702 turns failed that way in the
    # spec-prompting 60-message run on 2026-09-17, and 120-message contexts are worse. 8 MiB
    # (16 MiB limit) covers a 120-message rollout with headroom; it is a cap, not a preallocation.
    READ_BUFSIZE = 8 << 20

    async def on_startup(app: web.Application) -> None:
        # max_line_size: aiohttp's default 8 KB per line (512 KB in the stream reader) is smaller
        # than one SSE chunk when vLLM returns `prompt_token_ids` (the sidecar's return_token_ids)
        # on a 120-message context: ~120k ids serialise to ~800 KB on a single `data:` line. The
        # read then raised LineTooLong, the turn 502'd, and Inspect retried it forever (2026-09-22,
        # six optc rollouts at 90k-126k prompt tokens). 16 MB covers the 131k-token context.
        app[_SESSION] = aiohttp.ClientSession(connector=KeepaliveConnector(limit=0),
                                              max_line_size=16 * 1024 * 1024, max_field_size=16 * 1024 * 1024)

    async def on_cleanup(app: web.Application) -> None:
        await app[_SESSION].close()
        if ids_fh is not None:
            ids_fh.close()

    def write_dump(name: str, obj: dict) -> None:
        os.makedirs(dump_dir, exist_ok=True)
        tmp = os.path.join(dump_dir, name + ".tmp")
        with open(tmp, "w") as fh:
            json.dump(obj, fh)
        os.replace(tmp, os.path.join(dump_dir, name))

    async def handle(request: web.Request) -> web.StreamResponse:
        session = request.app[_SESSION]
        body = await request.read()
        is_chat = request.path.endswith("/chat/completions")
        data = None
        if is_chat or request.path.endswith("/completions"):
            data = json.loads(body)
        rkey = rollout_key(data.get("messages", [])) if (is_chat and data is not None) else None
        spans = None   # steer_positions sent with this request (recorded in the dump)
        history_skipped = None   # history spans dropped from steer_positions (history table)
        boundary = None          # token index where the recorded history ends in this prompt
        n_render = None  # length of the /tokenize render the spans came from
        req_sha = None  # ids sidecar join key: hash of the forwarded messages + tools

        up_for_request = pick_upstream() if (is_chat and data is not None) else None

        if is_chat and data is not None:
            for m in data.get("messages", []):
                if m.get("role") == "assistant":
                    for k in ("reasoning", "reasoning_content"):
                        if m.get(k): m[k] = strip_think(m[k])
            if ids_fh is not None:
                # join key = the messages as forwarded after strip_think (what eval_records.proxy_view
                # recomputes from the Inspect log) + tools — before --replay-reasoning and
                # mutate_request, which the exporter cannot reproduce
                data["return_token_ids"] = True
                req_sha = request_sha(data.get("messages", []), data.get("tools"))
            if replay_reasoning:
                for m in data.get("messages", []):
                    if m.get("role") == "assistant" and m.get("reasoning") and not m.get("reasoning_content"):
                        m["reasoning_content"] = m.pop("reasoning")
            stamp_max_tokens(data, max_tokens)
            if history_table is not None:
                h = history_table.get(fingerprint(data.get("messages", [])))
                if h is not None:
                    history_n[rkey] = int(h["n_history"]) if isinstance(h, dict) else int(h)
                    print(f"[HISTORY] {(h.get('task_id') if isinstance(h, dict) else '-')} n_history={history_n[rkey]} rollout {rkey}", flush=True)
                if rkey not in history_n:
                    raise web.HTTPBadGateway(text=f"history table set but rollout {rkey} ({len(data.get('messages', []))} messages) never matched it")
            if steer_payload:
                payload = steer_payload
                try:
                    spans, ids, hdr = await steer_positions_for(session, up_for_request, data)
                    n_render = len(ids)
                    if rkey in history_n:
                        k = sum(1 for m in data["messages"][:history_n[rkey]] if m.get("role") == "assistant")
                        boundary = history_boundary(ids, hdr, k)
                        if steer_history:   # "as if the whole transcript had been generated under steering"
                            history_skipped = 0
                        else:
                            spans, history_skipped = skip_history_spans(spans, boundary)
                    payload = with_positions(steer_payload, spans)
                    position_failures.pop(rkey, None)
                except Exception as e:  # noqa: BLE001
                    # never send a steered request without its positions: the prefill would be
                    # unsteered and a later replay could not tell. A 502 makes Inspect retry the
                    # turn (a transient /tokenize hiccup); after MAX_POSITION_FAILURES in a row
                    # the turn gets a 400, which Inspect records as the sample's error instead
                    # of retrying forever against a server that has stopped rendering.
                    n = position_failures[rkey] = position_failures.get(rkey, 0) + 1
                    print(f"steer_positions failed ({e!r}) for {rkey} ({n}/{MAX_POSITION_FAILURES}): refusing the request", flush=True)
                    if n >= MAX_POSITION_FAILURES:
                        raise web.HTTPBadRequest(text=f"steer_positions failed {n} times in a row: {e}") from e
                    raise web.HTTPBadGateway(text=f"steer_positions failed: {e}") from e
                data.setdefault("vllm_xargs", {})["apply_steering_vectors"] = payload
            if cache_salt:
                data["cache_salt"] = cache_salt
            if reasoning_budget and rkey is not None:
                used = budget_used.get(rkey, 0)
                if used >= reasoning_budget:
                    synth = {"id": "chatcmpl-budget", "object": "chat.completion", "created": 0, "model": data.get("model", ""),
                             "choices": [{"index": 0, "message": {"role": "assistant", "content": "[reasoning budget exhausted]"}, "finish_reason": "stop"}],
                             "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
                    return web.Response(status=200, body=json.dumps(synth).encode(), content_type="application/json")
                data["max_tokens"] = min(int(data["max_tokens"]), reasoning_budget - used + 1500)
            if mutate_request is not None:
                mutate_request(data)
            body = json.dumps(data).encode()

        if dump_dir and is_chat and data is not None:
            try:
                write_dump(f"{rkey}.req.json", {"ts": time.time(), "seed": data.get("seed"), "model": data.get("model"),
                                                "n_messages": len(data.get("messages", [])), "messages": data.get("messages", []),
                                                "tools": data.get("tools"), "steer_positions": spans,
                                                "history_spans_skipped": history_skipped, "n_history": history_n.get(rkey),
                                                "history_boundary": boundary, "steer_history": steer_history})
            except Exception as e:  # noqa: BLE001
                print(f"dump skipped: {e!r}", flush=True)

        upstream_headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}
        client_streams = bool(data and data.get("stream"))
        stream_it = stream_upstream and is_chat and data is not None and not client_streams
        timeout = aiohttp.ClientTimeout(total=UPSTREAM_TIMEOUT_S, sock_read=SOCK_READ_TIMEOUT_S)

        async def send(req_body: bytes):
            to_send = req_body
            if stream_it:
                d2 = json.loads(req_body); d2["stream"] = True; d2["stream_options"] = {"include_usage": True}
                to_send = json.dumps(d2).encode()
            async with session.request(request.method, (up_for_request or pick_upstream()) + request.path_qs, data=to_send,
                                       headers=upstream_headers, timeout=timeout) as resp:
                if not stream_it or resp.status != 200 or "text/event-stream" not in (resp.headers.get("Content-Type") or ""):
                    return resp.status, dict(resp.headers), await resp.read()
                chunks = []
                async for line in iter_sse_lines(resp.content):
                    if not line.startswith("data:"): continue
                    pl = line[5:].strip()
                    if pl == "[DONE]": break
                    try: chunks.append(json.loads(pl))
                    except Exception: continue
                out = json.dumps(assemble_stream(chunks, data.get("model", ""))).encode()
                return 200, {"Content-Type": "application/json", "Content-Length": str(len(out))}, out

        async def watched(coro_factory):
            # Cancel the upstream generation if the client goes away.
            task = asyncio.create_task(coro_factory())
            while True:
                done, _ = await asyncio.wait({task}, timeout=5)
                if done:
                    return task.result()
                tr = request.transport
                if tr is None or tr.is_closing():
                    task.cancel()
                    print("client disconnected; upstream request cancelled", flush=True)
                    raise web.HTTPRequestTimeout(text="client disconnected")

        async def prefill_upstream(hit: dict):
            up = pick_upstream()
            async def post(path, obj):
                async with session.post(up + path, json=obj, timeout=timeout) as r:
                    return r.status, await r.read()
            st, b = await post("/tokenize", {"model": data.get("model"), "messages": data["messages"], "tools": data.get("tools"), "add_generation_prompt": True})
            if st != 200: return st, {"Content-Type": "application/json"}, b
            tk = json.loads(b); ids = tk["tokens"]; maxlen = tk.get("max_model_len") or 131072
            pids = []
            if hit["prefix"]:
                st, b = await post("/tokenize", {"model": data.get("model"), "prompt": hit["prefix"], "add_special_tokens": False})
                if st != 200: return st, {"Content-Type": "application/json"}, b
                pids = json.loads(b)["tokens"]
            prompt = ids + pids
            max_new = min(int(data.get("max_tokens") or max_tokens), maxlen - len(prompt) - CTX_MARGIN)
            if max_new <= 64:
                synth = synthetic_length_response(data.get("model", "")); synth["choices"][0]["message"]["reasoning_content"] = hit["prefix"]
                bb = json.dumps(synth).encode(); return 200, {"Content-Type": "application/json"}, bb
            req = {"model": data.get("model"), "prompt": prompt, "max_tokens": max_new, "temperature": data.get("temperature", 0.7),
                   "top_p": data.get("top_p", 0.95), "seed": data.get("seed"), "skip_special_tokens": False,
                   "stream": True, "stream_options": {"include_usage": True}}
            if "vllm_xargs" in data: req["vllm_xargs"] = data["vllm_xargs"]
            parts, finish, usage = [], None, None
            async with session.post(up + "/v1/completions", json=req, timeout=timeout) as resp:
                if resp.status != 200:
                    return resp.status, {"Content-Type": "application/json"}, await resp.read()
                async for line in iter_sse_lines(resp.content):
                    if not line.startswith("data:"): continue
                    pl = line[5:].strip()
                    if pl == "[DONE]": break
                    try: c = json.loads(pl)
                    except Exception: continue
                    if c.get("usage"): usage = c["usage"]
                    for ch in c.get("choices", []):
                        if ch.get("text"): parts.append(ch["text"])
                        if ch.get("finish_reason"): finish = ch["finish_reason"]
            text = "".join(parts).replace("<|im_end|>", "")
            gen_reason, rest = text.split("</think>", 1) if "</think>" in text else (text, "")
            calls = parse_qwen_tool_calls(rest, data.get("tools"))
            content = CALL_RE.sub("", rest).strip()
            msg = {"role": "assistant", "content": content or "\n", "reasoning_content": (hit["prefix"] + gen_reason).strip("\n")}
            if calls: msg["tool_calls"] = calls
            out = {"id": f"chatcmpl-prefill-{int(time.time())}", "object": "chat.completion", "created": int(time.time()), "model": data.get("model", ""),
                   "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if calls else (finish or "stop"), "logprobs": None}],
                   "usage": usage or {"prompt_tokens": len(prompt), "completion_tokens": 0, "total_tokens": len(prompt)},
                   "prefill": {"task_id": hit.get("task_id"), "cut": hit.get("cut"), "n_prompt_ids": len(prompt), "n_prefix_ids": len(pids), "finish": finish}}
            print(f"[PREFILL] {hit.get('task_id')} cut{hit.get('cut')} prompt_ids={len(prompt)} prefix_ids={len(pids)} "
                  f"gen_tokens={(usage or {}).get('completion_tokens')} finish={finish} calls={[c['function']['name'] for c in calls]}", flush=True)
            bb = json.dumps(out).encode()
            return 200, {"Content-Type": "application/json", "Content-Length": str(len(bb))}, bb

        hit = None
        if is_chat and data is not None:
            hit = prefill_table.get(fingerprint(data.get("messages", [])))
            if hit is None and all_completions:
                hit = {"task_id": "-", "cut": "-", "prefix": ""}
        if hit:
            status, headers, payload = await watched(lambda: prefill_upstream(hit))
        else:
            status, headers, payload = await watched(lambda: send(body))

        # context-overflow repair (chat path). Attempt 1: the window vLLM reports (a lower
        # bound, so it can overflow again). Attempt 2: cap removed, vLLM fits the true window.
        # Still overflowing: the prompt alone fills the window -> synthetic length finish.
        if status == 400 and is_chat and data is not None and not hit:
            allowed = parse_overflow(payload)
            if allowed is not None and allowed > 64:
                print(f"400 overflow; retrying with max_tokens={allowed}", flush=True)
                data["max_tokens"] = allowed
                status, headers, payload = await watched(lambda: send(json.dumps(data).encode()))
                if status == 400 and parse_overflow(payload) is not None:
                    print("400 overflow again; retrying with max_tokens removed", flush=True)
                    data.pop("max_tokens", None); data.pop("max_completion_tokens", None)
                    status, headers, payload = await watched(lambda: send(json.dumps(data).encode()))
                    if status == 400 and parse_overflow(payload) is not None:
                        allowed = 0
            if status == 400 and allowed is not None and allowed <= 64:
                print("400 overflow; window exhausted -> synthetic length finish", flush=True)
                synth = synthetic_length_response(data.get("model", ""))
                return web.Response(status=200, body=json.dumps(synth).encode(), content_type="application/json")

        n_fixed = 0
        if status == 200 and is_chat:
            try:
                resp = json.loads(payload)
                n_fixed = normalize_tool_args(resp, data.get("tools"))
                if n_fixed: print(f"renamed {n_fixed} mis-keyed tool argument(s) for {rkey}", flush=True)
                if collapse_tool_calls(resp) or n_fixed:
                    payload = json.dumps(resp).encode()
                if reasoning_budget and rkey is not None:
                    n_r = 0
                    for ch in resp.get("choices", []):
                        r_ = (ch.get("message") or {}).get("reasoning_content") or (ch.get("message") or {}).get("reasoning") or ""
                        if hit and hit.get("prefix") and r_.startswith(hit["prefix"]): r_ = r_[len(hit["prefix"]):]
                        if r_:
                            async with session.post(pick_upstream() + "/tokenize", json={"model": data.get("model"), "prompt": r_, "add_special_tokens": False},
                                                    timeout=aiohttp.ClientTimeout(total=120)) as tr:
                                n_r += (await tr.json()).get("count", 0) if tr.status == 200 else 0
                    budget_used[rkey] = budget_used.get(rkey, 0) + n_r
                    print(f"[BUDGET] {rkey} +{n_r} = {budget_used[rkey]}/{reasoning_budget}", flush=True)
            except Exception as e:  # never break the response path
                print(f"response post-processing skipped: {e!r}", flush=True)

        if ids_fh is not None and is_chat and status == 200 and req_sha is not None and not hit:
            try:
                resp = json.loads(payload)
            except Exception as e:  # noqa: BLE001
                print(f"ids-log: unparseable reply for {rkey}: {e!r}", flush=True)
            else:
                # steer_positions provenance: None = this proxy steers nothing; [] = steered, first
                # turn (no earlier reply); a list = the spans sent. A steered request whose
                # /tokenize failed never gets here (the request is refused), so a null never means
                # "positions unknown".
                line = ids_sidecar_line(resp, req_sha, spans, n_fixed, seed=data.get("seed"))
                payload = json.dumps(resp).encode()   # ids popped: Inspect never sees them, the eval log is unchanged
                if line is None:
                    if (resp.get("usage") or {}).get("prompt_tokens"):
                        print(f"ids-log: upstream returned no token_ids for {rkey} (return_token_ids ignored?)", flush=True)
                else:
                    line["steered"] = steer_payload is not None
                    line["n_render_ids"] = n_render
                    ids_fh.write(json.dumps(line) + "\n"); ids_fh.flush()
                    if n_render is not None and line["n_prompt_ids"] not in (None, n_render):
                        # the spans were computed on a render the server did not prefill: the
                        # steered positions of this turn are shifted, so the arm is invalid —
                        # fail the sample loudly rather than let the run finish looking normal
                        print(f"RENDER_MISMATCH {rkey}: /tokenize rendered {n_render} ids, the server prefilled "
                              f"{line['n_prompt_ids']} — steer_positions were shifted", flush=True)
                        raise web.HTTPBadRequest(text=f"steer_positions render mismatch: {n_render} vs {line['n_prompt_ids']} prompt ids")

        if dump_dir and is_chat and status == 200 and rkey is not None:
            try:
                write_dump(f"{rkey}.resp.json", {"ts": time.time(), "response": json.loads(payload)})
            except Exception as e:  # noqa: BLE001
                print(f"resp dump skipped: {e!r}", flush=True)

        return web.Response(status=status, body=payload, headers={
            k: v for k, v in headers.items() if k.lower() not in ("content-length", "transfer-encoding", "content-encoding")})

    app = web.Application(client_max_size=1 << 26)
    app.router.add_route("*", "/{tail:.*}", handle)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


async def serve(port: int, upstream: str = "http://localhost:8000", **kw) -> None:
    """Bind the proxy on 127.0.0.1:port and serve forever. kw as in make_app."""
    app = make_app(upstream, **kw)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    await asyncio.Event().wait()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--upstream", default="http://localhost:8000",
                   help="URL, comma-separated URLs (round-robin), or a file listing URLs one per line (re-read per request)")
    p.add_argument("--max-tokens", type=int, default=MAX_TOKENS, help=f"per-turn cap (default {MAX_TOKENS})")
    p.add_argument("--artifact", default=None, help="steering vector .safetensors (with .json sidecar); with --threshold enables clamp steering")
    p.add_argument("--mode", default="clamp", choices=["clamp", "add", "ablate"], help="steering mode (default clamp)")
    p.add_argument("--threshold", type=float, default=None, help="clamp mode: ceiling on h.u for every decoded token")
    p.add_argument("--scale", type=float, default=1.0, help="add mode: h += scale * v (v as stored in the artifact)")
    p.add_argument("--no-steer", action="store_true", help="explicit baseline: cap and hygiene only")
    p.add_argument("--replay-reasoning", action="store_true", help="rename assistant 'reasoning' -> 'reasoning_content' (diagnostic)")
    p.add_argument("--dump-dir", default=None, help="write each request's history and reply here (resumable rollouts)")
    p.add_argument("--no-stream-upstream", action="store_true", help="plain non-streaming upstream requests")
    p.add_argument("--prefill-table", default=None, help="JSON {fingerprint: {task_id, cut, prefix}}; see module docstring")
    p.add_argument("--all-completions", action="store_true", help="route every turn through /tokenize + /v1/completions with the proxy's own tool-call parser")
    p.add_argument("--reasoning-budget", type=int, default=None, help="stop each rollout after this many generated reasoning tokens")
    p.add_argument("--cache-salt", default=None, help="vLLM cache_salt stamped on every chat request (one per arm; see module docstring)")
    p.add_argument("--history-table", default=None, help="JSON {fingerprint: {task_id, n_history}}: never steer the recorded history; see module docstring")
    p.add_argument("--require-positions", action="store_true", help="(no-op since the merge of 2026-09-15: a steered request is always refused when its positions cannot be computed)")
    p.add_argument("--steer-history", action="store_true", help="with --history-table: steer the history's model-written spans too (as if the whole transcript had been generated under steering)")
    p.add_argument("--ids-log", default=None, help="append one JSON line per completed turn with the exact generated token ids "
                                                   "(and the prompt-ids hash + steer_positions) for later exact replay; see module docstring")
    args = p.parse_args()

    steer = (not args.no_steer) and args.artifact is not None
    if steer and args.mode == "clamp" and args.threshold is None:
        p.error("clamp mode needs --threshold (or pass --no-steer)")
    if not steer and args.threshold is not None and not args.no_steer:
        p.error("--threshold needs --artifact (or pass --no-steer)")
    payload = build_steer_payload(args.artifact, mode=args.mode, threshold=args.threshold, scale=args.scale) if steer else None
    table = json.load(open(args.prefill_table)) if args.prefill_table else None
    htable = json.load(open(args.history_table)) if args.history_table else None
    dose = {"clamp": f"thr={args.threshold}", "add": f"scale={args.scale}", "ablate": ""}[args.mode]
    modes = [f"{args.mode} {dose} {args.artifact}".replace("  ", " ") if steer else "PASSTHROUGH (cap only)"]
    if args.replay_reasoning: modes.append("replay-reasoning")
    if table: modes.append(f"prefill table ({len(table)} entries)")
    if args.all_completions: modes.append("all-completions")
    if args.reasoning_budget: modes.append(f"reasoning budget {args.reasoning_budget}")
    if args.cache_salt: modes.append(f"cache salt {args.cache_salt}")
    if htable is not None: modes.append(f"history table ({len(htable)} entries, history {'STEERED' if args.steer_history else 'never steered'})")
    if args.require_positions: modes.append("require-positions")
    if args.dump_dir: modes.append(f"dumps -> {args.dump_dir}")
    if args.ids_log: modes.append(f"ids sidecar -> {args.ids_log}")
    print(f"proxy :{args.port} -> {args.upstream} | max_tokens={args.max_tokens} | " + " | ".join(modes), flush=True)
    asyncio.run(serve(args.port, args.upstream, max_tokens=args.max_tokens, steer_payload=payload,
                      replay_reasoning=args.replay_reasoning, dump_dir=args.dump_dir,
                      stream_upstream=not args.no_stream_upstream, prefill_table=table,
                      all_completions=args.all_completions, reasoning_budget=args.reasoning_budget,
                      cache_salt=args.cache_salt, history_table=htable, steer_history=args.steer_history, ids_log=args.ids_log))


if __name__ == "__main__":
    main()
