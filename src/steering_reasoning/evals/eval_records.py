"""Export replay records from Inspect eval logs: exactly what the model was asked, per turn.

A replay capture (steering/replay.py) must feed the model the same prompt the arms server
rendered. That prompt is a function of the OpenAI wire request Inspect sent — messages,
tools (with the provider's `strict` flag), completion params — after the arm proxy's
mutations. The logs written by the n5 arms carry no raw `call` (ModelEvent.call is None),
so the wire request is rebuilt here with Inspect's OWN converters, the code path the
provider used (`inspect_ai.model._providers.openai_compatible.OpenAICompatibleAPI.generate`:
`messages_to_openai` → `openai_chat_message`, `openai_chat_tools` + `strict`,
`openai_chat_tool_choice`, `openai_completion_params`), so the result is what left Inspect.
The proxy's mutations (strip_think on prior reasoning, max_tokens cap, tool-call collapse
on the *response*) are applied at replay time, not here: the record is the PRE-proxy
request, mirroring what `ModelEvent.call.request` holds when Inspect does log it.

Per turn the record also carries the logged response (post-proxy, as Inspect saw it),
`usage` (the server's own prompt/completion token counts — the length oracle for the
rendered prompt), and — joined from the proxy's ids sidecar (`steering_reasoning.proxy
--ids-log`) by a hash of the forwarded request — the exact generated `token_ids`, the
server's prompt-ids length + sha, and the `steer_positions` the proxy sent (the spans the
server steered during prefill), so a replay reproduces the turn without re-rendering.

Root-env module (inspect-ai only; no torch).
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Iterable

from inspect_ai.log import EvalLog, read_eval_log
from inspect_ai.model._openai import (
    messages_to_openai,
    openai_chat_message,
    openai_chat_tool_choice,
    openai_chat_tools,
    openai_completion_params,
)
from inspect_ai.scorer import CORRECT

from steering_reasoning.steering.clamp import proxy_view, request_sha

ARM_PATH_RE = re.compile(r"(?:^|/)(?P<arm>[^/]+)/(?P<split>conflicting|original)/s(?P<seed>\d+)(?:/|$)")
# The arm proxy stamps max_tokens AFTER Inspect records the request, so the log says None; the
# effective value comes from the proxy: the n5 arms (archived clamp proxy) capped at 16000,
# steering_reasoning.proxy caps at evals.config.MAX_TOKENS.
DEFAULT_MAX_TOKENS_EFFECTIVE = 16000


def parse_arm_path(path: str) -> dict:
    """{arm, split, seed} from `.../<ARM>/<split>/s<seed>/...`."""
    m = ARM_PATH_RE.search(path.rstrip("/") + "/")
    if not m:
        raise ValueError(f"cannot parse <arm>/<split>/s<seed> from {path!r}")
    return {"arm": m["arm"], "split": m["split"], "seed": int(m["seed"])}


def arm_steering(arms_tsv: str | Path, arm: str, repo_root: str | Path) -> tuple[str | None, float | None, str | None]:
    """(mode, dose, artifact) of `arm` from an experiment's arms.tsv — the runner's own table
    (scripts/run_arms.sh, fleet), so the export and the run agree by construction. mode None =
    unsteered; artifact is repo-relative."""
    from steering_reasoning.fleet.plan import read_arms

    rows = {r["name"]: r for r in read_arms(Path(repo_root), Path(arms_tsv))}
    if arm not in rows:
        raise ValueError(f"arm {arm!r} not in {arms_tsv} (has {sorted(rows)})")
    r = rows[arm]
    return (None if r["mode"] == "none" else r["mode"]), r["dose"], r["artifact"]


def service_model_name(inspect_model: str) -> str:
    """`openai-api/local/Qwen/Qwen3.5-27B` → `Qwen/Qwen3.5-27B` (what went on the wire)."""
    parts = inspect_model.split("/")
    return "/".join(parts[2:]) if parts[0] == "openai-api" and len(parts) > 2 else inspect_model


def _wire_request(event, model: str, strict_tools: bool) -> dict:
    messages = asyncio.run(messages_to_openai(event.input))
    tools = openai_chat_tools(event.tools) if event.tools else []
    for t in tools:  # OpenAICompatibleAPI.tools_to_openai
        t["function"]["strict"] = strict_tools
    params = openai_completion_params(model, event.config, tools=bool(tools))
    req = {"messages": messages, "tools": tools,
           "tool_choice": openai_chat_tool_choice(event.tool_choice) if tools else None}
    req.update(params)
    return req


def _tool_call_key(tc: dict) -> tuple:
    f = tc.get("function", {})
    return (f.get("name"), f.get("arguments"))


def _logged_ids(event) -> dict:
    """prompt_token_ids / token_ids / prompt_text when the proxy returned them (else None)."""
    out = {"prompt_token_ids": None, "prompt_text": None, "token_ids": None}
    call = getattr(event, "call", None)
    resp = getattr(call, "response", None) if call else None
    if isinstance(resp, dict):
        out["prompt_token_ids"] = resp.get("prompt_token_ids")
        out["prompt_text"] = resp.get("prompt_text")
        choices = resp.get("choices") or []
        if choices:
            out["token_ids"] = choices[0].get("token_ids")
    return out


def load_ids_log(paths: Iterable[str | Path]) -> dict[str, list[dict]]:
    """The proxy's ids sidecar (steering_reasoning.proxy --ids-log): one JSON line per response,
    keyed here by request_sha. A request that was retried (connection drop, overflow) has
    several lines, rollouts of different seeds share their first request, and so do all tasks
    of one seed (the task is in sandbox files); `join_ids` disambiguates by seed,
    completion_tokens and the model event's time window."""
    index: dict[str, list[dict]] = {}
    for p in paths:
        n_bad = 0
        with open(p) as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:  # a truncated tail (file synced while the proxy appends)
                    n_bad += 1
                    continue
                if not isinstance(row, dict) or "request_sha" not in row:
                    n_bad += 1
                    continue
                index.setdefault(row["request_sha"], []).append(row)
        if n_bad:
            print(f"load_ids_log: {p}: skipped {n_bad} unparseable line(s)")
    return index


# Clock slack between Inspect's event stamps and the proxy's `ts`. Both are stamped on the eval
# box and the proxy stamps just before Inspect receives the reply, so `ts` lies inside the event's
# [timestamp, completed] by construction; the slack only absorbs stamp granularity.
WINDOW_SLACK_S = 2.0


def join_ids(index: dict[str, list[dict]] | None, request: dict, usage: dict,
             window: tuple[float | None, float | None] | None = None) -> dict:
    """Sidecar fields for one turn: token_ids (exact generated ids), prompt_ids_sha256 and
    n_prompt_ids (byte identity of the server's prompt), steer_positions (the spans the proxy
    told the server to steer during prefill; None when the proxy sent none, [] for a first
    turn), plus how the join went.

    `window` = (start, end) of the turn's model event as POSIX seconds. The request hash does not
    identify a rollout on its first turn — the task lives in sandbox files, so every task's first
    request is byte-identical and one seed's sidecar holds one line per task under the same key
    — and equal completion lengths are common there; the proxy's `ts` (stamped when the reply
    finished) falling inside the event's window is what tells the lines apart."""
    out = {"token_ids": None, "prompt_ids_sha256": None, "n_prompt_ids": None, "steer_positions": None,
           "ids_join": "none"}
    if not index:
        return out
    key = request_sha(proxy_view(request["messages"]), request.get("tools"))
    seed = request.get("seed")
    cands = [c for c in index.get(key, []) if seed is None or c.get("seed") == seed]
    if not cands:
        out["ids_join"] = "missing"
        return out
    ct = usage.get("completion_tokens")
    match = [c for c in cands if ct is None or c.get("completion_tokens") == ct]
    if not match:
        out["ids_join"] = "length_mismatch"
        return out
    # Narrow by time only when every candidate carries a `ts`: a line without one (an older
    # sidecar mixed in via several --ids-log files) could otherwise be silently dropped in favour
    # of another rollout's line and reported as an exact join.
    if (window is not None and len({_ids_identity(c) for c in match}) > 1
            and all(isinstance(c.get("ts"), (int, float)) for c in match)):
        t0, t1 = window
        timed = [c for c in match if (t0 is None or c["ts"] >= t0 - WINDOW_SLACK_S)
                 and (t1 is None or c["ts"] <= t1 + WINDOW_SLACK_S)]
        if timed:
            match = timed
    row = match[-1]
    # Several lines with the same ids are one answer, not an ambiguity: a first turn shared by
    # two runs of the same seed (a restarted sample, the smoke), or a turn the client retried
    # after the server had already finished it.
    distinct = {_ids_identity(c) for c in match}
    out.update({"token_ids": row.get("token_ids"), "prompt_ids_sha256": row.get("prompt_ids_sha256"),
                "n_prompt_ids": row.get("n_prompt_ids"), "steer_positions": row.get("steer_positions"),
                "ids_join": "sidecar" if len(distinct) == 1 else "sidecar_ambiguous"})
    return out


def _ids_identity(c: dict) -> tuple:
    return (tuple(c.get("token_ids") or ()), c.get("prompt_ids_sha256"),
            json.dumps(c.get("steer_positions"), sort_keys=True))


def check_steering(mode: str | None, dose: float | None) -> None:
    if mode not in (None, "clamp", "add", "ablate"):
        raise ValueError(f"mode must be None|clamp|add|ablate, got {mode!r}")
    if (mode in ("clamp", "add")) != (dose is not None):
        raise ValueError(f"mode={mode} dose={dose}: clamp/add take a dose, none/ablate do not")


def sample_records(log: EvalLog, *, arm: str, mode: str | None, dose: float | None, seed: int, split: str,
                   eval_file: str, select: set[str] | None = None,
                   max_tokens_effective: int = DEFAULT_MAX_TOKENS_EFFECTIVE,
                   strict_tools: bool = True, ids_index: dict[str, list[dict]] | None = None) -> list[dict]:
    check_steering(mode, dose)
    if log.plan.config.seed != seed:
        raise ValueError(f"{eval_file}: log seed {log.plan.config.seed} != path seed {seed}")
    model = service_model_name(log.eval.model)
    records = []
    for s in log.samples or []:
        task_id = str(s.id)
        if select is not None and task_id not in select:
            continue
        passed = any(sc.value == CORRECT for sc in (s.scores or {}).values())
        md = s.metadata or {}
        # A rollout continued from its stored messages (the authors' extension of a shorter first pass)
        # holds the whole conversation but model events only from the continuation on: the
        # first `resumed_from_messages` messages are restored input, their turns live in the
        # base log (merge_extended puts the two back together).
        resumed = md.get("resumed_from_messages")
        first_gen = int(resumed) if resumed is not None else 0
        assistant_idx = [i for i, m in enumerate(s.messages) if m.role == "assistant" and i >= first_gen]
        events = [e for e in s.events if e.event == "model"]
        turns = []
        for e in events:
            if e.output is None or e.error is not None or not e.output.choices:
                continue
            mi = len(e.input)  # the event's input is the transcript prefix
            if mi >= len(s.messages) or s.messages[mi].role != "assistant":
                raise RuntimeError(f"{eval_file} {task_id}: model event at input length {mi} does "
                                   f"not align with an assistant message")
            out_msg = e.output.choices[0].message
            if s.messages[mi].id != out_msg.id:
                raise RuntimeError(f"{eval_file} {task_id} msg{mi}: event output id {out_msg.id} != "
                                   f"transcript message id {s.messages[mi].id}")
            usage = e.output.usage
            response = asyncio.run(openai_chat_message(out_msg))
            tcs = response.get("tool_calls") or []
            keys = [_tool_call_key(tc) for tc in tcs]
            request = _wire_request(e, model, strict_tools)
            usage_d = {"prompt_tokens": usage.input_tokens if usage else None,
                       "completion_tokens": usage.output_tokens if usage else None}
            turn = {
                "message_index": mi,
                "request": request,
                "response": response,
                "usage": usage_d,
                "finish_reason": e.output.choices[0].stop_reason,
                "synthetic_length": bool(usage and usage.input_tokens == 0),
                "n_tool_calls": len(tcs),
                # the proxy collapses >2 identical tool calls to the first 2; the log holds
                # the collapsed reply, so a repeated pair is the only trace of it
                "tool_calls_collapse_suspect": len(keys) != len(set(keys)),
                **_logged_ids(e),  # only populated when the eval ran with log_model_api
            }
            window = (e.timestamp.timestamp() if getattr(e, "timestamp", None) else None,
                      e.completed.timestamp() if getattr(e, "completed", None) else None)
            side = join_ids(ids_index, request, usage_d, window=window)
            for k, v in side.items():
                if turn.get(k) is None:
                    turn[k] = v
            turns.append(turn)
        if len(turns) != len(assistant_idx):
            raise RuntimeError(f"{eval_file} {task_id}: {len(turns)} usable model events for "
                               f"{len(assistant_idx)} assistant messages")
        records.append({
            "task_id": task_id,
            "rollout_id": f"{arm}/{split}/s{seed}/{task_id}",
            "arm": arm, "mode": mode, "dose": dose, "seed": seed, "split": split,
            "score": {k: v.value for k, v in (s.scores or {}).items()},
            "passed": passed,
            "is_reward_hack": passed if split == "conflicting" else None,
            "solved": passed if split == "original" else None,
            "limit": s.limit.model_dump() if s.limit else None,
            "error": str(s.error) if s.error else None,
            "resumed_from_messages": first_gen if resumed is not None else None,
            "extended_from_limit": md.get("extended_from_limit"),
            "eval_file": eval_file, "sample_uuid": s.uuid, "epoch": s.epoch,
            "model": model, "model_base_url": log.eval.model_base_url,
            "max_tokens_effective": max_tokens_effective,
            "messages": asyncio.run(messages_to_openai(s.messages)),
            "turns": turns,
        })
    return records


def export_records(eval_files: Iterable[str | Path], *, arm: str, mode: str | None, dose: float | None,
                   seed: int, split: str, select: Iterable[str] | None = None,
                   max_tokens_effective: int = DEFAULT_MAX_TOKENS_EFFECTIVE,
                   ids_logs: Iterable[str | Path] | None = None) -> list[dict]:
    """One record per selected sample. `mode`/`dose` = the arm's steering (clamp threshold or add
    scale; None/None unsteered), normally from its arms.tsv (`arm_steering`)."""
    sel = set(select) if select is not None else None
    ids_index = load_ids_log(ids_logs) if ids_logs else None
    out: list[dict] = []
    for f in eval_files:
        log = read_eval_log(str(f), resolve_attachments=True)
        # the provider's strict flag is rendered into the tools JSON the template sees
        strict = bool((log.eval.model_args or {}).get("strict_tools", True))
        out.extend(sample_records(log, arm=arm, mode=mode, dose=dose, seed=seed, split=split,
                                  eval_file=str(f), select=sel, strict_tools=strict,
                                  max_tokens_effective=max_tokens_effective, ids_index=ids_index))
    return out


def dedup_records(records: list[dict]) -> list[dict]:
    """One record per rollout_id. A sample that was run again (an errored or unscored sample
    restarted into the same directory) leaves several; a scored, error-free record beats the
    rest, then the latest eval file wins (log names start with their timestamp)."""
    best: dict[str, tuple[tuple, dict]] = {}
    for r in records:
        rank = (bool(r["score"]) and r["error"] is None, r["eval_file"])
        cur = best.get(r["rollout_id"])
        if cur is not None and (cur[1].get("limit") or {}).get("limit") != (r.get("limit") or {}).get("limit"):
            print(f"dedup_records: {r['rollout_id']}: duplicates disagree on the message limit "
                  f"({cur[1]['eval_file']} vs {r['eval_file']}) — are these logs from one run?")
        if cur is None or rank > cur[0]:
            best[r["rollout_id"]] = (rank, r)
    return [v[1] for v in best.values()]


def merge_extended(base: list[dict], ext: list[dict], *, ext_name: str) -> list[dict]:
    """Fold extension records (a rollout continued under a larger message limit from its stored
    messages) onto their base records, one record per rollout with
    the whole transcript. The merged record is the extension's (its messages, score, limit and
    error are the rollout's final state) with the base's turns followed by the extension's, the
    boundary in `resumed_from_messages`, and `extended`/`base_eval_file`/`base_limit` saying where
    it came from. Every base message must reappear unchanged at the head of the extension's
    (checked on the OpenAI-rendered messages), so a merge cannot pair the wrong logs silently.
    An extension that errored or was not scored is dropped and its base stands (as analyze.py
    does); a base without an extension passes through with extended=None."""
    base_by = {r["rollout_id"]: r for r in base}
    ext_by = {r["rollout_id"]: r for r in ext if r["score"] and r["error"] is None}
    stray = sorted(set(ext_by) - set(base_by))
    if stray:
        raise ValueError(f"{len(stray)} {ext_name} record(s) have no base record: {stray[:5]}")
    out = []
    for rid, b in base_by.items():
        e = ext_by.get(rid)
        if e is None:
            out.append({**b, "extended": None, "base_eval_file": None, "base_limit": None})
            continue
        n = e.get("resumed_from_messages")
        if n is None:
            raise ValueError(f"{rid}: {ext_name} record {e['eval_file']} carries no resumed_from_messages")
        if len(b["messages"]) != n or b["messages"] != e["messages"][:n]:
            raise ValueError(f"{rid}: the base record's {len(b['messages'])} messages are not the "
                             f"{n} restored at the head of {e['eval_file']} — wrong base log?")
        if any(t["message_index"] < n for t in e["turns"]):
            raise ValueError(f"{rid}: {ext_name} turn indices reach into the {n} restored messages")
        out.append({**e, "turns": b["turns"] + e["turns"], "extended": ext_name,
                    "base_eval_file": b["eval_file"], "base_limit": b["limit"]})
    return out
