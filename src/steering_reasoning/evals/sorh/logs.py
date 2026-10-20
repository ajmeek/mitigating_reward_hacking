"""Reading SoRH Inspect logs: the one FINISHED rule, resume bookkeeping, and per-sample extraction.

A sample is FINISHED when `sample.error` is None and the model returned something: a non-empty final
answer (`output.completion`), non-empty reasoning, or `usage.output_tokens > 0`. There is no scorer,
so nothing keys on scores. An answer that is empty because the proxy's cap was hit inside the
reasoning is a real, finished generation and is NOT re-run. An errored sample (after Inspect's
retries, with fail_on_error 0) IS present in the .eval with `error` set and must be re-run. An id that
appears once errored and once finished counts as finished. `finished_dict` (raw zip) and
`finished_sample` (inspect_ai object) are the same rule; keep them identical.

Every .eval under an arm's log dir (recursively, one rule for resume, status, grading and analysis) is
merged: a finished copy beats an errored one, and among finished copies the first sorted log wins
(`merge_views`, `collect_arm`). Only epoch-1 samples are analysed (the eval runs one epoch); records of
other epochs are skipped with a warning. Reading goes through `inspect_ai.log.read_eval_log`
when inspect_ai is importable and otherwise straight from the zip (`samples/*.json`); both give the
same rows, so `status`/`summarize` work on a box without the eval env.
"""

from __future__ import annotations

import collections
import glob
import json
import os
import re
import sys
import zipfile
from pathlib import Path

# ----------------------------------------------------------------------------- finished
def finished_dict(d: dict) -> bool:
    """FINISHED on a raw samples/<id>_epoch_N.json dict from a .eval zip."""
    if d.get("error") is not None:
        return False
    out = d.get("output") or {}
    if (out.get("completion") or "").strip():
        return True
    if ((out.get("usage") or {}).get("output_tokens") or 0) > 0:
        return True
    for ch in out.get("choices") or []:
        content = (ch.get("message") or {}).get("content")
        if isinstance(content, str) and content.strip():
            return True
        if isinstance(content, list):
            for c in content:
                if isinstance(c, dict) and (c.get("reasoning") or c.get("text") or "").strip():
                    return True
    return False


def finished_sample(s) -> bool:
    """FINISHED on an inspect_ai EvalSample (same rule as finished_dict)."""
    if s.error is not None:
        return False
    out = s.output
    if out is None:
        return False
    if (out.completion or "").strip():
        return True
    if out.usage is not None and (out.usage.output_tokens or 0) > 0:
        return True
    for ch in out.choices or []:
        content = ch.message.content
        if isinstance(content, str) and content.strip():
            return True
        if isinstance(content, list):
            for c in content:
                t = getattr(c, "type", "")
                part = (getattr(c, "reasoning", "") if t == "reasoning" else getattr(c, "text", "") if t == "text" else "") or ""
                if part.strip():
                    return True
    return False


# ----------------------------------------------------------------------------- resume / status
def eval_files(log_dir: str | Path, recursive: bool = True) -> list[str]:
    pat = os.path.join(str(log_dir), "**", "*.eval") if recursive else os.path.join(str(log_dir), "*.eval")
    return sorted(glob.glob(pat, recursive=recursive))


def finished_ids(log_dir: str | Path) -> set[str]:
    """Ids with a FINISHED sample in any .eval under log_dir (authoritative:
    inspect_ai.log.read_eval_log; an unreadable log is skipped with a warning and its ids re-run)."""
    from inspect_ai.log import read_eval_log

    done = set()
    for f in eval_files(log_dir):
        try:
            log = read_eval_log(f)
        except Exception as e:  # a half-written zip: its samples are simply re-run
            print(f"WARN unreadable {f}: {e!r}", file=sys.stderr)
            continue
        for s in (log.samples or []):
            if finished_sample(s):
                done.add(str(s.id))
    return done


def remaining_ids(log_dir: str | Path, ids: list[str]) -> list[str]:
    """The ids (input order kept) that do NOT yet have a FINISHED sample under log_dir."""
    done = finished_ids(log_dir)
    return [i for i in ids if i not in done]


def scan_dir(log_dir: str | Path) -> dict:
    """Cheap raw zip scan (no inspect_ai): finished, errored-only, unreadable files, files, cap hits."""
    finished, errored, files, bad, caps = set(), set(), 0, 0, set()
    for f in eval_files(log_dir):
        files += 1
        try:
            with zipfile.ZipFile(f) as z:
                for n in z.namelist():
                    if not (n.startswith("samples/") and n.endswith(".json")):
                        continue
                    try:
                        d = json.loads(z.read(n))
                    except Exception:
                        continue
                    sid = str(d.get("id"))
                    if finished_dict(d):
                        finished.add(sid)
                        chs = (d.get("output") or {}).get("choices") or []
                        if chs and chs[0].get("stop_reason") == "max_tokens":
                            caps.add(sid)
                    elif d.get("error") is not None:
                        errored.add(sid)
        except (zipfile.BadZipFile, OSError):
            bad += 1
    return dict(finished=len(finished), errored_only=len(errored - finished), unreadable=bad,
                files=files, cap_hits=len(caps))


# ----------------------------------------------------------------------------- views for the analysis
def _reasoning_text(msg) -> str:
    c = msg.content
    if isinstance(c, str):
        return ""
    return "\n".join((getattr(x, "reasoning", "") or "") for x in c if getattr(x, "type", "") == "reasoning")


def view_from_sample(s) -> dict:
    """The fields the analysis needs, from an inspect_ai EvalSample."""
    epoch = int(getattr(s, "epoch", 1) or 1)
    if not finished_sample(s):
        return dict(id=str(s.id), epoch=epoch, finished=False, error=str(s.error)[:160] if s.error else "no output",
                    total_time=s.total_time)
    out = s.output
    ch = out.choices[0] if out.choices else None
    usage = out.usage
    if usage is None and s.model_usage:
        usage = next(iter(s.model_usage.values()))
    return dict(id=str(s.id), epoch=epoch, finished=True, total_time=s.total_time,
                output_tokens=(usage.output_tokens if usage else None),
                input_tokens=(usage.input_tokens if usage else None),
                stop_reason=(ch.stop_reason if ch else ""), answer=out.completion or "",
                reasoning_chars=(len(_reasoning_text(ch.message)) if ch else 0))


def view_from_dict(d: dict) -> dict:
    """The same fields from the raw samples/<id>_epoch_N.json of a .eval zip (no inspect_ai). Mirrors
    the inspect_ai objects: output.completion is the "\\n"-join of the text parts of the first
    choice's message; ModelUsage tokens default to 0; str(EvalError) is pydantic's "field=value" form."""
    epoch = int(d.get("epoch", 1) or 1)
    if not finished_dict(d):
        err = d.get("error")
        text = (" ".join(f"{k}={v!r}" for k, v in err.items()) if isinstance(err, dict) else str(err))[:160] if err else "no output"
        return dict(id=str(d.get("id")), epoch=epoch, finished=False, error=text, total_time=d.get("total_time"))
    out = d.get("output") or {}
    chs = out.get("choices") or []
    ch = chs[0] if chs else None
    usage = out.get("usage")
    if usage is None and d.get("model_usage"):
        usage = next(iter(d["model_usage"].values()))
    content = (ch.get("message") or {}).get("content") if ch else None
    if isinstance(content, str):
        answer, reasoning_chars = content, 0
    elif isinstance(content, list):
        answer = "\n".join((x.get("text") or "") for x in content if x.get("type") == "text")
        reasoning_chars = len("\n".join((x.get("reasoning") or "") for x in content if x.get("type") == "reasoning"))
    else:
        answer, reasoning_chars = "", 0
    return dict(id=str(d.get("id")), epoch=epoch, finished=True, total_time=d.get("total_time"),
                output_tokens=(usage.get("output_tokens", 0) if usage else None),
                input_tokens=(usage.get("input_tokens", 0) if usage else None),
                stop_reason=(ch.get("stop_reason", "unknown") if ch else ""), answer=answer or "",
                reasoning_chars=reasoning_chars)


def read_views(path: str) -> list[dict]:
    """[view] for every sample of one .eval: inspect_ai when importable, else the raw zip. Raises on an
    unreadable log (the caller skips the whole file, as read_eval_log would)."""
    try:
        from inspect_ai.log import read_eval_log
    except ImportError:
        read_eval_log = None
    if read_eval_log is not None:
        return [view_from_sample(s) for s in (read_eval_log(path).samples or [])]
    views = []
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            if n.startswith("samples/") and n.endswith(".json"):
                views.append(view_from_dict(json.loads(z.read(n))))
    return views


def merge_views(log_dir: str | Path) -> dict[str, tuple[dict, str]]:
    """{id: (view, log basename)} over every .eval under log_dir: a finished copy beats an errored one;
    among finished copies the first sorted log wins. Epoch-1 samples only (others are skipped, warned once)."""
    seen: dict[str, tuple[dict, str]] = {}
    skipped_epochs = 0
    for f in eval_files(log_dir):
        try:
            views = read_views(f)
        except Exception as e:
            print(f"WARN unreadable {f}: {e!r}", file=sys.stderr)
            continue
        for v in views:
            if v.get("epoch", 1) != 1:
                skipped_epochs += 1
                continue
            sid = v["id"]
            if sid in seen and (seen[sid][0]["finished"] or not v["finished"]):
                continue
            seen[sid] = (v, os.path.basename(f))
    if skipped_epochs:
        print(f"WARN {log_dir}: {skipped_epochs} sample(s) with epoch > 1 skipped; only epoch 1 is analysed", file=sys.stderr)
    return seen


# ----------------------------------------------------------------------------- extraction for the judge
def _get(obj, name, default=None):
    """Attribute or dict lookup that tolerates None (inspect objects and plain dicts alike)."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def split_content(content):
    """(reasoning, text) concatenated from a message content that is a str or a list of parts."""
    if content is None:
        return "", ""
    if isinstance(content, str):
        return "", content
    reasoning, text = [], []
    for part in content:
        typ = _get(part, "type")
        if typ == "reasoning":
            reasoning.append(_get(part, "reasoning") or "")
        elif typ == "text":
            text.append(_get(part, "text") or "")
    return "".join(reasoning), "".join(text)


def repeat_stats(text: str) -> int:
    """Longest run of one >40-char sentence repeated (a mechanical loop signal)."""
    sents = [x.strip() for x in re.split(r"(?<=[.!?])\s+|\n", text or "") if len(x.strip()) > 40]
    if not sents:
        return 0
    return collections.Counter(sents).most_common(1)[0][1]


def extract_sample(sample) -> dict:
    """Everything the judge and the report need from one Inspect EvalSample (or its raw dict).

    Final answer = the text parts of the LAST assistant message; reasoning = its reasoning parts.
    Stop reason and token counts come from sample.output, falling back to the sample's model events.
    A sample with `.error` is ungradable and never sent to the judge; so is one the runner does not
    count as finished. A cap hit inside the reasoning (empty answer, tokens spent) and a
    reasoning-only sample are finished and stay gradable.
    """
    sid = str(_get(sample, "id"))
    epoch = int(_get(sample, "epoch", 1) or 1)
    err = _get(sample, "error")
    msgs = _get(sample, "messages") or []
    asst = [m for m in msgs if _get(m, "role") == "assistant"]
    users = [m for m in msgs if _get(m, "role") == "user"]
    reasoning, answer = split_content(_get(asst[-1], "content")) if asst else ("", "")
    user_text = split_content(_get(users[0], "content"))[1] if users else ""

    stop_reason, in_tok, out_tok, reas_tok, finish_raw = None, None, None, None, None
    out = _get(sample, "output")
    choices = _get(out, "choices") or []
    if choices:
        stop_reason = _get(choices[0], "stop_reason")
    usage = _get(out, "usage")
    if usage is not None:
        in_tok = _get(usage, "input_tokens")
        out_tok = _get(usage, "output_tokens")
        reas_tok = _get(usage, "reasoning_tokens")

    events = [e for e in (_get(sample, "events") or []) if _get(e, "event") == "model"]
    if events:
        last = events[-1]
        ech = _get(_get(last, "output"), "choices") or []
        if stop_reason is None and ech:
            stop_reason = _get(ech[0], "stop_reason")
        if out_tok is None:
            toks = [_get(_get(_get(e, "output"), "usage"), "output_tokens") for e in events]
            toks = [t for t in toks if t is not None]
            out_tok = sum(toks) if toks else None
        if reas_tok is None:
            toks = [_get(_get(_get(e, "output"), "usage"), "reasoning_tokens") for e in events]
            toks = [t for t in toks if t is not None]
            reas_tok = sum(toks) if toks else None
        resp = _get(_get(last, "call"), "response")
        if isinstance(resp, dict):
            ch = (resp.get("choices") or [{}])[0]
            finish_raw = ch.get("finish_reason") if isinstance(ch, dict) else None
            ru = resp.get("usage") or {}
            if out_tok is None and isinstance(ru, dict):
                out_tok = ru.get("completion_tokens")

    ungradable = None
    if err is not None:
        ungradable = "sample_error: " + str(_get(err, "message", err) or err)[:300]
    elif not asst:
        ungradable = "no_assistant_message"
    elif not answer.strip() and not reasoning.strip() and not (_get(usage, "output_tokens") or 0) > 0:
        ungradable = "empty_output"

    return dict(
        sample_id=sid, epoch=epoch, answer=answer, reasoning=reasoning, user_text=user_text,
        stop_reason=stop_reason, finish_reason_raw=finish_raw, input_tokens=in_tok,
        output_tokens=out_tok, reasoning_tokens=reas_tok,
        limit=_get(_get(sample, "limit"), "type"), n_assistant=len(asst),
        answer_chars=len(answer), reasoning_chars=len(reasoning), max_repeat=repeat_stats(answer),
        ungradable=ungradable,
    )


def collect_arm(logs_dir: str | Path, arm: str, rel_to: str | Path | None = None) -> tuple[dict, list[str]]:
    """Merge every .eval under <logs_dir>/<arm>/ into {(sample_id, epoch): extraction}; a gradable copy
    beats an ungradable one. Returns (merged, log files)."""
    from inspect_ai.log import read_eval_log

    seen: dict = {}
    files = eval_files(os.path.join(str(logs_dir), arm))
    for f in files:
        try:
            ev = read_eval_log(f)
        except Exception as ex:  # a log still being written, or corrupt
            print(f"WARN {arm}: cannot read {f}: {ex}", file=sys.stderr)
            continue
        for s in (ev.samples or []):
            ex = extract_sample(s)
            ex["log_file"] = os.path.relpath(f, rel_to) if rel_to else f
            ex["log_status"] = ev.status
            key = (ex["sample_id"], ex["epoch"])
            prev = seen.get(key)
            if prev is None or (prev["ungradable"] and not ex["ungradable"]):
                seen[key] = ex
    return seen, files
