#!/usr/bin/env python
"""Render Inspect .eval samples as readable Markdown: per assistant turn the reasoning, the text,
each tool call with its arguments, and the tool result; header with score, turns and token usage.

    uv run python scripts/render_eval_transcript.py <log.eval or dir> [--out transcript.md] [--max-chars 2500]
"""
import argparse, glob, json, os
from inspect_ai.log import read_eval_log


def text_of(m):
    if isinstance(m.content, str): return m.content
    return "".join(getattr(c, "text", "") for c in m.content if getattr(c, "type", "") == "text")


def reasoning_of(m):
    if isinstance(m.content, str): return ""
    return "".join(getattr(c, "reasoning", "") for c in m.content if getattr(c, "type", "") == "reasoning")


def clip(s, n):
    s = s or ""
    return s if len(s) <= n else s[:n] + f"\n… [{len(s) - n} more chars]"


def render(path, max_chars, sample=None):
    log = read_eval_log(path); out = []
    for s in log.samples or []:
        if sample is not None and str(s.id) != str(sample): continue
        sc = list(s.scores.values())[0] if s.scores else None; score = sc.value if sc else None
        n_asst = sum(1 for m in s.messages if m.role == "assistant")
        usage = getattr(log.stats, "model_usage", {}) or {}
        out.append(f"# {s.id} — {log.eval.task} (status {log.status}, score {score}, {n_asst} assistant turns, {len(s.messages)} messages)\n")
        if sc is not None and getattr(sc, "explanation", None): out.append(f"Scorer: {str(sc.explanation)[:1500]}\n")
        if s.limit: out.append(f"Ended on limit: {s.limit.type} = {s.limit.limit}\n")
        out.append(f"Log: `{os.path.basename(path)}`  \nModel: {log.eval.model}  \nUsage: {json.dumps({k: {'in': v.input_tokens, 'out': v.output_tokens} for k, v in usage.items()})}\n")
        if s.error: out.append(f"**Sample error:** {s.error}\n")
        k = 0
        for m in s.messages:
            if m.role == "system":
                out.append(f"<details><summary>system</summary>\n\n```\n{clip(text_of(m), max_chars)}\n```\n</details>\n")
            elif m.role == "user":
                out.append(f"### user\n\n```\n{clip(text_of(m), max_chars)}\n```\n")
            elif m.role == "assistant":
                k += 1; r = reasoning_of(m); t = text_of(m)
                out.append(f"## assistant turn {k}\n")
                if r: out.append(f"**reasoning** ({len(r)} chars)\n\n```\n{clip(r, max_chars)}\n```\n")
                if t.strip(): out.append(f"**text**\n\n```\n{clip(t, max_chars)}\n```\n")
                for tc in m.tool_calls or []:
                    a = tc.arguments or {}
                    if tc.function == "text_editor":
                        head = {kk: v for kk, v in a.items() if kk not in ("file_text", "new_str", "old_str")}
                        out.append(f"**tool call** `text_editor` {json.dumps(head)}\n")
                        for kk in ("old_str", "new_str", "file_text"):
                            if kk in a and a[kk] is not None: out.append(f"`{kk}`:\n\n```python\n{clip(str(a[kk]), max_chars)}\n```\n")
                    elif tc.function in ("bash", "python"):
                        code = a.get("cmd") or a.get("code") or json.dumps(a)
                        out.append(f"**tool call** `{tc.function}`\n\n```\n{clip(code, max_chars)}\n```\n")
                    else:
                        out.append(f"**tool call** `{tc.function}` {json.dumps(a)[:500]}\n")
            elif m.role == "tool":
                err = getattr(m, "error", None)
                out.append(f"**tool result** (`{m.function}`{', ERROR' if err else ''})\n\n```\n{clip(text_of(m), max_chars)}\n```\n")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("path"); ap.add_argument("--out", default=None); ap.add_argument("--max-chars", type=int, default=2500)
    ap.add_argument("--sample", default=None, help="render only this sample id (a batch log holds several)")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.path, "**", "*.eval"), recursive=True)) if os.path.isdir(a.path) else [a.path]
    md = "\n\n---\n\n".join(render(f, a.max_chars, a.sample) for f in files)
    if a.out:
        open(a.out, "w").write(md); print("wrote", a.out)
    else:
        print(md)


if __name__ == "__main__":
    main()
