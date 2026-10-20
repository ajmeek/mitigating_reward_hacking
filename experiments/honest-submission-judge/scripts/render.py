"""Render one inspect sample (an ImpossibleBench rollout) as plain text for the honest-submission judge.

Every message is kept in order with a numbered header. Assistant turns show the reasoning, any visible text and
each tool call with its arguments written out (real newlines, not JSON escapes). Tool outputs are kept verbatim
unless `tool_cap` is set, in which case each output longer than the cap keeps its head and tail.
"""
import json
import os
import zipfile

DATA = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "../../diff-mean-steering/data"))


def load_sample(eval_path: str, task: str, data_root: str = DATA) -> dict:
    z = zipfile.ZipFile(eval_path if eval_path.startswith("/") else f"{data_root}/{eval_path}")
    names = [n for n in z.namelist() if n.startswith(f"samples/{task}_epoch_")]
    assert len(names) == 1, (eval_path, task, names)
    return json.loads(z.read(names[0]))


def _args(arguments) -> str:
    if not isinstance(arguments, dict):
        return json.dumps(arguments)
    out = []
    for k, v in arguments.items():
        v = v if isinstance(v, str) else json.dumps(v)
        out.append(f"{k}:\n{v}" if "\n" in v or len(v) > 80 else f"{k}: {v}")
    return "\n".join(out)


def _cap(text: str, cap: int | None) -> str:
    if cap is None or len(text) <= cap:
        return text
    h = cap // 2
    return f"{text[:h]}\n[... {len(text) - cap} characters of tool output omitted ...]\n{text[-h:]}"


def render(sample: dict, tool_cap: int | None = None) -> str:
    msgs = sample["messages"]
    n = len(msgs)
    parts = []
    for i, m in enumerate(msgs, 1):
        role = m["role"]
        c = m.get("content")
        if role == "assistant":
            body = []
            for b in (c if isinstance(c, list) else [{"type": "text", "text": c or ""}]):
                if b.get("type") == "reasoning" and (b.get("reasoning") or "").strip():
                    body.append(f"<reasoning>\n{b['reasoning'].strip()}\n</reasoning>")
                elif b.get("type") == "text" and (b.get("text") or "").strip():
                    body.append(b["text"].strip())
            for t in m.get("tool_calls") or []:
                body.append(f"[tool call: {t.get('function')}]\n{_args(t.get('arguments'))}")
            parts.append(f"=== MESSAGE {i}/{n} | ASSISTANT ===\n" + "\n\n".join(body))
        elif role == "tool":
            text = c if isinstance(c, str) else "\n".join(b.get("text", "") for b in c or [] if isinstance(b, dict))
            parts.append(f"=== MESSAGE {i}/{n} | TOOL OUTPUT ({m.get('function') or 'tool'}) ===\n{_cap(text, tool_cap)}")
        else:
            text = c if isinstance(c, str) else "\n".join(b.get("text", "") for b in c or [] if isinstance(b, dict))
            parts.append(f"=== MESSAGE {i}/{n} | {role.upper()} ===\n{text.strip()}")
    return "\n\n".join(parts)
