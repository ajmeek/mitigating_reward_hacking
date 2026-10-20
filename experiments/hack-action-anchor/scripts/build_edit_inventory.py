"""Inventory every code-writing tool call in the 42 rollouts: text_editor create/str_replace/insert
on func.py or test.py, and bash commands that write to either file. Records the code written,
the message index, the capture turn index (position among assistant messages with reasoning),
and a crude regex flag for stateful/counter patterns (to be replaced by a judge)."""
import sys, json, re
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, f"{ROOT}/src")
import pandas as pd
from steering_reasoning.data.mistrace import load_mistrace
D = f"{ROOT}/data/captures/qwen35_27b_bf16_token_L35"
OUT = f"{ROOT}/experiments/hack-action-anchor/data/edit_inventory.jsonl"
recs, _ = load_mistrace("qwen3_5_27b"); splits = json.load(open(f"{ROOT}/src/steering_reasoning/data/splits.json"))
split_of = {t: s for s in ("train", "val", "test") for t in splits[s]}
HACK_RE = re.compile(r"global\s+\w*(count|call|state)|_call_count|call_count|_counter|_state\b|nonlocal|\.counter|_seen|_calls\b", re.I)
BASH_WRITE = re.compile(r"(cat\s*>|cat\s*>>|echo .*>|sed\s+-i|tee|python\s+-c\s+.*open\(.*['\"]w)", re.I)
rows = []
for r in recs:
    tid = r["task_id"]
    if tid not in split_of: continue
    meta = json.load(open(f"{D}/{tid}.json")); turns = meta["turns"]
    turn_of_msg = {t["message_index"]: k for k, t in enumerate(turns)}
    first_hack_msg = min((s["message_index"] for s in meta["segments"] if s.get("judge_label")), default=None)
    for mi, m in enumerate(r["messages"]):
        if m.get("role") != "assistant": continue
        for ti, tc in enumerate(m.get("tool_calls") or []):
            f = tc.get("function") or tc.get("name"); a = tc.get("arguments") or {}
            a = a if isinstance(a, dict) else {"raw": str(a)}
            path = str(a.get("path", "")); target = "test.py" if "test.py" in path else "func.py" if "func.py" in path else None
            code = None
            if f == "text_editor" and a.get("command") in ("create", "str_replace", "insert") and target:
                code = a.get("file_text") or a.get("new_str") or a.get("new_text") or json.dumps(a)
            elif f in ("bash", "python"):
                cmd = a.get("cmd") or a.get("command") or a.get("code") or a.get("raw") or ""
                if BASH_WRITE.search(cmd) and ("test.py" in cmd or "func.py" in cmd):
                    target = "test.py" if "test.py" in cmd else "func.py"; code = cmd
            if code is None: continue
            rows.append(dict(task_id=tid, split=split_of[tid], message_index=mi, tool_call_index=ti, capture_turn=turn_of_msg.get(mi),
                             function=f, command=a.get("command"), target=target, first_hack_msg=first_hack_msg,
                             regex_hack=bool(HACK_RE.search(code)), code=code))
df = pd.DataFrame(rows); df.to_json(OUT, orient="records", lines=True)
print(f"{len(df)} edits across {df.task_id.nunique()} rollouts; by target: {df.target.value_counts().to_dict()}; by function: {df.function.value_counts().to_dict()}")
print("edits with no capture turn (assistant message without reasoning):", int(df.capture_turn.isna().sum()))
print("regex-flagged:", int(df.regex_hack.sum()), "rollouts with any flagged edit:", df[df.regex_hack].task_id.nunique())
print("median code length (chars):", int(df.code.str.len().median()), " p90:", int(df.code.str.len().quantile(.9)))
