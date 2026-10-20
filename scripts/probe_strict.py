#!/usr/bin/env python3
"""Probe a vLLM server for the strict-tool-calling grammar defect: same prompt through
/v1/chat/completions with tool_choice auto and none, raw tokens via logprobs, plus the parsed
tool calls. Usage: python3 probe_strict.py http://localhost:8000 Qwen/Qwen3.5-27B tools.json"""
import json, sys, urllib.request
U, MODEL, TOOLS = sys.argv[1], sys.argv[2], json.load(open(sys.argv[3]))
msgs=[{"role":"system","content":"You are a coding agent. Use the text_editor tool."},
      {"role":"user","content":"In /func.py replace the exact text 'raise NotImplementedError' with a two-line implementation 'x = 1\\n    return x'. Use the text_editor str_replace command now, no explanation."}]
for choice in ("auto","none"):
    body={"model":MODEL,"messages":msgs,"tools":TOOLS,"temperature":0,"max_tokens":600,"logprobs":True,"top_logprobs":0,"tool_choice":choice}
    d=json.load(urllib.request.urlopen(urllib.request.Request(U+"/v1/chat/completions",json.dumps(body).encode(),{"Content-Type":"application/json"}),timeout=900))
    ch=d["choices"][0]; raw="".join(t["token"] for t in ((ch.get("logprobs") or {}).get("content") or []))
    i=raw.find("<tool_call>")
    print(f"tool_choice={choice}: raw generation from <tool_call>:\n{raw[i:i+500] if i>=0 else raw[-400:]}\n")
    print("  parsed:", [(tc["function"]["name"], tc["function"]["arguments"][:160]) for tc in ch["message"].get("tool_calls") or []], "\n")
