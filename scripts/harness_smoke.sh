#!/usr/bin/env bash
# Smoke-test a freshly provisioned GPU box end to end. Runs ON the box (it has the eval env,
# Docker, and the server on localhost:8000). Usage: bash scripts/harness_smoke.sh [MODEL] [SAMPLE_ID]
# Checks, in order: server up with all four vllm-lens patch lines; tool-calling grammar off
# (probe_strict: new_str present under tool_choice=auto); steering live (steer_probe: base == direct,
# base != clamp); one short Inspect rollout through the cap-only proxy and one through the clamp
# proxy, then every str_replace in both logs carries new_str.
set -uo pipefail
cd ~/steering_reasoning; export PATH=$HOME/.local/bin:$PATH; export LOCAL_API_KEY=dummy
MODEL="${1:-Qwen/Qwen3.5-27B}"; SID="${2:-lcbhard_6}"
ART=src/steering_reasoning/canonical/u_L35_steering.safetensors   # canonical bf16 u (CLAUDE.md)
echo "== 1. server"; for i in $(seq 1 120); do curl -s -m 3 localhost:8000/v1/models | grep -q Qwen && break; sleep 15; done
curl -s -m 3 localhost:8000/v1/models | grep -q Qwen && echo "UP" || { echo "server not up"; exit 1; }
grep -c "schema patched\|worker patched\|decode-only gate\|prefix-cache reads" ~/serve.log | sed 's/^/patch lines (expect 4): /'
grep -o "SERVE_STARTING.*" ~/serve.log | tail -1
echo; echo "== 2. tool-calling grammar (probe_strict)"
python3 scripts/probe_strict.py http://localhost:8000 "$MODEL" scripts/inspect_tools.json 2>&1 | grep -A1 "tool_choice=auto" | head -3
python3 scripts/probe_strict.py http://localhost:8000 "$MODEL" scripts/inspect_tools.json 2>&1 | grep "parsed" | head -1
echo; echo "== 3. steering (steer_probe)"
MODEL="$MODEL" uv run python scripts/steer_probe.py http://localhost:8000 -8 2>&1 | tail -1
pkill -f 'steering_reasoning\.proxy .*--port 810[0-9]' 2>/dev/null; sleep 1   # only this script's :8100/:8104, never an arm's proxy
echo; echo "== 4. one rollout each through the cap proxy (:8100) and the clamp proxy (:8104)"
mkdir -p ~/smoke
nohup uv run python -m steering_reasoning.proxy --port 8100 --upstream http://localhost:8000 --dump-dir ~/smoke/dumps_base > ~/smoke/proxy_base.log 2>&1 &
nohup uv run python -m steering_reasoning.proxy --port 8104 --upstream http://localhost:8000 --artifact $ART --threshold -8 --dump-dir ~/smoke/dumps_clamp > ~/smoke/proxy_clamp.log 2>&1 &
for i in $(seq 1 30); do curl -s -m 3 localhost:8100/v1/models | grep -q Qwen && curl -s -m 3 localhost:8104/v1/models | grep -q Qwen && break; sleep 2; done
PIDS=""
for arm in base clamp; do
  port=8100; [ $arm = clamp ] && port=8104
  uv run scripts/run_lcb_eval.py --split conflicting --sample-ids "$SID" --model "openai-api/local/$MODEL" --base-url http://localhost:$port/v1 \
    --seed 42 --max-connections 1 --max-sandboxes 1 --message-limit 14 --fail-on-error 0 --log-dir ~/smoke/logs/$arm > ~/smoke/eval_$arm.log 2>&1 &
  PIDS="$PIDS $!"
done
wait $PIDS   # the evals only; the proxies are background daemons and would block a bare wait
uv run python - <<'EOF'
import glob, os
from inspect_ai.log import read_eval_log
for arm in ("base", "clamp"):
    for f in glob.glob(os.path.expanduser(f"~/smoke/logs/{arm}/*.eval")):
        log = read_eval_log(f)
        for s in log.samples or []:
            sr = ok = 0; calls = []
            for m in s.messages:
                if m.role != "assistant": continue
                for tc in m.tool_calls or []:
                    a = tc.arguments or {}; calls.append(tc.function + (":" + str(a.get("command")) if tc.function == "text_editor" else ""))
                    if tc.function == "text_editor" and a.get("command") == "str_replace": sr += 1; ok += bool(a.get("new_str"))
            print(f"{arm}: {s.id} status={log.status} error={bool(s.error)} messages={len(s.messages)} calls={calls} str_replace={sr} with_new_str={ok}")
EOF
grep -h "PASSTHROUGH\|clamp thr" ~/smoke/proxy_*.log
pkill -f 'steering_reasoning\.proxy .*--port 810[0-9]' 2>/dev/null
echo "== done"
