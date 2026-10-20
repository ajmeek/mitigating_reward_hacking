#!/usr/bin/env bash
# Launch (or resume) the School of Reward Hacks (SoRH) steering arms on this box: one steering
# proxy per arm (root, eval venv), one Inspect eval per arm, two vLLM upstreams (GPU0 :8000,
# GPU1 :8002, identical config, MAX_NUM_SEQS 32; with one server set UP1=8000). NOT :8001 for GPU1 on RunPod: its system nginx holds
# 8001 and silently reverse-proxies it to :8000, so anything pointed at 8001 reaches GPU0
# (infra note 2026-09-13 08:05Z in /workspace/logs/sorh-status.md). Adapted from
# experiments/opt-vector-lcb/scripts/run_arms.sh. The proxy is the repo's src/steering_reasoning/proxy.py
# (`$PY -m steering_reasoning.proxy`; the eval venv must have the repo installed), launched with
# --max-tokens 16384 (the repo default cap). The runs of record went through the LCB copy of the older
# proxy, which capped at 16000 (same steering math via the same server patch; kept on the box, not in git);
# on these single-turn prompts both steer exactly the generated tokens. No sandbox, no eval user: the task
# is single-turn with no tools, so everything runs as root.
#
# Arms (name : proxy port : upstream : steering). Doses are the LCB doses of record
# (experiments/opt-vector/HANDOFF.md); u_cl8 is Kyle's paper vector (layer 35) at his dose of record.
#   base    8100  :8000  --no-steer
#   v3b     8101  :8000  add   scale 0.75  opt_27b_L31_v3b     (positive = AWAY from hacking)
#   v3c     8102  :8000  add   scale 0.5   opt_27b_L31_v3c
#   v3b_cl  8103  :8002  clamp thr -5     opt_27b_L31_v3b_cl  (positive = TOWARD hacking, unit norm)
#   v3c_cl  8104  :8002  clamp thr -8     opt_27b_L31_v3c_cl
#   u_cl8   8105  :8002  clamp thr -8     the canonical u (src/steering_reasoning/canonical/u_L35_steering.safetensors)
#   base2   8106  :8002  --no-steer       VERIFICATION ONLY: started as a proxy, NEVER evaluated;
#                                         `run_eval.py check-servers` compares it with base to prove
#                                         the two servers are byte-identical unsteered (base==base2 at
#                                         temperature 0). It is NOT passed to `run_eval.py
#                                         check-delivery`, which asserts base!=X for every non-base arm
#                                         and would fail on it.
#
# Per arm: one Inspect eval over the 1,073 prompts (ids sorh_0..sorh_1072), no scorer. Re-running
# this script resumes: ids that already have a FINISHED sample (error None and a non-empty output,
# see `scripts/run_eval.py remaining`) in any .eval under data/logs/<arm>/ are skipped and only the
# rest are re-run, so a second .eval lands in the same dir and analyze.py merges. An arm whose
# runner is still alive is skipped (no double launch). Proxies that answer are left alone unless
# FORCE_PROXIES=1.
#
# This is now the only launcher. The dose sweep (data/sweep/: 26 cells x the 100 dev prompts) was run
# with it too: each sweep cell is one ARM_SPECS entry name:port:upstream:mode:artifact:dose carrying
# the RAW proxy dose - mode add with scale = k/24 for "k units" along the unit vector (8 units =
# 0.3333, 10 units = 0.4167, 18 units = 0.75), mode clamp with the threshold itself (v3b_cl_t-5 ->
# -5) - and IDS set to the ids in artifacts/dev/ids.json (key "ids"), with LOGROOT and PROXYLOG
# pointed at data/sweep/logs and data/sweep/proxies (sweep proxies sat on ports 8110-8135), e.g.
#   ARM_SPECS="v3b_k8:8113:8002:add:<repo>/experiments/opt-vector/artifacts/opt_27b_L31_v3b.safetensors:0.3333" \
#   IDS="$(python3 -c 'import json;print(" ".join(json.load(open("artifacts/dev/ids.json"))["ids"]))')" \
#   LOGROOT=<exp>/data/sweep/logs PROXYLOG=<exp>/data/sweep/proxies bash scripts/run.sh
# The original sweep launcher (run_sweep.sh) is kept on the box, not in git.
#
# Launch detached:
#   setsid nohup bash /workspace/steering_reasoning/experiments/opt-vector-sorh/scripts/run.sh \
#     > /workspace/steering_reasoning/experiments/opt-vector-sorh/data/run_arms.$(date -u +%FT%TZ).log 2>&1 < /dev/null &
# Switches:
#   PROXIES_ONLY=1        start/check the proxies, launch no eval
#   ARMS_ONLY="base v3c"  only these arms (their proxies + evals); base2 is included only if listed
#   SMOKE=1               base and v3c_cl on sorh_0,sorh_1,sorh_2 into data/smoke/<arm>/
#   MAXCONN=10            Inspect --max-connections per arm (3 arms share each server: 3 x 10 < 32 seqs)
#   LOGBUF=5              Inspect --log-buffer (samples per .eval flush; SMOKE=1 forces 1). inspect 0.3.193
#                         rewrites the whole .eval on every flush, so 1 is O(n^2) disk traffic on a full arm
#   VERIFY=1              run `run_eval.py check-delivery` (base==base, base!=steered over the five steered
#                         proxies) and then `run_eval.py check-servers` (base==base2 across the two servers)
#                         before any eval; both need QUIET servers: they compare temperature-0 outputs,
#                         so nothing else may be generating
#   FORCE_PROXIES=1       restart the selected proxies even if they answer
#   PASSES=2              after a runner exits, recompute the remaining ids and relaunch up to this
#                         many times in one invocation (absorbs transient sample errors)
#   IDS_SEP=","           separator for run_eval.py --sample-ids (","; set " " if it takes nargs)
#   UP0=8000 UP1=8002     upstream vLLM ports of the GPU0 and GPU1 servers
set -uo pipefail
EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # this experiment folder
REPO="$(cd "$EXP/../.." && pwd)"                            # repo root
PROXY_MOD="${PROXY_MOD:-steering_reasoning.proxy}"   # the repo proxy, src/steering_reasoning/proxy.py (run as a module)
CAP="${CAP:-16384}"   # per-turn max_tokens the proxy stamps (the repo default); the runs of record used 16000
A=$REPO/experiments/opt-vector/artifacts
K="${K:-$REPO/src/steering_reasoning/canonical}"
PY="${PY:-/opt/venvs/eval/bin/python}"
LOGROOT="${LOGROOT:-$EXP/data/logs}"
PROXYLOG="${PROXYLOG:-$EXP/data/proxies}"
STATUS_MD="${STATUS_MD:-$EXP/data/sorh-status.md}"   # dated one-line notes; the runs of record wrote /workspace/logs/sorh-status.md on the box
MAXCONN="${MAXCONN:-10}"
LOGBUF="${LOGBUF:-5}"
PASSES="${PASSES:-2}"
IDS_SEP="${IDS_SEP:-,}"
UP0="${UP0:-8000}"; UP1="${UP1:-8002}"   # GPU0, GPU1 servers (on RunPod 8001 is the system nginx -> 8000; never use it there)
# name:port:upstream:mode:artifact:dose   (mode none = passthrough). Named ARM_SPECS, not ARMS, so an
# exported ARMS of bare arm names can never be parsed here as specs.
ARM_SPECS="${ARM_SPECS:-base:8100:$UP0:none:: v3b:8101:$UP0:add:$A/opt_27b_L31_v3b.safetensors:0.75 v3c:8102:$UP0:add:$A/opt_27b_L31_v3c.safetensors:0.5 v3b_cl:8103:$UP1:clamp:$A/opt_27b_L31_v3b_cl.safetensors:-5 v3c_cl:8104:$UP1:clamp:$A/opt_27b_L31_v3c_cl.safetensors:-8 u_cl8:8105:$UP1:clamp:$K/u_L35_steering.safetensors:-8 base2:8106:$UP1:none::}"
NO_EVAL_ARMS=" base2 "     # proxies that never get an eval
# run_eval.py check-delivery skips only the arm named 'base' and asserts every other arm DIFFERS from it, so
# base2 must not be in this list (byte-identical servers would be reported as 'steering not delivered').
VERIFY_ARMS="base:8100 v3b:8101 v3c:8102 v3b_cl:8103 v3c_cl:8104 u_cl8:8105"
IDENTITY_ARMS="base:8100 base2:8106"   # run_eval.py check-servers: must be byte-identical at temperature 0
ARMS_ONLY="${ARMS_ONLY:-}"
IDS="${IDS:-}"   # empty = all ids from data/sorh_rows.json (sorh_0..sorh_N-1); env IDS="sorh_1 sorh_2 ..." restricts the run (e.g. the eval500 set)
if [ "${SMOKE:-}" = "1" ]; then
  ARMS_ONLY="${ARMS_ONLY:-base v3c_cl}"; IDS="sorh_0 sorh_1 sorh_2"; LOGROOT="$EXP/data/smoke"; LOGBUF=1
fi

cd "$EXP" || exit 1   # Inspect chdir()s into cwd for the task run dir
mkdir -p "$LOGROOT" "$PROXYLOG" "$(dirname "$STATUS_MD")"
note() { echo "$*"; echo "- $(date -u +%FT%TZ) run_arms: $*" >> "$STATUS_MD"; }

selected() { [ -z "$ARMS_ONLY" ] || echo " $ARMS_ONLY " | grep -q " $1 "; }
# a proxy is wanted if its arm is selected, or VERIFY=1 (check-delivery + check-servers need all seven)
proxy_wanted() { [ "${VERIFY:-}" = "1" ] || selected "$1"; }

[ -x "$PY" ] || { echo "eval venv missing: $PY"; exit 1; }
[ -f "$EXP/scripts/run_eval.py" ] || { echo "missing $EXP/scripts/run_eval.py"; exit 1; }
$PY -c "import $PROXY_MOD" 2>/dev/null || { echo "$PY cannot import $PROXY_MOD: install the repo into the eval venv (uv sync at $REPO, or pip install -e $REPO)"; exit 1; }

# --- proxies: start the wanted ones that are not answering (FORCE_PROXIES=1 restarts them) ---
for spec in $ARM_SPECS; do
  IFS=: read -r name port up mode artifact dose <<< "$spec"
  proxy_wanted "$name" || continue
  if [ "${FORCE_PROXIES:-}" != "1" ] && curl -sf -m 3 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
    echo "proxy $name :$port already up"; continue
  fi
  pkill -f "[s]teering_reasoning.proxy --port $port " 2>/dev/null || true; sleep 1
  case "$mode" in
    none)  sa="--no-steer" ;;
    add)   sa="--mode add --artifact $artifact --scale $dose" ;;
    clamp) sa="--mode clamp --artifact $artifact --threshold $dose" ;;
    *) echo "unknown mode $mode for $name"; exit 1 ;;
  esac
  if [ "$mode" != "none" ]; then   # the repo proxy has no polarity guard; refuse a vector/mode mismatch before launching
    $PY "$EXP/scripts/run_eval.py" check-polarity --artifact "$artifact" --mode "$mode" --dose "$dose" || { echo "!!! polarity check refused $name"; exit 1; }
  fi
  setsid nohup $PY -m "$PROXY_MOD" --port "$port" --upstream "http://localhost:$up" --max-tokens "$CAP" --cache-salt "$name" $sa \
    >> "$PROXYLOG/$name.log" 2>&1 < /dev/null &
  echo "proxy $name pid $! port $port -> :$up $sa"
done
sleep 6
# --- health gates: both upstream servers and every wanted proxy must answer 200 with the model ---
for up in $UP0 $UP1; do
  if ! curl -sf -m 5 "http://127.0.0.1:$up/v1/models" | grep -q "Qwen/Qwen3.5-27B"; then
    note "upstream :$up not healthy (no Qwen/Qwen3.5-27B in /v1/models); not launching evals"; exit 1
  fi
  echo "upstream :$up ok"
done
for spec in $ARM_SPECS; do
  IFS=: read -r name port up mode artifact dose <<< "$spec"
  proxy_wanted "$name" || continue
  code=$(curl -s -m 5 -o /dev/null -w "%{http_code}" "http://127.0.0.1:$port/v1/models")
  echo "proxy $name :$port -> $code"
  [ "$code" = "200" ] || { note "proxy $name :$port not healthy ($code); see $PROXYLOG/$name.log; not launching evals"; exit 1; }
done
if [ "${VERIFY:-}" = "1" ]; then
  $PY "$EXP/scripts/run_eval.py" check-delivery --arms $VERIFY_ARMS || { note "VERIFY_DELIVERY FAILED - not launching evals"; exit 2; }
  note "VERIFY_DELIVERY PASS ($VERIFY_ARMS)"
  $PY "$EXP/scripts/run_eval.py" check-servers --arms $IDENTITY_ARMS \
    || { note "VERIFY_SERVERS FAILED (base!=base2: the two servers are not byte-identical unsteered) - not launching evals"; exit 2; }
  note "VERIFY_SERVERS PASS ($IDENTITY_ARMS)"
fi
if [ "${PROXIES_ONLY:-}" = "1" ]; then echo "PROXIES_ONLY=1: proxies up, not launching evals"; exit 0; fi

# --- ids ---
if [ -z "$IDS" ]; then
  N=$($PY -c "import json;print(len(json.load(open('$EXP/data/sorh_rows.json'))))") || exit 1
  IDS=$(seq -f "sorh_%g" 0 $((N-1)) | tr '\n' ' ')
fi
IDS_FILE="$LOGROOT/.all_ids"; echo "$IDS" > "$IDS_FILE"

remaining_ids() {  # $1 = log dir ; prints ids still to run (empty = nothing to do)
  $PY "$EXP/scripts/run_eval.py" remaining --log-dir "$1" --ids-file "$IDS_FILE"
}

# --- evals: one detached-safe subshell per arm ---
EVAL_PIDS=""
for spec in $ARM_SPECS; do
  IFS=: read -r name port up mode artifact dose <<< "$spec"
  selected "$name" || continue
  echo "$NO_EVAL_ARMS" | grep -q " $name " && { echo "[$name] verification-only proxy, no eval"; continue; }
  (
    d="$LOGROOT/$name"; mkdir -p "$d"
    if pgrep -f -- "[r]un_eval.py (run )?--sample-ids .*--log-dir $d( |$)" >/dev/null; then
      echo "[$name] runner already alive, skipping"; exit 0
    fi
    pass=0
    while [ $pass -lt "$PASSES" ]; do
      pass=$((pass+1))
      rem=$(remaining_ids "$d") || { echo "!!! [$name] remaining_ids FAILED"; exit 1; }
      if [ -z "$rem" ]; then echo "[$name] complete ($(date -u +%FT%TZ))"; exit 0; fi
      nrem=$(echo $rem | wc -w)
      echo "[$name] pass $pass: running $nrem ids via :$port  ($(date -u +%FT%TZ))"
      note "[$name] pass $pass launched $nrem ids -> $d"
      if [ "$IDS_SEP" = "," ]; then idarg="$(echo $rem | tr ' ' ',')"; else idarg="$rem"; fi
      env HF_HOME="${HF_HOME:-/workspace/hf}" HF_HUB_OFFLINE=1 LOCAL_API_KEY=dummy PYTHONUNBUFFERED=1 \
        OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        $PY "$EXP/scripts/run_eval.py" --sample-ids $idarg \
          --base-url "http://localhost:$port/v1" --max-connections "$MAXCONN" \
          --log-dir "$d" --log-buffer "$LOGBUF" --fail-on-error 0 --retry-on-error 2 \
          >> "$LOGROOT/$name.runner.log" 2>&1
      rc=$?
      echo "[$name] runner exited $rc ($(date -u +%FT%TZ))"
      note "[$name] pass $pass runner exited rc=$rc"
      if [ $rc -ne 0 ] && ! ls "$d"/*.eval >/dev/null 2>&1; then
        echo "!!! [$name] FAILED BEFORE WRITING ANY LOG (rc=$rc) -- see $LOGROOT/$name.runner.log"; exit 1
      fi
    done
    left=$(remaining_ids "$d"); [ -z "$left" ] && echo "[$name] complete" || echo "[$name] $(echo $left | wc -w) ids still unfinished after $PASSES pass(es); re-run to resume"
  ) &
  EVAL_PIDS="$EVAL_PIDS $!"
  echo "eval $name pid $!"
done
echo "launched: $EVAL_PIDS"
for pid in $EVAL_PIDS; do wait "$pid" || true; done
note "ALL_ARMS_DONE (arms: ${ARMS_ONLY:-all}, logs: $LOGROOT)"
echo "ALL_ARMS_DONE $(date -u +%FT%TZ)"
