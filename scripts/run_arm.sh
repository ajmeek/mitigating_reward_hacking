#!/usr/bin/env bash
# Runs ON the eval box (Docker + Inspect). One steering arm against one vLLM endpoint: a persistent
# proxy (systemd user unit) + one Inspect eval per (variant, seed), resumable: only tasks without a
# complete score in the arm's log dir are (re)run, so adding seeds later is additive.
#
#   UPSTREAM=http://host:port E=experiments/<exp> ARM=cn8 MODE=clamp DOSE=-8 PORT=8113 bash scripts/run_arm.sh
#
# Required: UPSTREAM, E (experiment dir; logs go to $E/data/logs/<ARM>/<variant>/s<seed>/), ARM, PORT.
# Steering: MODE=none|clamp|add|ablate (default: clamp if DOSE/THR is set, else none); DOSE is the
#   clamp threshold on h.u (raw projection units of the artifact's own vector) or the add scale;
#   THR is accepted as an alias for a clamp DOSE. ART is the artifact (.safetensors + .json sidecar
#   with layer_indices); default: the canonical u (src/steering_reasoning/canonical/INDEX.md).
#   Thresholds do not transfer between vectors: calibrate per vector
#   (scripts/calibrate_thresholds.py --artifact <v>).
# Optional: SEEDS="42" MAXCONN=5 MSG_LIMIT=120 VARIANTS="conflicting original" TASKS="id id ..."
#   MODEL=openai-api/local/Qwen/Qwen3.5-27B DUMP_DIR=~/dumps/<ARM> MAX_PARALLEL=2.
#   SUFFIX_FILE=<path>: a prompt arm — the file's text is appended to the system message after the
#   verbatim MisTrace template (run_lcb_eval.py --system-suffix-file); its sha256 is written to
#   $LOGROOT/system_suffix.sha256 so a later edit of the file cannot silently change the arm.
#   Combines with any MODE (a prompt-only arm is MODE=none + SUFFIX_FILE).
#   Tasks default to splits.json impossible.test (conflicting) / original.test (original).
#   MSG_LIMIT is the per-rollout message limit (120, the paper's); limit-hit rollouts are scored as they stand.
# Every proxy stamps cache_salt=<ARM> so arms sharing a server never share prefix-cache blocks.
# Usually launched per arm by scripts/run_arms.sh from an experiment's arms.tsv.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
export PATH=$HOME/.local/bin:$PATH
export LOCAL_API_KEY=dummy
: "${UPSTREAM:?UPSTREAM=http://host:port}"; : "${E:?E=experiments/<exp>}"; : "${ARM:?ARM=<name>}"; : "${PORT:?PORT=<proxy port>}"
ART="${ART:-src/steering_reasoning/canonical/u_L35_steering.safetensors}"
DOSE="${DOSE:-${THR:-}}"; MODE="${MODE:-}"; [ -z "$MODE" ] && { [ -n "$DOSE" ] && MODE=clamp || MODE=none; }
LOGROOT="$E/data/logs/$ARM"; PROXYLOG="$E/data/logs/proxies"; mkdir -p "$LOGROOT" "$PROXYLOG"
MODEL="${MODEL:-openai-api/local/Qwen/Qwen3.5-27B}"; SEEDS="${SEEDS:-42}"; MAXCONN="${MAXCONN:-5}"; MSG_LIMIT="${MSG_LIMIT:-120}"
IDS="${TASKS:-$(uv run python -c "import json; s=json.load(open('src/steering_reasoning/data/splits.json')); print(' '.join(s['impossible']['test']))")}"
[ -z "$IDS" ] && { echo "no task ids (TASKS unset and splits.json unreadable?)"; exit 2; }
case "$MODE" in
  none)   SA="--no-steer" ;;
  clamp)  [ -z "$DOSE" ] && { echo "clamp needs DOSE (threshold)"; exit 2; }; SA="--artifact $ART --mode clamp --threshold $DOSE" ;;
  add)    [ -z "$DOSE" ] && { echo "add needs DOSE (scale)"; exit 2; }; SA="--artifact $ART --mode add --scale $DOSE" ;;
  ablate) SA="--artifact $ART --mode ablate" ;;
  *) echo "MODE must be none|clamp|add|ablate"; exit 2 ;;
esac
SUF=""; SHAFILE="$LOGROOT/system_suffix.sha256"
if [ -n "${SUFFIX_FILE:-}" ]; then
  [ -f "$SUFFIX_FILE" ] || { echo "SUFFIX_FILE $SUFFIX_FILE not found (ship it to the box; scripts/provision_evalbox.sh ships spec/)"; exit 2; }
  SUF="--system-suffix-file $SUFFIX_FILE"
  # One suffix per arm log dir: refuse to add seeds under an edited prompt (never overwrite the hash).
  if [ -f "$SHAFILE" ] && ! sha256sum -c --quiet "$SHAFILE" >/dev/null 2>&1; then
    echo "$SHAFILE records a different $SUFFIX_FILE than the current one; use a new arm name/log dir"; exit 2
  fi
  sha256sum "$SUFFIX_FILE" > "$SHAFILE"
elif [ -f "$SHAFILE" ]; then
  echo "$LOGROOT was run with a system suffix ($SHAFILE) and SUFFIX_FILE is now unset; refusing to mix"; exit 2
fi
# --ids-log: the exact token ids of every turn (for replaying rollouts with scripts/capture_rollouts.py)
SA="$SA --dump-dir ${DUMP_DIR:-$HOME/dumps/$ARM} --cache-salt $ARM --ids-log $LOGROOT/ids.jsonl"
REPO="$(pwd)"

# UNIT_PREFIX (from run_arms.sh) namespaces the proxy unit like the arm unit, so a prefixed
# launch (a smoke) never stops a live arm's proxy; use PORT_OFFSET there too so ports differ.
PUNIT="proxy-${UNIT_PREFIX:-}$ARM"
cat > ~/proxy_${UNIT_PREFIX:-}$ARM.sh <<PEOF
cd "$REPO"; export PATH=\$HOME/.local/bin:\$PATH
exec uv run python -m steering_reasoning.proxy --port $PORT --upstream $UPSTREAM $SA >> $PROXYLOG/$ARM.log 2>&1
PEOF
systemctl --user stop "$PUNIT" 2>/dev/null
systemd-run --user --unit "$PUNIT" --collect bash ~/proxy_${UNIT_PREFIX:-}$ARM.sh
for i in $(seq 1 30); do curl -s -m 3 "http://localhost:$PORT/v1/models" | grep -q Qwen && break; sleep 2; done
echo "proxy $ARM on :$PORT -> $UPSTREAM ($SA)${SUF:+; system suffix $SUFFIX_FILE}"

remaining_ids() {  # <logdir> <ids>: ids without a complete score in any .eval under logdir
  uv run python - "$1" "$2" <<'PY'
import glob, sys
from inspect_ai.log import read_eval_log
d, ids = sys.argv[1], sys.argv[2].split(); done = set()
for f in glob.glob(f"{d}/*.eval"):
    try: l = read_eval_log(f)
    except Exception: continue
    for s in (l.samples or []):
        if s.scores and all(x.value not in (None, "") for x in s.scores.values()): done.add(s.id)
print(" ".join(i for i in ids if i not in done))
PY
}

MAX_PARALLEL="${MAX_PARALLEL:-2}"; RUNNING=0
for seed in $SEEDS; do
  for variant in ${VARIANTS:-conflicting original}; do        # Inspect's names: conflicting = impossible tests, original = passable tests
    d="$LOGROOT/$variant/s$seed"; mkdir -p "$d"
    rem=$(remaining_ids "$d" "$IDS")
    [ -z "$rem" ] && { echo "done $ARM $variant s$seed"; continue; }
    if [ "$RUNNING" -ge "$MAX_PARALLEL" ]; then wait -n || true; RUNNING=$((RUNNING-1)); fi
    uv run scripts/run_lcb_eval.py --split "$variant" --sample-ids "$(echo $rem | tr ' ' ',')" \
      --model "$MODEL" --base-url "http://localhost:$PORT/v1" --seed "$seed" $SUF \
      --max-connections "$MAXCONN" --max-sandboxes "$MAXCONN" --message-limit "$MSG_LIMIT" --log-dir "$d" --log-buffer 1 --retry-on-error 2 --fail-on-error 0 \
      > "$d/run.log" 2>&1 &
    RUNNING=$((RUNNING+1)); echo "started $ARM $variant s$seed: $(echo $rem | wc -w) tasks"
  done
done
wait; echo "ARM_DONE $ARM"
