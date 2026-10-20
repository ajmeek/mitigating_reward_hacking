#!/usr/bin/env bash
# Launch (or resume) School of Reward Hacks (SoRH) steering arms on one box: one steering proxy per arm
# (src/steering_reasoning/proxy.py) and one Inspect runner per arm, from an arms.tsv.
#
#   UPSTREAM=http://localhost:8000 bash scripts/run_sorh_arms.sh experiments/<exp> [experiments/<exp>/arms.tsv]
#
# arms.tsv: whitespace-separated, '#' comments, columns  name  mode  dose  port  [artifact  [upstream]]
#   mode none | clamp | add | ablate; dose = clamp threshold or add scale ('-' for none/ablate);
#   artifact '-' or omitted = the canonical u; the optional upstream routes that arm to another vLLM server
#   (a two-GPU box; on RunPod port 8001 is the system nginx -> 8000, never use it there). The same format drives the LCB arm
#   runner (scripts/run_arms.sh) when the sixth column is left out. Ports must be unique.
# Per arm: one Inspect eval over the ids (default: every row of $ROWS), no scorer. Re-running resumes:
#   ids that already have a FINISHED sample (error None and non-empty output) in any .eval under
#   $LOGROOT/<arm>/ are skipped; an arm whose runner is still alive is skipped. A proxy that already
#   answers is kept only if its command line carries exactly the arguments the table asks for now
#   (upstream, mode, artifact, dose, cap, cache salt); otherwise it is restarted, and FORCE_PROXIES=1
#   restarts all of them. Every proxy stamps cache_salt=<arm>.
# Environment:
#   UPSTREAM        default vLLM server (required unless every arm has its own upstream column)
#   ROWS            dataset rows JSON (default $E/data/sorh_rows.json; `sorh fetch --out` writes one)
#   LOGROOT         logs root (default $E/data/logs); PROXYLOG proxy logs (default $LOGROOT/proxies)
#   STATUS_MD       dated one-line notes (default $E/data/sorh-status.md)
#   IDS / IDS_FILE / IDS_JSON  restrict the prompts: space-separated ids, a file of them, or an ids.json
#                   (`sorh dev-set` / `eval-set` output; its "ids" list is used). Paths that do not exist
#                   from the repo root are also tried relative to $E.
#   ARMS            "name name": only these rows (each must be a row).  PROXIES_ONLY=1  start/check proxies, no evals
#   BASE            the unsteered arm's name (default base; must be a row). Any other `none` row (say base2)
#                   is started as a passthrough for check-servers and never evaluated or delivery-checked.
#   VERIFY=1        run check-delivery (base==base, base!=steered) before any eval; needs QUIET servers
#   VERIFY_SERVERS="base:8100 base2:8106"  also run check-servers on these passthrough proxies
#   CAP             per-turn max_tokens the proxy stamps (default: the repo's MAX_TOKENS, 16384; the SoRH
#                   runs of record used 16000, and `summarize --cap` must match what the proxy stamped)
#   MAXCONN=10 LOGBUF=5 PASSES=2 MODEL=openai-api/local/Qwen/Qwen3.5-27B
#   PY              python with the repo installed (default: /opt/venvs/eval/bin/python if present, else python3)
# Exit status: 0 only when every selected arm ends with no remaining ids; 1 when an arm failed or ids remain;
# 2 for a bad table / missing input.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
REPO="$(pwd)"
E="${1:?usage: run_sorh_arms.sh experiments/<exp> [arms.tsv]}"; TABLE="${2:-$E/arms.tsv}"
[ -f "$TABLE" ] || { echo "no arms file $TABLE"; exit 2; }
if [ -z "${PY:-}" ]; then if [ -x /opt/venvs/eval/bin/python ]; then PY=/opt/venvs/eval/bin/python; else PY=python3; fi; fi
$PY -c "import steering_reasoning.proxy, steering_reasoning.evals.sorh" 2>/dev/null \
  || { echo "$PY cannot import the repo: uv sync at $REPO (or pip install -e $REPO) into that python"; exit 2; }
SORH="$PY -m steering_reasoning.evals.sorh"
# inputs given relative to the experiment folder are accepted too
for v in ROWS IDS_FILE IDS_JSON; do
  [ -n "${!v:-}" ] && [ ! -f "${!v}" ] && [ -f "$E/${!v}" ] && printf -v "$v" '%s' "$E/${!v}"
done
ROWS="${ROWS:-$E/data/sorh_rows.json}"; [ -f "$ROWS" ] || { echo "no rows file $ROWS (sorh fetch --out $ROWS)"; exit 2; }
LOGROOT="${LOGROOT:-$E/data/logs}"; PROXYLOG="${PROXYLOG:-$LOGROOT/proxies}"; STATUS_MD="${STATUS_MD:-$E/data/sorh-status.md}"
CAP="${CAP:-$($PY -c 'from steering_reasoning.evals.config import MAX_TOKENS; print(MAX_TOKENS)')}" || exit 2
MAXCONN="${MAXCONN:-10}"; LOGBUF="${LOGBUF:-5}"; PASSES="${PASSES:-2}"; MODEL="${MODEL:-openai-api/local/Qwen/Qwen3.5-27B}"
BASE="${BASE:-base}"; ONLY="${ARMS:-}"; SERVED="${MODEL#openai-api/local/}"
mkdir -p "$LOGROOT" "$PROXYLOG" "$(dirname "$STATUS_MD")"
note() { echo "$*"; echo "- $(date -u +%FT%TZ) run_sorh_arms: $*" >> "$STATUS_MD"; }
selected() { [ -z "$ONLY" ] || tr ' ,' '\n\n' <<<"$ONLY" | grep -qxF -- "$1"; }

# --- the table: name port upstream mode artifact dose (upstream '-' = $UPSTREAM) ---
ROWS_TSV="$($SORH arms --arms-tsv "$TABLE")" || exit 2
NAMES="$(cut -f1 <<<"$ROWS_TSV")"
grep -qxF -- "$BASE" <<<"$NAMES" || { echo "BASE=$BASE is not a row of $TABLE"; exit 2; }
for a in $(tr ',' ' ' <<<"$ONLY"); do grep -qxF -- "$a" <<<"$NAMES" || { echo "ARMS: no row named '$a' in $TABLE"; exit 2; }; done
dup=$(cut -f2 <<<"$ROWS_TSV" | sort | uniq -d); [ -z "$dup" ] || { echo "duplicate proxy port(s) in $TABLE: $dup"; exit 2; }
if [ -z "${UPSTREAM:-}" ] && cut -f3 <<<"$ROWS_TSV" | grep -qx -- '-'; then echo "set UPSTREAM=http://host:port (rows without their own upstream column need it)"; exit 2; fi

# --- ids (resolved before anything starts, so a bad path fails first) ---
if [ -n "${IDS_JSON:-}" ]; then
  [ -f "$IDS_JSON" ] || { echo "IDS_JSON not found: $IDS_JSON"; exit 2; }
  IDS="$($PY -c "import json,sys; d=json.load(open(sys.argv[1])); print(' '.join(d['ids'] if isinstance(d, dict) else d))" "$IDS_JSON")" || exit 2
fi
if [ -n "${IDS_FILE:-}" ]; then [ -f "$IDS_FILE" ] || { echo "IDS_FILE not found: $IDS_FILE"; exit 2; }; IDS="$(cat "$IDS_FILE")"; fi
if [ -z "${IDS:-}" ]; then
  N=$($PY -c "import json;print(len(json.load(open('$ROWS'))))") || exit 2
  IDS=$(seq -f "sorh_%g" 0 $((N-1)) | tr '\n' ' ')
fi
IDS_FILE_ALL="$LOGROOT/.ids.$$"; echo "$IDS" > "$IDS_FILE_ALL"; trap 'rm -f "$IDS_FILE_ALL"' EXIT

# --- polarity: refuse a vector/mode pair that would steer the wrong way, before any proxy starts ---
$SORH check-polarity --arms-tsv "$TABLE" --root "$REPO" ${ONLY:+--arms "$ONLY"} || { echo "!!! polarity check refused an arm"; exit 2; }

# --- proxies: a wanted arm's proxy is kept only if it runs with exactly the arguments the table asks for ---
VERIFY_ARMS=""
while IFS=$'\t' read -r name port up mode artifact dose; do
  selected "$name" || { [ "${VERIFY:-}" = "1" ] || continue; }
  [ "$up" = "-" ] && up="${UPSTREAM:-}"
  [[ "$up" =~ ^[0-9]+$ ]] && up="http://localhost:$up"
  [ -n "$up" ] || { echo "[$name] no upstream (set UPSTREAM or a sixth column)"; exit 2; }
  # check-delivery asserts base != every other arm, so a second passthrough proxy stays out of its list
  { [ "$mode" != "none" ] || [ "$name" = "$BASE" ]; } && VERIFY_ARMS="$VERIFY_ARMS $name:$port"
  sa="$($SORH proxy-args --arms-tsv "$TABLE" --arm "$name" --cap "$CAP" --root "$REPO")" || exit 2
  want="--port $port --upstream $up $sa"
  if [ "${FORCE_PROXIES:-}" != "1" ] && curl -sf -m 3 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
    live="$(pgrep -af -- "[s]teering_reasoning.proxy --port $port " | head -1)"
    case "$live" in
      *"$want"*) echo "proxy $name :$port already up (arguments match)"; continue ;;
      *) note "proxy $name :$port is up with different arguments; restarting it (want: $want)" ;;
    esac
  fi
  pkill -f "[s]teering_reasoning.proxy --port $port " 2>/dev/null || true; sleep 1
  setsid nohup $PY -m steering_reasoning.proxy $want >> "$PROXYLOG/$name.log" 2>&1 < /dev/null &
  echo "proxy $name pid $! $want"
done <<<"$ROWS_TSV"
sleep 6
# --- health: every wanted proxy answers 200 and serves the model ---
while IFS=$'\t' read -r name port up mode artifact dose; do
  selected "$name" || { [ "${VERIFY:-}" = "1" ] || continue; }
  body="$(curl -s -m 5 -w '\n%{http_code}' "http://127.0.0.1:$port/v1/models")"; code="${body##*$'\n'}"
  echo "proxy $name :$port -> $code"
  [ "$code" = "200" ] || { note "proxy $name :$port not healthy ($code); see $PROXYLOG/$name.log; not launching evals"; exit 1; }
  grep -qF "\"$SERVED\"" <<<"$body" || { note "proxy $name :$port does not serve $SERVED; not launching evals"; exit 1; }
done <<<"$ROWS_TSV"
if [ "${VERIFY:-}" = "1" ]; then
  $SORH check-delivery --base "$BASE" --arms $VERIFY_ARMS || { note "VERIFY_DELIVERY FAILED - not launching evals"; exit 2; }
  note "VERIFY_DELIVERY PASS ($VERIFY_ARMS)"
  if [ -n "${VERIFY_SERVERS:-}" ]; then
    $SORH check-servers --rows "$ROWS" --arms $VERIFY_SERVERS || { note "VERIFY_SERVERS FAILED - not launching evals"; exit 2; }
    note "VERIFY_SERVERS PASS ($VERIFY_SERVERS)"
  fi
fi
[ "${PROXIES_ONLY:-}" = "1" ] && { echo "PROXIES_ONLY=1: proxies up, not launching evals"; exit 0; }

# --- evals: one subshell per arm, resumable; exit 1 from a subshell = that arm is not done ---
EVAL_PIDS=""
while IFS=$'\t' read -r name port up mode artifact dose; do
  selected "$name" || continue
  if [ "$mode" = "none" ] && [ "$name" != "$BASE" ]; then echo "[$name] passthrough proxy for check-servers only, no eval"; continue; fi
  (
    d="$LOGROOT/$name"; mkdir -p "$d"
    if pgrep -f -- "steering_reasoning.evals.sorh run .*--log-dir $d( |$)" >/dev/null; then echo "[$name] runner already alive, skipping"; exit 1; fi
    pass=0
    while [ $pass -lt "$PASSES" ]; do
      pass=$((pass+1))
      rem=$($SORH remaining --log-dir "$d" --ids-file "$IDS_FILE_ALL") || { echo "!!! [$name] remaining FAILED"; exit 1; }
      [ -z "$rem" ] && { echo "[$name] complete ($(date -u +%FT%TZ))"; exit 0; }
      nrem=$(echo $rem | wc -w)
      note "[$name] pass $pass launched $nrem ids -> $d"
      env HF_HUB_OFFLINE=1 LOCAL_API_KEY=dummy PYTHONUNBUFFERED=1 INSPECT_DISPLAY=plain OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        $SORH run --rows "$ROWS" --model "$MODEL" --sample-ids "$(echo $rem | tr ' ' ',')" \
          --base-url "http://localhost:$port/v1" --max-connections "$MAXCONN" \
          --log-dir "$d" --log-buffer "$LOGBUF" --fail-on-error 0 --retry-on-error 2 >> "$LOGROOT/$name.runner.log" 2>&1
      rc=$?; note "[$name] pass $pass runner exited rc=$rc"
      if [ $rc -ne 0 ] && [ "$($SORH status --logs "$LOGROOT" --arms "$name" | cut -f2)" = "0" ]; then
        echo "!!! [$name] FAILED WITH NO FINISHED SAMPLE (rc=$rc) -- see $LOGROOT/$name.runner.log"; exit 1
      fi
    done
    left=$($SORH remaining --log-dir "$d" --ids-file "$IDS_FILE_ALL")
    [ -z "$left" ] && { echo "[$name] complete"; exit 0; }
    echo "[$name] $(echo $left | wc -w) ids still unfinished after $PASSES pass(es); re-run to resume"; exit 1
  ) &
  EVAL_PIDS="$EVAL_PIDS $!"; echo "eval $name pid $!"
done <<<"$ROWS_TSV"
fail=0; for pid in $EVAL_PIDS; do wait "$pid" || fail=1; done
note "ALL_ARMS_DONE (arms: ${ONLY:-all}, logs: $LOGROOT, status: $([ $fail = 0 ] && echo complete || echo INCOMPLETE))"
exit $fail
