#!/usr/bin/env bash
# Runs ON the eval box. Launches every arm in an experiment's arms.tsv against one upstream server,
# each as its own systemd user unit (arm-<unit-prefix><name>) running scripts/run_arm.sh.
#
#   UPSTREAM=http://host:port bash scripts/run_arms.sh experiments/<exp> [experiments/<exp>/arms.tsv]
#
# arms.tsv: whitespace-separated, '#' comments, columns  name  mode  dose  port  [artifact]  [suffix]
#   mode  none | clamp | add | ablate      dose  clamp threshold or add scale ('-' for none)
#   artifact  '-' or omitted = the canonical u; anything else is passed to run_arm.sh as ART.
#   suffix  '-' or omitted = none; a file whose text is appended to the system message (a prompt
#           arm; passed to run_arm.sh as SUFFIX_FILE). Ship it to the box yourself (not under src/).
# Environment passed through to run_arm.sh: SEEDS MAXCONN MSG_LIMIT VARIANTS TASKS MODEL MAX_PARALLEL.
# UNIT_PREFIX (default empty) namespaces the arm AND proxy units, e.g. UNIT_PREFIX=smoke- ->
# arm-smoke-cn8 / proxy-smoke-cn8; PORT_OFFSET (default 0) is added to every row's port, so a
# prefixed launch (a smoke next to a live run) never stops or rebinds a live arm's proxy.
# ARMS="name name" restricts to those rows. Exit 2 if any selected row's suffix file is missing
# (that row is not launched). Progress: journalctl --user -u arm-<name> -f and
# $E/data/logs/<arm>/<variant>/s<seed>/run.log.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
: "${UPSTREAM:?set UPSTREAM=http://host:port}"
E="${1:?usage: run_arms.sh experiments/<exp> [arms.tsv]}"; TABLE="${2:-$E/arms.tsv}"
[ -f "$TABLE" ] || { echo "no arms file $TABLE"; exit 2; }
ONLY="${ARMS:-}"; PREFIX="${UNIT_PREFIX:-}"; OFFSET="${PORT_OFFSET:-0}"; RUNNER="$(pwd)/scripts/run_arm.sh"
PASS=(); for v in SEEDS MAXCONN MSG_LIMIT VARIANTS TASKS MODEL MAX_PARALLEL; do [ -n "${!v:-}" ] && PASS+=("$v=${!v}"); done   # array: TASKS/VARIANTS contain spaces
MISSING=0
while read -r name mode dose port art suffix _; do   # '#' starts a comment anywhere, as in fleet/plan.py
  [ -n "$ONLY" ] && ! grep -qw "$name" <<<"$ONLY" && continue
  [ "$dose" = "-" ] && dose=""; [ "${art:--}" = "-" ] && art=""; [ "${suffix:--}" = "-" ] && suffix=""
  [ -n "$suffix" ] && [ ! -f "$suffix" ] && { echo "row $name: suffix file $suffix not found (ship it to this box first); NOT launched" >&2; MISSING=1; continue; }
  port=$((port + OFFSET)); unit="arm-$PREFIX$name"
  systemctl --user stop "$unit" 2>/dev/null
  systemd-run --user --unit "$unit" --collect \
    env UPSTREAM="$UPSTREAM" E="$E" ARM="$name" MODE="$mode" DOSE="$dose" PORT="$port" ${art:+ART=$art} ${suffix:+SUFFIX_FILE=$suffix} ${PASS[@]+"${PASS[@]}"} \
        UNIT_PREFIX="$PREFIX" DUMP_DIR="$HOME/dumps/$PREFIX$name" \
    bash "$RUNNER"
  echo "launched $unit (mode $mode, dose '${dose}', port $port, artifact ${art:-canonical u}${suffix:+, system suffix $suffix})"
done < <(sed 's/#.*//' "$TABLE" | grep -v '^\s*$')
[ "$MISSING" = 0 ] || exit 2
