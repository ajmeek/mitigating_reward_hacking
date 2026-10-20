#!/usr/bin/env bash
# Runs ON the box. Launches both arms of ../arms.tsv against the local server via the shared
# runner (scripts/run_arms.sh -> one systemd user unit per arm running scripts/run_arm.sh).
#   bash experiments/spec-prompting/scripts/run_all.sh            # seeds 42 43, both variants
#   SEEDS="44 45 46" bash .../run_all.sh                             # add seeds later (additive)
#   ARMS=spec bash .../run_all.sh                                    # one arm
# Run parameters (decisions 2026-09-14, Austin): seeds 42 43 first (matching the regeneration
# sweep; more later), MSG_LIMIT 120,
# per-turn cap 16384 (the proxy's default). Concurrency: 2 seeds x 2 variants = 4 evals per arm,
# all in flight (MAX_PARALLEL=4); MAXCONN sandboxes per eval follows the box's vCPUs unless set:
#   48 vCPU box (MAXSEQS 48): MAXCONN=3 -> <=24 streams, <=24 sandboxes + 8 Inspect
#     processes + 4 vLLM workers (the n5 lesson: orphaned tool processes alone drove load to 85
#     on 16 cores; re-tune from the smoke's wall clock)
#   16 vCPU box (MAXSEQS 24):  MAXCONN=2 -> <=16 streams, <=16 sandboxes
# Re-running is safe: scored tasks are skipped; a running arm must NOT be relaunched (run_arm.sh
# restarts its proxy under in-flight requests) — this script refuses while arm units exist.
# Progress: journalctl --user -u arm-<name> -f; $E/data/logs/<arm>/<variant>/s<seed>/run.log;
# ARM_DONE <name> in the journal means the unit exited (check the logs, not just the line).
set -uo pipefail
E="${E:-experiments/spec-prompting}"   # another experiment with the same arms reuses this script by setting E
cd "$(dirname "$(readlink -f "$0")")/../../.."
if systemctl --user list-units --state=running,activating,deactivating 'arm-*' 2>/dev/null | grep -q 'arm-'; then
  systemctl --user list-units --state=running,activating,deactivating 'arm-*'
  echo "arm units are running; wait for ARM_DONE or stop them before relaunching" >&2; exit 3
fi
# Every suffix file in the table must be on this box (the provisioner ships spec/).
while read -r name mode dose port art suffix _; do
  [ "${suffix:--}" = "-" ] && continue
  [ -f "$suffix" ] || { echo "row $name: suffix file $suffix missing on this box (scripts/provision_evalbox.sh)" >&2; exit 2; }
done < <(sed 's/#.*//' "$E/arms.tsv" | grep -v '^\s*$')
export UPSTREAM="${UPSTREAM:-http://localhost:8000}"
[ "$(nproc)" -ge 32 ] && DEFCONN=3 || DEFCONN=2
export SEEDS="${SEEDS:-42 43}" MSG_LIMIT="${MSG_LIMIT:-120}" MAXCONN="${MAXCONN:-$DEFCONN}" MAX_PARALLEL="${MAX_PARALLEL:-4}"
echo "box: $(nproc) vCPU -> MAXCONN=$MAXCONN MAX_PARALLEL=$MAX_PARALLEL SEEDS='$SEEDS' MSG_LIMIT=$MSG_LIMIT"
bash scripts/run_arms.sh "$E" "$E/arms.tsv"
