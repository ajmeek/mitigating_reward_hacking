#!/usr/bin/env bash
# Runs ON the eval box. Launches every arm of the dose sweep (../arms.tsv) against one upstream server.
#   UPSTREAM=http://<gpu-host>:<port> [SEEDS="42"] bash experiments/diff-mean-steering/scripts/run_all_arms.sh
# Thin wrapper over scripts/run_arms.sh; the arm table is ../arms.tsv. Progress via
# `journalctl --user -u arm-<name>` and $E/data/logs/<arm>/<variant>/s<seed>/run.log.
set -uo pipefail
E=experiments/diff-mean-steering
cd "$(dirname "$(readlink -f "$0")")/../../.."
bash scripts/run_arms.sh "$E" "$E/arms.tsv"
