#!/usr/bin/env bash
# Runs ON the box. The spec comparison on the 73 non-test MisTrace tasks (../tasks.json),
# conflicting variant only, seeds 42 43: the same two arms (../arms.tsv, identical to
# spec-prompting's) through the same runner, so the two experiments differ only in task set
# and variant. Wraps spec-prompting/scripts/run_all.sh with E, TASKS and VARIANTS set.
#   bash experiments/spec-prompting-basetasks/scripts/run_all.sh
#   SEEDS="44" bash .../run_all.sh            # additive
# Concurrency: 2 seeds x 1 variant = 2 evals per arm, 4 in flight; MAXCONN follows the vCPUs so
# the stream count matches v2's (24 on 48 vCPU, 16 on 16 vCPU).
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/../../.."; export PATH=$HOME/.local/bin:$PATH
export E=experiments/spec-prompting-basetasks
[ -f "$E/tasks.json" ] || { echo "$E/tasks.json missing on this box (scripts/provision_evalbox.sh ships it)" >&2; exit 2; }
export TASKS="$(uv run python -c "import json; print(' '.join(json.load(open('$E/tasks.json'))['ids']))")"
[ "$(echo $TASKS | wc -w)" = 73 ] || { echo "expected 73 task ids, got $(echo $TASKS | wc -w)" >&2; exit 2; }
export VARIANTS="${VARIANTS:-conflicting}"
[ "$(nproc)" -ge 32 ] && DEF=6 || DEF=4
export MAXCONN="${MAXCONN:-$DEF}" MAX_PARALLEL="${MAX_PARALLEL:-4}"
echo "tasks: 73 (non-test MisTrace ids), variants: $VARIANTS"
exec bash experiments/spec-prompting/scripts/run_all.sh
