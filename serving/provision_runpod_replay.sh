#!/usr/bin/env bash
# Provision a RunPod pod for REPLAY capture (tools/runpod/bootstrap_replay_runpod.sh), and pull the
# shards back afterwards. Usage, from the repo root:
#
#   serving/provision_runpod_replay.sh <podId> --records <records.jsonl ...> [VAR=value ...]
#   serving/provision_runpod_replay.sh --pull <podId> <local dir>
#
#   e.g. serving/provision_runpod_replay.sh abc123 \
#          --records data/cache/replay-records/records_base_{original,conflicting}_s42.jsonl \
#          ARMS=base SPLITS="original conflicting" LAYERS=35 SEED=42
#
# Each records file must be named records_<arm>_<split>_s<seed>.jsonl (what the bootstrap looks for).
# VAR=value pairs go into the bootstrap's environment. Create the pod first with
# tools/runpod/rp create <name> "NVIDIA H200" 1, and delete it yourself when the pull is done.
# =====================================================================================
# COMPUTE PROVIDER. This is the authors' RunPod tooling, kept as a worked example.
# >>> ADD COMPUTE TOOLING FOR YOUR PROVIDER HERE. <<<  Any GPU box that can run
# serving/bootstrap.sh and expose vLLM over HTTP works. Credentials come from the repo-root
# .env (COMPUTE_API_KEY for the provider, HF_TOKEN for the model; see .env.example).
# =====================================================================================
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
RP="tools/runpod/rp"
set -a; . ./.env; set +a

pod_ssh() {   # echoes "<host> <port>" once the pod answers ssh
  local pod=$1 host port info
  for i in $(seq 1 60); do
    info=$("$RP" raw "pods/$pod")
    host=$(jq -r '.publicIp // empty' <<<"$info")
    port=$(jq -r '[.portMappings//{}|to_entries[]|select(.key=="22")][0].value // empty' <<<"$info")
    [ -n "$host" ] && [ -n "$port" ] && break
    sleep 10
  done
  [ -z "${host:-}" ] && { echo "no public ip/port after 10 min" >&2; exit 1; }
  for i in $(seq 1 30); do ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -p "$port" "root@$host" true 2>/dev/null && break; sleep 10; done
  echo "$host $port"
}

if [ "${1:-}" = "--pull" ]; then
  POD=$2; DEST=${3:?usage: --pull <podId> <local dir>}
  read -r host port < <(pod_ssh "$POD")
  mkdir -p "$DEST"
  # the bootstrap prints its dumps root; ask the pod rather than guessing the mount
  OUT=$(ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p "$port" "root@$host" \
        'grep -o "out=[^ ]*" ~/replay.log | tail -1 | cut -d= -f2')
  [ -n "$OUT" ] || { echo "could not find the dumps root in ~/replay.log" >&2; exit 1; }
  echo "pulling $OUT -> $DEST"
  rsync -az --info=progress2 -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $port" \
        "root@$host:$OUT/" "$DEST/"
  echo "pulled; delete the pod with: $RP rm $POD"
  exit 0
fi

POD=$1; shift
RECORDS=()
ENV_LINE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --records) shift; while [ $# -gt 0 ] && [ "${1#--}" = "$1" ] && [ -f "$1" ]; do RECORDS+=("$1"); shift; done ;;
    *=*) ENV_LINE="$ENV_LINE export $1;"; shift ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done
[ ${#RECORDS[@]} -gt 0 ] || { echo "no --records given" >&2; exit 2; }

read -r host port < <(pod_ssh "$POD")
SSH="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -p $port root@$host"
BUNDLE_DIR=$(mktemp -d)
# capture_rollouts.py needs serving/ (the patched vllm-lens env), src/ (replay + canonical vectors)
# and scripts/; COPYFILE_DISABLE keeps macOS from adding ._* sidecars that break hash walks.
COPYFILE_DISABLE=1 tar czf "$BUNDLE_DIR/repo.tar.gz" \
  --exclude='serving/.venv' --exclude='__pycache__' --exclude='*.pyc' \
  --exclude='data/cache' --exclude='data/scratch' \
  serving src scripts
scp -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P "$port" \
  "$BUNDLE_DIR/repo.tar.gz" tools/runpod/bootstrap_replay_runpod.sh "${RECORDS[@]}" "root@$host:~/"
rm -rf "$BUNDLE_DIR"
$SSH "echo 'export HF_TOKEN=$HF_TOKEN' > ~/.hf_env; chmod 600 ~/.hf_env; \
      $ENV_LINE nohup bash ~/bootstrap_replay_runpod.sh > ~/replay.log 2>&1 & echo REPLAY_STARTED"
echo "pod $POD  ssh: ssh -p $port root@$host   log: ~/replay.log"
echo "progress:  ssh -p $port root@$host 'tail -f ~/replay.log'"
