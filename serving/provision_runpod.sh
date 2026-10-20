#!/usr/bin/env bash
# Provision a RunPod pod as a model server: ship serving/ + serving/bootstrap.sh, start it under
# nohup (pods have no systemd). Usage (repo root):  serving/provision_runpod.sh <podId> [VAR=value ...]
#   e.g. serving/provision_runpod.sh abc123 MAXSEQS=32            # H200, bf16, defaults otherwise
# Create the pod first with tools/runpod/rp create <name>; VAR=value pairs go into the bootstrap's
# environment (see serving/bootstrap.sh). Waits up to 10 min for SSH.
# =====================================================================================
# COMPUTE PROVIDER. This is the authors' RunPod tooling, kept as a worked example.
# >>> ADD COMPUTE TOOLING FOR YOUR PROVIDER HERE. <<<  Any GPU box that can run
# serving/bootstrap.sh and expose vLLM over HTTP works. Credentials come from the repo-root
# .env (COMPUTE_API_KEY for the provider, HF_TOKEN for the model; see .env.example).
# =====================================================================================
set -euo pipefail
POD=$1; shift
RP="tools/runpod/rp"
set -a; . ./.env; set +a
for i in $(seq 1 60); do
  info=$("$RP" raw "pods/$POD"); host=$(jq -r '.publicIp // empty' <<<"$info"); port=$(jq -r '[.portMappings//{}|to_entries[]|select(.key=="22")][0].value // empty' <<<"$info")
  [ -n "$host" ] && [ -n "$port" ] && break; sleep 10
done
[ -z "${host:-}" ] && { echo "no public ip/port after 10 min"; exit 1; }
SSH="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -p $port root@$host"
for i in $(seq 1 30); do $SSH true 2>/dev/null && break; sleep 10; done
ENV_LINE=""; for kv in "$@"; do ENV_LINE="$ENV_LINE export $kv;"; done
# bootstrap parameters may also come from the caller's environment (VAR=value bash provision_*.sh ...)
for v in MODEL DTYPE TP MAXSEQS GPU_UTIL APC HOST STEER_LOG STEER_READ_PC EXTRA_ARGS; do
  case " $* " in *" $v="*) ;; *) [ -n "${!v:-}" ] && ENV_LINE="$ENV_LINE export $v=${!v};";; esac
done
# Fleet replicas provision concurrently; each invocation needs its own local archive.
BUNDLE_DIR=$(mktemp -d)
tar czf "$BUNDLE_DIR/repo.tar.gz" --exclude='serving/.venv' --exclude='__pycache__' serving
scp -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P "$port" "$BUNDLE_DIR/repo.tar.gz" serving/bootstrap.sh root@$host:~/
rm "$BUNDLE_DIR/repo.tar.gz"
rmdir "$BUNDLE_DIR"
$SSH "echo 'export HF_TOKEN=$HF_TOKEN' > ~/.hf_env; chmod 600 ~/.hf_env; $ENV_LINE nohup bash ~/bootstrap.sh > ~/serve.log 2>&1 & echo SERVE_STARTED"
echo "pod $POD  ssh: ssh -p $port root@$host   vllm: http://$host:$(jq -r '[.portMappings//{}|to_entries[]|select(.key=="8000")][0].value // "?"' <<<"$info")"
