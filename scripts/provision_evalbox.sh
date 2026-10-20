#!/usr/bin/env bash
# Provision a CPU eval box (Docker + Inspect + proxies) over SSH: any Ubuntu host with sudo,
# 16 vCPU or more. Usage: [BOX_USER=ubuntu] [KEY=~/.ssh/id_ed25519] scripts/provision_evalbox.sh <ip>
# =====================================================================================
# COMPUTE PROVIDER. Launching the box itself is up to you: >>> ADD COMPUTE TOOLING FOR
# YOUR PROVIDER <<< (the authors used an EC2 instance); this script only needs its IP.
# =====================================================================================
set -euo pipefail
IP=$1; KEY="${KEY:-$HOME/.ssh/id_ed25519}"; U="${BOX_USER:-ubuntu}"
SSH="ssh -i $KEY -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 $U@$IP"
COPYFILE_DISABLE=1 tar czf /tmp/evalrepo.tar.gz --exclude='.venv' --exclude='__pycache__' --exclude='.git' \
  --exclude='.env' --exclude='.env.*' \
  --exclude='experiments/*/data' --exclude='.agent-sandbox' --exclude='data/cache' --exclude='data/scratch' --exclude='logs' \
  README.md pyproject.toml uv.lock src scripts serving tools/runpod/rp external/impossiblebench \
  src/steering_reasoning/canonical $(ls experiments/*/artifacts/*.safetensors experiments/*/artifacts/*.json 2>/dev/null) \
  $(ls -d experiments/*/scripts) $(ls experiments/*/*.tsv 2>/dev/null) $(ls experiments/*/tasks.json 2>/dev/null) spec \
  $(ls experiments/*/fleet*.toml 2>/dev/null)
scp -q -i $KEY /tmp/evalrepo.tar.gz $U@$IP:~/
$SSH "sudo loginctl enable-linger $U; mkdir -p ~/steering_reasoning && tar xzf ~/evalrepo.tar.gz -C ~/steering_reasoning --no-same-owner; command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH=\$HOME/.local/bin:\$PATH; cd ~/steering_reasoning && uv sync > ~/evalenv.log 2>&1 && echo EVAL_ENV_OK; sudo usermod -aG docker $U; sudo docker pull aisiuk/inspect-tool-support:latest > /dev/null 2>&1 && echo IMAGE_OK; docker ps >/dev/null 2>&1 && echo DOCKER_OK || echo DOCKER_NEEDS_RELOGIN"
echo "provisioned $IP"
