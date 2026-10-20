# Box bootstrap + PHASE 1 of the full capture (3-rollout GPU smoke-slice).
# Phase 2 (--rollouts all, resume skips phase-1 rollouts) is launched separately
# after on-box verification of phase-1 outputs. Expects in ~: repo.tar.gz,
# hf_token, ~/.aws/. Runs under systemd-run --user (linger on).
set -uo pipefail
cd ~
mkdir -p steering_reasoning
tar xzf ~/repo.tar.gz -C ~/steering_reasoning --no-same-owner
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
[ -f ~/hf_token ] && export HF_TOKEN=$(cat ~/hf_token)

SCRATCH=$HOME
for d in /opt/dlami/nvme /mnt/nvme; do [ -d "$d" ] && [ -w "$d" ] && SCRATCH=$d && break; done
export HF_HOME=$SCRATCH/hf
mkdir -p "$SCRATCH/all103"
echo "=== SCRATCH=$SCRATCH ==="

cd ~/steering_reasoning/serving
uv python pin 3.12 > /dev/null
uv sync
uv run python patch_vllm_lens_prefix_reuse.py
uv run python patch_vllm_lens_disable_zstd.py
export VLLM_LENS_DISABLE_ZSTD=1   # bf16 activations barely compress; skip ~5.6s/call zstd

TP=$(nvidia-smi -L | wc -l)
echo "=== PHASE1_STARTING tp=$TP $(date -u +%H:%M) ==="
uv run python ../scripts/capture_all_tokens.py \
  --out-dir "$SCRATCH/all103" \
  --tensor-parallel "$TP" \
  --rollouts lcbhard_2,lcbhard_22,lcbhard_100 \
  --s3-prefix s3://steering-reasoning-artifacts/activations/qwen35_27b_bf16_all103
rc=$?
echo "=== PHASE1_EXIT=$rc ==="
