#!/usr/bin/env bash
# Box-side bootstrap for REPLAY capture (the authors' RunPod version). Shards stay on the
# box (pull them with serving/provision_runpod_replay.sh --pull) and nothing self-terminates —
# delete the box yourself (tools/runpod/rp rm <id> on RunPod).
# =====================================================================================
# COMPUTE PROVIDER. This is the authors' RunPod tooling, kept as a worked example.
# >>> ADD COMPUTE TOOLING FOR YOUR PROVIDER HERE. <<<  Any GPU box that can run
# serving/bootstrap.sh and expose vLLM over HTTP works. Credentials come from the repo-root
# .env (COMPUTE_API_KEY for the provider, HF_TOKEN for the model; see .env.example).
# =====================================================================================
#
# Expects in ~: repo.tar.gz (serving/ + src/ + scripts/) and records_<ARM>_<SPLIT>_s<SEED>.jsonl
# per arm and split. Pods have no systemd, so the provisioner starts this under nohup.
#
# Env (set by the provisioner):
#   ARMS    "base"                  arms to replay, records_<arm>_<split>_s<seed>.jsonl in ~
#   SPLITS  "original conflicting"  splits to replay per arm (the AWS script does conflicting only)
#   LAYERS  35                      'all' or comma-separated layer indices
#   SEED    42
#   MODE_<arm>/DOSE_<arm>           steering of a steered arm, as in arms.tsv (omit for base)
#   ART     steering artifact (default: the canonical u)
#   OUT     dumps root (default: the largest writable mount)
set -uo pipefail
cd ~
mkdir -p steering_reasoning
tar xzf ~/repo.tar.gz -C ~/steering_reasoning --no-same-owner
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
[ -f ~/.hf_env ] && . ~/.hf_env
[ -n "${HF_TOKEN:-}" ] && export HF_TOKEN
[ "$(id -u)" = 0 ] && export HF_HOME="${HF_HOME:-/root/hf}"

# Shards are ~10 KB per captured row per layer, so pick the roomiest writable mount.
if [ -z "${OUT:-}" ]; then
  OUT=$HOME/dumps
  for d in /workspace /runpod-volume; do [ -d "$d" ] && [ -w "$d" ] && OUT=$d/dumps && break; done
fi
mkdir -p "$OUT"
echo "=== REPLAY_START gpus=$(nvidia-smi -L | wc -l) out=$OUT free=$(df -h "$OUT" | awk 'NR==2{print $4}') $(date -u +%H:%M) ==="
nvidia-smi -L

cd ~/steering_reasoning/serving
uv python pin 3.12 > /dev/null
uv sync || exit 2
bash patch_vllm_lens.sh patch_vllm_lens_disable_zstd.py || exit 3
export VLLM_LENS_DISABLE_ZSTD=1
export VLLM_LENS_STEER_DECODE_ONLY=1   # positions-aware gate, as at generation (inert for base)
export VLLM_FLASH_ATTN_VERSION=2 VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_FLASHINFER_SAMPLER=0
TP=$(nvidia-smi -L | wc -l)
ART="${ART:-../src/steering_reasoning/canonical/u_L35_steering.safetensors}"
LAYERS="${LAYERS:-35}"
SEED="${SEED:-42}"
rc_all=0
for ARM in ${ARMS:-base}; do
  mode_var="MODE_${ARM}"; dose_var="DOSE_${ARM}"
  MODE="${!mode_var:-}"; DOSE="${!dose_var:-}"
  SA=""; [ -n "$MODE" ] && SA="--steer-artifact $ART --mode $MODE ${DOSE:+--dose $DOSE}"
  for SPLIT in ${SPLITS:-conflicting}; do
    REC=~/records_${ARM}_${SPLIT}_s${SEED}.jsonl
    [ -f "$REC" ] || { echo "=== NO_RECORDS $REC — skipping ==="; rc_all=4; continue; }
    echo "=== CELL_STARTING $ARM/$SPLIT mode=${MODE:-none} layers=$LAYERS tp=$TP $(date -u +%H:%M) ==="
    uv run python ../scripts/capture_rollouts.py \
      --records "$REC" --model Qwen/Qwen3.5-27B --layers "$LAYERS" \
      --max-model-len 131072 --tensor-parallel "$TP" --gpu-memory-utilization 0.90 \
      --host-mem-margin-gib 64 --expect-tp "${EXPECT_TP:-1}" \
      $SA --out-dir "$OUT/$ARM/$SPLIT/s$SEED"
    rc=$?; echo "=== CELL_EXIT $ARM/$SPLIT rc=$rc free=$(df -h "$OUT" | awk 'NR==2{print $4}') $(date -u +%H:%M) ==="
    [ $rc -eq 0 ] || rc_all=$rc
  done
done
echo "=== EXIT=$rc_all $(date -u +%H:%M) ==="
echo "=== PULL_WITH: serving/provision_runpod_replay.sh --pull <podId> <local dir> ==="
exit $rc_all
