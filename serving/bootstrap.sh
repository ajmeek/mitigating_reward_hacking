#!/usr/bin/env bash
# The one serving bootstrap: turns a fresh GPU box (any provider) into the patched vLLM server.
# Runs ON the box as the login user. Ship it with serving/ via serving/provision_runpod.sh (the
# authors' RunPod example) or by hand on any box; never edit copies in experiment folders.
#
# Steps: extract ~/repo.tar.gz (the serving/ tree) if present, install uv, sync the serving env,
# apply the three vllm-lens patches (steering modes; decode-only gate; steered prefix-cache reads),
# export the serving env vars,
# exec vllm serve. Parameters are environment variables with the bf16 single-GPU defaults:
#
#   MODEL=Qwen/Qwen3.5-27B
#   DTYPE=bfloat16
#   TP=1                          tensor-parallel size (4 on a 4x L40S box)
#   MAXSEQS=16                    --max-num-seqs (32 on H200; 4 for STEER_LOG runs)
#   GPU_UTIL=0.92                 --gpu-memory-utilization (0.88 on L40S)
#   APC=1                         prefix caching (0 to disable)
#   STEER_READ_PC=1               VLLM_LENS_STEER_READ_PREFIX_CACHE: steered requests read the prefix cache
#                                 (decision 2026-09-13; 0 restores vllm-lens's forced re-prefill)
#   HOST=0.0.0.0                  bind address (0.0.0.0 behind a provider port mapping; 127.0.0.1 keeps a box private)
#   STEER_LOG=0                   VLLM_LENS_STEER_LOG (1 prints a [STEER] line per step; sequential use only)
#   EXTRA_ARGS=                   appended to vllm serve
#
# Verify after start: grep "PATCHES_OK\|prefix-cache reads" ~/serve.log (both present), curl localhost:8000/v1/models,
# and scripts/probe_strict.py for the tool-calling grammar.
set -uo pipefail
cd ~
if [ -f ~/repo.tar.gz ]; then mkdir -p steering_reasoning && tar xzf ~/repo.tar.gz -C ~/steering_reasoning --no-same-owner; fi
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
export PATH="/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
[ "$(id -u)" = 0 ] && export HF_HOME="${HF_HOME:-/root/hf}"
[ -f ~/.hf_env ] && . ~/.hf_env
[ -n "${HF_TOKEN:-}" ] && export HF_TOKEN
cd ~/steering_reasoning/serving
uv python pin 3.12 > /dev/null
uv sync
# patches 1+2 (patch_vllm_lens.sh, shared with the replay box): steering modes (ablate/clamp), and the
# decode-only gate with positions-aware prefill. Generated tokens (decode: every prompt token already
# computed) are always steered. Prefill steers only positions inside the request's steer_positions
# (the spans the model itself wrote in earlier turns, sent by the proxy), so a re-read of the model's
# own reply gets the same steered KV whether it comes from the prefix cache or a fresh prefill;
# prompts and tool outputs are never steered. Replaces the older "single-token forward pass"
# heuristic (2026-09-13) and the "skip prefill entirely" gate (2026-09-14). A failed patch stops the
# box: a server that came up on an unpatched worker would steer differently and say so only here.
bash patch_vllm_lens.sh patch_vllm_lens_fastpath.py || exit 3   # + patch 4 (2026-09-24): steering fast path, patch_vllm_lens_fastpath.py
# patch 3: steering-only requests may read the prefix cache. vllm-lens forces skip_reading_prefix_cache
# on every hooked request because cached tokens never run the forward pass (right for activation
# capture). With patch 2 a cached block and a fresh prefill hold the same KV (prompt tokens unsteered,
# model-written tokens steered either way), so for steering the cache is only a speed-up and reads are
# safe (decision 2026-09-13/14). Capture requests and explicit skip_reading_prefix_cache still force a
# fresh prefill.
uv run python - <<'PYEOF'
from pathlib import Path
import vllm_lens
p = Path(vllm_lens.__file__).parent / "_activations_plugin.py"
t = p.read_text()
if "VLLM_LENS_STEER_READ_PREFIX_CACHE" not in t:
    a = "    needs_hooks = wants_activations or steering_vectors is not None\n    if needs_hooks or skip_kv_cache:\n"
    b = "    needs_hooks = wants_activations or has_steering\n    if needs_hooks or any_skip_kv_cache:\n"
    assert a in t and b in t, "plugin anchors"
    t = t.replace(a, "    needs_hooks = wants_activations or steering_vectors is not None\n"
                     "    if wants_activations or skip_kv_cache or (steering_vectors is not None and not _STEER_READ_PC):  # local patch\n", 1)
    t = t.replace(b, "    needs_hooks = wants_activations or has_steering\n"
                     "    if wants_activations or any_skip_kv_cache or (has_steering and not _STEER_READ_PC):  # local patch\n", 1)
    i = t.index("\ndef ") if "\ndef " in t else t.index("\nclass ")
    t = t[:i] + ('\n_STEER_READ_PC = (__import__("os").environ.get("VLLM_LENS_STEER_READ_PREFIX_CACHE") == "1"\n'
                 '                  and __import__("os").environ.get("VLLM_LENS_STEER_DECODE_ONLY") == "1")  # local patch\n') + t[i:]
    p.unlink(); p.write_text(t)   # replace, don't truncate: uv hardlinks the venv to its wheel cache
    print("steered prefix-cache reads patched")
else:
    print("steered prefix-cache reads present")
PYEOF
[ $? = 0 ] || { echo "=== PATCH_FAILED prefix-cache reads ==="; exit 3; }

MODEL="${MODEL:-Qwen/Qwen3.5-27B}"; TP="${TP:-1}"; MAXSEQS="${MAXSEQS:-16}"; GPU_UTIL="${GPU_UTIL:-0.92}"
APC="${APC:-1}"; HOST="${HOST:-0.0.0.0}"; STEER_LOG="${STEER_LOG:-0}"; STEER_READ_PC="${STEER_READ_PC:-1}"; EXTRA_ARGS="${EXTRA_ARGS:-}"
DTYPE="${DTYPE:-bfloat16}"
APC_ARG=""; [ "$APC" = 1 ] && APC_ARG="--enable-prefix-caching"

export VLLM_FLASH_ATTN_VERSION=2
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_LENS_STEER_DECODE_ONLY=1
export VLLM_LENS_STEER_LOG="$STEER_LOG"
export VLLM_LENS_STEER_READ_PREFIX_CACHE="$STEER_READ_PC"
# vLLM 0.24 defaults this to 1: an xgrammar structural tag compiled from the tool schemas
# constrains tool-call decoding under tool_choice=auto, and for Inspect's text_editor schema the
# grammar forbids new_str, so every str_replace deleted code.
export VLLM_ENFORCE_STRICT_TOOL_CALLING=0

echo "=== SERVE_STARTING $(date -u +%H:%M) model=$MODEL dtype=$DTYPE tp=$TP maxseqs=$MAXSEQS gpu_util=$GPU_UTIL apc=$APC host=$HOST steer_log=$STEER_LOG read_pc=$STEER_READ_PC ==="
exec uv run vllm serve "$MODEL" \
  --port 8000 --host "$HOST" --dtype "$DTYPE" --enforce-eager \
  --max-model-len 131072 --max-num-seqs "$MAXSEQS" --tensor-parallel-size "$TP" \
  --gpu-memory-utilization "$GPU_UTIL" $APC_ARG \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 $EXTRA_ARGS
