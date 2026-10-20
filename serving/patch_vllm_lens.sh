#!/usr/bin/env bash
# Apply the local vllm-lens patches to the CURRENT serving env (run inside serving/ after `uv sync`):
# steering modes (+ seq_lens) and the positions-aware decode-only gate. A failed patch exits non-zero
# so a bootstrap stops instead of starting a server or a replay on an unpatched worker. Extra patch
# scripts may be passed as arguments (e.g. patch_vllm_lens_disable_zstd.py for replay transport).
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")"
for patch in patch_vllm_lens_modes.py patch_vllm_lens_gate.py "$@"; do
  uv run python "$patch" || { echo "=== PATCH_FAILED $patch ==="; exit 3; }
done
echo "=== PATCHES_OK modes gate $* ==="
