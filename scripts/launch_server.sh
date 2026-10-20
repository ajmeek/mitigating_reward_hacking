#!/usr/bin/env bash
# Bring up a FRESH bf16 serving box and verify it end to end. Prints UPSTREAM=<url> for
# scripts/run_arms.sh. The runpod case never reuses an existing pod: every run gets new infrastructure.
#
#   bash scripts/launch_server.sh --provider runpod   [--gpu "NVIDIA H200"] [--name <pod name>]
#   bash scripts/launch_server.sh --provider external --upstream http://<host>:<port>
#
# =====================================================================================
# COMPUTE PROVIDER. `runpod` is the authors' provider, kept as a worked example (tools/runpod/rp,
# serving/provision_runpod.sh; needs COMPUTE_API_KEY and HF_TOKEN in .env, see .env.example).
# `external` is any box you launched yourself: copy serving/ over, run serving/bootstrap.sh
# there, and pass its URL; this script then only runs the verification probes against it.
# >>> ADD COMPUTE TOOLING FOR YOUR PROVIDER <<< as a third case below (create the box,
# provision it with serving/provision_runpod.sh's recipe, resolve UPSTREAM and SSHCMD).
# =====================================================================================
#
# Checks after the server is up: four vllm-lens patch lines in serve.log (when SSH is available);
# the strict-tool-calling probe (new_str present in a parsed str_replace under tool_choice=auto);
# scripts/steer_probe.py (base == direct, base != clamp); scripts/steer_cache_probe.py (cache
# engaged; unsteered output unchanged; clamp differs; cache salt isolates arms). Needs uv and the
# repo's eval env for the probes; the runpod case also needs ~/.ssh/id_ed25519.pub (rp create sends it).
# Read serving/README.md first; scripts/run_arms.sh's header is the end-to-end recipe.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
PROVIDER=""; GPU="NVIDIA H200"; UPSTREAM=""; NAME="${POD_NAME:-${USER:-user}-serve-$(date -u +%m%d%H%M)}"
while [ $# -gt 0 ]; do case "$1" in --provider) PROVIDER=$2; shift 2;; --gpu) GPU=$2; shift 2;; --upstream) UPSTREAM=$2; shift 2;; --name) NAME=$2; shift 2;; *) echo "unknown arg $1"; exit 2;; esac; done
[ -z "$PROVIDER" ] && { echo "usage: --provider runpod|external [--upstream URL]"; exit 2; }
MODEL="${MODEL:-Qwen/Qwen3.5-27B}"; ART="${ART:-src/steering_reasoning/canonical/u_L35_steering.safetensors}"
case "$PROVIDER" in
  runpod)
    CREATED=$(tools/runpod/rp create "$NAME" "$GPU" 1); echo "$CREATED"
    POD=$(echo "$CREATED" | python3 -c "import json,sys; d=json.load(sys.stdin); assert d.get('id') and not d.get('error'), d; print(d['id'])")
    echo "pod $POD ($NAME) created; provisioning (bf16, TP=1, MAXSEQS=${MAXSEQS:-32})"
    bash serving/provision_runpod.sh "$POD" MODEL="$MODEL" TP=1 MAXSEQS="${MAXSEQS:-32}"
    INFO=$(tools/runpod/rp info "$POD"); HOST=$(echo "$INFO" | python3 -c "import json,sys; print(json.load(sys.stdin)['publicIp'])"); P8=$(echo "$INFO" | python3 -c "import json,sys; print(json.load(sys.stdin)['portMappings']['8000'])")
    UPSTREAM="http://$HOST:$P8"; SSHCMD=$(tools/runpod/rp sshcmd "$POD") ;;
  external)
    [ -z "$UPSTREAM" ] && { echo "--provider external needs --upstream http://<host>:<port> (a box running serving/bootstrap.sh)"; exit 2; }
    POD="external"; SSHCMD="${SSHCMD:-}" ;;   # export SSHCMD="ssh <user>@<host>" to also check serve.log
  # your_provider)
  #   create the box, run serving/bootstrap.sh on it (see serving/provision_runpod.sh), then set
  #   UPSTREAM=http://<host>:<port> and SSHCMD="ssh ..." ;;
  *) echo "provider must be runpod or external"; exit 2;;
esac
echo "waiting for $UPSTREAM/v1/models (model download + load: ~15 min on a fresh box; the encoder-profiling step sits at 0% GPU for several minutes)"
for i in $(seq 1 120); do curl -s -m 5 "$UPSTREAM/v1/models" | grep -q Qwen && break; sleep 15; done
if [ -n "$SSHCMD" ]; then echo "--- patches in serve.log (expect PATCHES_OK + prefix-cache reads):"; $SSHCMD 'grep "PATCHES_OK\|prefix-cache reads" ~/serve.log'; fi
echo "--- strict tool-calling probe (expect new_str in the parsed str_replace under tool_choice=auto):"
python3 scripts/probe_strict.py "$UPSTREAM" "$MODEL" scripts/inspect_tools.json | grep -E "^tool_choice=|parsed" || echo "PROBE FAILED"
echo "--- steer probe (expect: base == direct: True   base == clamp: False):"
ART="$ART" MODEL="$MODEL" uv run python scripts/steer_probe.py "$UPSTREAM" -8 | tail -1 || echo "STEER PROBE FAILED"
echo "--- prefix-cache / positions probe (expect: cache engaged > 0; unsteered True; clamp CONSISTENT; prefill ACTIVE; salt isolates True):"
ART="$ART" MODEL="$MODEL" uv run python scripts/steer_cache_probe.py "$UPSTREAM" -16 | tail -5 || echo "CACHE PROBE FAILED"
echo; echo "UPSTREAM=$UPSTREAM"; if [ "$PROVIDER" = runpod ]; then echo "(pod $POD; tear it down when the arms are done: tools/runpod/rp rm $POD)"; fi
