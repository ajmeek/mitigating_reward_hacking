# Serving env

Isolated uv project for vLLM — the eval env (repo root) never imports torch/vllm; the two
communicate only over HTTP. Setup notes are in `bootstrap.sh`'s header.

```bash
cd serving
uv sync
bash bootstrap.sh      # MODEL, TP, DTYPE and the rest are environment variables; see its header
```


## Bootstrap and provisioning

`bootstrap.sh` is the single script that turns a fresh GPU box into the patched server (see its
header for the env-var parameters). `provision_runpod.sh` ships this folder and starts it on a RunPod
pod, the authors' provider, and is the place to add compute tooling for your own provider; on any other
box, copy `serving/` over and run `bootstrap.sh` there. Verification: `scripts/steer_probe.py`,
`scripts/steer_cache_probe.py`, `scripts/harness_smoke.sh`.
