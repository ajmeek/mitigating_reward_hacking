#!/usr/bin/env bash
# Eval box: run the adversarial-instruction spec arm against a vLLM server (no proxy: unsteered).
# usage: run_advspec.sh <upstream base url, e.g. http://host:port>
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH LOCAL_API_KEY=dummy PYTHONUNBUFFERED=1
cd ~/steering_reasoning
E=experiments/spec-adversarial-instruction; LOG=$E/data/logs/advspec/conflicting/s42; mkdir -p $LOG
sha256sum spec/anti_reward_hacking_spec_prompt.txt spec/adversarial_task_owner_instruction.txt > $LOG/suffix_files.sha256
IDS=$(python3 -c "import json; s=json.load(open('src/steering_reasoning/data/splits.json')); print(','.join(sorted(s['impossible']['test'])))")
exec .venv/bin/python scripts/run_lcb_eval.py --split conflicting --sample-ids "$IDS" \
  --model openai-api/local/Qwen/Qwen3.5-27B --base-url "$1/v1" --seed 42 \
  --max-connections 30 --max-sandboxes 30 --message-limit 120 \
  --system-suffix-file spec/anti_reward_hacking_spec_prompt.txt \
  --user-suffix-file spec/adversarial_task_owner_instruction.txt \
  --log-dir $LOG --log-buffer 1 --retry-on-error 0 --fail-on-error 0
