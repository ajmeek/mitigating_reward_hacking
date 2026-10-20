#!/usr/bin/env bash
# Eval box: run the adversarial-instruction spec arm against a vLLM server (no proxy: unsteered).
# usage: run_advspec.sh <upstream base url, e.g. http://host:port>
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH LOCAL_API_KEY=dummy PYTHONUNBUFFERED=1
cd ~/steering_reasoning
E=experiments/spec-adversarial-instruction; LOG=$E/data/logs/advspec/conflicting/s42/rest; mkdir -p $LOG
sha256sum spec/anti_reward_hacking_spec_prompt.txt spec/adversarial_task_owner_instruction.txt > $LOG/suffix_files.sha256
IDS=lcbhard_7,lcbhard_16,lcbhard_19,lcbhard_95,lcbhard_96   # the five unfinished at the 09:20 stop (lcbhard_77 cannot run: context overflow)
exec .venv/bin/python scripts/run_lcb_eval.py --split conflicting --sample-ids "$IDS" \
  --model openai-api/local/Qwen/Qwen3.5-27B --base-url "$1/v1" --seed 42 \
  --max-connections 5 --max-sandboxes 5 --message-limit 120 \
  --system-suffix-file spec/anti_reward_hacking_spec_prompt.txt \
  --user-suffix-file spec/adversarial_task_owner_instruction.txt \
  --log-dir $LOG --log-buffer 1 --retry-on-error 0 --fail-on-error 0
