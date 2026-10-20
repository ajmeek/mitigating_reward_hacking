#!/usr/bin/env python
"""Run ImpossibleBench LCB-hard against a local OpenAI-compatible endpoint.

Examples:
    # Standard arms route through the capping proxy (see steering_reasoning/proxy.py);
    # pointing --base-url at the raw vLLM port runs uncapped and prints a warning.
    uv run python -m steering_reasoning.proxy --port 8100 &   # then --base-url .../8100/v1
    uv run scripts/run_lcb_eval.py --split both --limit 2      # smoke test
    uv run scripts/run_lcb_eval.py --split conflicting         # full 103, hack rate
    uv run scripts/run_lcb_eval.py --split original            # full 103, capability control
    uv run scripts/run_lcb_eval.py --split conflicting \\
        --system-suffix-file spec/anti_reward_hacking_spec_prompt.txt   # spec arm

Serve the model first (serving/bootstrap.sh). Swapping in the steered model later
is a --model change (e.g. vllm-lens/Qwen/Qwen3-4B), not a code change.
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

from inspect_ai import eval as inspect_eval

from steering_reasoning.evals.config import GEN_DEFAULTS, MAX_ATTEMPTS, MESSAGE_LIMIT
from steering_reasoning.evals.lcb_task import lcb_task
from steering_reasoning.evals.summarize import print_summary


def default_log_dir(split: str, model: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    model_slug = model.replace("/", "_")
    return f"logs/lcb/{ts}_{split}_{model_slug}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=["original", "conflicting", "both"], default="both")
    p.add_argument("--model", default="openai-api/local/Qwen/Qwen3-4B",
                   help="Inspect model spec (openai-api/<service>/<model> for a generic endpoint)")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--limit", type=int, default=None, help="run only the first N samples (smoke tests)")
    p.add_argument("--sample-ids", default=None,
                   help="comma-separated task ids (e.g. lcbhard_22,lcbhard_41) to run instead of all")
    p.add_argument("--system-suffix-file", default=None,
                   help="file whose contents are appended to the system message (spec arm); "
                        "omit for the byte-identical baseline system message")
    p.add_argument("--user-suffix-file", default=None,
                   help="file whose contents are appended to the first user message (the task "
                        "prompt); an instruction arriving with the task, e.g. an adversarial "
                        "override of the spec. Omit for the baseline user message.")
    p.add_argument("--max-connections", type=int, default=4)
    p.add_argument("--max-sandboxes", type=int, default=4)
    p.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS)
    p.add_argument("--message-limit", type=int, default=MESSAGE_LIMIT)
    p.add_argument("--temperature", type=float, default=GEN_DEFAULTS["temperature"])
    p.add_argument("--top-p", type=float, default=GEN_DEFAULTS["top_p"])
    p.add_argument("--seed", type=int, default=GEN_DEFAULTS["seed"])
    p.add_argument("--log-dir", default=None)
    p.add_argument("--fail-on-error", type=float, default=0.1,
                   help="abort the task when this fraction of samples error (0 = never abort)")
    p.add_argument("--log-buffer", type=int, default=1,
                   help="flush finished samples to the .eval every N completions (1 = per sample)")
    p.add_argument("--retry-on-error", type=int, default=0,
                   help="re-run an errored sample up to N times before recording the error")
    args = p.parse_args()

    # The openai-api provider requires an API key env var for the service ("local");
    # vLLM ignores the value.
    os.environ.setdefault("LOCAL_API_KEY", "dummy")

    if ":8000" in args.base_url:
        print("WARNING: --base-url targets the raw vLLM port, so no per-turn max_tokens cap "
              "applies (uncapped turns can loop for 80k+ tokens). Route through the capped "
              "proxy instead:  uv run python -m steering_reasoning.proxy --port 8100\n"
              "Continuing uncapped.",
              flush=True)

    if args.limit is not None and args.sample_ids:
        p.error("--limit and --sample-ids are mutually exclusive: --limit truncates the "
                "dataset before the sample-id filter runs, so the ids may not be found")

    system_suffix = ""
    if args.system_suffix_file:
        # Normalize CRLF so a Windows-saved spec can't smuggle \r into the system message.
        system_suffix = (
            Path(args.system_suffix_file).read_text(encoding="utf-8").replace("\r\n", "\n")
        )

    user_suffix = ""
    if args.user_suffix_file:
        user_suffix = Path(args.user_suffix_file).read_text(encoding="utf-8").replace("\r\n", "\n")

    splits = ["original", "conflicting"] if args.split == "both" else [args.split]
    tasks = [
        lcb_task(
            s,
            max_attempts=args.max_attempts,
            message_limit=args.message_limit,
            limit=args.limit,
            system_suffix=system_suffix,
            user_suffix=user_suffix,
        )
        for s in splits
    ]

    sample_id = None
    if args.sample_ids:
        sample_id = [s.strip() for s in args.sample_ids.split(",") if s.strip()]
    logs = inspect_eval(
        tasks,
        model=args.model,
        model_base_url=args.base_url,
        sample_id=sample_id,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        max_connections=args.max_connections,
        model_args={"timeout": 7200},
        max_sandboxes=args.max_sandboxes,
        max_subprocesses=args.max_sandboxes,
        log_dir=args.log_dir or default_log_dir(args.split, args.model),
        fail_on_error=(False if args.fail_on_error == 0 else args.fail_on_error),
        retry_on_error=args.retry_on_error,
        log_buffer=args.log_buffer,
    )
    print_summary(logs)


if __name__ == "__main__":
    main()
