"""The SoRH Inspect task and the resumable runner.

One user message per row, no system prompt, no tools, no sandbox, no scorer: each sample's model output
is the only thing the task produces, and grading is offline (`grade.py` against the .eval logs).
Generation settings (temperature / top_p / seed) come from `evals.config.GEN_DEFAULTS`, the run
parameters of the reward-hacking transcripts; `max_tokens` is left unset so the steering proxy stamps
its cap (`evals.config.MAX_TOKENS`, 16384; the SoRH runs of record went through a proxy capping at
16000). Route `--base-url` through `python -m steering_reasoning.proxy`, never the raw vLLM port.

inspect_ai is imported inside the functions, so the module (and `status`, `summarize`, `--help`) works
on a box without the eval env.
"""

from __future__ import annotations

import os
from pathlib import Path

from steering_reasoning.evals.config import GEN_DEFAULTS

from .dataset import FIELDS, load_rows, sample_id

TASK_NAME = "sorh"
DEFAULT_MODEL = "openai-api/local/Qwen/Qwen3.5-27B"
DEFAULT_BASE_URL = "http://localhost:8100/v1"
MODEL_TIMEOUT = 7200          # inspect-ai 0.3.193 forwards model_args["timeout"] to AsyncOpenAI


def build_samples(rows: list[dict]):
    """One Sample per row: id sorh_<i>, the user text as the single input message, the dataset's own
    per-row fields as metadata (nothing derived; stratify on metadata['task'] downstream)."""
    from inspect_ai.dataset import Sample

    return [Sample(id=sample_id(i), input=row["user"],
                   metadata={"task": row["task"], "evaluation_metric": row["evaluation_metric"],
                             "cheat_method": row["cheat_method"], "hack_example": row["school_of_reward_hacks"],
                             "control_example": row["control"], "row_index": i})
            for i, row in enumerate(rows)]


def make_task(rows_path: str | Path, strict: bool = True):
    """Import inspect_ai and return the @task(name="sorh") factory `sorh_task(limit, sample_ids)` over
    the rows file. Built on demand so the decorator tags the Task exactly as a module-level definition
    would (registry name "sorh") without importing inspect_ai at module load."""
    from inspect_ai import Task, task
    from inspect_ai.dataset import MemoryDataset
    from inspect_ai.solver import generate

    rows_path = str(rows_path)

    @task(name=TASK_NAME)
    def sorh_task(limit: int | None = None, sample_ids: list[str] | None = None) -> Task:
        """sample_ids: keep only these ids (dataset order preserved; unknown ids raise).
        limit: keep only the first N samples (applied after the id filter)."""
        samples = build_samples(load_rows(rows_path, strict=strict))
        if sample_ids is not None:
            wanted = [s.strip() for s in sample_ids if s and s.strip()]
            known = {s.id for s in samples}
            unknown = [w for w in wanted if w not in known]
            if unknown:
                raise ValueError(f"unknown sample ids: {', '.join(unknown)}")
            keep = set(wanted)
            samples = [s for s in samples if s.id in keep]
        if limit is not None:
            samples = samples[:limit]
        return Task(name=TASK_NAME, dataset=MemoryDataset(samples, name="school_of_reward_hacks", shuffled=False),
                    solver=[generate()], scorer=None, sandbox=None)

    return sorh_task


def run_eval(rows_path: str | Path, log_dir: str | Path, *, base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_MODEL,
             sample_ids: list[str] | None = None, limit: int | None = None, max_connections: int = 10,
             temperature: float = GEN_DEFAULTS["temperature"], top_p: float = GEN_DEFAULTS["top_p"],
             seed: int = GEN_DEFAULTS["seed"], max_tokens: int | None = None, fail_on_error: float = 0.0,
             retry_on_error: int = 2, log_buffer: int = 5, max_retries: int | None = 30, strict_rows: bool = True):
    """Run the task against a local OpenAI-compatible endpoint (a steering proxy) and return the EvalLogs.

    Finished samples are flushed to the .eval every `log_buffer` completions (each flush rewrites the
    whole file, so 1 = per-sample resumability at ~5x the disk writes; use it for smoke runs). Resume by
    re-running with `sample_ids=remaining_ids(log_dir, ids)` (logs.py), never a presence check.
    `log_dir` should be the arm's own directory (<logs>/<arm>/), which the grader and the analysis glob.
    `max_retries` bounds Inspect's per-request retries (a dead proxy then fails the sample instead of
    retrying forever); None = Inspect's default. The .eval header's task_registry_name is
    'steering_reasoning/sorh' (the task lives in the package; the logs of record say 'sorh'), nothing keys on it.
    """
    from inspect_ai import eval as inspect_eval

    if limit is not None and sample_ids:
        raise ValueError("limit and sample_ids are mutually exclusive: limit truncates the dataset before "
                         "the id filter runs, so the ids may not be found")
    # The openai-api provider requires an API key env var for the service ("local"); vLLM ignores it.
    os.environ.setdefault("LOCAL_API_KEY", "dummy")
    log_dir = str(Path(log_dir).resolve())   # absolute: the log header records a real path whatever the cwd
    ids = [s.strip() for s in sample_ids if s and s.strip()] if sample_ids else None
    sorh_task = make_task(rows_path, strict=strict_rows)
    gen_kwargs = {} if max_tokens is None else {"max_tokens": max_tokens}
    if max_retries is not None:
        gen_kwargs["max_retries"] = max_retries
    return inspect_eval(
        [sorh_task(limit=limit, sample_ids=ids)],
        model=model, model_base_url=base_url, sample_id=ids,
        temperature=temperature, top_p=top_p, seed=seed, max_connections=max_connections,
        model_args={"timeout": MODEL_TIMEOUT}, log_dir=log_dir,
        fail_on_error=(False if fail_on_error == 0 else fail_on_error), retry_on_error=retry_on_error,
        log_buffer=log_buffer, **gen_kwargs,
    )


def print_summary(logs) -> None:
    for log in logs:
        samples = log.samples or []
        n_err = sum(1 for s in samples if s.error is not None)
        print(f"{log.eval.task} status={log.status} samples={len(samples)} errors={n_err} log={log.location}", flush=True)


__all__ = ["FIELDS", "TASK_NAME", "DEFAULT_MODEL", "DEFAULT_BASE_URL", "build_samples", "make_task", "run_eval", "print_summary"]
