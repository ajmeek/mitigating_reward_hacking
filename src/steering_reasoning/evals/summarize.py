"""Summarize ImpossibleBench LCB eval logs: solve rate (original) / hack rate (conflicting)."""

from inspect_ai.log import EvalLog, read_eval_log
from inspect_ai.scorer import CORRECT


def _split_from_task_name(task_name: str) -> str:
    # Task names look like lcb_<split>_canmod_tools.
    parts = task_name.split("_")
    return parts[1] if len(parts) > 1 and parts[0] == "lcb" else task_name


def summarize_log(log: EvalLog) -> dict:
    task_name = log.eval.task
    split = _split_from_task_name(task_name)

    passed_ids: list[str] = []
    error_ids: list[str] = []
    n_samples = 0
    for sample in log.samples or []:
        n_samples += 1
        if sample.error is not None:
            error_ids.append(str(sample.id))
            continue
        for score in (sample.scores or {}).values():
            if score.value == CORRECT:
                passed_ids.append(str(sample.id))
                break

    n_scored = n_samples - len(error_ids)
    return {
        "task": task_name,
        "split": split,
        "status": log.status,
        "n_samples": n_samples,
        "n_errors": len(error_ids),
        "error_ids": error_ids,
        "n_passed": len(passed_ids),
        "passed_ids": passed_ids,
        "pass_rate": len(passed_ids) / n_scored if n_scored else 0.0,
    }


def print_summary(logs: list[EvalLog]) -> None:
    for log in logs:
        s = summarize_log(log)
        rate_label = "HACK RATE" if s["split"] in ("conflicting", "oneoff") else "solve rate"
        print(f"\n=== {s['task']} ({s['status']}) ===")
        print(f"samples: {s['n_samples']}  errors: {s['n_errors']}")
        print(f"{rate_label}: {s['n_passed']}/{s['n_samples'] - s['n_errors']} = {s['pass_rate']:.1%}")
        if s["split"] in ("conflicting", "oneoff") and s["passed_ids"]:
            print(f"hacked task ids: {', '.join(s['passed_ids'])}")
        if s["error_ids"]:
            print(f"errored task ids: {', '.join(s['error_ids'])}")


def summarize_log_dir(log_dir: str) -> None:
    from inspect_ai.log import list_eval_logs

    infos = list_eval_logs(log_dir)
    if not infos:
        print(f"No eval logs found in {log_dir}")
        return
    print_summary([read_eval_log(info) for info in infos])
