"""Small fleet CLI. Planning is offline; up provisions, run performs evaluations."""

import argparse
import asyncio
import csv
import json
import time
from collections import Counter
from pathlib import Path

from .plan import make_plan, verify_sources
from .providers import down, load_resources, up
from .runner import execute, export_results
from .state import State, locked, write_json


def run_directory(root, plan):
    return root / plan["experiment"] / "data/fleet" / plan["run_id"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["plan", "up", "run", "status", "drain", "export", "down"]
    )
    parser.add_argument("config", type=Path, help="experiment fleet.toml (paths relative to repo)")
    parser.add_argument("--resume", action="store_true", help="run: clear an earlier drain request")
    parser.add_argument(
        "--retry-failed", action="store_true", help="run: allow one more attempt for failed jobs"
    )
    parser.add_argument(
        "--yes", action="store_true", help="down: confirm deletion of this run's owned pods"
    )
    parser.add_argument(
        "--tokens-per-rollout",
        type=float,
        default=33000,
        help="plan estimate assumption, not measured",
    )
    parser.add_argument(
        "--tokens-per-second",
        type=float,
        default=None,
        help="plan: measured aggregate throughput per GPU (homogeneous estimate)",
    )
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    config = args.config if args.config.is_absolute() else root / args.config
    # Find existing plans without rehashing source files: status/drain must work after edits.
    import tomllib
    from .plan import name, relative

    cfg = tomllib.loads(config.read_text())
    directory = root / relative(root, config.parent) / "data/fleet" / name(cfg["run_id"])
    manifest_path = directory / "plan.json"
    if args.command == "status":
        plan = json.loads(manifest_path.read_text())
        state = State(directory, plan)
        metrics = {}
        path = directory / "metrics.jsonl"
        if path.exists():
            with path.open() as f:
                for line in f:
                    try:
                        row = json.loads(line)
                        metrics[row["replica"]] = row
                    except json.JSONDecodeError:
                        pass
        print(
            json.dumps(
                {
                    "run": plan["run_id"],
                    "jobs": state.counts(),
                    "metrics": {
                        k: {**v, "age_seconds": time.time() - v["time"]} for k, v in metrics.items()
                    },
                    "resources": load_resources(directory),
                },
                indent=2,
            )
        )
        return 0
    if args.command == "drain":
        if not manifest_path.exists():
            raise ValueError("Run has not been planned")
        # This one control write intentionally bypasses the coordinator's exclusive lock.
        write_json(directory / "drain.json", {"requested_at": time.time()})
        print("Drain requested: active batches finish; pending jobs remain resumable.")
        return 0
    with locked(directory):
        if args.command == "plan":
            plan = make_plan(config, root)
            if manifest_path.exists():
                previous = json.loads(manifest_path.read_text())
                if previous["fingerprint"] != plan["fingerprint"]:
                    raise ValueError(
                        "This run_id already has a different plan; choose a new run_id"
                    )
            write_json(manifest_path, plan)
            State(directory, plan).save()
            with (directory / "assignments.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(plan["jobs"][0]))
                writer.writeheader()
                writer.writerows(plan["jobs"])
            estimate = None
            if args.tokens_per_second is not None:
                if args.tokens_per_second <= 0 or args.tokens_per_rollout <= 0:
                    raise ValueError("Throughput and token assumptions must be positive")
                gpu_hours = (
                    len(plan["jobs"]) * args.tokens_per_rollout / args.tokens_per_second / 3600
                )
                estimate = {
                    "gpu_hours": gpu_hours,
                    "ideal_wall_hours": gpu_hours / len(plan["replicas"]),
                    "assumption_tokens_per_rollout": args.tokens_per_rollout,
                    "assumption_tok_s_per_gpu": args.tokens_per_second,
                    "excludes": "startup, tools, queueing, retries, load imbalance",
                }
                for key in ("hourly_cost", "out_of_pocket_hourly_cost"):
                    costs = [r[key] for r in plan["replicas"]]
                    estimate[key.replace("hourly_cost", "estimated_cost")] = (
                        gpu_hours / len(costs) * sum(costs)
                        if all(c is not None for c in costs)
                        else None
                    )
            print(
                json.dumps(
                    {
                        "plan": str(manifest_path),
                        "rollouts": len(plan["jobs"]),
                        "preferred_assignments": dict(
                            Counter(j["preferred_replica"] for j in plan["jobs"])
                        ),
                        "max_live_rollouts": plan["scheduling"]["max_live_rollouts"],
                        "estimate": estimate,
                    },
                    indent=2,
                )
            )
            return 0
        plan = json.loads(manifest_path.read_text())
        if args.command in ("up", "run"):
            verify_sources(plan, root)
            if make_plan(config, root)["fingerprint"] != plan["fingerprint"]:
                raise ValueError("Config changed since planning; choose a new run_id")
        if args.command == "up":
            print(json.dumps(asyncio.run(up(plan, directory, root)), indent=2))
            print("Provisioning dispatched. Model readiness is checked when run starts.")
        elif args.command == "down":
            print(json.dumps(asyncio.run(down(directory, root, confirmed=args.yes)), indent=2))
        elif args.command == "export":
            print(export_results(plan, directory))
        elif args.command == "run":
            if args.resume and (directory / "drain.json").exists():
                (directory / "drain.json").unlink()
            if args.retry_failed:
                state = State(directory, plan)
                for record in state.data["jobs"].values():
                    if record["status"] == "failed":
                        record["status"] = "pending"
                state.save()
            counts = asyncio.run(execute(plan, directory, root, load_resources(directory)))
            print(export_results(plan, directory))
            print(json.dumps(counts))
            return 0 if counts.get("complete", 0) == len(plan["jobs"]) else 1
    return 0
