"""One eval-box coordinator, bounded Inspect batches, and a pinned proxy per batch."""

import asyncio
import csv
import json
import os
import signal
import socket
import sys
import time
import uuid
from collections import Counter
from contextlib import asynccontextmanager

import aiohttp
from aiohttp import web

from .metrics import monitor
from .plan import cache_salt, endpoint, verify_sources
from .state import State, write_json


def inspect_results(log_dir):
    """Read fully scored, error-free samples; a successful process exit alone proves nothing."""
    from inspect_ai.log import read_eval_log

    results = {}
    for path in sorted(log_dir.glob("*.eval")):
        try:
            log = read_eval_log(str(path))
        except Exception:
            continue  # Partial/corrupt logs remain pending or failed, never complete.
        for sample in log.samples or []:
            if (
                sample.error is not None
                or not sample.scores
                or any(s.value is None or s.value == "" for s in sample.scores.values())
            ):
                continue
            results[str(sample.id)] = {
                "log": str(path),
                "scores": {k: s.value for k, s in sample.scores.items()},
                "out_tokens": sum(
                    u.output_tokens or 0 for u in (sample.model_usage or {}).values()
                ),
                "turns": sum(m.role == "assistant" for m in sample.messages),
                "at_limit": bool(sample.limit and sample.limit.type == "message"),
            }
    return results


def reconcile(state, directory):
    """Recover scored work before retrying; refuse to duplicate a possibly live orphan worker."""
    checked = {}
    for job in state.plan["jobs"]:
        record = state.data["jobs"][job["id"]]
        if record["status"] != "running":
            continue
        attempt = record["attempts"][-1]
        pid = attempt.get("pid")
        if pid:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError(
                    f"Worker PID {pid} may still be alive; stop its old service first"
                )
        log_dir = directory / attempt["log_dir"] if "log_dir" in attempt else None
        if log_dir and str(log_dir) not in checked:
            checked[str(log_dir)] = inspect_results(log_dir)
        result = checked.get(str(log_dir), {}).get(job["task"])
        state.finish(job["id"], result=result, error=None if result else "Coordinator interrupted")


def eval_command(plan, jobs, port, log_dir, root):
    job = jobs[0]
    work = plan["work"]
    return [
        sys.executable,
        str(root / "scripts/run_lcb_eval.py"),
        "--split",
        job["variant"],
        "--sample-ids",
        ",".join(j["task"] for j in jobs),
        "--model",
        "openai-api/local/" + plan["model"],
        "--base-url",
        f"http://127.0.0.1:{port}/v1",
        "--seed",
        str(job["seed"]),
        "--max-connections",
        str(len(jobs)),
        "--max-sandboxes",
        str(len(jobs)),
        "--message-limit",
        str(work["message_limit"]),
        "--max-attempts",
        str(work["max_attempts"]),
        "--temperature",
        str(work["temperature"]),
        "--top-p",
        str(work["top_p"]),
        "--log-dir",
        str(log_dir),
        "--log-buffer",
        "1",
        "--retry-on-error",
        "0",
        "--fail-on-error",
        "0",
    ]


@asynccontextmanager
async def pinned_proxy(plan, arm, root, upstream, dump_dir=None, ids_log=None):
    from steering_reasoning.proxy import make_app
    from steering_reasoning.steering import build_steer_payload

    payload = None
    if arm["mode"] != "none":
        payload = build_steer_payload(
            str(root / arm["artifact"]),
            mode=arm["mode"],
            threshold=arm["dose"] if arm["mode"] == "clamp" else None,
            scale=arm["dose"] if arm["mode"] == "add" else 1.0,
        )
    app = make_app(
        upstream,
        max_tokens=plan["work"]["max_tokens"],
        steer_payload=payload,
        cache_salt=cache_salt(plan, arm),
        dump_dir=str(dump_dir) if dump_dir else None,
        ids_log=str(ids_log) if ids_log else None,   # exact token ids per turn, for later replay
    )
    runner = web.AppRunner(app)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(128)
    port = sock.getsockname()[1]
    try:
        await runner.setup()
        await web.SockSite(runner, sock).start()
        yield port
    finally:
        await runner.cleanup()
        sock.close()


async def run_batch(plan, state, directory, root, replica, upstream, jobs):
    # Per-batch source check. On a shared eval box another session's file edits would disable
    # every replica mid-run (2026-09-14); FLEET_SKIP_SOURCE_CHECK=1 keeps only the start-of-run
    # check (cli). The proxies live in this process, so later edits cannot change the
    # intervention; eval-side modules are imported per batch, so a skipped check is a
    # provenance gap the run's rebaseline.json / git history has to cover.
    if os.environ.get("FLEET_SKIP_SOURCE_CHECK") != "1":
        verify_sources(plan, root)
    arm = next(a for a in plan["arms"] if a["name"] == jobs[0]["arm"])
    batch_id = uuid.uuid4().hex
    batch_dir = directory / "batches" / replica / batch_id
    batch_dir.mkdir(parents=True)
    async with pinned_proxy(plan, arm, root, upstream, batch_dir / "dumps", batch_dir / "ids.jsonl") as port:
        command = eval_command(plan, jobs, port, batch_dir, root)
        write_json(
            batch_dir / "batch.json",
            dict(
                replica=replica,
                upstream=upstream,
                jobs=jobs,
                arm=arm,
                cache_salt=cache_salt(plan, arm),
                command=command,
                git_commit=plan["git_commit"],
                plan_fingerprint=plan["fingerprint"],
            ),
        )
        for j in jobs:
            state.data["jobs"][j["id"]]["attempts"][-1]["log_dir"] = str(
                batch_dir.relative_to(directory)
            )
        state.save()
        proc = None
        try:
            with (batch_dir / "run.log").open("a") as log:
                proc = await asyncio.create_subprocess_exec(
                    *command,
                    cwd=root,
                    stdout=log,
                    stderr=asyncio.subprocess.STDOUT,
                    env={**os.environ, "LOCAL_API_KEY": "dummy", "PYTHONUNBUFFERED": "1"},
                    start_new_session=True,
                )
                for j in jobs:
                    state.data["jobs"][j["id"]]["attempts"][-1]["pid"] = proc.pid
                state.save()
                code = await proc.wait()
        finally:
            if proc and proc.returncode is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    await asyncio.wait_for(proc.wait(), 15)
                except TimeoutError:
                    os.killpg(proc.pid, signal.SIGKILL)
                    await proc.wait()
        results = await asyncio.to_thread(inspect_results, batch_dir)
        for j in jobs:
            result = results.get(j["task"])
            state.finish(
                j["id"],
                result=result,
                error=None if result else f"No complete error-free score (exit {code})",
            )
        if not results:
            raise RuntimeError(
                f"Batch produced no valid samples (exit {code}); inspect {batch_dir}"
            )


async def preflight(plan, resources):
    endpoints = {}
    observations = {}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as session:
        for r in plan["replicas"]:
            upstream = resources.get(r["name"], {}).get("upstream", r.get("upstream"))
            if not upstream:
                raise ValueError(f"No upstream for {r['name']}; run fleet up first")
            upstream = endpoint(upstream)
            if upstream in endpoints.values():
                raise ValueError(
                    "Two replicas point to the same endpoint; this would double its load"
                )
            async with session.get(upstream + "/v1/models") as response:
                response.raise_for_status()
                body = await response.json()
            if plan["model"] not in [m["id"] for m in body.get("data", [])]:
                raise ValueError(f"{r['name']} is not serving {plan['model']}")
            endpoints[r["name"]] = upstream
            observations[r["name"]] = {
                "upstream": upstream,
                "models": body,
                "checked_at": time.time(),
                "steering_probe": "not certified by fleet health check",
            }
    return endpoints, observations


async def execute(plan, directory, root, resources, *, batch_fn=run_batch, endpoints=None):
    verify_sources(plan, root)
    state = State(directory, plan)
    reconcile(state, directory)
    if endpoints is None:
        endpoints, observations = await preflight(plan, resources)
        write_json(directory / "health.json", observations)
    stop_file = directory / "drain.json"
    if stop_file.exists():
        raise RuntimeError("Run is drained. Use fleet run --resume to clear the drain request")
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    metrics_task = asyncio.create_task(
        monitor(
            endpoints, directory / "metrics.jsonl", stop, plan["scheduling"]["metrics_interval"]
        )
    )
    active = {}
    occupied = Counter()
    disabled = set()
    state.data["coordinator"] = {"pid": os.getpid(), "started_at": time.time()}
    state.save()
    try:
        while True:
            if stop_file.exists():
                stop.set()
            if not stop.is_set():
                # Fill one batch per replica per pass, so the global CPU cap doesn't favor #1.
                while True:
                    dispatched = False
                    for replica in sorted(
                        plan["replicas"], key=lambda r: occupied[r["name"]] / r["live_rollouts"]
                    ):
                        rid = replica["name"]
                        if rid in disabled:
                            continue
                        capacity = min(
                            replica["live_rollouts"] - occupied[rid],
                            plan["scheduling"]["max_live_rollouts"] - sum(occupied.values()),
                        )
                        jobs = state.claim(rid, capacity)
                        if not jobs:
                            continue
                        task = asyncio.create_task(
                            batch_fn(plan, state, directory, root, rid, endpoints[rid], jobs)
                        )
                        active[task] = (rid, jobs)
                        occupied[rid] += len(jobs)
                        dispatched = True
                    if not dispatched:
                        break
            if not active:
                break
            done, _ = await asyncio.wait(active, timeout=2, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                rid, jobs = active.pop(task)
                occupied[rid] -= len(jobs)
                try:
                    task.result()
                except Exception as e:
                    # Infrastructure errors disable this replica for this coordinator invocation.
                    disabled.add(rid)
                    for j in jobs:
                        if state.data["jobs"][j["id"]]["status"] == "running":
                            state.finish(j["id"], error=f"Replica failure: {e}")
                    print(f"disabled {rid}: {e}", flush=True)
            if done:
                print(
                    json.dumps(
                        {
                            "jobs": state.counts(),
                            "active_slots": dict(occupied),
                            "draining": stop.is_set(),
                        }
                    ),
                    flush=True,
                )
    finally:
        for task in active:
            task.cancel()
        await asyncio.gather(*active, return_exceptions=True)
        stop.set()
        await metrics_task
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.remove_signal_handler(sig)
            signal.signal(sig, old_handlers[sig])
        state.data["coordinator"]["ended_at"] = time.time()
        state.save()
    return state.counts()


def export_results(plan, directory):
    state = State(directory, plan)
    rows = []
    arms = {a["name"]: a for a in plan["arms"]}
    replicas = {r["name"]: r for r in plan["replicas"]}
    for job in plan["jobs"]:
        record = state.data["jobs"][job["id"]]
        if record["status"] != "complete":
            continue
        last = record["attempts"][-1]
        result = record["result"]
        arm = arms[job["arm"]]
        replica = replicas.get(last["replica"], {"provider": "retired", "gpu": "unknown"})   # a re-baselined plan may no longer list the replica that ran it
        rows.append(
            {
                **job,
                "replica": last["replica"],
                "provider": replica["provider"],
                "gpu_declared": replica.get("gpu", "unknown"),
                "mode": arm["mode"],
                "dose": arm["dose"],
                "artifact": arm["artifact"],
                "scores": json.dumps(result["scores"], sort_keys=True),
                "out_tokens": result["out_tokens"],
                "turns": result["turns"],
                "at_limit": result["at_limit"],
                "log": result["log"],
                "attempts": len(record["attempts"]),
                "plan_fingerprint": plan["fingerprint"],
            }
        )
    path = directory / "results.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["id", "arm", "scores"])
        writer.writeheader()
        writer.writerows(rows)
    return path
