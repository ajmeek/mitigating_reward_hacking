"""Synthetic concurrency sweep through the same steering proxy; outputs are not eval results."""

import argparse
import asyncio
import json
import time
from pathlib import Path

import aiohttp

from .metrics import interval, snapshot
from .plan import digest, endpoint, positive, verify_sources
from .providers import load_resources
from .runner import pinned_proxy
from .state import locked, write_json


async def measure(plan, root, upstream, arm, concurrency, requests_per_slot, tokens, messages):
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=7200)) as session:
        before = await snapshot(session, upstream)
        if before.get("running", 0) or before.get("waiting", 0):
            raise RuntimeError(
                "Benchmark requires a quiet server; model requests are already active"
            )
        # Separate namespace per setting; otherwise order-dependent warm caches bias the curve.
        plan = {**plan, "fingerprint": digest([plan["fingerprint"], "benchmark", time.time_ns()])}
        async with pinned_proxy(plan, arm, root, upstream) as port:
            sem = asyncio.Semaphore(concurrency)

            async def request():
                async with sem:
                    started = time.monotonic()
                    body = {
                        "model": plan["model"],
                        "messages": messages,
                        "temperature": 0,
                        "max_tokens": tokens,
                        "min_tokens": tokens,
                        "seed": 7,
                    }
                    async with session.post(
                        f"http://127.0.0.1:{port}/v1/chat/completions", json=body
                    ) as r:
                        r.raise_for_status()
                        response = await r.json()
                    if "usage" not in response or "completion_tokens" not in response["usage"]:
                        raise ValueError("Response has no completion token usage")
                    return {
                        "seconds": time.monotonic() - started,
                        "out_tokens": response["usage"]["completion_tokens"],
                    }

            started = time.monotonic()
            outcomes = await asyncio.gather(
                *(request() for _ in range(concurrency * requests_per_slot)), return_exceptions=True
            )
            duration = time.monotonic() - started
        after = await snapshot(session, upstream)
    successes = [r for r in outcomes if not isinstance(r, Exception)]
    return {
        "concurrency": concurrency,
        "requested_requests": len(outcomes),
        "completed_requests": len(successes),
        "seconds": duration,
        "generation_tok_s": sum(r["out_tokens"] for r in successes) / duration,
        "request_seconds_mean": sum(r["seconds"] for r in successes) / len(successes)
        if successes
        else None,
        "errors": [str(r) for r in outcomes if isinstance(r, Exception)],
        "server_interval": interval(before, after),
        "final_metrics": after,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("plan", type=Path, help="generated data/fleet/<run_id>/plan.json")
    p.add_argument("--replica", required=True)
    p.add_argument("--arm", default="cn8")
    p.add_argument("--concurrency", type=int, nargs="+", default=[8, 16, 32])
    p.add_argument("--requests-per-slot", type=int, default=2)
    p.add_argument("--tokens", type=int, default=512)
    p.add_argument(
        "--messages", type=Path, help="JSON chat messages, including long representative history"
    )
    args = p.parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    plan = json.loads(args.plan.read_text())
    verify_sources(plan, root)
    directory = args.plan.resolve().parent
    replica = next(r for r in plan["replicas"] if r["name"] == args.replica)
    arm = next(a for a in plan["arms"] if a["name"] == args.arm)
    upstream = endpoint(
        load_resources(directory).get(args.replica, {}).get("upstream", replica.get("upstream", ""))
    )
    for value in [*args.concurrency, args.requests_per_slot, args.tokens]:
        positive(value, "benchmark load")
    if args.tokens > plan["work"]["max_tokens"]:
        raise ValueError("Benchmark token count exceeds proxy cap")
    messages = (
        json.loads(args.messages.read_text())
        if args.messages
        else [
            {
                "role": "user",
                "content": "Explain step by step how to debug contradictory unit tests.",
            }
        ]
    )
    out = directory / f"benchmark-{args.replica}-{time.time_ns()}.json"

    async def sweep():
        report = {
            "kind": "synthetic throughput, not rollout cost",
            "replica": args.replica,
            "arm": args.arm,
            "plan_fingerprint": plan["fingerprint"],
            "messages_hash": digest(messages),
            "tokens_per_request": args.tokens,
            "results": [],
        }
        for concurrency in args.concurrency:
            row = await measure(
                plan,
                root,
                upstream,
                arm,
                concurrency,
                args.requests_per_slot,
                args.tokens,
                messages,
            )
            report["results"].append(row)
            write_json(out, report)
            print(json.dumps(row), flush=True)
            if row["errors"]:
                raise RuntimeError(f"Benchmark errors; stopping higher loads. Results: {out}")
        print(out)

    with locked(directory):
        asyncio.run(sweep())
