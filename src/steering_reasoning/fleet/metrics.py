"""Read-only, interval-based vLLM metrics (never interpret missing counters as zero)."""

import asyncio
import json
import math
import re
import time

import aiohttp

METRICS = {
    "generation_tokens": "vllm:generation_tokens_total",
    "prompt_tokens": "vllm:prompt_tokens_total",
    "cache_hits": "vllm:prefix_cache_hits_total",
    "cache_queries": "vllm:prefix_cache_queries_total",
    "running": "vllm:num_requests_running",
    "waiting": "vllm:num_requests_waiting",
    "preemptions": "vllm:num_preemptions_total",
    "kv_usage": "vllm:kv_cache_usage_perc",
}


def parse_metrics(text):
    values = {}
    for line in text.splitlines():
        match = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{.*\})?\s+(\S+)", line)
        if match:
            try:
                value = float(match[2])
            except ValueError:
                continue
            if math.isfinite(value):
                values.setdefault(match[1], []).append(value)
    result = {
        k: (max(values[v]) if k == "kv_usage" else sum(values[v]))
        for k, v in METRICS.items()
        if v in values
    }
    if "kv_usage" not in result and "vllm:gpu_cache_usage_perc" in values:
        result["kv_usage"] = max(values["vllm:gpu_cache_usage_perc"])
    return result


def interval(before, after):
    elapsed = after["time"] - before["time"]
    result = {"seconds": elapsed}
    for key in ("generation_tokens", "prompt_tokens", "cache_hits", "cache_queries", "preemptions"):
        if key in before and key in after and after[key] >= before[key] and elapsed > 0:
            result[key] = after[key] - before[key]
    if "generation_tokens" in result:
        result["generation_tok_s"] = result["generation_tokens"] / elapsed
    if "cache_hits" in result and result.get("cache_queries", 0) > 0:
        result["cache_hit_fraction"] = result["cache_hits"] / result["cache_queries"]
    return result


async def snapshot(session, upstream):
    async with session.get(upstream + "/metrics", timeout=aiohttp.ClientTimeout(total=10)) as r:
        r.raise_for_status()
        return {"time": time.time(), **parse_metrics(await r.text())}


async def monitor(endpoints, path, stop, seconds):
    previous = {}
    async with aiohttp.ClientSession() as session:

        async def collect(replica, upstream):
            try:
                current = await snapshot(session, upstream)
                row = dict(replica=replica, **current)
                if replica in previous:
                    row["interval"] = interval(previous[replica], current)
                previous[replica] = current
                return row
            except Exception as e:
                previous.pop(replica, None)
                return dict(replica=replica, time=time.time(), error=str(e))

        while True:
            rows = await asyncio.gather(*(collect(k, v) for k, v in endpoints.items()))
            with path.open("a") as f:
                for row in rows:
                    f.write(json.dumps(row, allow_nan=False) + "\n")
            if stop.is_set():
                break
            try:
                await asyncio.wait_for(stop.wait(), timeout=seconds)
            except TimeoutError:
                pass
