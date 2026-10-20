"""Fleet correctness without GPUs: assignment, ownership, restart recovery, and real HTTP routing."""

import asyncio
import json
import os
from collections import Counter, defaultdict

import pytest

from steering_reasoning.fleet import providers, runner
from steering_reasoning.fleet.metrics import interval, parse_metrics
from steering_reasoning.fleet.plan import cache_salt, make_plan, verify_sources
from steering_reasoning.fleet.state import State, locked, write_json


@pytest.fixture
def fixture_plan(tmp_path):
    experiment = tmp_path / "experiments/demo"
    experiment.mkdir(parents=True)
    data = tmp_path / "src/steering_reasoning/data"
    data.mkdir(parents=True)
    tasks = [f"lcbhard_{i}" for i in range(12)]
    (data / "splits.json").write_text(
        json.dumps(
            {"impossible": {"test": tasks, "train": ["lcbhard_99"]}, "original": {"test": tasks}}
        )
    )
    (experiment / "arms.tsv").write_text("base none - 8110\ncontrol none - 8111\n")
    cfg = experiment / "fleet.toml"
    cfg.write_text("""run_id = "test-run"
model = "test-model"
[work]
seeds = [42, 43]
[scheduling]
batch_size = 2
max_live_rollouts = 3
max_job_attempts = 2
metrics_interval = 1
[[replicas]]
name = "r0"
upstream = "http://127.0.0.1:9000"
live_rollouts = 2
[[replicas]]
name = "r1"
upstream = "http://127.0.0.1:9001"
live_rollouts = 2
""")
    return tmp_path, cfg, make_plan(cfg, tmp_path)


def test_balanced_complete_deterministic_plan(fixture_plan):
    root, cfg, plan = fixture_plan
    assert make_plan(cfg, root) == plan
    assert len(plan["jobs"]) == 2 * 2 * 2 * 12
    keys = {(j["arm"], j["variant"], j["seed"], j["task"]) for j in plan["jobs"]}
    assert len(keys) == len(plan["jobs"])
    groups = defaultdict(Counter)
    for job in plan["jobs"]:
        groups[job["arm"], job["variant"], job["seed"]][job["preferred_replica"]] += 1
    assert all(c == {"r0": 6, "r1": 6} for c in groups.values())


@pytest.mark.parametrize(
    "edit",
    [
        lambda s: s.replace("seeds = [42, 43]", "seeds = [42, 42]"),
        lambda s: s.replace("seeds = [42, 43]", 'tasks = ["lcbhard_99"]'),
        lambda s: s.replace("batch_size = 2", "batch_size = 0"),
        lambda s: s.replace('name = "r1"', 'name = "r0"'),
        lambda s: s.replace("http://127.0.0.1:9000", "http://127.0.0.1:9000/v1"),
        lambda s: s.replace('run_id = "test-run"', 'run_id = "../escape"'),
    ],
)
def test_reject_invalid_plan(fixture_plan, edit):
    root, cfg, _ = fixture_plan
    cfg.write_text(edit(cfg.read_text()))
    with pytest.raises(ValueError):
        make_plan(cfg, root)


def test_hashes_and_salts_follow_artifact_bytes(fixture_plan):
    import numpy as np
    from safetensors.numpy import save_file

    root, cfg, _ = fixture_plan
    artifact = cfg.parent / "v.safetensors"
    save_file({"vector": np.array([[1, 0, 0]], dtype=np.float32)}, artifact)
    artifact.with_suffix(".json").write_text('{"layer_indices": [35]}')
    table = cfg.parent / "arms.tsv"
    table.write_text("base none - 8110\nother add -4 8111 experiments/demo/v.safetensors\n")
    plan = make_plan(cfg, root)
    assert cache_salt(plan, plan["arms"][0]) != cache_salt(plan, plan["arms"][1])
    verify_sources(plan, root)
    save_file({"vector": np.array([[0, 1, 0]], dtype=np.float32)}, artifact)
    with pytest.raises(ValueError, match="Files changed"):
        verify_sources(plan, root)
    changed = make_plan(cfg, root)
    assert cache_salt(plan, plan["arms"][1]) != cache_salt(changed, changed["arms"][1])


def test_claims_steal_without_duplicates_and_resume_completed(fixture_plan):
    root, _, plan = fixture_plan
    state = State(root / "run", plan)
    claimed = []
    while jobs := state.claim("r0", 2):
        claimed.extend(j["id"] for j in jobs)
        for job in jobs:
            state.finish(job["id"], result={"scores": {"score": "I"}})
    assert len(claimed) == len(set(claimed)) == len(plan["jobs"])
    assert State(root / "run", plan).claim("r1", 2) == []
    assert state.counts() == {"complete": len(plan["jobs"])}


def test_retry_exhaustion_and_recovery_of_scored_interrupted_batch(fixture_plan, monkeypatch):
    root, _, plan = fixture_plan
    directory = root / "run"
    state = State(directory, plan)
    jobs = state.claim("r0", 2)
    first, second = jobs
    for j in jobs:
        state.data["jobs"][j["id"]]["attempts"][-1]["log_dir"] = "batch"
    monkeypatch.setattr(
        runner, "inspect_results", lambda p: {first["task"]: {"scores": {"s": "C"}}}
    )
    runner.reconcile(state, directory)
    assert state.data["jobs"][first["id"]]["status"] == "complete"
    assert state.data["jobs"][second["id"]]["status"] == "pending"
    state.data["jobs"][second["id"]]["attempts"].append({"replica": "r1"})
    state.finish(second["id"], error="second failure")
    assert state.data["jobs"][second["id"]]["status"] == "failed"


def test_refuses_live_orphan_and_second_coordinator(fixture_plan):
    root, _, plan = fixture_plan
    directory = root / "run"
    state = State(directory, plan)
    job = state.claim("r0", 1)[0]
    state.data["jobs"][job["id"]]["attempts"][-1]["pid"] = os.getpid()
    with pytest.raises(RuntimeError, match="still be alive"):
        runner.reconcile(state, directory)
    with locked(directory), pytest.raises(RuntimeError, match="active coordinator"):
        with locked(directory):
            pass


def test_metric_labels_missing_counters_and_restarts():
    parsed = parse_metrics("""# HELP ignored
vllm:generation_tokens_total{engine="0"} 100
vllm:generation_tokens_total{engine="1"} 200
vllm:num_requests_running{engine="0"} 3
vllm:kv_cache_usage_perc{engine="0"} 0.6
vllm:kv_cache_usage_perc{engine="1"} 0.8
""")
    assert parsed == {"generation_tokens": 300, "running": 3, "kv_usage": 0.8}
    assert "cache_hit_fraction" not in interval({"time": 0}, {"time": 1})
    assert "generation_tok_s" not in interval(
        {"time": 0, "generation_tokens": 100}, {"time": 5, "generation_tokens": 20}
    )
    assert (
        interval({"time": 0, "generation_tokens": 0}, {"time": 10, "generation_tokens": 100})[
            "generation_tok_s"
        ]
        == 10
    )


def test_scheduler_capacity_sticky_batches_and_drain(fixture_plan, monkeypatch):
    root, _, plan = fixture_plan
    directory = root / "run"
    directory.mkdir()
    current = Counter()
    maximum = 0
    seen = set()

    async def fake_monitor(*args):
        await args[2].wait()

    async def fake_batch(plan, state, directory, root, rid, endpoint, jobs):
        nonlocal maximum
        current[rid] += len(jobs)
        maximum = max(maximum, sum(current.values()))
        assert current[rid] <= 2
        assert sum(current.values()) <= 3
        for job in jobs:
            assert job["id"] not in seen
            seen.add(job["id"])
        await asyncio.sleep(0.002 if rid == "r0" else 0.006)
        for job in jobs:
            state.finish(job["id"], result={"scores": {"s": "I"}})
        current[rid] -= len(jobs)

    monkeypatch.setattr(runner, "monitor", fake_monitor)
    counts = asyncio.run(
        runner.execute(
            plan, directory, root, {}, batch_fn=fake_batch, endpoints={"r0": "a", "r1": "b"}
        )
    )
    assert counts == {"complete": len(plan["jobs"])}
    assert maximum == 3
    assert len(seen) == len(plan["jobs"])


def test_drain_preserves_pending_and_running_jobs(fixture_plan, monkeypatch):
    root, _, plan = fixture_plan
    directory = root / "run"
    directory.mkdir()

    async def fake_monitor(*args):
        await args[2].wait()

    async def batch(plan, state, directory, root, rid, endpoint, jobs):
        write_json(directory / "drain.json", {"drain": True})
        for job in jobs:
            state.finish(job["id"], result={"scores": {"s": "I"}})

    monkeypatch.setattr(runner, "monitor", fake_monitor)
    counts = asyncio.run(
        runner.execute(plan, directory, root, {}, batch_fn=batch, endpoints={"r0": "a", "r1": "b"})
    )
    assert counts == {"complete": 3, "pending": len(plan["jobs"]) - 3}


def test_down_only_owned_pods_requires_confirmation(tmp_path, monkeypatch):
    write_json(
        tmp_path / "resources.json",
        {
            "new": {"pod_id": "owned-1", "owned": True},
            "aws": {"upstream": "http://aws", "owned": False},
            "borrowed": {"pod_id": "existing-1", "owned": False},
        },
    )
    calls = []

    async def fake_rp(root, *args):
        calls.append(args)
        return "deleted (http 204)"

    monkeypatch.setattr(providers, "rp", fake_rp)
    assert asyncio.run(providers.down(tmp_path, tmp_path))["would_delete"] == {"new": "owned-1"}
    assert not calls
    asyncio.run(providers.down(tmp_path, tmp_path, confirmed=True))
    assert calls == [("rm", "owned-1")]
    asyncio.run(providers.down(tmp_path, tmp_path, confirmed=True))
    assert len(calls) == 1


def test_up_attaches_existing_pod_without_reprovisioning(fixture_plan, monkeypatch):
    root, _, plan = fixture_plan
    plan["replicas"] = [{"name": "attached", "provider": "runpod", "pod_id": "borrowed"}]
    directory = root / "run"
    directory.mkdir()
    calls = []

    async def fake_rp(root, *args):
        calls.append(args)
        assert args == ("info", "borrowed")
        return json.dumps({"publicIp": "1.2.3.4", "portMappings": {"8000": 1234}})

    monkeypatch.setattr(providers, "rp", fake_rp)
    for _ in range(2):
        resources = asyncio.run(providers.up(plan, directory, root))
        assert resources["attached"]["upstream"] == "http://1.2.3.4:1234"
        assert resources["attached"]["owned"] is False
    assert len(calls) == 2


def test_up_recovers_created_pod_id_without_duplicate_create(fixture_plan, monkeypatch):
    root, _, plan = fixture_plan
    directory = root / "run"
    directory.mkdir()
    plan["replicas"] = [
        {"name": "r0", "provider": "runpod", "gpu": "NVIDIA H200", "max_sequences": 32}
    ]
    write_json(
        directory / "resources.json",
        {"r0": {"pod_id": "already-created", "owned": True, "provisioned_at": 1}},
    )

    async def fake_rp(root, *args):
        assert args == ("info", "already-created")
        return json.dumps({"publicIp": "1.2.3.4", "portMappings": {"8000": 1234}})

    monkeypatch.setattr(providers, "rp", fake_rp)
    resources = asyncio.run(providers.up(plan, directory, root))
    assert resources["r0"]["pod_id"] == "already-created"


def test_error_samples_are_not_complete_even_when_scored(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import inspect_ai.log

    (tmp_path / "sample.eval").touch()
    bad = SimpleNamespace(id="bad", error="error", scores={"s": SimpleNamespace(value="C")})
    missing = SimpleNamespace(id="missing", error=None, scores={"s": SimpleNamespace(value=None)})
    good = SimpleNamespace(
        id="good",
        error=None,
        scores={"s": SimpleNamespace(value=0)},
        model_usage={},
        messages=[],
        limit=None,
    )
    monkeypatch.setattr(
        inspect_ai.log, "read_eval_log", lambda p: SimpleNamespace(samples=[bad, missing, good])
    )
    assert set(runner.inspect_results(tmp_path)) == {"good"}


def test_real_proxy_and_subprocess_stay_on_assigned_gpu(fixture_plan, monkeypatch):
    """Two fake GPU HTTP servers + real fleet proxies + real subprocesses, no Inspect/GPU cost."""
    from aiohttp import web

    root, _, plan = fixture_plan
    plan["jobs"] = plan["jobs"][:8]
    scripts = root / "scripts"
    scripts.mkdir()
    (scripts / "run_lcb_eval.py").write_text("""import argparse, json, urllib.request
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("--base-url"); p.add_argument("--sample-ids"); p.add_argument("--log-dir")
a, _ = p.parse_known_args()
results = {}
for task in a.sample_ids.split(","):
    messages = [{"role": "user", "content": task}]
    for turn in range(2):
        data = {"model": "test-model", "messages": messages}
        req = urllib.request.Request(a.base_url + "/chat/completions", data=json.dumps(data).encode(),
                                     headers={"Content-Type": "application/json"})
        response = json.load(urllib.request.urlopen(req))
        messages += [response["choices"][0]["message"], {"role": "user", "content": "continue"}]
    results[task] = {"scores": {"s": "I"}, "out_tokens": 2, "turns": 2, "at_limit": False,
                     "log": str(Path(a.log_dir) / "fake.eval")}
Path(a.log_dir, "results.json").write_text(json.dumps(results))
""")
    monkeypatch.setattr(
        runner, "inspect_results", lambda p: json.loads((p / "results.json").read_text())
    )
    directory = root / "run"
    directory.mkdir()
    calls = defaultdict(list)

    async def exercise():
        servers = []
        resources = {}
        try:
            for replica in plan["replicas"]:
                rid = replica["name"]

                async def chat(request, rid=rid):
                    data = await request.json()
                    calls[rid].append(data)
                    return web.json_response(
                        {
                            "choices": [
                                {
                                    "index": 0,
                                    "finish_reason": "stop",
                                    "message": {"role": "assistant", "content": "ok"},
                                }
                            ],
                            "usage": {"completion_tokens": 1, "prompt_tokens": 1},
                        }
                    )

                async def models(request):
                    return web.json_response({"data": [{"id": "test-model"}]})

                async def metrics(request):
                    return web.Response(text="vllm:num_requests_running 0\n")

                app = web.Application()
                app.router.add_post("/v1/chat/completions", chat)
                app.router.add_get("/v1/models", models)
                app.router.add_get("/metrics", metrics)
                server = web.AppRunner(app)
                await server.setup()
                site = web.TCPSite(server, "127.0.0.1", 0)
                await site.start()
                port = site._server.sockets[0].getsockname()[1]
                resources[rid] = {"upstream": f"http://127.0.0.1:{port}"}
                servers.append(server)
            result = await runner.execute(plan, directory, root, resources)
            assert result == {"complete": 8}
        finally:
            for server in servers:
                await server.cleanup()

    asyncio.run(exercise())
    state = State(directory, plan)
    assert all(calls.values()) and len(calls) == 2
    for job in plan["jobs"]:
        rid = state.data["jobs"][job["id"]]["attempts"][-1]["replica"]
        arm = next(a for a in plan["arms"] if a["name"] == job["arm"])
        salt = cache_salt(plan, arm)
        matching = [
            c
            for c in calls[rid]
            if c["messages"][0]["content"] == job["task"] and c["cache_salt"] == salt
        ]
        assert len(matching) >= 2  # both turns retain the same replica and salt
    assert runner.export_results(plan, directory).exists()


def test_fleet_proxy_refuses_silent_prefill_fallback():
    import aiohttp
    from aiohttp import web
    from steering_reasoning.proxy import make_app

    async def check():
        async def tokenize(request):
            return web.Response(status=500, text="broken tokenizer")

        app = web.Application()
        app.router.add_post("/tokenize", tokenize)
        server = web.AppRunner(app)
        proxy = None
        await server.setup()
        site = web.TCPSite(server, "127.0.0.1", 0)
        await site.start()
        try:
            port = site._server.sockets[0].getsockname()[1]
            proxy = web.AppRunner(
                make_app(f"http://127.0.0.1:{port}", steer_payload="[]")
            )
            await proxy.setup()
            proxy_site = web.TCPSite(proxy, "127.0.0.1", 0)
            await proxy_site.start()
            proxy_port = proxy_site._server.sockets[0].getsockname()[1]
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
                    json={"model": "test", "messages": []},
                ) as response:
                    assert response.status == 502
                    assert "steer_positions failed" in await response.text()
        finally:
            if proxy:
                await proxy.cleanup()
            await server.cleanup()

    asyncio.run(check())


def test_benchmark_measures_real_proxy_concurrency(fixture_plan):
    from aiohttp import web
    from steering_reasoning.fleet.benchmark import measure

    root, _, plan = fixture_plan
    active = maximum = generated = 0

    async def exercise():
        async def metrics(request):
            return web.Response(
                text=f"vllm:num_requests_running {active}\n"
                f"vllm:num_requests_waiting 0\n"
                f"vllm:generation_tokens_total {generated}\n"
            )

        async def chat(request):
            nonlocal active, maximum, generated
            data = await request.json()
            active += 1
            maximum = max(maximum, active)
            await asyncio.sleep(0.02)
            active -= 1
            generated += data["max_tokens"]
            return web.json_response(
                {
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "length",
                            "message": {"role": "assistant", "content": "ok"},
                        }
                    ],
                    "usage": {"completion_tokens": data["max_tokens"], "prompt_tokens": 4},
                }
            )

        app = web.Application()
        app.router.add_get("/metrics", metrics)
        app.router.add_post("/v1/chat/completions", chat)
        server = web.AppRunner(app)
        await server.setup()
        site = web.TCPSite(server, "127.0.0.1", 0)
        await site.start()
        try:
            port = site._server.sockets[0].getsockname()[1]
            row = await measure(
                plan,
                root,
                f"http://127.0.0.1:{port}",
                plan["arms"][0],
                concurrency=3,
                requests_per_slot=2,
                tokens=16,
                messages=[{"role": "user", "content": "test"}],
            )
            assert row["errors"] == []
            assert row["completed_requests"] == 6
            assert row["server_interval"]["generation_tokens"] == 96
            assert row["generation_tok_s"] > 0
        finally:
            await server.cleanup()

    asyncio.run(exercise())
    assert maximum == 3


def test_arms_table_suffix_column_parses_and_fleet_refuses_prompt_arms(tmp_path):
    """The optional 6th column (system-message suffix, a prompt arm)
    parses everywhere the table is read, but make_plan refuses such arms: the fleet runner
    never passes the suffix, so planning one would silently run it without its prompt."""
    from steering_reasoning.fleet.plan import read_arms

    table = tmp_path / "arms.tsv"
    table.write_text(
        "# name mode dose port artifact suffix\n"
        "base none - 8130 - -\n"
        "spec none - 8131 - spec/prompt.txt   # trailing comment\n"
        "old  none - 8132\n"
    )
    rows = {r["name"]: r for r in read_arms(tmp_path, table)}
    assert rows["base"]["system_suffix"] is None
    assert rows["old"]["system_suffix"] is None
    assert rows["spec"]["system_suffix"] == "spec/prompt.txt"
    assert rows["spec"]["mode"] == "none" and rows["spec"]["artifact"] is None
    bad = tmp_path / "bad.tsv"
    bad.write_text("x none - 8133 - spec/prompt.txt extra\n")
    with pytest.raises(ValueError, match="name mode dose port"):
        read_arms(tmp_path, bad)

    experiment = tmp_path / "experiments" / "exp"
    experiment.mkdir(parents=True)
    data = tmp_path / "src/steering_reasoning/data"
    data.mkdir(parents=True)
    data.joinpath("splits.json").write_text(json.dumps(
        {"impossible": {"test": ["lcbhard_1"], "train": []}, "original": {"test": ["lcbhard_1"]}}
    ))
    (experiment / "arms.tsv").write_text(table.read_text())
    cfg = experiment / "fleet.toml"
    cfg.write_text('run_id = "t"\nmodel = "m"\n[[replicas]]\nname = "r0"\nupstream = "http://127.0.0.1:9000"\nlive_rollouts = 2\n')
    with pytest.raises(ValueError, match="system suffix"):
        make_plan(cfg, tmp_path)
