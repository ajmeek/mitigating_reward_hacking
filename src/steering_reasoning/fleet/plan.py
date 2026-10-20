"""Offline, deterministic plans. Paths in manifests are relative to the repository."""

import hashlib
import itertools
import json
import math
import os
import re
import subprocess
import tomllib
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

CANONICAL = "src/steering_reasoning/canonical/u_L35_steering.safetensors"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", value):
        raise ValueError(f"Invalid identifier: {value!r}")
    return value


def positive(value, field):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def endpoint(value):
    parsed = urlsplit(value)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in ("", "/")
    ):
        raise ValueError("upstream must be one HTTP(S) server origin, without /v1 or credentials")
    return value.rstrip("/")


def relative(root, path):
    return str(path.resolve().relative_to(root.resolve()))


def source_hashes(root):
    paths = {root / "scripts/run_lcb_eval.py", root / "pyproject.toml", root / "uv.lock"}
    for folder in ("src/steering_reasoning", "serving", "external/impossiblebench/src"):
        for directory, dirs, files in os.walk(root / folder):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
            paths.update(
                Path(directory) / f
                for f in files
                if Path(f).suffix in (".py", ".sh", ".toml", ".lock", ".yaml") and not f.startswith(".")   # skips macOS ._ sidecars
            )
    return {relative(root, p): file_hash(p) for p in sorted(paths) if p.exists()}


def read_arms(root, table):
    arms = []
    for line in table.read_text().splitlines():
        fields = line.split("#", 1)[0].split()
        if not fields:
            continue
        if len(fields) not in (4, 5, 6):
            raise ValueError(f"Expected name mode dose port [artifact] [suffix]: {line}")
        arm, mode, dose, port, *artifact = fields[:5]
        # optional 6th column: a system-message suffix file (a prompt arm; scripts/run_arms.sh
        # passes it as SUFFIX_FILE). Recorded here so the table parses everywhere; the fleet
        # runner itself does not pass it (make_plan refuses such arms).
        system_suffix = fields[5] if len(fields) == 6 and fields[5] != "-" else None
        name(arm)
        if mode not in ("none", "clamp", "add", "ablate"):
            raise ValueError(f"Unknown mode {mode}")
        # Legacy ports are validated but not used: fleet proxies bind ephemeral loopback ports.
        if not port.isdigit() or not 1 <= int(port) <= 65535:
            raise ValueError(f"Invalid proxy port: {port}")
        dose = None if dose == "-" else float(dose)
        if dose is not None and not math.isfinite(dose):
            raise ValueError("Dose must be finite")
        if (mode in ("clamp", "add")) != (dose is not None):
            raise ValueError(f"{mode}: dose required for clamp/add, '-' for none/ablate")
        spec = dict(name=arm, mode=mode, dose=dose, artifact=None, artifact_hashes={}, system_suffix=system_suffix)
        if mode != "none":
            art = artifact[0] if artifact and artifact[0] != "-" else CANONICAL
            path = root / art
            if path.suffix != ".safetensors":
                raise ValueError("Artifact must be .safetensors with a .json sidecar")
            spec["artifact"] = relative(root, path)
            spec["artifact_hashes"] = {
                relative(root, p): file_hash(p) for p in (path, path.with_suffix(".json"))
            }
            from safetensors.numpy import load_file
            import numpy as np

            vector = load_file(str(path))["vector"]
            layers = json.loads(path.with_suffix(".json").read_text())["layer_indices"]
            if (
                vector.ndim != 2
                or not layers
                or vector.shape[0] != len(layers)
                or any(type(i) is not int or i < 0 for i in layers)
                or len(set(layers)) != len(layers)
                or not np.isfinite(vector).all()
                or np.any(np.linalg.norm(vector.astype(float), axis=1) == 0)
            ):
                raise ValueError(f"Invalid vector/layer metadata: {art}")
        arms.append(spec)
    if not arms or len({a["name"] for a in arms}) != len(arms):
        raise ValueError("Arms must be nonempty and have unique names")
    return arms


def make_plan(config_path, root):
    cfg = tomllib.loads(config_path.read_text())
    unknown = set(cfg) - {"run_id", "arms", "model", "work", "replicas", "scheduling"}
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    run_id = name(cfg["run_id"])
    experiment = relative(root, config_path.parent)
    arms = read_arms(root, config_path.parent / cfg.get("arms", "arms.tsv"))
    prompt_arms = [a["name"] for a in arms if a.get("system_suffix")]
    if prompt_arms:
        raise ValueError(f"arms with a system suffix are not supported by the fleet runner "
                         f"(use scripts/run_arms.sh): {prompt_arms}")
    model = cfg.get("model", "Qwen/Qwen3.5-27B")
    if not isinstance(model, str) or not re.fullmatch(r"[A-Za-z0-9_./-]+", model):
        raise ValueError("model must be a model identifier, without shell syntax or whitespace")
    work = {
        "variants": ["conflicting", "original"],
        "seeds": [42],
        "message_limit": 120,
        "max_tokens": 16384,
        "max_attempts": 10,
        "temperature": 0.7,
        "top_p": 0.95,
        **cfg.get("work", {}),
    }
    if set(work) - {
        "variants",
        "seeds",
        "message_limit",
        "max_tokens",
        "max_attempts",
        "temperature",
        "top_p",
        "tasks",
    }:
        raise ValueError("Unknown work configuration key")
    for key in ("message_limit", "max_tokens", "max_attempts"):
        positive(work[key], key)
    if (
        not work["seeds"]
        or any(type(s) is not int or s < 0 for s in work["seeds"])
        or len(set(work["seeds"])) != len(work["seeds"])
    ):
        raise ValueError("seeds must be unique nonnegative integers")
    if (
        not work["variants"]
        or len(set(work["variants"])) != len(work["variants"])
        or set(work["variants"]) - {"conflicting", "original"}
    ):
        raise ValueError("variants must be a unique subset of conflicting/original")
    if (
        not math.isfinite(work["temperature"])
        or work["temperature"] < 0
        or not 0 < work["top_p"] <= 1
    ):
        raise ValueError("Invalid temperature or top_p")
    split_path = root / "src/steering_reasoning/data/splits.json"
    splits = json.loads(split_path.read_text())
    tasks = {}
    for variant in work["variants"]:
        allowed = splits["impossible" if variant == "conflicting" else "original"]["test"]
        selected = work.get("tasks", allowed)
        if not selected or len(set(selected)) != len(selected) or set(selected) - set(allowed):
            raise ValueError(f"tasks must be unique {variant} TEST task IDs")
        tasks[variant] = sorted(selected)
    scheduling = {
        "batch_size": 4,
        "max_live_rollouts": 32,
        "max_job_attempts": 2,
        "metrics_interval": 30,
        **cfg.get("scheduling", {}),
    }
    if set(scheduling) - {
        "batch_size",
        "max_live_rollouts",
        "max_job_attempts",
        "metrics_interval",
    }:
        raise ValueError("Unknown scheduling configuration key")
    for key, value in scheduling.items():
        positive(value, key)
    replicas = []
    for raw in cfg["replicas"]:
        if set(raw) - {
            "name",
            "provider",
            "upstream",
            "gpu",
            "live_rollouts",
            "weight",
            "hourly_cost",
            "out_of_pocket_hourly_cost",
            "max_sequences",
            "pod_id",
        }:
            raise ValueError("Unknown replica configuration key")
        r = {
            "provider": "external",
            "live_rollouts": 16,
            "weight": 1.0,
            "hourly_cost": None,
            "out_of_pocket_hourly_cost": None,
            "max_sequences": 32,
            **raw,
        }
        name(r["name"])
        if r.get("pod_id"):
            name(r["pod_id"])
        if r["provider"] not in ("external", "runpod"):
            raise ValueError("provider must be external (any server you bring, given as upstream) or runpod "
                             "(the authors' reference provider; add tooling for yours in fleet/providers.py)")
        positive(r["live_rollouts"], "live_rollouts")
        positive(r["max_sequences"], "max_sequences")
        if not math.isfinite(r["weight"]) or r["weight"] <= 0:
            raise ValueError("Replica weight must be finite and positive")
        for key in ("hourly_cost", "out_of_pocket_hourly_cost"):
            if r[key] is not None and (not math.isfinite(r[key]) or r[key] < 0):
                raise ValueError(f"{key} must be finite and nonnegative")
        if r.get("upstream"):
            r["upstream"] = endpoint(r["upstream"])
        elif r["provider"] == "external":
            raise ValueError("External replicas need an upstream")
        elif not r.get("gpu") and not r.get("pod_id"):
            raise ValueError("RunPod replicas need gpu or upstream")
        replicas.append(r)
    if not replicas or len({r["name"] for r in replicas}) != len(replicas):
        raise ValueError("Replicas must be nonempty and uniquely named")
    jobs, total, groups = [], Counter(), Counter()
    # Cycle arms inside task/seed so partial runs don't exhaust one dose before others start.
    for seed, task, variant, arm in itertools.product(
        work["seeds"], sorted(set().union(*map(set, tasks.values()))), work["variants"], arms
    ):
        if task not in tasks[variant]:
            continue
        group = (arm["name"], variant, seed)
        replica = min(
            replicas,
            key=lambda r: (
                groups[group, r["name"]] / r["weight"],
                total[r["name"]] / r["weight"],
                r["name"],
            ),
        )
        key = dict(arm=arm["name"], variant=variant, seed=seed, task=task)
        jobs.append(dict(id=digest(key)[:20], **key, preferred_replica=replica["name"]))
        groups[group, replica["name"]] += 1
        total[replica["name"]] += 1
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True)
    plan = dict(
        version=1,
        run_id=run_id,
        experiment=experiment,
        model=model,
        work=work,
        scheduling=scheduling,
        arms=arms,
        replicas=replicas,
        jobs=jobs,
        git_commit=git.stdout.strip(),
        source_hashes=source_hashes(root),
        split_hash=file_hash(split_path),
    )
    # Provisioned eval bundles have no .git directory. Content hashes remain authoritative there.
    plan["fingerprint"] = digest({k: v for k, v in plan.items() if k != "git_commit"})
    return plan


def verify_sources(plan, root):
    expected = {
        **plan["source_hashes"],
        "src/steering_reasoning/data/splits.json": plan["split_hash"],
    }
    for arm in plan["arms"]:
        expected.update(arm["artifact_hashes"])
    changed = [
        p for p, h in expected.items() if not (root / p).exists() or file_hash(root / p) != h
    ]
    if changed:
        raise ValueError(f"Files changed since planning; use a new run_id: {changed}")


def cache_salt(plan, arm):
    # Isolate runs and configurations, including baseline. No dependence on proxy port or GPU.
    return "fleet-" + digest({"run": plan["fingerprint"], "arm": arm})
