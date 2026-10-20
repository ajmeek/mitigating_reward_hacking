"""Replica lifecycle. An `external` replica is any server you bring (its `upstream` URL); a `runpod`
replica is created, provisioned and deleted through tools/runpod/rp.

=====================================================================================
COMPUTE PROVIDER. This is the authors' RunPod tooling, kept as a worked example.
>>> ADD COMPUTE TOOLING FOR YOUR PROVIDER HERE. <<<  Any GPU box that can run
serving/bootstrap.sh and expose vLLM over HTTP works. Credentials come from the repo-root
.env (COMPUTE_API_KEY for the provider, HF_TOKEN for the model; see .env.example).

To add a provider: give it a name in plan.py's provider check, and branch on replica["provider"]
in up() (create + provision + resolve the endpoint into resources.json) and down() (delete).
"""

import asyncio
import json
import time

from .plan import endpoint
from .state import write_json


async def rp(root, *args):
    proc = await asyncio.create_subprocess_exec(
        str(root / "tools/runpod/rp"),
        *args,
        cwd=root,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate()
    if proc.returncode:
        raise RuntimeError(f"RunPod {args[0]} failed: {err.decode()[:500]}")
    return out.decode()


def load_resources(directory):
    path = directory / "resources.json"
    return json.loads(path.read_text()) if path.exists() else {}


async def up(plan, directory, root):
    """Create at most one pod per replica. Persist IDs immediately; retries never blindly create."""
    resources = load_resources(directory)
    path = directory / "resources.json"

    async def provision(replica):
        rid = replica["name"]
        if replica.get("upstream"):
            resources[rid] = {"upstream": endpoint(replica["upstream"]), "owned": False}
            write_json(path, resources)
            return
        record = resources.setdefault(rid, {})
        if record.get("deleted_at"):
            raise ValueError(f"{rid} was deleted; use a new run_id for replacement infrastructure")
        pod_name = "fleet-" + plan["fingerprint"][:12] + "-" + rid[:30]
        if not record.get("pod_id"):
            if replica.get("pod_id"):
                record.update(pod_id=replica["pod_id"], owned=False)
            else:
                # Recover an interrupted create response by deterministic, run-unique pod name.
                pods = json.loads(await rp(root, "raw", "pods"))
                if not isinstance(pods, list):
                    raise RuntimeError("Could not list RunPod pods; no create attempted")
                matches = [p for p in pods if p.get("name") == pod_name]
                if len(matches) > 1:
                    raise RuntimeError(f"Multiple pods named {pod_name}; resolve before retrying")
                if matches:
                    if not record.get("creating"):
                        raise RuntimeError(
                            f"Pod {pod_name} exists without this run's ownership record"
                        )
                    created = matches[0]
                else:
                    record.update(creating=True, name=pod_name, created_at=time.time(), owned=True)
                    write_json(path, resources)
                    created = json.loads(await rp(root, "create", pod_name, replica["gpu"], "1"))
                if not created.get("id") or created.get("error"):
                    raise RuntimeError(f"RunPod creation failed for {rid}: {created}")
                record.update(
                    pod_id=created["id"], owned=True, hourly_cost=created.get("costPerHr")
                )
            write_json(path, resources)
        if record.get("owned") and not record.get("provisioned_at"):
            log_path = directory / f"provision-{rid}.log"
            with log_path.open("a") as log:
                proc = await asyncio.create_subprocess_exec(
                    "bash",
                    str(root / "serving/provision_runpod.sh"),
                    record["pod_id"],
                    f"MODEL={plan['model']}",
                    "DTYPE=bfloat16",
                    "TP=1",
                    f"MAXSEQS={replica['max_sequences']}",
                    cwd=root,
                    stdout=log,
                    stderr=asyncio.subprocess.STDOUT,
                )
                if await proc.wait():
                    raise RuntimeError(f"Provisioning failed; pod retained. See {log_path}")
            record["provisioned_at"] = time.time()
            write_json(path, resources)
        info = json.loads(await rp(root, "info", record["pod_id"]))
        port = (info.get("portMappings") or {}).get("8000")
        if not info.get("publicIp") or not port:
            raise RuntimeError(f"{rid}: pod exists but endpoint is not available yet; rerun up")
        record.update(
            upstream=endpoint(f"http://{info['publicIp']}:{port}"),
            gpu=info.get("gpu"),
            hourly_cost=info.get("costPerHr"),
        )
        write_json(path, resources)

    outcomes = await asyncio.gather(
        *(provision(r) for r in plan["replicas"]), return_exceptions=True
    )
    errors = [str(x) for x in outcomes if isinstance(x, Exception)]
    if errors:
        raise RuntimeError("; ".join(errors) + ". Resource IDs are retained in resources.json")
    return resources


async def down(directory, root, *, confirmed=False):
    resources = load_resources(directory)
    targets = {
        k: v["pod_id"]
        for k, v in resources.items()
        if v.get("owned") and v.get("pod_id") and not v.get("deleted_at")
    }
    if not confirmed:
        return {"would_delete": targets, "instruction": "Pass --yes after collecting results"}
    for rid, pod_id in targets.items():
        output = await rp(root, "rm", pod_id)
        if not any(f"http {code}" in output for code in (200, 204)):
            raise RuntimeError(f"Deletion not confirmed for {pod_id}: {output}")
        resources[rid]["deleted_at"] = time.time()
        write_json(directory / "resources.json", resources)
    return {"deleted": targets, "external_endpoints": "left running"}
