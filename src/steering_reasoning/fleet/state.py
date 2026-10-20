"""A single coordinator owns state; atomic JSON and an OS lock survive interrupted writes."""

import fcntl
import json
import os
import time
from collections import Counter
from contextlib import contextmanager


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


@contextmanager
def locked(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "coordinator.lock").open("a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                "This run already has an active coordinator/lifecycle command"
            ) from None
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


class State:
    def __init__(self, directory, plan):
        self.path = directory / "state.json"
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
            if self.data["fingerprint"] != plan["fingerprint"]:
                raise ValueError("Run state belongs to a different plan")
        else:
            self.data = {
                "fingerprint": plan["fingerprint"],
                "created_at": time.time(),
                "jobs": {j["id"]: {"status": "pending", "attempts": []} for j in plan["jobs"]},
            }
        self.plan = plan

    def save(self):
        self.data["updated_at"] = time.time()
        write_json(self.path, self.data)

    def counts(self):
        return dict(Counter(j["status"] for j in self.data["jobs"].values()))

    def claim(self, replica, capacity):
        pending = [
            j for j in self.plan["jobs"] if self.data["jobs"][j["id"]]["status"] == "pending"
        ]
        if not pending or capacity <= 0:
            return []
        # Prefer local balanced assignments, then steal only unstarted/retried work.
        first = min(pending, key=lambda j: j["preferred_replica"] != replica)
        group = tuple(first[k] for k in ("arm", "variant", "seed"))
        same = [j for j in pending if tuple(j[k] for k in ("arm", "variant", "seed")) == group]
        same.sort(key=lambda j: j["preferred_replica"] != replica)
        chosen = same[: min(capacity, self.plan["scheduling"]["batch_size"])]
        for job in chosen:
            record = self.data["jobs"][job["id"]]
            record["status"] = "running"
            record["attempts"].append({"replica": replica, "started_at": time.time()})
        self.save()
        return chosen

    def finish(self, job_id, result=None, error=None):
        record = self.data["jobs"][job_id]
        attempt = record["attempts"][-1]
        attempt.update(ended_at=time.time(), error=error, result=result)
        if result is not None:
            record.update(status="complete", result=result)
        else:
            exhausted = len(record["attempts"]) >= self.plan["scheduling"]["max_job_attempts"]
            record["status"] = "failed" if exhausted else "pending"
        self.save()
