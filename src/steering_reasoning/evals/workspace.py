"""Capture the sandbox workspace at scoring time.

Inspect keeps the conversation of every finished sample in the .eval, but the files the agent
left in its sandbox die with the container. Recording them at final scoring makes a rollout
that stopped at the message limit resumable: the resumed sample gets the stored messages as input and
these files written back before its first turn.

`lcb_scorer` wraps upstream's swe_style_humaneval_scorer: it captures first (upstream reverts an
edited test.py before running it, which must not leak into the record), then delegates, and
returns upstream's score with `metadata["workspace"]` added. basic_agent scores on every submit
through the same scorer, so intermediate captures happen too; only the final one is stored in
the sample's score.
"""

from __future__ import annotations

import json

from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState
from inspect_ai.util import sandbox

from .impossiblebench_compat import livecodebench_tasks

# The sandbox (aisiuk/inspect-tool-support, compose.yaml) has workdir "/": the agent's files are
# /func.py, /test.py and whatever it creates next to them. Regular files directly under "/" are
# captured, plus the trees under REC (/tmp, /home, /root, /workspace); text only, size-capped,
# caches and dotfiles skipped.
CAPTURE_SCRIPT = r"""
import json, os
ROOT = os.environ.get("WS_ROOT", "/")
REC = [os.path.join(ROOT, d) for d in ("tmp", "home", "root", "workspace")]
SKIP_DIRS = {"__pycache__", ".cache", ".git", "node_modules", ".venv", ".pytest_cache"}
SKIP_ROOT = {".dockerenv"}   # dotfiles are skipped everywhere (stock /root/.bashrc etc.)
MAX_FILE, MAX_TOTAL = 262144, 4 << 20
files, skipped = {}, []
def add(p):
    if os.path.basename(p).startswith(".") or os.path.islink(p) or not os.path.isfile(p):
        return
    try:
        n = os.path.getsize(p)
        if n > MAX_FILE:
            skipped.append([p, "size", n]); return
        b = open(p, "rb").read()
        files[p] = b.decode("utf-8")
    except UnicodeDecodeError:
        skipped.append([p, "binary", len(b)])
    except Exception as e:
        skipped.append([p, repr(e), 0])
for name in sorted(os.listdir(ROOT)):
    if name not in SKIP_ROOT:
        add(os.path.join(ROOT, name))
for r in REC:
    for d, dn, fn in os.walk(r):
        dn[:] = sorted(x for x in dn if x not in SKIP_DIRS)
        for f in sorted(fn):
            add(os.path.join(d, f))
total = sum(len(v.encode("utf-8")) for v in files.values())
for p in sorted(files, key=lambda k: -len(files[k])):
    if total <= MAX_TOTAL:
        break
    total -= len(files[p].encode("utf-8")); skipped.append([p, "total", len(files[p])]); del files[p]
print(json.dumps({"root": ROOT, "files": files, "skipped": skipped}))
"""


async def capture_workspace(timeout: int = 60) -> dict:
    """{"root", "files": {abs path: text}, "skipped": [[path, reason, size]]} or {"error": ...}.
    Never raises: a failed capture must not fail the score."""
    try:
        res = await sandbox().exec(["python", "-c", CAPTURE_SCRIPT], timeout=timeout, timeout_retry=False)
    except Exception as e:  # noqa: BLE001
        return {"error": f"exec failed: {e!r}"}
    if not res.success:
        return {"error": f"exit {res.returncode}: {res.stderr[-2000:]}"}
    try:
        return json.loads(res.stdout)
    except json.JSONDecodeError as e:
        return {"error": f"bad capture output: {e}"}


def workspace_files(ws: dict | None) -> dict[str, str]:
    """Sample.files form of a capture: paths relative to the sandbox workdir ("/")."""
    if not ws or "files" not in ws:
        return {}
    root = ws.get("root", "/")
    out = {}
    for p, text in ws["files"].items():
        rel = p[len(root):] if p.startswith(root) else p
        out[rel.lstrip("/")] = text
    return out


@scorer(metrics=[accuracy(), stderr()])
def lcb_scorer() -> Scorer:
    """Upstream's swe_style_humaneval_scorer plus a workspace capture in the score metadata."""
    inner = livecodebench_tasks.swe_style_humaneval_scorer()

    async def score(state: TaskState, target: Target) -> Score:
        ws = await capture_workspace()
        s = await inner(state, target)
        md = dict(s.metadata or {})
        md["workspace"] = ws
        return Score(value=s.value, answer=s.answer, explanation=s.explanation, metadata=md)

    return score
