"""Stream large per-rollout files from S3 one at a time, under a disk budget.

The bf16 captures (`s3://steering-reasoning-artifacts/activations/...`) are tens of GB per
set and up to ~10 GB per rollout, so the fit stages pull one file, reduce it, delete it,
and prefetch the next in the background while they work. `Prefetch` is that loop's
downloader: it starts the next download only when the resident file plus the next one fit
in `budget` bytes, verifies sizes against the S3 listing, and retries foreground pulls.

Uses the `aws` CLI (credentials from the environment), no boto3 dependency.

    sizes = s3_sizes(f"{S3}/segmeans/")
    pf = Prefetch(f"{S3}/segmeans", cache_dir, sizes, budget=20e9)
    pf.start(ids[0])
    for i, tid in enumerate(ids):
        path = pf.wait()                      # local file of tid, size-checked
        if i + 1 < len(ids): pf.start(ids[i + 1], resident=tid)
        ...                                   # reduce
        os.remove(path)
"""

import subprocess
import time
from pathlib import Path


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def s3_sizes(prefix: str) -> dict:
    """{stem: bytes} for every object directly under `prefix` (an `aws s3 ls` listing).

    Keys are file names without their extension (so `lcbhard_5.safetensors` -> `lcbhard_5`).
    """
    if not prefix.endswith("/"):
        prefix += "/"
    out = subprocess.run(["aws", "s3", "ls", prefix], check=True, capture_output=True, text=True).stdout
    sizes = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[0] != "PRE":
            sizes[Path(parts[3]).stem] = int(parts[2])
    return sizes


def s3_cp(src: str, dst: Path, retries: int = 4, expect_bytes=None) -> Path:
    """Foreground `aws s3 cp src dst` with retries and an optional size check."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        if dst.exists():
            dst.unlink()   # stale or partial
        rc = subprocess.run(["aws", "s3", "cp", src, str(dst), "--only-show-errors"]).returncode
        if rc == 0 and dst.exists() and (expect_bytes is None or dst.stat().st_size == expect_bytes):
            return dst
        log(f"  retry {attempt + 1}/{retries} for {src}")
    raise RuntimeError(f"download failed for {src} after {retries} attempts")


class Prefetch:
    """One-ahead downloader of `<prefix>/<id><ext>` into `cache_dir`, under a byte budget.

    sizes: {id: bytes} from `s3_sizes(prefix)`, used for the budget and the size check.
    start(id, resident=None): begin fetching id in the background, unless the resident file
        plus id would exceed the budget, in which case the fetch is deferred to wait().
    wait(): block until the started file is complete and size-checked; returns its path.
    """

    def __init__(self, prefix: str, cache_dir, sizes: dict, budget: float = 20e9, ext: str = ".safetensors",
                 retries: int = 4):
        self.prefix = prefix.rstrip("/")
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.sizes, self.budget, self.ext, self.retries = sizes, budget, ext, retries
        self.proc = None
        self.tid = None
        self.pending = None

    def path(self, tid) -> Path:
        return self.cache / f"{tid}{self.ext}"

    def _url(self, tid) -> str:
        return f"{self.prefix}/{tid}{self.ext}"

    def complete(self, tid) -> bool:
        p = self.path(tid)
        return p.exists() and (tid not in self.sizes or p.stat().st_size == self.sizes[tid])

    def start(self, tid, resident=None):
        self.tid, self.pending, self.proc = tid, None, None
        if self.complete(tid):
            return
        if resident is not None and self.sizes.get(resident, 0) + self.sizes.get(tid, 0) > self.budget:
            self.pending = tid   # fetch in wait(), once the caller has deleted the resident file
            return
        dst = self.path(tid)
        if dst.exists():
            dst.unlink()
        self.proc = subprocess.Popen(["aws", "s3", "cp", self._url(tid), str(dst), "--only-show-errors"])

    def wait(self) -> Path:
        tid = self.tid
        if self.pending is not None:
            s3_cp(self._url(tid), self.path(tid), self.retries, self.sizes.get(tid))
            self.pending = None
        if self.proc is not None:
            rc = self.proc.wait()
            self.proc = None
            if rc != 0 or not self.complete(tid):
                log(f"  background download of {tid} failed (rc={rc}); refetching")
                s3_cp(self._url(tid), self.path(tid), self.retries, self.sizes.get(tid))
        if not self.complete(tid):
            p = self.path(tid)
            have = p.stat().st_size if p.exists() else None
            raise RuntimeError(f"{p.name}: size {have} != S3 {self.sizes.get(tid)}")
        return self.path(tid)
