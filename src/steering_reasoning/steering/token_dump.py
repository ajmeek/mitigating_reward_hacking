"""Reader for the per-layer bf16 token dump (`activations/qwen35_27b_bf16_all103`).

Format reference: experiments/full-token-capture/DATASET.md. The dump root may be a local
directory or an ``s3://bucket/prefix``. S3 objects are fetched whole into memory with
parallel ranged GETs and parsed zero-copy, so streaming a per-layer shard costs one
download and no temp files — the intended pattern for a CPU box that projects shards
and discards them.
"""

from __future__ import annotations

import json
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from steering_reasoning.kinds import KIND_NAMES, KIND_REASONING  # noqa: F401  (re-exported)

N_LAYERS = 64
D_MODEL = 5120  # Qwen3.5-27B; the all103 dump's hidden size (readers of other models pass their own)

_NP_DTYPES = {
    "I8": np.int8,
    "I32": np.int32,
    "I64": np.int64,
    "F16": np.float16,
    "F32": np.float32,
    "BF16": np.uint16,  # bit pattern; viewed as torch.bfloat16 below
}


def tensors_from_bytes(buf: bytearray) -> dict[str, torch.Tensor]:
    """Parse a safetensors payload held in a (writable) bytearray without copying.

    Tensors alias ``buf``; keep the dict alive as long as you use them. bf16 tensors are
    reinterpreted from their uint16 storage.
    """
    header_len = struct.unpack("<Q", buf[:8])[0]
    header = json.loads(bytes(buf[8 : 8 + header_len]))
    base = 8 + header_len
    out: dict[str, torch.Tensor] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        np_dtype = np.dtype(_NP_DTYPES[info["dtype"]])
        start, end = info["data_offsets"]
        arr = np.frombuffer(buf, dtype=np_dtype, offset=base + start, count=(end - start) // np_dtype.itemsize)
        if (base + start) % np_dtype.itemsize:
            # capture_all's shard writer puts the i8 kind array before the i32 index arrays,
            # so for odd-N rollouts those offsets are unaligned; torch cannot view them —
            # copy (they are the small (N,) arrays; token_acts starts the data block).
            arr = arr.copy()
        t = torch.from_numpy(arr.reshape(info["shape"]))
        if info["dtype"] == "BF16":
            t = t.view(torch.bfloat16)
        out[name] = t
    return out


def _s3_get_bytes(client, bucket: str, key: str, *, part_bytes: int, workers: int, attempts: int = 6) -> bytearray:
    size = client.head_object(Bucket=bucket, Key=key)["ContentLength"]
    buf = bytearray(size)
    view = memoryview(buf)

    def fetch(start: int) -> None:
        end = min(start + part_bytes, size) - 1
        off = start
        # botocore retries the request, not a stream that dies mid-body; the range is
        # idempotent, so re-issue from wherever the previous attempt got to.
        for attempt in range(attempts):
            try:
                body = client.get_object(Bucket=bucket, Key=key, Range=f"bytes={off}-{end}")["Body"]
                while chunk := body.read(1 << 20):
                    view[off : off + len(chunk)] = chunk
                    off += len(chunk)
                if off == end + 1:
                    return
                raise IOError(f"short read: {off - start} of {end + 1 - start} bytes")
            except Exception as e:  # noqa: BLE001 — botocore raises a zoo of stream errors
                if attempt == attempts - 1:
                    raise IOError(f"s3://{bucket}/{key} range {start}-{end} failed after {attempts} attempts") from e
                time.sleep(min(2**attempt, 30))

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(fetch, range(0, size, part_bytes)))
    return buf


class TokenDump:
    """Handle on one dump root. Every accessor fetches the object whole and returns tensors."""

    def __init__(self, root: str, *, part_bytes: int = 64 << 20, workers: int = 16):
        self.root = root.rstrip("/")
        self.part_bytes, self.workers = part_bytes, workers
        if self.root.startswith("s3://"):
            import boto3
            from botocore.config import Config

            bucket, _, prefix = self.root[5:].partition("/")
            self._bucket, self._prefix = bucket, prefix.rstrip("/")
            self._s3 = boto3.client("s3", config=Config(max_pool_connections=max(64, 4 * workers)))
            self._dir = None
        else:
            self._s3 = None
            self._dir = Path(self.root)
        self._manifest: dict | None = None

    # -- raw access -------------------------------------------------------------------
    def _key(self, rel: str) -> str:
        return f"{self._prefix}/{rel}" if self._prefix else rel

    def exists(self, rel: str) -> bool:
        if self._s3 is None:
            return (self._dir / rel).exists()
        try:
            self._s3.head_object(Bucket=self._bucket, Key=self._key(rel))
            return True
        except self._s3.exceptions.ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def read_bytes(self, rel: str) -> bytearray:
        if self._s3 is None:
            return bytearray((self._dir / rel).read_bytes())
        return _s3_get_bytes(self._s3, self._bucket, self._key(rel), part_bytes=self.part_bytes, workers=self.workers)

    def read_json(self, rel: str) -> dict:
        return json.loads(bytes(self.read_bytes(rel)))

    def read_tensors(self, rel: str) -> dict[str, torch.Tensor]:
        return tensors_from_bytes(self.read_bytes(rel))

    # -- dump objects -----------------------------------------------------------------
    def manifest(self) -> dict:
        if self._manifest is None:
            self._manifest = self.read_json("manifest.json")
        return self._manifest

    def task_ids(self, *, label: bool | None = None) -> list[str]:
        """All task ids in manifest order; ``label=True`` → hacked only, ``False`` → clean only."""
        rows = self.manifest()["transcripts"]
        return [r["task_id"] for r in rows if label is None or bool(r["label"]) == label]

    def meta(self, task: str) -> dict:
        return self.read_json(f"meta/{task}.json")

    def tokens(self, task: str) -> dict[str, torch.Tensor]:
        """token_ids + the four index arrays (token_kind/msg/turn/ctxpos)."""
        return self.read_tensors(f"tokens/{task}.safetensors")

    def layer(self, task: str, layer: int) -> dict[str, torch.Tensor]:
        """token_acts (N, 5120) bf16 for one layer + the four index arrays."""
        return self.read_tensors(f"L{layer:02d}/{task}.safetensors")

    def segmeans(self, task: str) -> dict[str, torch.Tensor]:
        """turn_means (T,64,5120) f32 [+ segment_means (S,64,5120) f16 for hacked rollouts]."""
        return self.read_tensors(f"segmeans/{task}.safetensors")
