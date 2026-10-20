"""Writers for the per-token activation dump format (one rollout at a time).

Layout (experiments/full-token-capture/DATASET.md):

    L{ll:02d}/<id>.safetensors   token_acts (N,d_model) bf16 + the 4 index arrays
    tokens/<id>.safetensors      token_ids (N,) i32 + the 4 index arrays
    segmeans/<id>.safetensors    turn_means (T,L,d) f32, turn_tokens [+ segment_means/tokens]
    meta/<id>.json               manifest row + turns[] [+ segments[]] — written LAST: its
                                 presence is the completion marker resume keys on

`create_acts_shard` hand-writes the safetensors header and returns a uint16 memmap over the
bf16 region so a capture writes rows straight into the final file. Lifted verbatim from
scripts/capture_all_tokens.py (verified byte-identical to
safetensors' save_file there).
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import torch

_ST_DTYPE = {"bfloat16": "BF16", "int8": "I8", "int32": "I32"}


def create_acts_shard(path, N: int, d_model: int, index_tensors: dict) -> np.memmap:
    """Create `path` with token_acts (N,d_model) bf16 preallocated and the small index
    tensors written; return an (N,d_model) uint16 memmap aliasing token_acts' bytes.

    NOTE: the i8 kind array precedes the i32 arrays, so for odd N the i32 offsets are
    unaligned — readers must copy those (token_dump.tensors_from_bytes does)."""
    acts_nbytes = N * d_model * 2
    header = {"token_acts": {"dtype": "BF16", "shape": [N, d_model],
                             "data_offsets": [0, acts_nbytes]}}
    off = acts_nbytes
    for name, t in index_tensors.items():
        nb = t.numel() * t.element_size()
        header[name] = {"dtype": _ST_DTYPE[str(t.dtype).split(".")[-1]],
                        "shape": list(t.shape), "data_offsets": [off, off + nb]}
        off += nb
    hjson = json.dumps(header, separators=(",", ":")).encode("utf-8")
    hjson += b" " * ((-(len(hjson) + 8)) % 8)  # 8-byte-align the data region
    data_start = 8 + len(hjson)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        f.truncate(data_start + off)
    with open(path, "r+b") as f:
        for name, t in index_tensors.items():
            f.seek(data_start + header[name]["data_offsets"][0])
            f.write(t.contiguous().numpy().tobytes())
    return np.memmap(path, dtype=np.uint16, mode="r+", offset=data_start, shape=(N, d_model))


def index_tensors(tree: dict) -> dict[str, torch.Tensor]:
    """The four per-row index arrays from a tree dict (kinds/msgs/turn_of/ctxpos lists)."""
    return {
        "token_kind": torch.tensor(tree["kinds"], dtype=torch.int8),
        "token_msg": torch.tensor(tree["msgs"], dtype=torch.int32),
        "token_turn": torch.tensor(tree["turn_of"], dtype=torch.int32),
        "token_ctxpos": torch.tensor(tree["ctxpos"], dtype=torch.int32),
    }


def write_tokens(out_dir: Path, rid: str, tok_ids: list[int], idx: dict) -> Path:
    from safetensors.torch import save_file

    p = Path(out_dir) / "tokens" / f"{rid}.safetensors"
    p.parent.mkdir(parents=True, exist_ok=True)
    save_file({"token_ids": torch.tensor(tok_ids, dtype=torch.int32), **idx}, str(p))
    return p


def write_segmeans(out_dir: Path, rid: str, turn_means: torch.Tensor, turn_tokens: list[int],
                   segment_means: torch.Tensor | None = None,
                   segment_tokens: list[int] | None = None) -> Path:
    """turn_means (T,L,d) f32; segment_means (S,L,d) stored f16 (clamped) as in the 103 dump."""
    from safetensors.torch import save_file

    p = Path(out_dir) / "segmeans" / f"{rid}.safetensors"
    p.parent.mkdir(parents=True, exist_ok=True)
    tensors = {"turn_means": turn_means.float().contiguous(),
               "turn_tokens": torch.tensor(turn_tokens, dtype=torch.int32)}
    if segment_means is not None:
        tensors["segment_means"] = segment_means.clamp(-65504, 65504).half().contiguous()
        tensors["segment_tokens"] = torch.tensor(segment_tokens or [], dtype=torch.int32)
    save_file(tensors, str(p))
    return p


def write_meta(out_dir: Path, rid: str, row: dict, extra: dict) -> Path:
    """meta/<id>.json = manifest row + per-turn detail. Call LAST (completion marker)."""
    p = Path(out_dir) / "meta" / f"{rid}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({**row, **extra}, indent=1) + "\n")
    return p


# ---------------------------------------------------------------- S3 (aws cli; the serving env has no boto3)

def s3_cp(local: Path, s3_prefix: str, rel: str) -> None:
    import subprocess

    subprocess.run(["aws", "s3", "cp", "--quiet", str(local), f"{s3_prefix.rstrip('/')}/{rel}"], check=True)


def s3_cp_unlink(local: Path, s3_prefix: str, rel: str) -> None:
    s3_cp(local, s3_prefix, rel)
    Path(local).unlink()


def s3_existing_metas(s3_prefix: str) -> set[str]:
    """Ids whose meta/<id>.json (the completion marker) is already in S3 — for resume."""
    import subprocess

    r = subprocess.run(["aws", "s3", "ls", f"{s3_prefix.rstrip('/')}/meta/"], capture_output=True, text=True)
    if r.returncode not in (0, 1):  # 1 = prefix doesn't exist yet (fresh run)
        raise RuntimeError(f"cannot list {s3_prefix}/meta/ for resume: {r.stderr}")
    return {line.split()[-1].removesuffix(".json") for line in r.stdout.splitlines()
            if line.strip().endswith(".json")}


# ---------------------------------------------------------------- provenance

def vllm_lens_provenance() -> tuple[str, str]:
    """(version, sha256-prefix of the patched plugin)."""
    import vllm_lens

    plugin = Path(vllm_lens.__file__).parent / "_activations_plugin.py"
    return (getattr(vllm_lens, "__version__", "unknown"),
            hashlib.sha256(plugin.read_bytes()).hexdigest()[:16])


def worker_patch_markers() -> dict[str, bool]:
    """Which local vllm-lens patches are present in the installed worker/plugin."""
    import vllm_lens

    pkg = Path(vllm_lens.__file__).parent
    worker = (pkg / "_worker_ext.py").read_text()
    plugin = (pkg / "_activations_plugin.py").read_text()
    return {
        "modes": "local patch: ablate/clamp modes" in worker,
        "seq_lens": "seq_lens (local patch)" in worker,
        "decode_only_gate": "VLLM_LENS_STEER_DECODE_ONLY" in worker,
        # the gate's positions-aware form (serving/patch_vllm_lens_gate.py): prefill steers only
        # the request's steer_positions, so a replay can pass the spans generation steered
        "positions_gate": 'getattr(_cfg, "steer_positions"' in worker,
        "prefix_reuse": "vllm_lens_reuse_prefix_cache" in plugin,
        "disable_zstd": "VLLM_LENS_DISABLE_ZSTD" in worker or "VLLM_LENS_DISABLE_ZSTD" in plugin,
    }


def template_sha256(tokenizer) -> str:
    """sha256 prefix of the tokenizer's chat template (the 27B and 2B templates differ)."""
    tmpl = getattr(tokenizer, "chat_template", None) or ""
    return hashlib.sha256(tmpl.encode("utf-8")).hexdigest()[:16]


def serving_provenance(capturer, *, layer_chunk_cap: int, prefix_reuse: bool,
                       tokenizer=None, extra: dict | None = None) -> dict:
    """The manifest `serving{}` block: everything that can change the numerics."""
    import os

    import vllm

    llm = capturer.llm
    vc = llm.llm_engine.vllm_config
    ver, sha = vllm_lens_provenance()
    block = {
        "model": capturer.model_name,
        "hf_revision": getattr(vc.model_config, "revision", None),
        "tensor_parallel_size": int(vc.parallel_config.tensor_parallel_size),
        "max_model_len": int(vc.model_config.max_model_len),
        "dtype": str(vc.model_config.dtype).replace("torch.", ""),
        "enable_prefix_caching": bool(vc.cache_config.enable_prefix_caching),
        "enable_chunked_prefill": bool(getattr(vc.scheduler_config, "enable_chunked_prefill", None)),
        "max_num_batched_tokens": int(getattr(vc.scheduler_config, "max_num_batched_tokens", 0) or 0),
        "vllm_version": vllm.__version__,
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "n_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "layer_chunk_cap": layer_chunk_cap,
        "batch_size": 1,
        "enforce_eager": True,
        "storage_dtype": "bfloat16",
        "prefix_reuse_patch": prefix_reuse,
        "vllm_lens_version": ver,
        "vllm_lens_plugin_sha256": sha,
        "patches": worker_patch_markers(),
        "env": {k: os.environ.get(k) for k in (
            "VLLM_ATTENTION_BACKEND", "VLLM_FLASH_ATTN_VERSION", "VLLM_USE_FLASHINFER_SAMPLER",
            "VLLM_LENS_STEER_DECODE_ONLY", "VLLM_LENS_CAPTURE_PREFIX_REUSE", "VLLM_LENS_DISABLE_ZSTD")},
    }
    if tokenizer is not None:
        block["template_sha256"] = template_sha256(tokenizer)
    if extra:
        block.update(extra)
    return block
