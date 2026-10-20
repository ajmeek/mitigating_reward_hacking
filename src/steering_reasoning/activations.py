"""Loading and label-joining for dump_activations.py output dirs.

A dump dir holds `manifest.json` plus one `{task_id}.safetensors` / `{task_id}.json` pair
per transcript. `load_dump` reads it into (manifest, rows); `join_judge_labels` attaches
MisTrace judge labels to the captured segment metadata.

Heavy deps (torch, safetensors) are imported lazily inside functions so this module stays
importable from the root eval env.
"""

import json
from pathlib import Path

#: tensors every row exposes, so these are always read
_ROW_TENSORS = ("turn_means", "turn_tokens", "segment_means")
#: tensors shaped (..., L, D) — the ones `layers=` can slice
_LAYERED = ("turn_means", "segment_means", "token_acts")


def load_dump(dump_dir: str | Path, layers=None, extra_tensors=()):
    """Read a dump_activations.py output dir into (manifest, rows).

    Reads only the tensors a row actually exposes. A shard also holds `token_acts`
    (N_tokens, L, D) — ~10 GB across the 27B uncapped dump — which nothing here returns
    by default. Ask for it by name if you want it.

    layers: subset of the captured layers, e.g. [35]. Layered tensors keep their
        (..., L, D) shape but carry only these layers, and the RETURNED manifest's
        "layers" list is filtered to match — so the usual
        `li = manifest["layers"].index(35)` idiom still indexes correctly, and memory
        drops by the number of layers dropped.
    extra_tensors: additional shard keys to read ("token_acts", "token_pos",
        "token_turn", "segment_tokens"); each lands in the row under its own name.

    Each row: task_id, label, gold, trace_mean (L, D) token-weighted over turns,
    turn_means (T, L, D), turn_tokens (T,), segment_means (S, L, D) or None,
    segment_meta (list of dicts from the sidecar JSON).
    """
    import torch
    from safetensors import safe_open

    dump_dir = Path(dump_dir)
    manifest = json.loads((dump_dir / "manifest.json").read_text())
    keep = None
    if layers is not None:
        keep = [manifest["layers"].index(int(x)) for x in layers]   # ValueError if absent
        if keep != list(range(len(manifest["layers"]))):
            manifest = {**manifest, "layers": [int(x) for x in layers]}
        else:
            keep = None

    wanted = tuple(_ROW_TENSORS) + tuple(extra_tensors)
    rows = []
    for t in manifest["transcripts"]:
        tensors = {}
        with safe_open(str(dump_dir / f"{t['task_id']}.safetensors"), "pt") as f:
            available = set(f.keys())
            for k in wanted:
                if k not in available:
                    continue
                if keep is not None and k in _LAYERED:
                    sl = f.get_slice(k)
                    tensors[k] = torch.stack([sl[:, i, :] for i in keep], dim=1)
                else:
                    tensors[k] = f.get_tensor(k)
        meta = json.loads((dump_dir / f"{t['task_id']}.json").read_text())
        w = tensors["turn_tokens"].float()
        trace_mean = (tensors["turn_means"] * w[:, None, None]).sum(0) / w.sum()
        row = {
            "task_id": t["task_id"],
            "label": t["label"],
            "gold": t["has_gold_spans"],
            "trace_mean": trace_mean,  # (L, D)
            "turn_means": tensors["turn_means"],  # (T, L, D)
            "turn_tokens": tensors["turn_tokens"],  # (T,) — lets analyses length-match turns
            "segment_means": tensors.get("segment_means"),  # (S, L, D) | None
            "segment_meta": meta["segments"],
        }
        for k in extra_tensors:
            row[k] = tensors.get(k)
        rows.append(row)
    return manifest, rows


def join_judge_labels(rows, model="qwen3_5_27b"):
    """Attach MisTrace judge_label to captured segment metadata, keyed by (msg, segment_idx).

    Mutates `rows` in place (sets meta["judge_label"] on matched segments) and returns
    (provenance, stats) where stats counts matched/unmatched segments. A segment only
    matches when its captured text equals the MisTrace segment text exactly.
    """
    from steering_reasoning.data.mistrace import load_mistrace

    records, provenance = load_mistrace(model)
    by_task = {r["task_id"]: r for r in records}
    stats = {"matched": 0, "unmatched": 0}
    for row in rows:
        rec = by_task.get(row["task_id"])
        if not rec or not rec.get("span_labels") or row["segment_means"] is None:
            continue
        lookup = {}
        for seg in rec["span_labels"]["segments"]:
            section = seg.get("section", "")
            if not section.endswith(":reasoning"):
                continue
            msg = int(section.split(":")[0][3:])
            lookup[(msg, seg["segment_idx"])] = seg
        for meta in row["segment_meta"]:
            key = (meta["message_index"], meta["segment_idx"])
            src = lookup.get(key)
            if src is None or src["text"] != meta["text"]:
                stats["unmatched"] += 1
                continue
            meta["judge_label"] = src.get("judge_label")
            stats["matched"] += 1
    return provenance, stats


# --- memmapped access to one tensor of a large safetensors file -----------------------
# The segment-mean and per-token shards are too big to load; the fit stages memmap one
# tensor and reduce it in row chunks. Graduated from probe-refit-bf16's streaming scripts.


def safetensors_header(path):
    """(header dict, byte offset of the data section) of a safetensors file, without loading it."""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        hdr = json.loads(f.read(n))
    return hdr, 8 + n


_NP_DTYPES = {"F16": "float16", "F32": "float32", "BF16": None, "I32": "int32", "I64": "int64", "I8": "int8"}


def memmap_tensor(path, name, hdr=None, base=None, dtype_expect=None):
    """Read-only numpy memmap of tensor `name` in a safetensors file (F16/F32/int dtypes).

    hdr/base from `safetensors_header` may be passed to avoid re-reading the header.
    dtype_expect ("F16", "F32", ...) asserts the stored dtype. BF16 has no numpy dtype;
    use safetensors' `safe_open(...).get_slice` for those shards.
    """
    import numpy as np

    if hdr is None:
        hdr, base = safetensors_header(path)
    info = hdr[name]
    if dtype_expect is not None:
        assert info["dtype"] == dtype_expect, (name, info["dtype"])
    np_dtype = _NP_DTYPES.get(info["dtype"])
    if np_dtype is None:
        raise ValueError(f"{name}: dtype {info['dtype']} has no numpy memmap")
    off = info["data_offsets"][0]
    return np.memmap(path, dtype=np_dtype, mode="r", offset=base + off, shape=tuple(info["shape"]))
