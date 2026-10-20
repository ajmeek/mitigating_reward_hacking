#!/usr/bin/env python
"""Idempotently patch an installed vllm-lens with steering modes: add | ablate | clamp.

- "add"    (default): h += scale * v            (upstream behavior, unchanged)
- "ablate": h -= (h_true . u) u                 (remove the component entirely)
- "clamp":  h -= max(0, (h_true . u) - thr) u   (cap the component at thr)

All modes take an optional per-token boolean ``mask`` (positions to steer within the slice);
the decode-only gate (bootstrap patch 2) uses it to steer, during prefill, only the positions the
model itself wrote (``steer_positions`` in the request payload).

where u = v/|v| and h_true is the TRUE residual stream. For fused-residual models the
hook's ``target`` is only the MLP-delta half of the layer output; vllm-lens's own capture
path reconstructs the stream as ``output[0] + output[1]``, so the vectors we fit live in
that basis and projections must be measured there too. Measuring on ``target`` would clamp
against the wrong stream. The correction is written into ``target``, which adds into the
stream, so the net effect lands on h_true.

Two vllm-lens layouts are supported:

- newer: ``_apply_steering`` already receives a ``norm_ref`` (the true stream) argument.
- older (<=1.1): ``_apply_steering`` has no ``norm_ref``; this script adds the parameter,
  computes the true stream in ``_hook_inner``, and threads it through the call site.

Set VLLM_LENS_STEER_LOG=1 in the server environment to print one ``[STEER]`` line per
hooked forward pass with the measured projection and the excess removed.

Composes with the decode-only gate patch (patch_vllm_lens_gate.py, VLLM_LENS_STEER_DECODE_ONLY):
with it active, modes apply to generated tokens and to the request's steer_positions.

Run inside the serving env:  uv run python patch_vllm_lens_modes.py
"""

from pathlib import Path

import vllm_lens

pkg = Path(vllm_lens.__file__).parent


def _write(path, text):
    """Replace the file rather than truncating it in place: uv hardlinks site-packages to its
    wheel cache, so an in-place write would also patch the cached wheel and a later reinstall
    would silently reproduce the old patched file (2026-09-14)."""
    path.unlink()
    path.write_text(text)

# --- 1. schema: accept mode/threshold through validation -----------------------------
types_py = pkg / "_helpers" / "types.py"
t = types_py.read_text()
if "steering mode (local patch)" not in t:
    anchor = '    position_indices: list[int] | None = None'
    assert anchor in t, "schema anchor not found"
    addition = (
        '    mode: str = "add"\n'
        '    """steering mode (local patch): add | ablate | clamp."""\n\n'
        '    threshold: float | None = None\n'
        '    """clamp ceiling on the (h.u) projection (local patch; clamp mode only)."""\n\n'
        '    steer_positions: list[list[int]] | None = None\n'
        '    """absolute [start, end) token ranges the model itself wrote (local patch): during prefill only these positions are steered."""\n\n'
    )
    t = t.replace(anchor, addition + anchor)
    _write(types_py, t)
    print("schema patched")
else:
    print("schema already patched")

# --- 2. worker: implement the modes in _apply_steering (2D broadcast branch) ---------
MODE_BODY = """        if vec.dim() == 1:
            # local patch: ablate/clamp modes operate on the projection onto the
            # TRUE residual stream (norm_ref), writing the correction into target.
            _ref = norm_ref if norm_ref is not None else target
            mode = getattr(cfg, "mode", "add")
            if mode in ("ablate", "clamp"):
                u = (vec / vec.norm()).float()
                proj = _ref[start:end].float() @ u
                if mode == "ablate":
                    excess = proj
                else:
                    thr = cfg.threshold if cfg.threshold is not None else 0.0
                    excess = (proj - thr).clamp(min=0.0)
                if mask is not None:
                    excess = excess * mask
                if _STEER_LOG:
                    print(
                        "[STEER] layer=%d mode=%s abs_start=%d masked=%s proj=%s excess=%s"
                        % (layer_idx, mode, abs_start, None if mask is None else int(mask.sum()),
                           proj.tolist(), excess.tolist()),
                        flush=True,
                    )
                delta = (excess.unsqueeze(1) * u.unsqueeze(0)).to(target.dtype)
                target[start:end] = target[start:end] - delta
                continue
            # 2D: broadcast to all positions
            v = vec.unsqueeze(0)
            if cfg.norm_match:
                v = norm_match(_ref[start:end], v)
            if mask is not None:
                v = v * mask.unsqueeze(1).to(v.dtype)
            target[start:end] = target[start:end] + v * cfg.scale"""

worker_py = pkg / "_worker_ext.py"
w = worker_py.read_text()

if "local patch: ablate/clamp modes" in w:
    print("worker already patched")
else:
    # -- module-level log flag (shared by both layouts) --
    log_anchor = "def _apply_steering(\n"
    assert log_anchor in w, "worker _apply_steering def not found"
    w = w.replace(
        log_anchor,
        'import os as _os\n\n'
        '_STEER_LOG = _os.environ.get("VLLM_LENS_STEER_LOG") == "1"\n\n\n'
        + log_anchor,
        1,
    )

    new_anchor = """        if vec.dim() == 1:
            # 2D: broadcast to all positions
            v = vec.unsqueeze(0)
            if cfg.norm_match:
                v = norm_match(norm_ref[start:end], v)
            target[start:end] = target[start:end] + v * cfg.scale"""
    old_anchor = """        if vec.dim() == 1:
            # 2D: broadcast to all positions
            v = vec.unsqueeze(0)
            if cfg.norm_match:
                v = norm_match(target[start:end], v)
            target[start:end] = target[start:end] + v * cfg.scale"""

    if new_anchor in w:
        w = w.replace(new_anchor, MODE_BODY)
        print("worker patched (norm_ref-native layout)")
    else:
        assert old_anchor in w, "worker anchor not found (neither layout matched)"
        w = w.replace(old_anchor, MODE_BODY)

        # -- add the norm_ref parameter to the signature --
        sig_anchor = """    end: int,
    abs_start: int,
) -> None:"""
        assert sig_anchor in w, "worker signature anchor not found"
        w = w.replace(
            sig_anchor,
            """    end: int,
    abs_start: int,
    norm_ref: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> None:""",
            1,
        )

        # -- compute the true residual stream in the hook --
        tgt_anchor = """        if isinstance(output, tuple):
            modified_output = (output[0].clone(), output[1])
            target = modified_output[0]
        else:
            modified_output = output.clone()
            target = modified_output"""
        assert tgt_anchor in w, "worker target anchor not found"
        w = w.replace(
            tgt_anchor,
            tgt_anchor
            + """

        # local patch: the TRUE residual stream at this layer's output. For
        # fused-residual models the hook's target is only the MLP-delta half;
        # vllm-lens's own capture path reconstructs output[0] + output[1], so
        # projections must be measured against that same quantity.
        if isinstance(output, tuple) and output[1] is not None:
            _norm_ref = target + output[1]
        else:
            _norm_ref = target""",
            1,
        )

        # -- thread it through the call site --
        call_anchor = """            _apply_steering(
                per_req_steering[i], layer_idx, target, start, end, abs_start
            )"""
        assert call_anchor in w, "worker call-site anchor not found"
        w = w.replace(
            call_anchor,
            """            _steer_mask = None  # local patch: set by the decode-only gate (serving/bootstrap.sh patch 2)
            _apply_steering(
                per_req_steering[i],
                layer_idx,
                target,
                start,
                end,
                abs_start,
                norm_ref=_norm_ref,
                mask=_steer_mask,
            )""",
            1,
        )
        print("worker patched (legacy layout, norm_ref threaded)")

    _write(worker_py, w)

# --- 3. seq_lens (local patch): abs_start was always 0 -------------------------------
# vllm-lens reads `getattr(attn_metadata, "seq_lens", None)` — but attn_metadata is a
# dict of per-backend metadata (vllm/forward_context.py), so that is always None and the
# fallback `abs_start = 0` runs on EVERY forward pass. Harmless for stock steering (2D
# vectors broadcast to all rows) but wrong for anything positional under chunked prefill:
# every chunk after the first would be treated as starting at position 0. Take seq_lens from
# the same per-backend metadata objects the query_start_loc lookup uses (found 2026-09-10 in
# review). The positions-aware gate (patch_vllm_lens_gate.py) takes its positions from the
# request state instead, so this only feeds abs_start in the [STEER] log line now.
w = worker_py.read_text()
if "seq_lens (local patch)" not in w:
    old = '        seq_lens: Any = getattr(attn_metadata, "seq_lens", None)\n'
    assert old in w, "worker seq_lens anchor not found"
    new = (
        '        seq_lens: Any = None  # seq_lens (local patch): lives on the per-backend metadata\n'
        '        for _meta in attn_metadata.values():\n'
        '            if getattr(_meta, "seq_lens", None) is not None:\n'
        '                seq_lens = getattr(_meta, "seq_lens")\n'
        '                break\n'
    )
    w = w.replace(old, new, 1)
    _write(worker_py, w)
    print("worker patched (seq_lens)")
else:
    print("worker already has seq_lens patch")

print("patch complete:", pkg)
