"""Idempotently patch the installed vllm-lens worker hook with a steering FAST PATH (local patch 4).

Why (measured 2026-09-24 on the spec-vector arms box, 4x L40S TP=4, 40 steered streams):
aggregate decode was flat at ~150-165 tok/s from 23 to 40 streams (4-5 tok/s per stream) while the
same box type serves UNSTEERED at ~264 tok/s for 24 streams (11 per stream). The hook is installed
on all 64 decoder layers; whenever any request carries steering data it runs, at EVERY layer and
every step, the per-request config lookup (Phase 1), clones the layer output and builds the
norm_ref (Phase 2), and does three GPU->CPU syncs per steered request (`query_start_loc[i].item()`,
`query_start_loc[i+1].item()`, `seq_lens[i].item()`) before `_apply_steering` finds that this layer
is not in the vector's layer_index_map. At 40 requests that is ~7,700 device syncs per step per TP
worker, which serialises CPU and GPU and defeats async scheduling.

What this patch changes (outputs are bit-identical; only wasted work is removed):
  1. Phases 1-2 are skipped at layers that NO live steering config touches (all but L35 for the
     canonical arms). The old path only returned an unmodified clone there.
  2. At a steered layer the per-request `.item()` calls are replaced by one `query_start_loc.tolist()`
     and one `seq_lens.tolist()` before the loop.
Phase 3 (activation capture) is untouched.

Apply AFTER patch_vllm_lens_modes.py and patch_vllm_lens_gate.py (its anchors are upstream lines
that those patches leave in place, and it asserts the modes patch is present so the order is
enforced); run as an extra argument of patch_vllm_lens.sh, e.g.
`bash patch_vllm_lens.sh patch_vllm_lens_fastpath.py`. Safe to re-run. The file is unlinked
before it is rewritten: uv hardlinks site-packages to its wheel cache (2026-09-14 lesson), and an
in-place write would patch the cached wheel too.
"""
import importlib.util
import sys
from pathlib import Path

MARK = "local patch (fastpath)"

spec = importlib.util.find_spec("vllm_lens._worker_ext")
if spec is None or spec.origin is None:
    print("vllm_lens._worker_ext not found", file=sys.stderr)
    sys.exit(2)
worker_py = Path(spec.origin)
t = worker_py.read_text()

if MARK in t:
    print("worker fastpath already patched")
    sys.exit(0)
assert "local patch: ablate/clamp modes" in t, "apply patch_vllm_lens_modes.py (and _gate.py) first"

# 1. skip Phases 1-2 at layers no live config steers
a = """    per_req_steering: list[list[SteeringVector]] = []
    needs_steering = False
    for i in range(num_reqs):
        req_id = req_ids[i]
        req_state = runner.requests.get(req_id)"""
assert a in t, "Phase 1 anchor not found"
t = t.replace(a, """    per_req_steering: list[list[SteeringVector]] = []
    needs_steering = False
    # local patch (fastpath): when no live steering config touches THIS layer there is nothing
    # Phases 1-2 could change (the old path only returned an unmodified clone), so skip them.
    _sd = extension._steering_data
    _steers_here = bool(_sd) and any(
        layer_idx in c.layer_indices for cfgs in _sd.values() for c in cfgs   # layer_index_map is a property that rebuilds a dict
    )
    for i in range(num_reqs if _steers_here else 0):
        req_id = req_ids[i]
        req_state = runner.requests.get(req_id)""", 1)

# 2. one host copy of the position arrays instead of three device syncs per steered request
b = """        for i in range(num_reqs):
            if not per_req_steering[i]:
                continue
            start = int(query_start_loc[i].item())
            end = int(query_start_loc[i + 1].item())"""
assert b in t, "Phase 2 loop anchor not found"
t = t.replace(b, """        # local patch (fastpath): one host copy of the position arrays per layer instead of
        # three device syncs per steered request
        _qsl = query_start_loc.tolist()
        _sls = seq_lens.tolist() if isinstance(seq_lens, torch.Tensor) else seq_lens
        for i in range(num_reqs):
            if not per_req_steering[i]:
                continue
            start = int(_qsl[i])
            end = int(_qsl[i + 1])""", 1)

c = """            if seq_lens is not None:
                sl = seq_lens[i]
                sl_val = sl.item() if isinstance(sl, torch.Tensor) else int(sl)
                abs_start = int(sl_val - n_query)"""
assert c in t, "seq_lens anchor not found"
t = t.replace(c, """            if _sls is not None:
                abs_start = int(int(_sls[i]) - n_query)""", 1)

worker_py.unlink()          # break uv's hardlink to the wheel cache before rewriting
worker_py.write_text(t)
print("worker fastpath patched (skip unsteered layers; host-side position arrays)")
