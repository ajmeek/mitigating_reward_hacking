"""Local vllm-lens patch 2: the decode-only gate with positions-aware prefill (idempotent).

Generated tokens (decode: every prompt token already computed) are always steered. Prefill steers
only positions inside the request's ``steer_positions`` (the spans the model itself wrote in earlier
turns, sent by the proxy), so a re-read of the model's own reply gets the same steered KV whether it
comes from the prefix cache or a fresh prefill; prompts and tool outputs are never steered. Replaces
the older "single-token forward pass" heuristic (2026-09-13) and the "skip prefill entirely" gate
(2026-09-14). Active only when VLLM_LENS_STEER_DECODE_ONLY=1 is in the worker's environment.

Apply after patch_vllm_lens_modes.py (it anchors on that patch's ``_steer_mask`` line); both run
from patch_vllm_lens.sh (serving/bootstrap.sh and tools/runpod/bootstrap_replay_runpod.sh). A replay
passes ``steer_positions`` = the spans generation steered + the generated branch, so one prefill
reproduces prefill-and-decode steering of a generation turn.
"""
from pathlib import Path
import vllm_lens
p = Path(vllm_lens.__file__).parent / "_worker_ext.py"


def _write(path, text):
    """Replace the file rather than truncating it in place: uv hardlinks site-packages to its
    wheel cache, so an in-place write would also patch the cached wheel and a later reinstall
    would silently reproduce the old patched file (2026-09-14)."""
    path.unlink()
    path.write_text(text)
t = p.read_text()
if "VLLM_LENS_STEER_DECODE_ONLY" in t and 'getattr(_cfg, "steer_positions"' not in t:
    raise SystemExit("this worker carries the OLDER decode-only gate (single-token-pass heuristic), which "
                     "cannot be upgraded in place: reinstall vllm-lens from a clean wheel (cd serving && "
                     "uv cache clean vllm-lens && uv sync --reinstall-package vllm-lens — the cache clean "
                     "matters, uv hardlinks the cache into the venv), then re-run patch_vllm_lens.sh")
if "VLLM_LENS_STEER_DECODE_ONLY" not in t:
    a = "def _hook_inner(\n"
    assert a in t, "hook_inner anchor"
    t = t.replace(a, '_STEER_DECODE_ONLY = __import__("os").environ.get("VLLM_LENS_STEER_DECODE_ONLY") == "1"\n\n\n' + a, 1)
    b = "            _steer_mask = None  # local patch: set by the decode-only gate (serving/bootstrap.sh patch 2)\n"
    assert b in t, "steer_mask anchor (apply patch_vllm_lens_modes.py first)"
    t = t.replace(b, b + "            if _STEER_DECODE_ONLY:\n"
                        "                _rs = runner.requests.get(req_ids[i])\n"
                        "                if _rs is None:\n"
                        "                    print('[STEER] gate: request %s not in runner.requests; not steered' % req_ids[i], flush=True)\n"
                        "                    continue\n"
                        "                if _rs.num_computed_tokens < _rs.num_prompt_tokens:   # prefill: steer only what the model wrote\n"
                        "                    _n = end - start\n"
                        "                    _pos = torch.arange(_rs.num_computed_tokens, _rs.num_computed_tokens + _n, device=target.device)\n"
                        "                    _m = torch.zeros(_n, dtype=torch.bool, device=target.device)\n"
                        "                    for _cfg in per_req_steering[i]:\n"
                        "                        for _a, _b in (getattr(_cfg, \"steer_positions\", None) or []):\n"
                        "                            _m |= (_pos >= _a) & (_pos < _b)\n"
                        "                    if not bool(_m.any()):\n"
                        "                        continue\n"
                        "                    _steer_mask = _m\n", 1)
    _write(p, t)
    print("decode-only gate patched")
else:
    print("decode-only gate present")
