#!/usr/bin/env python
"""Idempotently patch an installed vllm-lens to allow prefix-cache reuse for capture.

Upstream behavior (unchanged by default): any request that wants activations gets
``skip_reading_prefix_cache = True`` forced, because prefix-cached tokens skip the
forward pass entirely and the hooks would return activations with holes. Correct,
but for multi-turn transcript harvesting it re-prefills a ~95%-shared conversation
prefix once per turn (measured ×20.5 over the MisTrace corpus).

This patch adds an OPT-IN escape hatch on the offline (``LLM.generate``) path only:

- the request must pass ``extra_args={"vllm_lens_reuse_prefix_cache": True}``, AND
- the driver process must set ``VLLM_LENS_CAPTURE_PREFIX_REUSE=1``.

For such a request the skip is not forced: cached prefix positions are reused and
the capture covers ONLY the computed suffix. The caller maps rows to absolute
positions via ``RequestOutput.num_cached_tokens`` (see
``ActivationCapturer.capture_suffix`` in src/steering_reasoning). Everything else —
default captures, steering, the async/server path, other users of the package — is
byte-for-byte unaffected; with the env var unset the new branch is dead code.

An explicit per-request ``skip_reading_prefix_cache`` in extra_args still wins over
the reuse flag (matching upstream's stated contract).

Run inside the serving env:  uv run python patch_vllm_lens_prefix_reuse.py
"""

from pathlib import Path

import vllm_lens

MARK = "vllm_lens_reuse_prefix_cache"

ANCHOR_POP = """    # Pop skip_reading_prefix_cache from extra_args for each request.
    any_skip_kv_cache = False
    for sp in params_list:
        if (sp.extra_args or {}).pop("skip_reading_prefix_cache", None):
            any_skip_kv_cache = True
"""

INSERT_POP = """
    # [prefix-reuse patch] opt-in per-request cache reuse for suffix-only capture
    _reuse_flags = [
        bool((sp.extra_args or {}).pop("vllm_lens_reuse_prefix_cache", None))
        for sp in params_list
    ]
"""

ANCHOR_FORCE = """    if needs_hooks or any_skip_kv_cache:
        for sp in params_list:
            sp.skip_reading_prefix_cache = True
"""

REPLACE_FORCE = """    if needs_hooks or any_skip_kv_cache:
        import os as _os

        _reuse_ok = _os.environ.get("VLLM_LENS_CAPTURE_PREFIX_REUSE") == "1"
        for sp, _reuse in zip(params_list, _reuse_flags):
            if _reuse and _reuse_ok and not any_skip_kv_cache:
                # [prefix-reuse patch] suffix-only capture: cached prefix positions
                # are not re-computed; caller maps rows via num_cached_tokens
                continue
            sp.skip_reading_prefix_cache = True
"""


def main() -> None:
    p = Path(vllm_lens.__file__).parent / "_activations_plugin.py"
    t = p.read_text()
    if MARK in t:
        print("prefix-reuse patch present")
        return
    assert ANCHOR_POP in t, "pop anchor not found — vllm-lens layout changed, re-derive patch"
    assert ANCHOR_FORCE in t, "force anchor not found — vllm-lens layout changed, re-derive patch"
    t = t.replace(ANCHOR_POP, ANCHOR_POP + INSERT_POP)
    t = t.replace(ANCHOR_FORCE, REPLACE_FORCE)
    p.write_text(t)
    print("prefix-reuse patch applied")


if __name__ == "__main__":
    main()
