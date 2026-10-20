#!/usr/bin/env python
"""Idempotently patch an installed vllm-lens to skip zstd on the capture return.

Upstream ``HiddenStatesExtension.get_captured_states`` stacks all captured layers,
``pickle.dumps`` them, then ``ZstdCompressor(level=1).compress(...)`` the result before
returning the bytes over the collective_rpc / ZMQ path. On bf16 residual-stream captures
level-1 zstd buys almost nothing (~1.29x measured) yet dominates wall-clock: on a realistic
3 GiB payload (64 layers x 4915 rows x 5120 bf16) it costs ~2.77s to compress in the worker
and ~2.81s to decompress in the driver — ~5.6s of pure CPU per generate() call, on top of
the unavoidable ~4.4s of pickle.dumps + pickle.loads. Transport is per-call and blocks the
capture path synchronously, so this is the single biggest lever.

This patch adds an OPT-IN escape hatch on the worker return only:

- with ``VLLM_LENS_DISABLE_ZSTD=1`` in the worker environment, ``get_captured_states``
  returns the raw pickle bytes instead of the zstd frame.

No driver-side change is needed: ``_merge_captured_states`` already routes on the zstd
magic (``s[:4] == b"\\x28\\xb5\\x2f\\xfd"``) and falls back to ``pickle.loads(s)`` otherwise.
Default pickle bytes begin with ``0x80`` (protocol >=2), which never collides with the zstd
magic, so uncompressed frames are decoded correctly. With the env var unset the behavior is
byte-for-byte upstream (the new branch is dead code).

Correctness / message-size note: the ZMQ/msgspec bin payload cap is 2**32-1 (~4.0 GiB) and
applies to the *returned bytes* — today the compressed frame, after this patch the raw
pickle. capture_all_tokens.py splits so each sub-request holds <= PAYLOAD_CAP_BYTES = 3 GiB of RAW
activations; the compressed frame was therefore ALWAYS smaller than that 3 GiB cut, and the
raw pickle is ~3 GiB + a few hundred bytes of header — still comfortably under the 4.0 GiB
cap (~1 GiB headroom). Disabling zstd does not narrow the split margin; the split math was
already measured on raw bytes. The extra ~0.67 GiB/call on the wire (3.0 vs 2.33 GiB) costs
a fraction of a second on localhost ZMQ, far less than the ~5.6s of zstd it removes.

Run inside the serving env:  uv run python patch_vllm_lens_disable_zstd.py
"""

from pathlib import Path

import vllm_lens

MARK = "[no-zstd patch]"

ANCHOR_IMPORT = """import logging
import pickle
"""

INSERT_IMPORT = """import logging
import os as _os
import pickle
"""

ANCHOR_RETURN = """                return _ZSTD_COMPRESSOR.compress(
                    pickle.dumps(
                        {
                            "activations": {"residual_stream": stacked},
                        }
                    )
                )
"""

REPLACE_RETURN = """                _payload = pickle.dumps(
                    {
                        "activations": {"residual_stream": stacked},
                    }
                )
                # [no-zstd patch] level-1 zstd on bf16 activations buys only
                # ~1.29x but costs ~5.6s/call (compress+decompress) on a 3 GiB
                # payload. Skip it when VLLM_LENS_DISABLE_ZSTD=1: the driver's
                # _merge_captured_states routes on the zstd magic bytes, so raw
                # pickle (magic 0x80...) is loaded directly with no driver change.
                if _os.environ.get("VLLM_LENS_DISABLE_ZSTD") == "1":
                    return _payload
                return _ZSTD_COMPRESSOR.compress(_payload)
"""


def main() -> None:
    p = Path(vllm_lens.__file__).parent / "_worker_ext.py"
    t = p.read_text()
    if MARK in t:
        print("no-zstd patch present")
        return
    assert ANCHOR_RETURN in t, "return anchor not found — vllm-lens layout changed, re-derive patch"
    # The modes patch may already have added `import os as _os`; only add it if absent.
    if "import os as _os" not in t:
        assert ANCHOR_IMPORT in t, "import anchor not found — vllm-lens layout changed"
        t = t.replace(ANCHOR_IMPORT, INSERT_IMPORT, 1)
    t = t.replace(ANCHOR_RETURN, REPLACE_RETURN)
    p.unlink(); p.write_text(t)   # replace, don't truncate: uv hardlinks the venv to its wheel cache
    print("no-zstd patch applied")


if __name__ == "__main__":
    main()
