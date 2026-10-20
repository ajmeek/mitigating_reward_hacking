"""Residual-stream activation capture via vllm-lens (offline vLLM API).

vllm-lens patches LLM.generate: requests with extra_args={"output_residual_stream":
[layer, ...]} get PyTorch hooks installed and return activations on output.activations
["residual_stream"] with shape (n_captured_layers, n_positions, d_model), layers stacked in
ascending order. The plugin forces skip_reading_prefix_cache for hooked requests, so cached
prefixes cannot hole the capture. Hooks require enforce_eager=True.
"""

import os
import shutil
from pathlib import Path

import torch


def apply_no_nvcc_workarounds() -> None:
    """On machines without a CUDA toolkit, steer vLLM away from paths that JIT with nvcc.

    vLLM ≥0.24 JIT-compiles flashinfer's sampler at engine start. setdefault only — cloud
    images ship nvcc and skip this entirely, and explicit env vars always win.
    """
    if shutil.which("nvcc") or Path("/usr/local/cuda").exists():
        return
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    os.environ.setdefault("VLLM_FLASH_ATTN_VERSION", "2")


def _worker_source() -> str:
    from pathlib import Path

    import vllm_lens

    return (Path(vllm_lens.__file__).parent / "_worker_ext.py").read_text()


def _worker_has_positions_gate() -> bool:
    """The positions-aware decode-only gate (serving/patch_vllm_lens_gate.py) is installed."""
    return 'getattr(_cfg, "steer_positions"' in _worker_source()


class ActivationCapturer:
    """Thin wrapper around an offline vLLM engine with vllm-lens capture."""

    def __init__(
        self,
        model: str,
        *,
        max_model_len: int = 14336,
        gpu_memory_utilization: float = 0.85,
        tensor_parallel_size: int = 1,
        enable_prefix_caching: bool = False,
        enable_chunked_prefill: bool | None = None,
        max_num_batched_tokens: int | None = None,
    ):
        import vllm_lens  # noqa: F401  (import registers the plugin's LLM.generate patch)
        from vllm import LLM

        # Steered captures run under the positions-aware gate (VLLM_LENS_STEER_DECODE_ONLY=1,
        # serving/patch_vllm_lens_gate.py), which the WORKER reads from the environment it
        # inherited at spawn: it must be set before the engine is built. Unsteered captures
        # are unaffected by it either way. `steering_objects` refuses a steered capture that
        # the gate would silently leave unsteered.
        self.gate_env = os.environ.get("VLLM_LENS_STEER_DECODE_ONLY") == "1"
        self.model_name = model
        extra = {}
        if enable_chunked_prefill is not None:
            extra["enable_chunked_prefill"] = enable_chunked_prefill
        if max_num_batched_tokens is not None:
            extra["max_num_batched_tokens"] = max_num_batched_tokens
        self.llm = LLM(
            model=model,
            dtype="bfloat16",
            enforce_eager=True,  # hooks don't fire under torch.compile
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            # False for plain capture (vllm-lens forces a fresh prefill anyway);
            # True only for the suffix-capture path (capture_suffix), which needs
            # the prefix-reuse patch (serving/patch_vllm_lens_prefix_reuse.py)
            enable_prefix_caching=enable_prefix_caching,
            tensor_parallel_size=tensor_parallel_size,
            **extra,
        )
        self.prefix_caching_enabled = enable_prefix_caching

    @property
    def d_model(self) -> int:
        return int(self.llm.llm_engine.model_config.get_hidden_size())

    @property
    def n_layers(self) -> int:
        # Qwen3.5 checkpoints are *ForConditionalGeneration wrappers: the decoder config is
        # the nested text config (vLLM resolves it as hf_text_config)
        mc = self.llm.llm_engine.model_config
        return int(getattr(mc, "hf_text_config", mc.hf_config).num_hidden_layers)

    def assert_serving_matches(self, expected: dict) -> list[str]:
        """Hard-fail unless this engine matches the generation config of the rollouts being
        replayed — model, dtype, chat-template sha, required local patches, the gate env when
        `gate` is expected. Differences that only move numerics within rounding (tensor
        parallelism, GPU) come back as warnings for the manifest. `expected` keys: model,
        dtype, template_sha256, require_patches, gate, tensor_parallel_size, gpu (all optional)."""
        from steering_reasoning.steering.shards import serving_provenance

        prov = serving_provenance(self, layer_chunk_cap=0, prefix_reuse=False,
                                  tokenizer=self.llm.get_tokenizer())
        errs, warns = [], []
        for k in ("model", "dtype"):
            if expected.get(k) is not None and expected[k] != prov.get(k):
                errs.append(f"{k}: engine {prov.get(k)!r} != generation {expected[k]!r}")
        exp_sha, got_sha = expected.get("template_sha256"), prov.get("template_sha256") or ""
        if exp_sha and not (got_sha.startswith(exp_sha) or exp_sha.startswith(got_sha)):  # prefix-tolerant
            errs.append(f"template_sha256: engine {got_sha!r} != generation {exp_sha!r}")
        for p in expected.get("require_patches", ()):
            if not prov["patches"].get(p):
                errs.append(f"required vllm-lens patch missing: {p}")
        if expected.get("gate") and not self.gate_env:
            errs.append("steered replay but VLLM_LENS_STEER_DECODE_ONLY was not 1 when the engine was built")
        for k in ("tensor_parallel_size", "gpu"):
            if expected.get(k) is not None and expected[k] != prov.get(k):
                warns.append(f"{k}: replay {prov.get(k)!r} vs generation {expected[k]!r} "
                             "(same geometry up to floating-point rounding)")
        if errs:
            raise RuntimeError("serving config does not match generation:\n  " + "\n  ".join(errs))
        return warns

    @staticmethod
    def steering_objects(payload) -> list:
        """Turn a steering payload (the JSON string build_steer_payload returns, or a list
        of dicts) into validated vllm_lens SteeringVector objects for the OFFLINE path,
        which pickles objects as-is (the server path validates JSON per request).

        A steered capture is a prefill under the positions-aware gate with the payload's
        `steer_positions` (the rows generation steered): refuses whatever would capture an
        UNSTEERED replay that looks steered — pydantic silently drops unknown keys and the
        gate skips a prefill with no positions — so the schema field, the gate patch, the
        gate env and a non-empty single-vector span list are all required.
        """
        import json
        import os

        from vllm_lens._helpers.types import SteeringVector

        dicts = json.loads(payload) if isinstance(payload, str) else list(payload)
        if len(dicts) != 1:
            raise RuntimeError("a steered capture takes exactly one steering vector per payload "
                               "(the gate builds one mask per request)")
        if not dicts[0].get("steer_positions"):
            raise RuntimeError("steered capture without steer_positions: the gate would skip the whole "
                               "prefill (unsteered capture)")
        if "steer_positions" not in SteeringVector.model_fields:
            raise RuntimeError("the SteeringVector schema lacks steer_positions — run serving/patch_vllm_lens_modes.py")
        if not _worker_has_positions_gate():
            raise RuntimeError("the worker lacks the positions-aware gate — run serving/patch_vllm_lens_gate.py")
        if os.environ.get("VLLM_LENS_STEER_DECODE_ONLY") != "1":
            raise RuntimeError("VLLM_LENS_STEER_DECODE_ONLY is not 1: without the gate every prefill row would be steered")
        return [SteeringVector.model_validate(d) for d in dicts]

    def capture(self, prompts: list, layers: list[int], *, token_ids: bool = False,
                steering=None) -> list[torch.Tensor]:
        """Prefill each prompt and return residual streams.

        Returns one tensor per prompt, shape (len(layers), n_prompt_tokens, d_model),
        layer order ascending (matching sorted(layers)). ``token_ids=True`` passes lists
        of ids as TokensPrompts (no retokenization). ``steering`` (payload JSON string or
        list of dicts, see build_clamp_payload) applies steering during the prefill; the
        captured residual stream is the POST-steering one at and after the steered layer.
        """
        from vllm import SamplingParams

        extra = {"output_residual_stream": sorted(layers)}
        if steering is not None:
            extra["apply_steering_vectors"] = self.steering_objects(steering)
        params = [SamplingParams(max_tokens=1, temperature=0.0, extra_args=dict(extra))
                  for _ in prompts]
        gen_prompts = ([{"prompt_token_ids": list(p)} for p in prompts] if token_ids else prompts)
        outputs = self.llm.generate(gen_prompts, params)
        results = []
        for out in outputs:
            acts = getattr(out, "activations", None)
            if acts is None or "residual_stream" not in acts:
                raise RuntimeError(
                    f"No activations captured for request {out.request_id} — "
                    "is vllm-lens installed in this env and enforce_eager on?"
                )
            n_prompt = len(out.prompt_token_ids)
            # capture includes positions for the 1 generated token's forward pass;
            # keep prompt positions only
            results.append(acts["residual_stream"][:, :n_prompt, :])
        return results

    def capture_suffix(
        self, prompts: list, layers: list[int], *, token_ids: bool = False
    ) -> list[tuple[torch.Tensor, int]]:
        """Prefill with prefix-cache reuse; capture only the computed suffix.

        ``prompts`` are strings by default; with ``token_ids=True`` they are lists
        of ints, passed as vLLM TokensPrompts. Token-id prompts skip vLLM's
        retokenization and are exactly the ids given — essential for split
        sub-requests, where a string cut could BPE-remerge at the boundary and a
        char-level safe-cut search costs O(prompt) tokenization per candidate.

        Returns one ``(acts, first_pos)`` per prompt: ``acts`` has shape
        ``(len(layers), n_prompt - first_pos, d_model)`` covering absolute prompt
        positions ``[first_pos, n_prompt)``, where ``first_pos`` is the number of
        prefix tokens served from cache (0 when nothing was cached — then this is
        exactly ``capture``'s output).

        Requires all three of: the engine built with ``enable_prefix_caching=True``,
        the prefix-reuse patch applied to vllm-lens
        (``serving/patch_vllm_lens_prefix_reuse.py``), and
        ``VLLM_LENS_CAPTURE_PREFIX_REUSE=1`` in the environment. All three are
        checked here so a misconfigured box fails loudly instead of silently
        running at full-replay speed.

        Turns of one conversation are sent with batch 1 not for correctness —
        vLLM registers cache blocks at schedule time, so same-batch requests do
        deduplicate — but to keep each turn's ``first_pos`` predictable.
        """
        import os
        from pathlib import Path

        import vllm_lens
        from vllm import SamplingParams

        if os.environ.get("VLLM_LENS_CAPTURE_PREFIX_REUSE") != "1":
            raise RuntimeError(
                "capture_suffix called without VLLM_LENS_CAPTURE_PREFIX_REUSE=1 — "
                "set it (and apply patch_vllm_lens_prefix_reuse.py) or use capture()"
            )
        if not self.prefix_caching_enabled:
            raise RuntimeError(
                "capture_suffix needs ActivationCapturer(enable_prefix_caching=True)"
            )
        plugin = Path(vllm_lens.__file__).parent / "_activations_plugin.py"
        if "vllm_lens_reuse_prefix_cache" not in plugin.read_text():
            raise RuntimeError(
                f"prefix-reuse patch not applied to {plugin} — run "
                "serving/patch_vllm_lens_prefix_reuse.py (otherwise this would "
                "silently degrade to full-replay capture)"
            )
        params = [
            SamplingParams(
                max_tokens=1,
                temperature=0.0,
                extra_args={
                    "output_residual_stream": sorted(layers),
                    "vllm_lens_reuse_prefix_cache": True,
                },
            )
            for _ in prompts
        ]
        gen_prompts = ([{"prompt_token_ids": list(p)} for p in prompts]
                       if token_ids else prompts)
        outputs = self.llm.generate(gen_prompts, params)
        results = []
        for out in outputs:
            acts = getattr(out, "activations", None)
            if acts is None or "residual_stream" not in acts:
                raise RuntimeError(
                    f"No activations captured for request {out.request_id} — "
                    "is vllm-lens installed in this env and enforce_eager on?"
                )
            n_prompt = len(out.prompt_token_ids)
            n_cached = getattr(out, "num_cached_tokens", None)
            if n_cached is None:
                raise RuntimeError(
                    "RequestOutput.num_cached_tokens missing — cannot map "
                    "suffix-capture rows to absolute positions on this vLLM version"
                )
            n_cached = int(n_cached)
            rs = acts["residual_stream"]
            expected = n_prompt - n_cached
            # Strict equality: with max_tokens=1 vLLM 0.24's scheduler never runs
            # a surplus pass (verified in review against scheduler.py), and a
            # looser bound would let a mid-prefill preemption slip misaligned rows
            # through silently. Any mismatch means position bookkeeping is broken
            # and the data must not be used.
            if rs.shape[1] != expected:
                raise RuntimeError(
                    f"suffix capture shape mismatch for {out.request_id}: "
                    f"{rs.shape[1]} rows, expected exactly {expected} "
                    f"(n_prompt={n_prompt}, num_cached_tokens={n_cached})"
                )
            results.append((rs, n_cached))
        return results

    @property
    def block_size(self) -> int:
        """Prefix-cache block granularity (cache hits are block-aligned)."""
        eng = self.llm.llm_engine
        cc = getattr(eng, "cache_config", None) or eng.vllm_config.cache_config
        return int(cc.block_size)

    def warm_prefix(self, token_ids: list[int]) -> int:
        """Prefill a token-id prefix with NO capture hook, purely to populate the
        prefix cache, then confirm it landed. Used before splitting an oversized
        turn: warming [0,keep_start) with no activation payload (no msgspec/pickle/
        transport, no 4 GiB-cap risk) lets the subsequent capture_suffix subs hit
        the cache and compute only [keep_start, n) — replacing the dozens of
        cache-warming capture RPCs the from-0 split issued per big turn.

        Returns num_cached_tokens of a follow-up hit over the same prefix, so the
        caller can REFUSE to issue the (payload-bearing, cap-risking) first capture
        sub if prefix reuse is silently broken — converting an engine-killing >4 GiB
        transport into a clean raise. Both generates carry no activations, so
        neither can hit the cap regardless of prefix length.
        """
        from vllm import SamplingParams

        p = [{"prompt_token_ids": list(token_ids)}]
        sp = [SamplingParams(max_tokens=1, temperature=0.0)]
        self.llm.generate(p, sp)                 # warm: prefill + cache the prefix
        out = self.llm.generate(p, sp)[0]        # probe: should be a full cache hit
        return int(getattr(out, "num_cached_tokens", 0) or 0)

    def reset_prefix_cache(self) -> None:
        """Drop all cached prefix blocks (between layer-chunk passes: a repeated
        prompt would otherwise be a full cache hit and capture nothing)."""
        if not self.llm.reset_prefix_cache():
            # returns False when blocks are still referenced; proceeding would
            # make every turn a near-total cache hit → mass fallbacks, ~20x slower
            raise RuntimeError("reset_prefix_cache failed - blocks still referenced")
