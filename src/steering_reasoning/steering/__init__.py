"""Steering-vector extraction and artifact handling.

NOTE: most modules here run inside the SERVING env (vllm + vllm-lens + transformers) —
they must not import inspect_ai, pandas, or anything else that only exists in the root
eval env. The exception is `clamp`, which is env-agnostic (lazy heavy imports) and is
re-exported here so `steering_reasoning.steering.build_clamp_payload` works everywhere.
"""

from steering_reasoning.steering.clamp import assistant_spans, history_boundary, build_clamp_payload, build_steer_payload, strip_think, with_positions

__all__ = ["assistant_spans", "build_clamp_payload", "build_steer_payload", "strip_think", "with_positions"]
