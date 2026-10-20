"""Transcript-reproduction constants.

Single source of truth for the run parameters that produced the reward-hacking
transcripts in data/reward_hacking/ (see their generation_config blocks).
Steering-side code should import from here rather than redefining.
"""

# Sampling parameters used for the original transcript generation runs.
GEN_DEFAULTS = {
    "temperature": 0.7,
    "top_p": 0.95,
    "seed": 42,
}

MAX_ATTEMPTS = 10
MESSAGE_LIMIT = 120

# NOT a transcript-reproduction value: the original MisTrace runs deliberately unset
# max_tokens. Cap per-turn
# generation at 16k = 1.2x the p99 of natural turns, clipping only degenerate loops.
# 16384 (not 16000) is the community-convention power of two for reasoning-model caps;
# the Phase-3 pilot arms ran 16000 (2.4% smaller — treat as equivalent when comparing).
# Enforced per-request by steering_reasoning.proxy (Inspect's static max_tokens can't
# express min(cap, window - prompt)).
MAX_TOKENS = 16384

# All three HF splits have the same 103 task ids, lcbhard_0..lcbhard_102.
N_TASKS = 103
