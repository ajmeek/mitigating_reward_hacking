"""Token-kind enum for per-token activation dumps — the single source of truth.

Every row of a per-layer shard (`L{ll}/<task>.safetensors`, see
experiments/full-token-capture/DATASET.md) carries a kind: what the token was to the model
that computed it. Kinds 0-4 are context the model read (template markup, messages, tool
output, and its own earlier replies as re-read in later history); 5-7 are what it produced
(reasoning, then the reply text and tool-call block as generated).

Torch-free so the root eval env can import it.
"""

KIND = {
    "template": 0,       # chat markup incl. special tokens, think tags, generation suffix
    "system": 1,
    "user": 2,
    "assistant": 3,      # assistant reply as RE-READ in later history (reasoning-stripped)
    "tool": 4,           # tool output
    "reasoning": 5,
    "response_gen": 6,   # reply text AS GENERATED (incl. the "\n\n" after </think>)
    "tool_call_gen": 7,  # tool-call block AS GENERATED
}
KIND_NAMES = {
    0: "template",
    1: "system",
    2: "user",
    3: "assistant_reread",
    4: "tool_output",
    5: "reasoning",
    6: "response_gen",
    7: "toolcall_gen",
}
ROLE_KIND = {"system": 1, "user": 2, "assistant": 3, "tool": 4}
KIND_REASONING = KIND["reasoning"]
GENERATED_KINDS = (KIND["reasoning"], KIND["response_gen"], KIND["tool_call_gen"])
