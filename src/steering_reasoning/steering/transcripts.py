"""Turn transcript records into per-turn capture prompts with reasoning-token spans.

Transcripts (MisTrace, data/reward_hacking/) store full agentic conversations where each
assistant message carries its reasoning as a content part of type "reasoning". We process
one assistant turn at a time: render the conversation up to that turn with ALL prior
reasoning dropped, append the turn's reasoning wrapped in <think> tags, and record which
token positions belong to the reasoning.

Dropping all prior reasoning is a documented approximation of the generation-time context,
not a reproduction of it. The Qwen3/Qwen3.5 chat templates drop reasoning only from
assistant messages at or before the last real user message; assistant messages after it
(the tool-call loop) are rendered with their <think> block whenever the client sends the
reasoning back (`reasoning_content`, or <think> tags in content). Inspect's OpenRouter
provider, which produced the MisTrace transcripts, does send it back (as
`reasoning_details`); whether the upstream provider applied it is not known. So in-loop
turns are captured with less context than the model likely had, and in-loop history turns
render as empty "<think>\n\n</think>" blocks.

Because attention is causal, tokens after </think> cannot affect reasoning-token
activations, so each capture prompt is truncated right after the reasoning block.

Known v0 approximations (documented, acceptable for mean-pooled DiffMean):
- token positions at concatenation boundaries can merge (off-by-one at span edges);
- tool-call formatting in history relies on the tokenizer's chat template, which may not
  byte-match the serving-time rendering of the original provider;
- the Qwen3.5 template's generation prompt already ends in "<think>\n" and THINK_OPEN adds
  a second, so capture prompts read "<|im_start|>assistant\n<think>\n<think>\n...". Spans
  are computed on the rendered string, so they stay correct. Existing captures were made
  this way; changing it would shift every prompt by one token.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

THINK_OPEN = "<think>\n"
THINK_CLOSE = "\n</think>"


def load_transcripts(path: str | Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _part_text(content: Any, kind: str) -> str:
    """Concatenate content parts of a given type ("text" or "reasoning")."""
    if isinstance(content, str):
        return content if kind == "text" else ""
    out = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == kind:
            out.append(part.get(kind, "") if kind == "reasoning" else part.get("text", ""))
    return "".join(out)


def _ensure_mapping(arguments) -> dict:
    if isinstance(arguments, dict):
        return arguments
    try:
        parsed = json.loads(arguments)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except (json.JSONDecodeError, TypeError):
        return {"value": str(arguments)}


def _to_chat_message(msg: dict) -> dict:
    """Convert a transcript message to an OpenAI-style dict for apply_chat_template.

    Reasoning parts are dropped from every assistant message, so history renders
    reasoning-stripped throughout. This is an approximation: the Qwen3.5 template keeps
    <think> blocks for assistant messages after the last real user message when the client
    round-trips them, so the generation-time context inside a tool loop likely included
    earlier in-loop reasoning (see the module docstring). Inspect-style tool_calls are
    converted to OpenAI format.
    """
    role = msg["role"]
    out: dict[str, Any] = {"role": role, "content": _part_text(msg.get("content"), "text")}
    if role == "assistant" and msg.get("tool_calls"):
        out["tool_calls"] = [
            {
                "id": tc.get("id", f"call_{i}"),
                "type": "function",
                "function": {
                    "name": tc.get("function", tc.get("name", "")),
                    # keep arguments as a mapping: Qwen3.5's chat template iterates it with
                    # .items() and rejects the OpenAI-style JSON string (Qwen3 accepts both)
                    "arguments": _ensure_mapping(tc.get("arguments", {})),
                },
            }
            for i, tc in enumerate(msg["tool_calls"])
        ]
    if role == "tool":
        out["tool_call_id"] = msg.get("tool_call_id", "")
    return out


@dataclass
class ReasoningTurn:
    """One assistant turn's capture prompt and its reasoning token span."""

    prompt: str
    reasoning_start: int  # token index of first reasoning token
    reasoning_end: int  # token index one past last reasoning token
    turn_index: int
    n_tokens: int = 0  # total prompt tokens (set during build)
    meta: dict = field(default_factory=dict)


def iter_reasoning_turns(
    record: dict,
    tokenizer,
    *,
    max_tokens: int | None = None,
    max_turns: int | None = None,
    dedupe_think_open: bool = False,
) -> list[ReasoningTurn]:
    """Build per-turn capture prompts for every assistant turn with non-empty reasoning.

    max_tokens skips turns whose prompt exceeds the serving context window.
    max_turns caps turns per transcript (front-loaded) to bound extraction cost.
    dedupe_think_open: Qwen3.5's chat template already ends the generation prompt
    with "<think>\\n"; the historical default here appends THINK_OPEN on top,
    giving every capture prompt a doubled think-open (off-policy by ~2 tokens vs
    what the model saw at generation). False keeps that legacy behavior — every
    existing dump/vector was built with it; True skips the extra append when the
    template already provides it (generation-faithful; used by capture_all).
    """
    messages = record["messages"]
    turns: list[ReasoningTurn] = []
    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        reasoning = _part_text(msg.get("content"), "reasoning").strip()
        if not reasoning:
            continue

        history = [_to_chat_message(m) for m in messages[:i]]
        context = tokenizer.apply_chat_template(
            history, tokenize=False, add_generation_prompt=True
        )
        if dedupe_think_open and context.endswith(THINK_OPEN):
            prefix = context
        else:
            prefix = context + THINK_OPEN
        prompt = prefix + reasoning + THINK_CLOSE

        # Single tokenization with char offsets instead of tokenizing prefix /
        # prefix+reasoning / prompt separately (3 full passes per turn dominated CPU
        # time on full-dataset captures). Indices are derived from the same token
        # sequence the engine will see, so boundary-merge tokens can no longer put
        # the span off-by-one relative to the captured activations.
        offsets = tokenizer(
            prompt, add_special_tokens=False, return_offsets_mapping=True
        )["offset_mapping"]
        n_total = len(offsets)
        if max_tokens is not None and n_total > max_tokens:
            continue
        a, b = len(prefix), len(prefix) + len(reasoning)
        n_prefix = next((t for t, (_, e) in enumerate(offsets) if e > a), n_total)
        n_with_reasoning = next(
            (t for t, (s, _) in enumerate(offsets) if s >= b), n_total
        )

        turns.append(
            ReasoningTurn(
                prompt=prompt,
                reasoning_start=n_prefix,
                reasoning_end=n_with_reasoning,
                turn_index=i,
                n_tokens=n_total,
                meta={"task_id": record.get("task_id"), "rollout_id": record.get("rollout_id")},
            )
        )
        if max_turns is not None and len(turns) >= max_turns:
            break
    return turns
