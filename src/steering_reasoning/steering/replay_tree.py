"""Per-token kind/message attribution for REPLAYED rollouts (ids supplied, not rendered).

The MisTrace tree builder (scripts/capture_all_tokens.py::build_tree)
re-renders each turn from parsed messages. A replay of a rollout we generated ourselves
starts from exact token ids instead — the prompt ids vLLM's own renderer produced for that
turn's request and the branch ids the model emitted (from the log, or re-rendered with a
recorded delta) — so this module only ATTRIBUTES: which row is template / message / reasoning
/ reply, and which rows a turn contributes to the tree.

Storage follows the same LCP keep-range rule as the 103 dump: turn k stores prompt positions
[keep_start, n) where keep_start is the longest common token prefix with the previous kept
turn. Every turn is still COMPUTED in full (no prefix reuse for steered replays); the keep
range is a storage decision only.

Branch attribution is by token id, which needs no text offsets: reasoning runs from the first
generated position to the `</think>` token; `</think>` is template; `\n\n` + reply text is
kind 6 until the first `<tool_call>` token, after which everything is kind 7; a trailing
`<|im_end|>` is template and is flagged (`im_end_rows`) because at generation it was sampled
but never fed back, so its activation row never existed then.

Context markup follows the 103-dump convention (capture_all.build_tree): only the tokenizer's
TRUE special ids (`<|im_end|>`, `<|endoftext|>`, pads) are forced to template; `<|im_start|>`,
`<think>`, `<tool_call>`, `<tool_response>` are added-but-not-special in Qwen3.5 and keep
their block's role kind, so arm dumps and the 103 dump attribute rows identically.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass, field

import numpy as np

from steering_reasoning.kinds import KIND, ROLE_KIND

IM_START = "<|im_start|>"
TOOL_RESPONSE = "<tool_response>"


@dataclass
class ReplayTurn:
    message_index: int          # index of this assistant message in the transcript
    prompt_ids: list[int]       # exact prompt token ids (vLLM renderer / logged)
    prompt_text: str            # the rendered prompt string (must tokenize to prompt_ids)
    branch_ids: list[int]       # generated token ids (logged, or re-rendered)
    messages: list[dict]        # wire-format request messages (role, ...) for this turn
    token_ids_source: str = "logged"   # "logged" | "rendered"
    branch_delta: int = 0       # branch_tokens - completion_tokens (0 when logged)
    steer_positions: list | None = None  # spans the proxy sent at generation (sidecar), if any
    ids_join: str = "none"      # how the sidecar joined: sidecar | sidecar_ambiguous | missing | length_mismatch | none
    flags: dict = field(default_factory=dict)


def _special_id(tokenizer, tok: str) -> int:
    i = tokenizer.convert_tokens_to_ids(tok)
    if i is None or i == getattr(tokenizer, "unk_token_id", None):
        raise ValueError(f"tokenizer has no single token for {tok!r}")
    return int(i)


def _context_bounds(prompt_text: str, messages: list[dict], task: str, mi: int):
    """(msg_bounds, bound_msg, hist_end_chars) for the history blocks of one prompt.

    Ported from capture_all.build_tree: one `<|im_start|>` block per message, consecutive
    tool messages merge into one block and are sub-split at `<tool_response>` markers.
    The final block is the generation prompt (`<|im_start|>assistant\\n<think>\\n`)."""
    starts = [mo.start() for mo in re.finditer(re.escape(IM_START), prompt_text)]
    if not starts:
        raise RuntimeError(f"{task} msg{mi}: no {IM_START} blocks in rendered prompt")
    hist_end_chars = starts[-1]
    block_starts = starts[:-1]
    groups: list[list[int]] = []
    for i, m in enumerate(messages):
        if groups and m["role"] == "tool" and messages[groups[-1][-1]]["role"] == "tool":
            groups[-1].append(i)
        else:
            groups.append([i])
    if len(block_starts) != len(groups):
        raise RuntimeError(f"{task} msg{mi}: {len(block_starts)} history blocks for "
                           f"{len(groups)} message groups")
    msg_bounds, bound_msg = [], []
    block_ends = block_starts[1:] + [hist_end_chars]
    for g, s, e in zip(groups, block_starts, block_ends):
        if len(g) == 1:
            msg_bounds.append(s); bound_msg.append(g[0])
        else:
            occ = [s + mo.start() for mo in
                   re.finditer(re.escape(TOOL_RESPONSE), prompt_text[s:e])]
            if len(occ) == len(g):
                msg_bounds.append(s); bound_msg.append(g[0])
                msg_bounds.extend(occ[1:]); bound_msg.extend(g[1:])
            else:  # coarse: whole block to the first tool msg (same kind)
                msg_bounds.append(s); bound_msg.append(g[0])
    return msg_bounds, bound_msg, hist_end_chars


def replay_tree(task: str, turns: list[ReplayTurn], tokenizer, *, share_branch: bool = False) -> dict:
    """Attribute every row of a replayed rollout.

    share_branch=False (steered replays): a turn's keep range never reaches into the previous
    turn's branch, so the generated (kinds 5-7) and re-read (kind 3) views of a reply are both
    stored (they are different forward passes under steering). share_branch=True (UNSTEERED
    prefix-reuse replays): keep_start is the plain LCP with the previous turn's sequence, the
    103-dump rule — a re-read reply is the same tokens after the same prefix, hence the same
    computation, and the prefix cache serves exactly those positions; storing them once keeps
    the dump complete by construction and lets the cache hit reach the keep range.

    Returns the same dict shape the MisTrace builder produces (per-turn ids/keep_start/row_start + flat kinds/msgs/ctxpos/tok_ids/
    turn_of), plus per-turn `gen_start`, `reasoning_start/end`, `resp_tok_start`.

    Asserts, per turn, that HF-tokenizing `prompt_text` reproduces `prompt_ids` exactly
    (this is what licenses using the text's char offsets for message attribution)."""
    think_close = _special_id(tokenizer, "</think>")
    tool_call = _special_id(tokenizer, "<tool_call>")
    im_end = _special_id(tokenizer, "<|im_end|>")
    special_ids = set(tokenizer.all_special_ids)

    per_turn, prev_ids, keep_total = [], [], 0
    kinds, msgs, ctxpos, tok_ids, turn_of = [], [], [], [], []
    im_end_rows = []
    # the LAST branch row of every turn: that token was sampled but never fed back at
    # generation, so its activation row never existed then (whatever the token is —
    # <|im_end|> normally, anything when max_tokens cut the turn). Replay computes it;
    # consumers should drop these rows for generation-faithful analyses.
    never_computed_rows = []
    for ti, t in enumerate(turns):
        enc = tokenizer(t.prompt_text, add_special_tokens=False, return_offsets_mapping=True)
        if list(enc["input_ids"]) != list(t.prompt_ids):
            raise RuntimeError(f"{task} msg{t.message_index}: HF tokenization of the rendered "
                               f"prompt differs from the prompt ids ({len(enc['input_ids'])} "
                               f"vs {len(t.prompt_ids)} tokens)")
        offsets = enc["offset_mapping"]
        gen_start = len(t.prompt_ids)
        ids = list(t.prompt_ids) + list(t.branch_ids)
        n = len(ids)
        branch = list(t.branch_ids)
        # branch structure by token id. A thinking-off prompt (template rendered
        # "<think>\n\n</think>\n\n" itself) has no reasoning span: the whole branch is reply.
        if t.prompt_text.endswith("</think>\n\n"):
            r_end, resp_start, think_row = gen_start, gen_start, None
        else:
            r_end = next((gen_start + j for j, x in enumerate(branch) if x == think_close), n)
            resp_start = min(r_end + 1, n)
            think_row = r_end if r_end < n else None
        tc_pos = next((gen_start + j for j, x in enumerate(branch)
                       if x == tool_call and gen_start + j >= resp_start), None)
        msg_bounds, bound_msg, hist_end = _context_bounds(t.prompt_text, t.messages, task,
                                                          t.message_index)
        m_ = min(len(prev_ids), n)
        if m_ == 0:
            lcp = 0
        else:
            neq = np.array(ids[:m_]) != np.array(prev_ids[:m_])
            lcp = int(neq.argmax()) if neq.any() else m_
        # The branch is always this turn's own rows. The PREVIOUS turn's branch rows are
        # never shared either: in-loop reasoning re-renders token-identically in the next
        # prompt, but generated (kinds 5-7, clamped under decode-only steering) and re-read
        # (kind 3: unsteered under decode-only generations, steered by the span rule under
        # model-tokens ones) are different forward passes with different activations, so
        # both views are stored. Without this cap a steered replay would silently keep only
        # the clamped copy.
        prev_gen = per_turn[-1]["gen_start"] if per_turn else gen_start
        lcp = min(lcp, gen_start) if share_branch else min(lcp, gen_start, prev_gen)
        per_turn.append({
            "turn": t, "ids": ids, "keep_start": lcp, "row_start": keep_total,
            "gen_start": gen_start, "reasoning_start": gen_start, "reasoning_end": r_end,
            "resp_tok_start": resp_start, "tool_tok_start": tc_pos,
            "n_prompt": gen_start, "n_branch": len(branch),
        })
        for pos in range(lcp, n):
            if pos < gen_start:
                s = offsets[pos][0]
                if s >= hist_end or not msg_bounds:
                    k, m = KIND["template"], t.message_index
                else:
                    j = max(0, min(bisect_right(msg_bounds, s) - 1, len(bound_msg) - 1))
                    m = bound_msg[j]
                    k = ROLE_KIND.get(t.messages[m]["role"], KIND["template"])
                if ids[pos] in special_ids:
                    k = KIND["template"]
            elif pos < r_end:
                k, m = KIND["reasoning"], t.message_index
            elif think_row is not None and pos == think_row:
                k, m = KIND["template"], t.message_index  # the </think> token
            elif tc_pos is not None and pos >= tc_pos:
                k, m = KIND["tool_call_gen"], t.message_index
            else:
                k, m = KIND["response_gen"], t.message_index
            if pos >= gen_start and ids[pos] == im_end:
                k = KIND["template"]
                im_end_rows.append(keep_total + (pos - lcp))
            kinds.append(k); msgs.append(m); ctxpos.append(pos)
            tok_ids.append(ids[pos]); turn_of.append(ti)
        if share_branch and per_turn[-1:] and never_computed_rows and lcp >= len(prev_ids) and ti > 0:
            # this turn re-reads the whole previous sequence, so the previous turn's last branch row
            # (sampled, never fed back THEN) was computed by THIS turn's prefill at generation; under
            # the shared tree it is stored once, in the previous turn, and is not never-computed
            never_computed_rows.pop()
        if branch:
            never_computed_rows.append(keep_total + (n - 1 - lcp))
        keep_total += n - lcp
        prev_ids = ids

    collapses = [{"turn_row": i, "from": per_turn[i - 1]["keep_start"],
                  "to": per_turn[i]["keep_start"]}
                 for i in range(1, len(per_turn))
                 if per_turn[i]["keep_start"] < per_turn[i - 1]["keep_start"]]
    return {
        "turns": per_turn, "n_tree": keep_total,
        "n_reasoning": sum(pt["reasoning_end"] - pt["reasoning_start"] for pt in per_turn),
        "keep_start_collapses": collapses, "im_end_rows": im_end_rows,
        "never_computed_rows": never_computed_rows,
        "max_prompt": max(len(pt["ids"]) for pt in per_turn),
        "max_delta": max(len(pt["ids"]) - pt["keep_start"] for pt in per_turn),
        "share_branch": share_branch,
        "kinds": kinds, "msgs": msgs, "ctxpos": ctxpos, "tok_ids": tok_ids, "turn_of": turn_of,
    }
