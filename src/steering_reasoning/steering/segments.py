"""Map gold span-label segments to token spans inside per-turn capture prompts.

Gold-tagged transcripts (has_gold_spans) carry span_labels.segments: sentence-level
segments per section ("msg<N>:reasoning" etc.) with gold_label marking sentences that
reference reward hacking. Segment texts concatenate in order within each section
(verified: 100% in-order substring match on all gold transcripts), so we locate each
segment by an in-order find() over the message's reasoning text, then convert char spans
to token spans with the fast tokenizer's offset_mapping on the full capture prompt.

Token spans inherit the pipeline's documented ±1-token boundary tolerance (merges at
concatenation points); negligible under mean pooling.
"""

import re
from dataclasses import dataclass

_SECTION_RE = re.compile(r"msg(\d+):reasoning$")


@dataclass
class SegmentSpan:
    """One reasoning segment located within a turn's capture prompt."""

    message_index: int  # transcript message index (msgN)
    segment_idx: int  # index within its section
    position: int  # global segment position from the gold data
    gold_label: bool
    judge_label: bool | None
    char_start: int  # within the reasoning string
    char_end: int
    token_start: int  # index into THIS turn's capture prompt — NOT a global transcript
    token_end: int    # coordinate; prompts are not append-only across turns, so these can
                      # regress between messages. For cross-turn distances rebuild a
                      # monotone coordinate from cumulative turn n_tokens (see
                      # experiments/archive/unverbalized-hack-signal/scripts/segment_token_horizon.py).
    text: str


_ENUMERATION = re.compile(r"^\d+[.)]$")
_BARE_OPENER = re.compile(r"^(else|try|finally|do)\s*:$")
_FENCE = re.compile(r"^```\w*$")
# Colon-enders up to this length are lead-ins ("The problem says:", "Looking at line 66:");
# longer ones are usually real sentences that happen to introduce a formula, and merging
# them dilutes the unit (detection lift drops 41x -> 39x when the cap is raised to 120).
_MAX_INTRODUCER_CHARS = 60
_MAX_CHAIN = 4  # cap consecutive introducers so a run can't swallow a message

# Used only to close an unclosed code fence: keywords, comments, assignments, indented
# continuations, and bracket/brace lines. Deliberately conservative — a false negative
# just ends the block early, a false positive absorbs prose into it.
_CODE_LINE = re.compile(
    r"^(#|\}|\)|\]|"
    r"(def|class|return|import|from|global|nonlocal|if|elif|else|for|while|try|except|"
    r"finally|with|assert|raise|yield|pass|break|continue|lambda|print)\b|"
    r"[A-Za-z_][\w.\[\]'\"]*\s*(=[^=]|[-+*/|&^]=|\.append\(|\.add\(|\()|"
    r"\s{2,}\S)"
)


def _looks_like_code(text: str) -> bool:
    return bool(_CODE_LINE.match(text))


def _is_introducer(text: str) -> bool:
    """A fragment that exists to introduce what follows, not to stand alone.

    Covers enumeration markers ("1.", "2)"), bare block openers ("else:", "try:"),
    stubs of one or two characters, and colon-terminated lead-ins ("The problem says:",
    "Looking at line 66:", "What if:") — 5.7% of reasoning segments are the last kind.
    """
    return bool(
        _ENUMERATION.match(text)
        or _BARE_OPENER.match(text)
        or len(text) <= 2
        or (text.endswith(":") and len(text) <= _MAX_INTRODUCER_CHARS)
    )


def regroup_segments(segments: list[dict]) -> list[list[int]]:
    """Group a single message's segments into semantically whole units.

    The upstream segmenter splits on sentence boundaries after splitting on lines, which
    produces two kinds of unit that don't stand alone:

    - a fenced code block becomes one segment per line (```python, `x = {}`, `def f(...)`,
      ...), so a single hack implementation is scattered across a dozen units;
    - enumeration markers ("1.", "2)") and bare block openers ("else:", "try:") are split
      off as their own segments because they end in a period or colon.

    Returns index groups (in order) merging fence-delimited code into one unit and
    absorbing degenerate fragments into the segment that follows them. Everything else is
    left alone — ordinary prose sentences stay as they are.
    """
    groups: list[list[int]] = []
    pending: list[int] = []  # degenerate fragments awaiting the unit they introduce
    i = 0
    while i < len(segments):
        text = segments[i]["text"].strip()
        if _FENCE.match(text):
            block = pending + [i]
            j = i + 1
            closed = False
            while j < len(segments):  # run to the closing fence, inclusive
                block.append(j)
                if _FENCE.match(segments[j]["text"].strip()):
                    closed = True
                    j += 1
                    break
                j += 1
            if closed:
                pending = []
                groups.append(block)
                i = j
                continue
            # Unclosed fence (3 of 1593 messages): the model opened a code block and never
            # closed it, typically drifting back into prose mid-block. Merging to the end
            # of the message would swallow it whole (one case ran to 987 segments), and
            # fixed-size blocks would cut arbitrarily. Instead close the block where the
            # content stops looking like code.
            block = pending + [i]
            pending = []
            j = i + 1
            while j < len(segments) and _looks_like_code(segments[j]["text"].strip()):
                block.append(j)
                j += 1
            groups.append(block)
            i = j
            continue
        if _is_introducer(text) and len(pending) < _MAX_CHAIN:
            pending.append(i)
            i += 1
            continue
        groups.append(pending + [i])
        pending = []
        i += 1
    if pending:  # trailing fragments with nothing to attach to
        groups.append(pending) if not groups else groups[-1].extend(pending)
    return groups


def reasoning_segments_by_message(record: dict) -> dict[int, list[dict]]:
    """Group a gold record's reasoning-kind segments by message index, in order."""
    out: dict[int, list[dict]] = {}
    span_labels = record.get("span_labels") or {}
    for seg in span_labels.get("segments", []):
        m = _SECTION_RE.match(seg.get("section", ""))
        if m:
            out.setdefault(int(m.group(1)), []).append(seg)
    return out


def locate_segments(reasoning: str, segments: list[dict]) -> list[tuple[dict, int, int]]:
    """In-order find of each segment's text in the reasoning string → (seg, start, end).

    Raises ValueError if any segment text cannot be found at or after the cursor —
    that would mean the segmentation doesn't derive from this text and the transcript
    should be skipped loudly, not silently mis-mapped.
    """
    located = []
    cursor = 0
    for seg in segments:
        text = seg["text"]
        idx = reasoning.find(text, cursor)
        if idx < 0:
            raise ValueError(
                f"segment not found in-order in reasoning text: {text[:80]!r}"
            )
        located.append((seg, idx, idx + len(text)))
        cursor = idx + len(text)
    return located


def char_to_token_spans(
    prompt: str,
    prefix_len_chars: int,
    located: list[tuple[dict, int, int]],
    tokenizer,
) -> list[tuple[dict, int, int, int, int]]:
    """Convert reasoning-relative char spans to prompt-absolute token spans.

    Tokenizes the prompt once with offset_mapping (requires a fast tokenizer) and
    selects, per segment, the tokens whose char extent overlaps the segment's chars.
    Returns (seg, char_start, char_end, token_start, token_end) tuples; token_end is
    exclusive. Zero-token segments (fully merged into a neighbor) are dropped.
    """
    enc = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    out = []
    ti = 0
    for seg, cs, ce in located:
        a, b = prefix_len_chars + cs, prefix_len_chars + ce
        # offsets are monotonically non-decreasing; advance a shared cursor
        while ti < len(offsets) and offsets[ti][1] <= a:
            ti += 1
        t_start = ti
        tj = ti
        while tj < len(offsets) and offsets[tj][0] < b:
            tj += 1
        if tj > t_start:
            out.append((seg, cs, ce, t_start, tj))
        # next segment may share a boundary token; back cursor up one to allow overlap
        ti = max(t_start, tj - 1)
    return out
