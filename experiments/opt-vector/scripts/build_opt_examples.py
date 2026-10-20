#!/usr/bin/env python
"""Build src-only training examples for the optimized steering vector.

One row per judge-positive REASONING segment ("hack sentence") in the TRAIN split.
The prompt stops EXACTLY where that sentence begins; src_completions holds the sentence
itself, which the trainer suppresses via -log(1-p). dst_completions is empty by design
(v1 is src-only: the only honest-decline label set is regex-built and the repo itself
says to treat it as a weak prior, not ground truth).

Differences from experiments/prefill-intervention/scripts/build_prefills.py, which this is
modelled on:
  * every hack sentence, not just the first one per rollout;
  * the message history is TRUNCATED to ONE user message (full history is ~38k tokens
    median and will not fit a training batch). This is a SPLICE, not a prefix: the target
    assistant turn is never the message right after that user message, so the system
    prompt and every intervening assistant/tool turn are deleted (median 8 messages, max
    50). `gap` in each row's meta records how many were dropped, so the train/eval
    distribution shift is measurable rather than assumed;
  * segment offsets come from steering.segments.locate_segments (an in-order walk that
    fails loudly), NOT str.find -- 164/3405 segment texts occur more than once inside
    their own message's reasoning, so a bare find() returns the wrong occurrence;
  * all 8 prefill-intervention grid contexts are dropped, including the 4 that are in
    TRAIN (lcbhard_0, 2, 51, 66): fitting on them would leak into the eval this vector
    is built to be judged by.

The reasoning prefix is sliced out of the ORIGINAL reasoning string, never re-joined from
segment texts (segment texts do not carry inter-sentence whitespace).
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from steering_reasoning.data.mistrace import load_mistrace  # noqa: E402
from steering_reasoning.steering.segments import (  # noqa: E402
    locate_segments,
    reasoning_segments_by_message,
)

SPLITS = REPO / "src/steering_reasoning/data/splits.json"
OUT = REPO / "experiments/opt-vector/data/rh_src_train.jsonl"
# Every context in the prefill-intervention grid is excluded, not just the four that
# hack at baseline. Four of the eight are in TRAIN (lcbhard_0, 2, 51, 66), so fitting on
# them would leak into the evaluation this vector is built to be judged by. The findings
# table reports an "all 8 tasks" column alongside the 4-that-hack column, and the full
# prefill set is 34 contexts, so excluding the whole grid keeps both usable.
# Costs 42 rows relative to the 1-task exclusion (232 -> 190).
LEAK_EXCLUDE = {
    "lcbhard_72", "lcbhard_2", "lcbhard_66", "lcbhard_63",
    "lcbhard_41", "lcbhard_0", "lcbhard_33", "lcbhard_51",
}
TOKENIZER = "Qwen/Qwen3.5-27B"
THINK_OPEN = "<think>"


def blocks_to_parts(content):
    """MisTrace assistant content is a list of typed blocks (same cut as build_prefills)."""
    reasoning, text = [], []
    if isinstance(content, list):
        for b in content:
            if not isinstance(b, dict):
                text.append(str(b))
                continue
            if b.get("type") == "reasoning":
                reasoning.append(b.get("reasoning") or "")
            elif b.get("type") == "text":
                text.append(b.get("text") or "")
    elif isinstance(content, str):
        text.append(content)
    return "\n".join(x for x in reasoning if x), "\n".join(x for x in text if x)


def task_instruction(messages, upto, mode):
    """The user message carrying the task instruction for the turn at index `upto`.

    'last': the most recent user message before that turn (retry turns restate the task
    verbatim plus the failing-test output -- that is what the model actually had in front
    of it). 'first': always the opening instruction.
    """
    idxs = [i for i, m in enumerate(messages) if m.get("role") == "user" and i < upto]
    if not idxs:
        return None, None
    i = idxs[0] if mode == "first" else idxs[-1]
    _, text = blocks_to_parts(messages[i].get("content"))
    return (text, i) if text else (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-prompt-tokens", type=int, default=4096)
    ap.add_argument("--user-message", choices=["last", "first"], default="last")
    ap.add_argument("--tokenizer", default=TOKENIZER)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--min-completion-tokens", type=int, default=0,
                    help="drop rows whose hack sentence is shorter than this many tokens; "
                         "0 keeps everything (the unconfounded primary set). 21/190 rows "
                         "are <=3 tokens, incl. 5 bare ``` fences and 7 list ordinals.")
    ap.add_argument("--drop-duplicate-prompts", action="store_true",
                    help="keep only the first row of each identical prompt. Collisions are "
                         "ALWAYS printed; this flag decides whether they are written.")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    splits = json.loads(SPLITS.read_text())
    keep = [t for t in splits["train"] if t not in LEAK_EXCLUDE]
    revision = splits.get("mistrace_revision")
    print(f"train task_ids: {len(splits['train'])} -> {len(keep)} after excluding {sorted(LEAK_EXCLUDE)}")

    # The parquet itself is cached and hf_hub_download resolves an explicit SHA, but
    # load_mistrace's provenance call (HfApi().dataset_info) hits the API and the shared
    # IP gets 429-rate-limited. Retry rather than silently dropping the revision assert.
    records = prov = None
    for attempt in range(8):
        try:
            records, prov = load_mistrace("qwen3_5_27b", revision=revision)
            break
        except Exception as exc:  # noqa: BLE001
            if "429" not in str(exc) and "rate limit" not in str(exc).lower():
                raise
            wait = 35
            print(f"  hub rate-limited (attempt {attempt + 1}/8); retrying in {wait}s")
            time.sleep(wait)
    if records is None:
        raise SystemExit("could not reach the hub to resolve the pinned MisTrace revision")
    print(f"mistrace: {prov['n_records']} records @ {prov['revision']}")
    assert prov["revision"] == revision, f"revision mismatch: {prov['revision']} != {revision}"

    keep_set = set(keep)
    skips = {}
    oversize = {}

    def skip(reason, n=1):
        skips[reason] = skips.get(reason, 0) + n

    rows = []
    for rec in records:
        if rec.get("task_id") not in keep_set:
            continue
        if not rec.get("span_labels"):
            skip("rollout: no span_labels")
            continue
        by_msg = reasoning_segments_by_message(rec)
        if not by_msg:
            skip("rollout: no reasoning segments")
            continue
        messages = rec["messages"]
        for mi, segs in sorted(by_msg.items()):
            n_pos = sum(1 for s in segs if s.get("judge_label") and s.get("kind") == "reasoning")
            if not n_pos:
                continue
            reasoning, _ = blocks_to_parts(messages[mi].get("content"))
            if not reasoning:
                skip("message: empty reasoning", n_pos)
                continue
            try:
                # locate ALL segments of the message so the in-order cursor is correct
                located = locate_segments(reasoning, segs)
            except ValueError as exc:
                print(f"  skip {rec['task_id']} msg{mi}: {exc}")
                skip("message: locate_segments failed", n_pos)
                continue
            instr, ui = task_instruction(messages, mi, args.user_message)
            if instr is None:
                skip("message: no preceding user message", n_pos)
                continue
            base = tok.apply_chat_template(
                [{"role": "user", "content": instr}], tokenize=False, add_generation_prompt=True
            )
            if not base.rstrip().endswith(THINK_OPEN):
                base = base + THINK_OPEN + "\n"
            for seg, cs, _ce in located:
                if not (seg.get("judge_label") and seg.get("kind") == "reasoning"):
                    continue
                prompt = base + reasoning[:cs]
                completion = seg["text"]
                n_prompt = len(tok(prompt, add_special_tokens=False).input_ids)
                n_comp = len(tok(completion, add_special_tokens=False).input_ids)
                if n_prompt > args.max_prompt_tokens:
                    skip(f"segment: prompt > {args.max_prompt_tokens} tokens")
                    oversize[rec["task_id"]] = oversize.get(rec["task_id"], 0) + 1
                    continue
                if n_comp < args.min_completion_tokens:
                    skip(f"segment: completion < {args.min_completion_tokens} tokens")
                    continue
                # How much history the truncation deletes between the chosen user message
                # and the target assistant turn. gap==0 would mean a true prefix; it never is.
                dropped = messages[ui + 1:mi]
                gap_roles = {}
                for m in dropped:
                    gap_roles[m.get("role")] = gap_roles.get(m.get("role"), 0) + 1
                for m in messages[:ui]:
                    gap_roles[m.get("role")] = gap_roles.get(m.get("role"), 0) + 1
                rows.append(
                    {
                        "prompt": prompt,
                        "src_completions": [completion],
                        "dst_completions": [],
                        "is_negative": False,
                        "meta": {
                            "task_id": rec["task_id"],
                            "message_index": mi,
                            "segment_idx": seg.get("segment_idx"),
                            "position": seg.get("position"),
                            "gold_label": seg.get("gold_label"),
                            "char_start": cs,
                            "n_prompt_tokens": n_prompt,
                            "n_completion_tokens": n_comp,
                            "user_message_index": ui,
                            "gap": len(dropped),
                            "dropped_roles": gap_roles,
                        },
                    }
                )

    # Duplicate prompts: four lcbhard_67 rows (msg88/90/92/96, all char_start==0) collapse
    # to a byte-identical prompt once the intervening turns are truncated away, so the
    # optimizer is asked to suppress four mutually exclusive continuations at one context.
    # Always report; only drop when asked, so the primary set stays unconfounded.
    seen = {}
    dupes = []
    kept = []
    for r in rows:
        key = r["prompt"]
        if key in seen:
            dupes.append((seen[key], r))
            if args.drop_duplicate_prompts:
                continue
        else:
            seen[key] = r
        kept.append(r)
    if dupes:
        print(f"\n!! {len(dupes)} DUPLICATE-PROMPT collisions "
              f"({'dropped' if args.drop_duplicate_prompts else 'KEPT -- pass --drop-duplicate-prompts to drop'}):")
        for first, dup in dupes:
            fm, dm = first["meta"], dup["meta"]
            print(f"   {fm['task_id']} msg{fm['message_index']} {first['src_completions'][0][:48]!r}")
            print(f"    vs {dm['task_id']} msg{dm['message_index']} {dup['src_completions'][0][:48]!r}")
    rows = kept

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    def pct(vals, q):
        if not vals:
            return 0
        s = sorted(vals)
        return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]

    p = [r["meta"]["n_prompt_tokens"] for r in rows]
    c = [r["meta"]["n_completion_tokens"] for r in rows]
    tids = sorted({r["meta"]["task_id"] for r in rows})
    print(f"\nwrote {len(rows)} rows -> {args.out}")
    print(f"distinct task_ids: {len(tids)}  {tids}")
    print(f"distinct (task,msg) contexts: {len({(r['meta']['task_id'], r['meta']['message_index']) for r in rows})}")
    print(f"user-message mode: {args.user_message}   tokenizer: {args.tokenizer}")
    print(f"prompt tokens     p50={pct(p,.5)}  p90={pct(p,.9)}  max={max(p) if p else 0}  min={min(p) if p else 0}")
    print(f"completion tokens p50={pct(c,.5)}  p90={pct(c,.9)}  max={max(c) if c else 0}  min={min(c) if c else 0}")
    g = [r["meta"]["gap"] for r in rows]
    gd = {}
    for x in g:
        gd[x] = gd.get(x, 0) + 1
    print(f"splice gap (messages deleted between the user msg and the target turn): "
          f"p50={pct(g,.5)}  p90={pct(g,.9)}  max={max(g) if g else 0}  "
          f"gap==0 (true prefix): {gd.get(0,0)}/{len(rows)}")
    print(f"  gap distribution: {dict(sorted(gd.items()))}")
    roles = {}
    for r in rows:
        for k, v in r["meta"]["dropped_roles"].items():
            roles[k] = roles.get(k, 0) + v
    print(f"  roles deleted, summed over rows: {dict(sorted(roles.items()))}")
    short = [r for r in rows if r["meta"]["n_completion_tokens"] <= 3]
    print(f"degenerate completions (<=3 tokens): {len(short)}  "
          f"e.g. {[r['src_completions'][0][:14] for r in short[:6]]}")
    print(f"distinct prompts: {len({r['prompt'] for r in rows})} of {len(rows)}")
    print("skips:" if skips else "skips: none")
    for k, v in sorted(skips.items(), key=lambda kv: -kv[1]):
        print(f"  {v:5d}  {k}")
    if oversize:
        print("  oversize by task: " + ", ".join(f"{k}={v}" for k, v in sorted(oversize.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
