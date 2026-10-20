#!/usr/bin/env python
"""Full capture-everything run: all MisTrace rollouts x all layers, every tree token.

Scope (experiments/full-token-capture/spec.md): store activations for EVERY token the
model saw — the shared trunk (system/user messages, tool outputs, stripped assistant
responses, template markup) at first arrival plus legitimate re-reads when the
template re-renders history, each turn's reasoning branch, and each turn's own reply
as generated — labeled by kind, message, computing turn, and absolute prompt
position, so any token's exact seen-prefix is reconstructable.

Requires the prefix-reuse patch (serving/patch_vllm_lens_prefix_reuse.py); validated
2026-09-04 against the unpatched baseline (projection corr >= 0.99995,
experiments/full-token-capture/DATASET.md). Moved here from
experiments/full-token-capture/scripts/capture_all.py on 2026-10-13.

Runs in the SERVING env on the GPU box:

    cd serving && uv run python ../scripts/capture_all_tokens.py \
        --out-dir /opt/dlami/nvme/all103 \
        --s3-prefix s3://steering-reasoning-artifacts/activations/qwen35_27b_bf16_all103

CPU-only validation of the tree/template assumptions (run locally BEFORE box time):

    uv run --with transformers --with jinja2 python scripts/capture_all_tokens.py \
        --plan-only --out-dir experiments/full-token-capture/data/plan_tree

Output layout per rollout (uploaded per rollout, meta/<task>.json LAST as the
completion marker; resume skips rollouts whose meta exists in S3):

    L{ll:02d}/<task>.safetensors  token_acts (N,5120) bf16, token_kind (N,) i8,
                                  token_msg (N,) i32, token_turn (N,) i32,
                                  token_ctxpos (N,) i32
    tokens/<task>.safetensors     token_ids (N,) i32 (+ same index arrays)
    segmeans/<task>.safetensors   turn_means/turn_tokens [+ segment_means/tokens]
    meta/<task>.json              turns[] (with trunk_len/prompt_n_tokens), segments[]
    manifest.json                 written once at the very end

token_kind: 0=template (chat markup incl. all special tokens, think tags,
generation suffix), 1=system, 2=user, 3=assistant response as RE-READ in later
history (reasoning-stripped), 4=tool_output, 5=reasoning, 6=response text AS
GENERATED, 7=tool call AS GENERATED. Kinds 6/7 come from extending each turn's
prompt past </think> through the model's own emitted reply (decided 2026-09-04);
the same reply appears again as kind 3 when later turns re-read it. Caveats:
kind 6/7 text is the template's re-render of MisTrace's PARSED structure (exactly
inverse where the parse was lossless — 4,928/4,941 tool calls; turns with parse
loss get no extension, listed per rollout as response_ext_skipped_msgs); the
"\n\n" separator after </think> is kind 6 even on tool-call-only turns, so
kind==6 is not synonymous with prose.

Tree bookkeeping: each turn keeps prompt positions [keep_start, n) where
keep_start = longest common token prefix (LCP) with the previous kept turn's
prompt. Complete by construction (a position below keep_start carries the same
token id AND the same whole prefix as an earlier turn, so it was stored then).
keep_start is NOT monotone: Qwen's template re-renders earlier assistant messages
when a new user message arrives (empty-think markers track the last-user index),
so ~15% of rows are legitimate re-reads of history under its new rendering —
recorded per rollout as keep_start_collapses. Skipped overlong turns lose their
branch but their trunk delta is captured by the next kept turn.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from bisect import bisect_right
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from steering_reasoning.steering.segments import (  # noqa: E402
    char_to_token_spans,
    locate_segments,
    reasoning_segments_by_message,
)
from steering_reasoning.steering.transcripts import (  # noqa: E402
    THINK_CLOSE,
    THINK_OPEN,
    _part_text,
    _to_chat_message,
    iter_reasoning_turns,
)

from steering_reasoning.kinds import KIND, ROLE_KIND  # noqa: E402
from steering_reasoning.steering.shards import (  # noqa: E402
    create_acts_shard,
    vllm_lens_provenance as _vllm_lens_provenance,
)

MISTRACE_REVISION = "7b4689723dd16fc992b116a8d6d3bb1a41627b62"
D_MODEL = 5120
# vLLM ships activations across processes through msgspec, which hard-caps any
# bytes payload at 2**32-1; one request's capture must stay safely under it
# (incident 2026-09-04: a 25k-token turn at chunk 64 ≈ 16 GB killed the engine)
PAYLOAD_CAP_BYTES = 3 << 30


def host_mem_available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return -1


# ------------------------------------------------------------------ tree prep

def build_tree(rec, tokenizer, max_model_len: int,
               dedupe_think_open: bool = True) -> dict | None:
    """CPU-only per-rollout tree bookkeeping.

    Returns kept turns, per-turn token ids/offsets/boundaries, the keep plan, and
    kind/msg attribution for every kept token. All char/message attribution is
    computed PER TURN from that turn's own prompt: Qwen3.5's template renders
    history differently depending on where the last user message sits (assistant
    messages after it gain empty-think markers), so a single full-history render
    is in the wrong coordinate system for earlier turns.
    """
    import re

    all_turns = iter_reasoning_turns(rec, tokenizer, max_tokens=None,
                                     dedupe_think_open=dedupe_think_open)
    messages = rec["messages"]
    marker = re.escape("<|im_start|>")
    chat_msgs = [_to_chat_message(m) for m in messages]

    # Generation-time response extension: append the model's OWN emitted reply
    # (post-</think> text + tool-call block + <|im_end|>) to each turn's prompt,
    # recovered from the template's rendering of that message — the same renderer
    # that produces the trunk, so history stays byte-identical and only the branch
    # grows. kinds: 6 = response as generated, 7 = tool call as generated.
    _render_cache: dict[int, str] = {}

    def _render(i: int) -> str:
        if i not in _render_cache:
            _render_cache[i] = tokenizer.apply_chat_template(
                chat_msgs[:i], tokenize=False, add_generation_prompt=False)
        return _render_cache[i]

    def response_ext(turn) -> str:
        t = turn.turn_index
        tcs = messages[t].get("tool_calls") or []
        # No fabrication: where MisTrace's parse lost the emission (tool-call
        # arguments unparseable, or a reply swallowed into the reasoning string),
        # re-rendering would produce bytes the model never emitted. Skip the
        # extension for those turns — they get no kind 6/7 rows (26/4,954 turns).
        if any(tc.get("parse_error") is not None for tc in tcs):
            return ""
        if not tcs and not _part_text(messages[t].get("content"), "text").strip():
            return ""
        if "</think>" in _part_text(messages[t].get("content"), "text"):
            return ""  # template moves pre-</think> content into reasoning: lossy
        base = _render(t)
        with_msg = _render(t + 1)
        if not with_msg.startswith(base):
            raise RuntimeError(
                f"{rec['task_id']} msg{t}: render not concatenative for response ext")
        block = with_msg[len(base):]
        mark = "</think>\n\n"
        cut = block.find(mark)
        if cut < 0:
            raise RuntimeError(
                f"{rec['task_id']} msg{t}: no think-close marker in own block")
        tail = block[cut + len(mark):]
        end = tail.rfind("<|im_end|>")
        if end < 0:
            raise RuntimeError(f"{rec['task_id']} msg{t}: no <|im_end|> in own block")
        # "\n\n" separates </think> from the reply exactly as the model emitted it
        return "\n\n" + tail[:end] + "<|im_end|>"

    exts = {t.turn_index: response_ext(t) for t in all_turns}
    ext_skipped = sorted(t for t, e in exts.items() if e == "")
    turns = [t for t in all_turns
             if t.n_tokens + len(tokenizer(exts[t.turn_index],
                                           add_special_tokens=False)["input_ids"])
             <= max_model_len - 8]
    if not turns:
        return None

    # Per-turn keep range: everything past the longest common token prefix (LCP)
    # with the previous kept turn's prompt. No template assumption: BPE re-merges
    # a token or two at the trunk/branch splice between consecutive prompts, so a
    # strict "canonical trunk" is ill-defined at boundaries; the LCP definition is
    # complete by construction (any prompt position < keep_start carries the SAME
    # token id as an earlier prompt at that position, hence was stored then), at
    # the cost of ~1-2 duplicated boundary tokens per turn. vLLM's prefix cache
    # hashes actual token ids, so a cache hit can never exceed the LCP — the
    # first_pos > keep_start fallback in capture_one is pure safety.
    prev_ids: list[int] = []
    keep_total = 0
    per_turn = []
    for turn in turns:
        prompt = turn.prompt + exts[turn.turn_index]
        enc = tokenizer(prompt, add_special_tokens=False,
                        return_offsets_mapping=True)
        ids, offsets = enc["input_ids"], enc["offset_mapping"]
        # the base prompt must tokenize as a clean prefix of the extended one
        # (</think> is an ADDED token — atomic in pre-tokenization, though not in
        # all_special_ids — so no BPE merge crosses the boundary)
        resp_tok_start = next(
            (t for t, (s, _) in enumerate(offsets) if s >= len(turn.prompt)),
            len(ids))
        assert resp_tok_start > 0 and offsets[resp_tok_start - 1][1] <= len(turn.prompt), \
            f"{rec['task_id']} msg{turn.turn_index}: token straddles response boundary"
        assert resp_tok_start == turn.n_tokens, \
            f"{rec['task_id']} msg{turn.turn_index}: base tokenization shifted"
        ext = exts[turn.turn_index]
        # a skipped extension (parse loss) legitimately contains no tool calls
        expected_tc = (len(messages[turn.turn_index].get("tool_calls") or [])
                       if ext else 0)
        if ext.count("<tool_call>") != expected_tc:
            raise RuntimeError(f"{rec['task_id']} msg{turn.turn_index}: literal "
                               "<tool_call> in reply text — kind 6/7 boundary unsafe")
        tc = ext.find("<tool_call>")
        tool_char = len(turn.prompt) + tc if tc >= 0 else None
        reasoning = _part_text(
            messages[turn.turn_index].get("content"), "reasoning").strip()
        prefix_chars = len(turn.prompt) - len(reasoning) - len(THINK_CLOSE)
        assert turn.prompt[prefix_chars:prefix_chars + len(reasoning)] == reasoning, \
            f"{rec['task_id']} msg{turn.turn_index}: reasoning not at prefix_chars"
        ctx_chars = prefix_chars - len(THINK_OPEN)
        ctx_len = next((t for t, (_, e) in enumerate(offsets) if e > ctx_chars),
                       len(ids))

        # message boundaries from THIS turn's own context; last block is the
        # generation prompt. Consecutive tool messages (parallel tool calls)
        # merge into one block — sub-split at their <tool_response> markers.
        context = turn.prompt[:ctx_chars]
        starts = [mo.start() for mo in re.finditer(marker, context)]
        hist_end_chars = starts[-1]
        block_starts = starts[:-1]
        groups: list[list[int]] = []
        for i in range(turn.turn_index):
            if (groups and messages[i]["role"] == "tool"
                    and messages[groups[-1][-1]]["role"] == "tool"):
                groups[-1].append(i)
            else:
                groups.append([i])
        if len(block_starts) != len(groups):
            raise RuntimeError(
                f"{rec['task_id']} msg{turn.turn_index}: {len(block_starts)} "
                f"history blocks for {len(groups)} message groups"
            )
        msg_bounds: list[int] = []
        bound_msg: list[int] = []
        block_ends = block_starts[1:] + [hist_end_chars]
        for g, s, e in zip(groups, block_starts, block_ends):
            if len(g) == 1:
                msg_bounds.append(s); bound_msg.append(g[0])
            else:
                occ = [s + mo.start() for mo in
                       re.finditer(re.escape("<tool_response>"), context[s:e])]
                if len(occ) == len(g):
                    msg_bounds.append(s); bound_msg.append(g[0])
                    msg_bounds.extend(occ[1:]); bound_msg.extend(g[1:])
                else:  # coarse: attribute whole block to first tool msg (same kind)
                    msg_bounds.extend([s] * len(g)); bound_msg.extend(g)

        import numpy as np

        m = min(len(prev_ids), len(ids))
        if m == 0:
            lcp = 0
        else:
            neq = np.array(ids[:m]) != np.array(prev_ids[:m])
            lcp = int(neq.argmax()) if neq.any() else m
        assert lcp <= turn.reasoning_start, \
            f"{rec['task_id']} msg{turn.turn_index}: LCP {lcp} reaches into reasoning"
        per_turn.append({
            "turn": turn, "prompt": prompt, "ids": ids, "offsets": offsets,
            "ctx_len": ctx_len, "keep_start": lcp,
            "row_start": keep_total,
            "resp_tok_start": resp_tok_start, "tool_char": tool_char,
            "base_chars": len(turn.prompt),
            "msg_bounds": msg_bounds, "bound_msg": bound_msg,
            "hist_end_chars": hist_end_chars,
        })
        keep_total += len(ids) - lcp
        prev_ids = ids

    # keep_start is NOT monotone: a user retry message retroactively changes how
    # the template renders earlier assistant messages (empty-think markers appear/
    # disappear with the last-user index), collapsing the LCP and re-reading
    # already-stored history (~15% of rows corpus-wide). Correct — the model
    # genuinely saw both renderings — but record where it happens.
    collapses = [{"turn_row": t, "from": per_turn[t - 1]["keep_start"],
                  "to": per_turn[t]["keep_start"]}
                 for t in range(1, len(per_turn))
                 if per_turn[t]["keep_start"] < per_turn[t - 1]["keep_start"]]

    # kind/msg/ctxpos arrays in keep (computation) order. Special tokens
    # (<|im_start|>, <|im_end|>, ...) are kind 0 regardless of message role —
    # they are the high-norm markup tokens, not message content.
    special_ids = set(tokenizer.all_special_ids)
    kinds, msgs, ctxpos, tok_ids, turn_of = [], [], [], [], []
    for ti, pt in enumerate(per_turn):
        turn, ids, offsets = pt["turn"], pt["ids"], pt["offsets"]
        for pos in range(pt["keep_start"], len(ids)):
            if pos < pt["ctx_len"]:
                s = offsets[pos][0]
                if s >= pt["hist_end_chars"] or not pt["msg_bounds"]:
                    k, m = KIND["template"], turn.turn_index  # generation suffix
                else:
                    j = max(0, min(bisect_right(pt["msg_bounds"], s) - 1,
                                   len(pt["bound_msg"]) - 1))
                    m = pt["bound_msg"][j]
                    k = ROLE_KIND.get(messages[m]["role"], KIND["template"])
            elif turn.reasoning_start <= pos < turn.reasoning_end:
                k, m = KIND["reasoning"], turn.turn_index
            elif pos >= pt["resp_tok_start"]:
                s = offsets[pos][0]
                if pt["tool_char"] is not None and s >= pt["tool_char"]:
                    k = KIND["tool_call_gen"]
                else:
                    k = KIND["response_gen"]
                m = turn.turn_index
            else:
                k, m = KIND["template"], turn.turn_index  # think open/close
            if k != KIND["reasoning"] and ids[pos] in special_ids:
                # NOTE: Qwen3.5's true special ids are only <|im_end|>/<|endoftext|>
                # and pads; <|im_start|>/<think>/<tool_call> are added-but-not-special
                # tokens and KEEP their region's kind (in-history markup carries its
                # message's role kind; the kind-7 <tool_call> token stays kind 7)
                k = KIND["template"]
            kinds.append(k); msgs.append(m); ctxpos.append(pos)
            tok_ids.append(ids[pos]); turn_of.append(ti)

    assert len(kinds) == keep_total
    return {
        "turns": per_turn,
        "n_tree": keep_total,
        "n_reasoning": sum(t["turn"].reasoning_end - t["turn"].reasoning_start
                           for t in per_turn),
        "n_skipped_overlong": len(all_turns) - len(turns),
        "skipped_overlong": [{"message_index": t.turn_index, "n_tokens": t.n_tokens}
                             for t in all_turns if t not in turns],
        "keep_start_collapses": collapses,
        "response_ext_skipped_msgs": ext_skipped,  # upstream parse loss, no kind 6/7
        "max_delta": max(len(pt["ids"]) - pt["keep_start"] for pt in per_turn),
        "max_prompt": max(len(pt["ids"]) for pt in per_turn),
        "kinds": kinds, "msgs": msgs, "ctxpos": ctxpos,
        "tok_ids": tok_ids, "turn_of": turn_of,
    }


# ------------------------------------------------------------------- storage

from steering_reasoning.steering.shards import (  # noqa: E402  (moved to src, unchanged)
    s3_cp,
    s3_cp_unlink,
    s3_existing_metas,
)


# Oversized turns are split at exact token indices and captured as token-id
# sub-requests (see capture_one) — no string retokenization, which an earlier
# char-cut approach paid O(prompt) per candidate on, stalling long rollouts.


# ------------------------------------------------------------------- capture

def capture_one(rec, tree, tokenizer, capturer, layers, chunk, out_dir: Path,
                s3_prefix: str | None, prefix_reuse: bool, timing: list) -> dict | None:
    import numpy as np
    import torch
    from safetensors.torch import save_file

    from concurrent.futures import ThreadPoolExecutor

    task_id = rec["task_id"]
    N = tree["n_tree"]
    per_turn = tree["turns"]
    seg_by_msg = reasoning_segments_by_message(rec) if rec.get("span_labels") else {}
    chunks = [layers[i:i + chunk] for i in range(0, len(layers), chunk)]
    uploader = ThreadPoolExecutor(max_workers=6) if s3_prefix else None
    upload_futs = []

    turn_meta, seg_meta, seg_tok_spans, turn_tokens = [], [], [], []
    turn_means_parts, seg_means_parts = [], []
    n_fallbacks = n_nonfinite = 0

    idx = {
        "token_kind": torch.tensor(tree["kinds"], dtype=torch.int8),
        "token_msg": torch.tensor(tree["msgs"], dtype=torch.int32),
        "token_turn": torch.tensor(tree["turn_of"], dtype=torch.int32),
        "token_ctxpos": torch.tensor(tree["ctxpos"], dtype=torch.int32),
    }
    for ci, chunk_layers in enumerate(chunks):
        assert chunk_layers == sorted(chunk_layers)
        if prefix_reuse:
            # always start a chunk pass cold: within a pass, turns reuse each
            # other's trunk; a warm cache from the previous pass (same prompts!)
            # or the previous rollout would only trigger turn-0 fallbacks
            capturer.reset_prefix_cache()
        # Each layer's FINAL safetensors file is created now and capture writes bf16
        # bit-patterns (uint16) straight into its token_acts region — no .npy, no
        # read-rewrite. bf16 (not f16): f16's narrow exponent would turn large
        # residual coordinates into inf irrecoverably, and this dump covers early
        # layers the f16 legacy dumps never touched.
        mm = {li: create_acts_shard(
                  out_dir / f"L{li:02d}" / f"{task_id}.safetensors", N, D_MODEL, idx)
              for li in chunk_layers}
        t_means, s_means = [], []
        row_ptr = 0
        for ti, pt in enumerate(per_turn):
            turn = pt["turn"]
            t0 = time.monotonic()
            discarded = 0
            # payload cap is on returned rows = (prompt tokens) - (cache hit). The
            # cache hit is unknown ahead of time (eviction can drop it far below
            # keep_start), so split on the ABSOLUTE prompt length, never on delta.
            cap_rows = PAYLOAD_CAP_BYTES // (len(chunk_layers) * D_MODEL * 2)
            ks = pt["keep_start"]
            if prefix_reuse and len(pt["ids"]) > cap_rows:
                # Oversized turn. Warm the trunk [0, ks) in ONE non-capturing call
                # (no payload), then capture only [ks, n) in cap-sized token-id
                # sub-requests. This replaces the dozens of from-0 cache-warming
                # capture RPCs (which dominated big-rollout wall-clock) with a
                # single warm + ceil((n-ks)/cap) real captures. Token-id prompts =
                # no retokenization / no BPE-remerge risk.
                ids = pt["ids"]
                n = len(ids)
                bs = capturer.block_size  # cache-hit granularity (don't hardcode 16)
                if ks > 0:
                    # warm [0,ks) and CONFIRM it cached before issuing the first
                    # payload-bearing capture — if reuse is silently broken the
                    # confirm (no payload) raises cleanly instead of the first
                    # capture returning ks+cap rows and crashing the engine at >4 GiB
                    n_warm = capturer.warm_prefix(ids[:ks])
                    if n_warm < ks - bs:
                        raise RuntimeError(
                            f"{task_id} msg{turn.turn_index}: prefix warm did not "
                            f"cache (confirmed {n_warm} < ks {ks}); refusing unsafe "
                            "capture — prefix reuse may be broken")
                cut_bs = list(range(ks + cap_rows, n, cap_rows)) + [n]
                segs = []
                cover = ks  # highest absolute position captured so far
                computed = 0
                for b in cut_bs:
                    a, fp = capturer.capture_suffix([ids[:b]], chunk_layers,
                                                    token_ids=True)[0]
                    computed += int(a.shape[1])
                    if fp > cover:
                        raise RuntimeError(
                            f"{task_id} msg{turn.turn_index}: cache jumped past "
                            f"cover (fp {fp} > {cover}) — position gap")
                    if fp < cover - bs:
                        raise RuntimeError(
                            f"{task_id} msg{turn.turn_index}: cache hole mid-turn "
                            f"(fp {fp} << cover {cover}) — payload unsafe")
                    segs.append(a[:, cover - fp:, :])
                    cover = b
                    del a
                act = torch.cat(segs, dim=1)
                assert act.shape[1] == n - ks, \
                    (task_id, turn.turn_index, "split rows", act.shape[1], n - ks)
                first_pos = ks
                del segs
            elif prefix_reuse:
                act, first_pos = capturer.capture_suffix([pt["prompt"]], chunk_layers)[0]
                computed = int(act.shape[1])
                if first_pos > ks:
                    discarded = computed
                    act = capturer.capture([pt["prompt"]], chunk_layers)[0]
                    computed = int(act.shape[1])
                    first_pos = 0
                    n_fallbacks += 1
            else:
                act, first_pos = capturer.capture([pt["prompt"]], chunk_layers)[0], 0
                computed = int(act.shape[1])
            timing.append({"chunk_size": len(chunk_layers),
                           "prompt_tokens": len(pt["ids"]),
                           "computed_tokens": computed + discarded,
                           "seconds": time.monotonic() - t0})

            keep = act[:, pt["keep_start"] - first_pos:, :]  # canonical keep slice
            n_keep = keep.shape[1]
            for row, li in enumerate(chunk_layers):
                x = keep[row].to(torch.bfloat16)
                n_nonfinite += int((~torch.isfinite(x)).any(dim=-1).sum())
                mm[li][row_ptr:row_ptr + n_keep] = x.view(torch.uint16).numpy()
            # reasoning span, for means/segments (absolute -> keep-slice offset).
            # keep_start <= reasoning_start is asserted in build_tree, so these
            # indices are non-negative; empty spans would desync token_turn from
            # turns[], so they are fatal rather than skipped.
            span = keep[:, turn.reasoning_start - pt["keep_start"]
                        : turn.reasoning_end - pt["keep_start"], :]
            assert span.shape[1] > 0, (task_id, turn.turn_index, "empty reasoning span")
            if True:
                t_means.append(span.mean(dim=1, dtype=torch.float32).cpu())
                if ci == 0:
                    turn_tokens.append(span.shape[1])
                    turn_meta.append({
                        "turn_row": ti,  # joins token_turn <-> turns[] (identical order)
                        "row_start": pt["row_start"],  # this turn's first shard row
                        "message_index": turn.turn_index,
                        "n_tokens": span.shape[1],
                        "reasoning_start": turn.reasoning_start,
                        "trunk_len": pt["ctx_len"],
                        "prompt_n_tokens": len(pt["ids"]),  # incl. response ext
                        "base_n_tokens": turn.n_tokens,  # through </think> only
                        "response_start": pt["resp_tok_start"],
                        "keep_start": pt["keep_start"],
                        "first_pos": first_pos,  # cache offset (chunk-0 pass)
                    })
                    if turn.turn_index in seg_by_msg:
                        reasoning = _part_text(
                            rec["messages"][turn.turn_index].get("content"),
                            "reasoning").strip()
                        prefix_chars = len(turn.prompt) - len(reasoning) - len(THINK_CLOSE)
                        assert turn.prompt[prefix_chars:prefix_chars + len(reasoning)] \
                            == reasoning
                        located = locate_segments(reasoning, seg_by_msg[turn.turn_index])
                        for seg, _cs, _ce, ts, te in char_to_token_spans(
                                turn.prompt, prefix_chars, located, tokenizer):
                            if te <= ts:
                                continue
                            assert ts >= pt["keep_start"], (task_id, ti, ts)
                            seg_tok_spans.append((len(t_means) - 1, ti, ts, te))
                            seg_meta.append({
                                "message_index": turn.turn_index,
                                "segment_idx": seg["segment_idx"],
                                "position": seg["position"],
                                "gold_label": bool(seg["gold_label"]),
                                "judge_label": seg.get("judge_label"),
                                "n_tokens": te - ts,
                                "token_start": ts, "token_end": te,
                                "text_sha1": hashlib.sha1(seg["text"].encode()).hexdigest()[:10],
                                "text": seg["text"],
                            })
            act = keep = span = None  # noqa: F841
            row_ptr += n_keep

        assert row_ptr == N, (task_id, row_ptr, N)
        # segment means from the memmap (chunk's layers), fp32 pooling
        if seg_tok_spans:
            base_of_turn = {}
            rp = 0
            for tii, pt2 in enumerate(per_turn):
                base_of_turn[tii] = rp - pt2["keep_start"]
                rp += len(pt2["ids"]) - pt2["keep_start"]
            for trow, tti, ts, te in seg_tok_spans:
                rows = slice(base_of_turn[tti] + ts, base_of_turn[tti] + te)
                per_layer = [torch.from_numpy(np.asarray(mm[li][rows]))
                             .view(torch.bfloat16).float().mean(dim=0)
                             for li in chunk_layers]
                s_means.append(torch.stack(per_layer))
        turn_means_parts.append(torch.stack(t_means))
        if s_means:
            seg_means_parts.append(torch.stack(s_means))

        # shards are already the final safetensors files — just flush and upload
        for li in chunk_layers:
            mm[li].flush()
            del mm[li]  # drop the memmap handle before the uploader reads the file
            path = out_dir / f"L{li:02d}" / f"{task_id}.safetensors"
            if uploader:  # background upload overlaps segment-means + token/segmean saves
                upload_futs.append(uploader.submit(
                    s3_cp_unlink, path, s3_prefix, f"L{li:02d}/{task_id}.safetensors"))
        del mm

    # tokens / segmeans / meta (meta last locally AND last to S3)
    tok_path = out_dir / "tokens" / f"{task_id}.safetensors"
    save_file({"token_ids": torch.tensor(tree["tok_ids"], dtype=torch.int32), **idx},
              str(tok_path))
    seg_path = out_dir / "segmeans" / f"{task_id}.safetensors"
    tensors = {"turn_means": torch.cat(turn_means_parts, dim=1),
               "turn_tokens": torch.tensor(turn_tokens, dtype=torch.int32)}
    if seg_means_parts:
        tensors["segment_means"] = torch.cat(seg_means_parts, dim=1).clamp(
            -65504, 65504).half()
        tensors["segment_tokens"] = torch.tensor(
            [te - ts for _, _, ts, te in seg_tok_spans], dtype=torch.int32)
    save_file(tensors, str(seg_path))
    # the manifest row is embedded in meta so a resumed run can rebuild the full
    # manifest from S3 meta files alone
    row = {"task_id": task_id, "label": bool(rec.get("is_reward_hack")),
           "has_gold_spans": bool(rec.get("has_gold_spans")),
           "n_turns": len(turn_meta), "n_tree_tokens": N,
           "n_reasoning_tokens": tree["n_reasoning"],
           "n_segments": len(seg_meta),
           "n_skipped_overlong_turns": tree["n_skipped_overlong"],
           "n_prefix_reuse_fallbacks": n_fallbacks,
           "n_nonfinite_token_layer_rows": n_nonfinite,
           "layer_chunk_used": chunk}
    meta_path = out_dir / "meta" / f"{task_id}.json"
    meta_path.write_text(json.dumps({
        **row,
        "skipped_overlong": tree["skipped_overlong"],
        "keep_start_collapses": tree["keep_start_collapses"],
        "response_ext_skipped_msgs": tree["response_ext_skipped_msgs"],
        "turns": turn_meta,
        "segments": seg_meta,
    }, indent=1) + "\n")
    if s3_prefix:
        for f in upload_futs:  # every layer shard must land before the marker
            f.result()
        uploader.shutdown()
        s3_cp(tok_path, s3_prefix, f"tokens/{task_id}.safetensors")
        s3_cp(seg_path, s3_prefix, f"segmeans/{task_id}.safetensors")
        s3_cp(meta_path, s3_prefix, f"meta/{task_id}.json")  # completion marker

    return row


# ---------------------------------------------------------------------- main

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rollouts", default="all")
    p.add_argument("--plan-only", action="store_true",
                   help="CPU-only: tree accounting + template-assumption asserts")
    p.add_argument("--model", default="Qwen/Qwen3.5-27B")
    p.add_argument("--layers", default="all")
    p.add_argument("--layer-chunk", type=int, default=64)
    p.add_argument("--max-model-len", type=int, default=262144)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--tensor-parallel", type=int, default=4)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--s3-prefix", default=None,
                   help="upload per rollout and delete local (streaming); omit to keep local")
    p.add_argument("--no-prefix-reuse", action="store_true",
                   help="debug: full re-prefill per turn (~10x slower)")
    p.add_argument("--host-mem-margin-gib", type=int, default=128,
                   help="RAM kept free of the assembled-turn budget — cushions "
                        "memmap dirty-page writeback on the largest rollouts")
    args = p.parse_args()
    prefix_reuse = not args.no_prefix_reuse
    sys.stdout.reconfigure(line_buffering=True)  # progress visible through a log file
    if args.layer_chunk < 1:
        p.error("--layer-chunk must be >= 1")
    if prefix_reuse and not args.plan_only:
        import vllm_lens

        plugin = Path(vllm_lens.__file__).parent / "_activations_plugin.py"
        if "vllm_lens_reuse_prefix_cache" not in plugin.read_text():
            p.error(f"prefix-reuse patch not applied to {plugin} — run "
                    "serving/patch_vllm_lens_prefix_reuse.py (or --no-prefix-reuse)")

    from steering_reasoning.data.mistrace import load_mistrace
    from transformers import AutoTokenizer

    records, provenance = load_mistrace("qwen3_5_27b", config="reward-hacking",
                                        revision=MISTRACE_REVISION)
    records = [r for r in records if r.get("is_reward_hack") in (True, False)]
    if args.rollouts != "all":
        by_id = {r["task_id"]: r for r in records}
        missing = [t for t in args.rollouts.split(",") if t not in by_id]
        if missing:
            p.error(f"task_ids not in MisTrace: {missing}")
        records = [by_id[t] for t in args.rollouts.split(",")]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if not tokenizer.is_fast:
        p.error("needs a fast tokenizer (offset_mapping)")
    out_dir = Path(args.out_dir)
    if not args.plan_only:
        for d in ("tokens", "segmeans", "meta"):
            (out_dir / d).mkdir(parents=True, exist_ok=True)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    if args.plan_only:
        from collections import Counter

        totals = {"tree": 0, "reasoning": 0, "skipped": 0,
                  "trunk_unstable": [], "rollouts": []}
        for ri, rec in enumerate(records):
            try:
                tree = build_tree(rec, tokenizer, args.max_model_len)
            except Exception as e:  # noqa: BLE001 — one bad rollout must not kill the run
                print(f"[{ri+1}/{len(records)}] TRUNK UNSTABLE: {e}")
                totals["trunk_unstable"].append(rec["task_id"])
                continue
            if tree is None:
                print(f"[{ri+1}/{len(records)}] {rec['task_id']}: no usable turns")
                continue
            kc = Counter(tree["kinds"])
            totals["tree"] += tree["n_tree"]
            totals["reasoning"] += tree["n_reasoning"]
            totals["skipped"] += tree["n_skipped_overlong"]
            totals["rollouts"].append({
                "task_id": rec["task_id"], "n_tree": tree["n_tree"],
                "n_reasoning": tree["n_reasoning"],
                "n_skipped_overlong": tree["n_skipped_overlong"],
                "max_delta": tree["max_delta"],
                "n_ext_skipped": len(tree["response_ext_skipped_msgs"]),
                "kind_counts": {k: kc.get(v, 0) for k, v in KIND.items()},
            })
            print(f"[{ri+1}/{len(records)}] {rec['task_id']}: tree {tree['n_tree']:,} "
                  f"(reasoning {tree['n_reasoning']:,}, skipped turns "
                  f"{tree['n_skipped_overlong']})")
        (out_dir / "plan_tree.json").write_text(json.dumps(totals, indent=1) + "\n")
        print(f"\nTOTAL tree {totals['tree']:,} tokens "
              f"({totals['tree'] * 64 * D_MODEL * 2 / 1e12:.2f} TB at 64 layers), "
              f"reasoning {totals['reasoning']:,}, skipped turns {totals['skipped']}")
        return

    # ---- GPU run -----------------------------------------------------------
    from steering_reasoning.steering.activations import (
        ActivationCapturer,
        apply_no_nvcc_workarounds,
    )

    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(args.model)
    n_layers = getattr(cfg, "num_hidden_layers", None) or getattr(
        getattr(cfg, "text_config", None), "num_hidden_layers", None)
    if n_layers is None:
        p.error(f"cannot determine layer count for {args.model}")
    layers = (list(range(n_layers)) if args.layers == "all"
              else sorted({int(x) for x in args.layers.split(",")}))
    bad = [li for li in layers if not 0 <= li < n_layers]
    if bad:
        p.error(f"layers {bad} out of range ({n_layers} layers)")
    for d in layers:
        (out_dir / f"L{d:02d}").mkdir(parents=True, exist_ok=True)

    # resume: seed the manifest from completed rollouts' meta files so a resumed
    # run's manifest covers the whole dump, not just its own rollouts
    done = s3_existing_metas(args.s3_prefix) if args.s3_prefix else set()
    resumed_rows = []
    if done:
        print(f"resume: {len(done)} rollouts already complete in S3")
        for t in sorted(done):
            local = out_dir / "meta" / f"{t}.json"
            if not local.exists():
                subprocess.run(["aws", "s3", "cp", "--quiet",
                                f"{args.s3_prefix.rstrip('/')}/meta/{t}.json",
                                str(local)], check=True)
            meta = json.loads(local.read_text())
            resumed_rows.append({k: v for k, v in meta.items()
                                 if k not in ("turns", "segments")})

    apply_no_nvcc_workarounds()
    if prefix_reuse:
        os.environ["VLLM_LENS_CAPTURE_PREFIX_REUSE"] = "1"
    capturer = ActivationCapturer(
        args.model, max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel,
        enable_prefix_caching=prefix_reuse,
    )

    timing: list[dict] = []
    manifest_rows = list(resumed_rows)
    error_rows = []
    margin = args.host_mem_margin_gib << 30
    for ri, rec in enumerate(records):
        if rec["task_id"] in done:
            print(f"[{ri+1}/{len(records)}] {rec['task_id']}: done in S3, skipped")
            continue
        try:
            tree = build_tree(rec, tokenizer, args.max_model_len)
        except Exception as e:  # noqa: BLE001 — one bad rollout must not kill the run
            print(f"[{ri+1}/{len(records)}] MAPPING FAILED, skipped: {e}")
            error_rows.append({"task_id": rec["task_id"], "error": str(e)})
            continue
        if tree is None:
            print(f"[{ri+1}/{len(records)}] {rec['task_id']}: no usable turns, skipped")
            continue
        # Adaptive chunk from the in-flight host buffer. Under prefix-reuse the
        # split path bounds EVERY request to cap_rows (~PAYLOAD_CAP_BYTES of rows),
        # independent of chunk, so the per-call RAM is ~constant and the biggest
        # rollouts can still run all layers in one pass — size on cap_rows, not on
        # max_prompt (which needlessly halved big rollouts into 2 passes). Without
        # reuse there is no splitting: the whole prompt is one request, so size on
        # max_prompt AND cap it under the msgspec limit. 3x allowance for
        # worker+IPC+driver copies observed in the smoke.
        usable = host_mem_available_bytes() - margin
        if prefix_reuse:
            # The driver peak is NOT the per-call payload (splitting bounds that to
            # cap_rows); it is the worst turn's assembled activations —
            # torch.cat(segs) holds the retained sub-request views AND the cat
            # output at once ≈ 2 × max_delta × chunk rows. Size on that so the
            # biggest turn fits and --host-mem-margin-gib stays an effective lever
            # (the margin also cushions concurrent memmap dirty-page writeback).
            worst_rows = tree["max_delta"]
            fit = max(1, usable // (worst_rows * D_MODEL * 2 * 2))
        else:
            # no cache → no splitting: the whole prompt is one request
            worst_rows = tree["max_prompt"]
            fit = max(1, usable // (worst_rows * D_MODEL * 2 * 3))
        chunk = int(min(args.layer_chunk, fit))
        if not prefix_reuse:
            chunk = max(1, min(chunk,
                               PAYLOAD_CAP_BYTES // (tree["max_prompt"] * D_MODEL * 2)))
        if chunk < args.layer_chunk:
            print(f"    host RAM limits chunk to {chunk} "
                  f"(worst assembled turn {worst_rows:,} rows)")
        print(f"[{ri+1}/{len(records)}] {rec['task_id']}: tree {tree['n_tree']:,} "
              f"tokens, chunk {chunk}")
        try:
            row = capture_one(rec, tree, tokenizer, capturer, layers, chunk, out_dir,
                              args.s3_prefix, prefix_reuse, timing)
        except (RuntimeError, AssertionError) as e:
            # a guard/shape failure on one rollout must not kill the whole run
            # (it leaves no meta marker, so a rerun retries it cleanly)
            print(f"    CAPTURE FAILED, skipped: {e}")
            error_rows.append({"task_id": rec["task_id"], "error": str(e)[:500]})
            continue
        manifest_rows.append(row)
        if row["n_prefix_reuse_fallbacks"]:
            print(f"    {row['n_prefix_reuse_fallbacks']} fallbacks")

    total_s = sum(t["seconds"] for t in timing)
    computed = sum(t["computed_tokens"] for t in timing)
    import torch
    import vllm
    (out_dir / "manifest.json").write_text(json.dumps({
        "extraction_model": args.model,
        "transcript_source": provenance,
        "transcript_model": records[0].get("target_model", "unknown") if records else None,
        "label_field": "is_reward_hack",
        "scope": "all tree tokens (trunk + branches), kinds 0-7; see script docstring",
        "layers": layers,
        "n_model_layers": n_layers,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "serving": {
            "tensor_parallel_size": args.tensor_parallel,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "dtype": "bfloat16",
            "enable_prefix_caching": prefix_reuse,
            "vllm_version": vllm.__version__,
            "torch_version": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "n_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "layer_chunk_cap": args.layer_chunk,
            "batch_size": 1,
            "enforce_eager": True,
            "storage_dtype": "bfloat16",
            "prefix_reuse_patch": prefix_reuse,
            "dedupe_think_open": True,
            "vllm_lens_version": _vllm_lens_provenance()[0],
            "vllm_lens_plugin_sha256": _vllm_lens_provenance()[1],
            "capture_script_sha256": hashlib.sha256(
                Path(__file__).read_bytes()).hexdigest()[:16],
        },
        "capture_wall_s_this_run": round(total_s, 1),
        "computed_tokens_this_run": computed,
        "transcripts": sorted(manifest_rows, key=lambda r: r["task_id"]),
        "errors": error_rows,
    }, indent=1) + "\n")
    if args.s3_prefix:
        s3_cp(out_dir / "manifest.json", args.s3_prefix, "manifest.json")
    print(f"\nDONE: {len(manifest_rows)} rollouts, {computed:,} computed tokens, "
          f"{total_s/3600:.1f} h capture")


if __name__ == "__main__":
    main()
