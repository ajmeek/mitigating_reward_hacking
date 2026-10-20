# Dataset format: `qwen35_27b_bf16_all103`

`s3://steering-reasoning-artifacts/activations/qwen35_27b_bf16_all103/`

Per-token residual-stream activations for **all 103 MisTrace Qwen3.5-27B rollouts**
(42 reward-hacking + 61 clean; `ameek/MisTrace` rev `7b4689723d`), **all 64 layers**,
captured in **bf16** with generation-faithful per-turn context. 15.8M "tree" tokens,
10.44 TB. See `scripts/capture_all_tokens.py` for how it was produced and for the per-turn tree model.

## What a "tree token" is

Reasoning models strip prior turns' reasoning from history, so each assistant turn is
captured in its own prompt (history + that turn's `<think>` + that turn's emitted
reply). The union over turns — every token the model actually saw/produced, once — is
the "tree" (trunk + branches). 15.8M tree tokens
vs 5.9M reasoning tokens: the rest is tool output, user messages, and responses.

## Layout

```
qwen35_27b_bf16_all103/
  manifest.json                 model, layers, serving/numerics provenance, per-rollout stats, errors[]
  meta/<task>.json              turns[] + segments[] metadata (see below)
  tokens/<task>.safetensors     token_ids (N,) i32  + the 4 index arrays
  segmeans/<task>.safetensors   turn_means / segment_means (mean-pooled; see below)
  L00/<task>.safetensors        per LAYER: token_acts (N,5120) bf16 + the 4 index arrays
  L01/ … L63/                   (one directory per layer — pull a single layer cheaply)
```
`<task>` = `lcbhard_NN`. `N` = tree-token count for that rollout (same across all its files).

### Per-layer shards (`L{ll}/<task>.safetensors`) — the heavy per-token data
- `token_acts` `(N, 5120)` **bf16** — one row per tree token, in **computation order**
  (= experienced order: each turn contributes its new trunk delta, then its reasoning,
  then its emitted reply).
- `token_kind` `(N,)` i8 — `0`=template/special markup, `1`=system, `2`=user,
  `3`=assistant response *as re-read* in later history (reasoning-stripped),
  `4`=tool output, `5`=reasoning, `6`=response text *as generated*,
  `7`=tool call *as generated*. (Kinds 6/7 are the generation-moment states; the same
  reply also appears as kind 3 when a later turn re-reads it.)
- `token_msg` `(N,)` i32 — source message index in the transcript.
- `token_turn` `(N,)` i32 — which capture turn computed the token; indexes `meta.turns[]`
  by `turn_row`.
- `token_ctxpos` `(N,)` i32 — the token's absolute position in that turn's prompt (so the
  exact prefix the model saw is reconstructable).

### `segmeans/<task>.safetensors` — lightweight mean-pooled aggregates
Cheap summaries for DiffMean fitting / layer sweeps / segment analysis; **no per-token
data**. fp32 pooling.
- `turn_means` `(T, 64, 5120)` **f32** — mean over each turn's **reasoning-span** tokens,
  all layers. Row `t` aligns with `meta.turns[t]`.
- `turn_tokens` `(T,)` i32 — reasoning-span token count per turn.
- `segment_means` `(S, 64, 5120)` **f16** — mean over each judge-labeled reasoning
  **segment**; present only for span-labeled (hacked) rollouts. Row `s` aligns with
  `meta.segments[s]`.
- `segment_tokens` `(S,)` i32 — token count per segment.

### `meta/<task>.json`
`{task_id, label (is_reward_hack), has_gold_spans, n_tree_tokens, turns[], segments[],
skipped_overlong[], keep_start_collapses[], response_ext_skipped_msgs[], ...}`.
- `turns[]`: `{turn_row, row_start, message_index, n_tokens (reasoning), reasoning_start,
  trunk_len, prompt_n_tokens, base_n_tokens, response_start, keep_start, first_pos}`.
  `row_start` = this turn's first row in the shard arrays.
- `segments[]`: legacy schema — `{message_index, segment_idx, position, gold_label,
  judge_label, n_tokens, token_start, token_end (prompt-absolute), text_sha1, text}`.

## Loading recipes

- **One layer, all rollouts:** sync `L31/` (~one prefix, ~160 GB) — nothing else needed.
- **Reasoning-only view (ports legacy code):** `m = token_kind == 5`; `token_turn[m]`
  indexes `turn_means`; `token_ctxpos[m] - turns[t].reasoning_start` == legacy `token_pos`.
- **One turn's full prompt as the model saw it:** select rows owned by each turn walking
  `keep_start` (NOT monotone — a user retry re-renders history; see
  `keep_start_collapses`); verify with `tokens/`. Full recipe in `scripts/capture_all_tokens.py` (consumer
  review) / session notes.
- **Segment → shard rows:** `turns[t].row_start + seg.token_start - turns[t].keep_start`.

## Derived datasets

- `s3://steering-reasoning-artifacts/derived/projection-trajectories/v1/` — per-token
  projections of every tree token onto the diff-mean directions at L31/L35 (+ norms,
  index arrays, per-rollout means), 1.9 GB; reader/writer in
  `experiments/archive/projection-trajectories/`. A Python reader for this dump format is
  `src/steering_reasoning/steering/token_dump.py` (local dir or `s3://`, zero-copy).
- **Arm dumps** — `s3://steering-reasoning-artifacts/derived/<experiment>/dumps/<arm>/<split>/s<seed>/`:
  replays of steering-arm rollouts in this same shard format (steered turns clamped
  exactly as at generation, by scripts/capture_rollouts.py). No `segments[]`; per-turn `gen_start`/`steer_positions`/`clamp_invariant_ok`
  in `meta/`; `never_computed_rows` lists each turn's last branch row (replay-only).

## Caveats
- Storage dtype is **bf16** (uint16 bit-pattern on disk): `torch.from_numpy(x).view(torch.bfloat16)`.
- `token_kind==6` includes the `"\n\n"` separator after `</think>` even on tool-call-only
  turns, so it is not strictly "prose".
- Kinds 6/7 are the template's re-render of MisTrace's *parsed* reply (byte-exact where
  the parse was lossless); 26 turns with upstream parse loss have no 6/7 rows
  (`response_ext_skipped_msgs`).
- ~15% of tree tokens are legitimate re-reads of re-rendered history (template
  retroactivity), recorded per rollout in `keep_start_collapses`.
