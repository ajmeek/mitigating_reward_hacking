"""Replay a logged rollout through the model and capture its residual stream — exactly.

The requirement (Austin, 2026-09-10): the replayed latent space must be what the model had
during generation. This module makes that true by construction and checks it:

* Prompt ids come from vLLM's OWN chat renderer (`LLM._preprocess_chat_one`, the same
  `render_chat` the server calls) applied to the pre-proxy wire request from the eval log
  (evals/eval_records.py) after the arm proxy's one request-side mutation (`strip_think` on
  prior reasoning). The rendered length is asserted against the server's logged
  `usage.prompt_tokens` on every turn; when a log carries `prompt_token_ids` the render is
  asserted element-wise.
* Branch ids are the logged `token_ids` when present (exact, pre-collapse); otherwise the
  logged reply is re-rendered through the template (`render_branch_ids`) and the delta vs
  the logged `completion_tokens` is recorded and flagged.
* Steering is reproduced INSIDE one prefill, matching how generation steered (serving
  harness of 2026-09-14 on, serving/patch_vllm_lens_gate.py): decode steered every generated
  token and prefill steered the spans of the model's own earlier replies (`steer_positions`,
  sent by the proxy per request and recorded in its ids sidecar). The replay passes
  `steer_positions` = those spans (recomputed with the proxy's `assistant_spans` and asserted
  equal to the spans the sidecar recorded) + the generated branch, under the same gate. A
  turn the proxy sent without positions is replayed with only its branch steered. No prefix
  reuse.
* Every clamp-steered turn is checked against the clamp invariant at the steering layer on
  its steered rows (`proj_u <= threshold + 2**-8 ‖h‖`) — the only thing that catches
  vllm-lens swallowing a steering-hook exception and returning unsteered activations.
  add/ablate modes have no such row-wise oracle; their manifests say so.

Output is the per-layer shard format of the 103 dump (steering/shards.py), so
`TokenDump`, `project_trajectories.py --phase project` and the plot scripts read arm dumps
unchanged. Serving env (vLLM + vllm-lens with the local patches).
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import torch

from steering_reasoning.steering.clamp import (
    artifact_sha,
    assistant_spans,
    build_steer_payload,
    ids_sha,
    proxy_view,
    with_positions,
)
from steering_reasoning.steering.replay_tree import ReplayTurn, replay_tree
from steering_reasoning.steering.shards import (
    create_acts_shard,
    index_tensors,
    s3_cp,
    s3_cp_unlink,
    write_meta,
    write_segmeans,
    write_tokens,
)

# vLLM ships activations across processes through msgspec, which hard-caps any bytes
# payload at 2**32-1; one request's capture must stay safely under it. Without prefix
# reuse a request returns every prompt row, so this bounds layers-per-pass by prompt length.
PAYLOAD_CAP_BYTES = 3 << 30
# Clamp-invariant tolerance, relative to ‖h‖: the worker subtracts `excess·u` in bf16
# (target dtype), so the clamped coordinate carries one bf16 rounding of a value of order
# ‖h‖ (2^-8 relative); measured on the GPU smoke (tests/gpu) the excess is far below this.
CLAMP_TOL = 2.0**-8
THINK_OPEN = "<think>\n"
# request keys that would change the render if present; the arms sent none of them
RENDER_AFFECTING_KEYS = ("chat_template", "chat_template_kwargs", "reasoning_effort",
                         "continue_final_message", "add_generation_prompt")


@dataclass
class SteeringSpec:
    """The arm's steering: artifact + mode + dose, resolved once per run. `dose` is the clamp
    threshold or the add scale (None for ablate); `threshold` is set for clamp only."""

    artifact: str
    mode: str
    dose: float | None
    layer: int
    sha: str
    _unit: torch.Tensor | None = field(default=None, repr=False)
    _base_payload: str | None = field(default=None, repr=False)

    @classmethod
    def from_artifact(cls, artifact: str, dose: float | None, mode: str = "clamp") -> "SteeringSpec":
        if mode not in ("clamp", "add", "ablate"):
            raise ValueError(f"mode must be clamp|add|ablate, got {mode!r}")
        if mode != "ablate" and dose is None:
            raise ValueError(f"{mode} mode needs a dose")
        meta = json.loads(Path(artifact.replace(".safetensors", ".json")).read_text())
        (layer,) = meta["layer_indices"]
        return cls(artifact=artifact, mode=mode, dose=None if mode == "ablate" else float(dose),
                   layer=int(layer), sha=artifact_sha(artifact))

    @property
    def threshold(self) -> float | None:
        return self.dose if self.mode == "clamp" else None

    def payload(self, steer_positions: list[list[int]]) -> str:
        """The request payload for one turn: the artifact serialised once, the spans stamped
        per turn (as the proxy does with `with_positions`)."""
        if not steer_positions:
            raise ValueError("a steered turn needs at least one span")
        if self._base_payload is None:
            self._base_payload = build_steer_payload(self.artifact, mode=self.mode, threshold=self.threshold,
                                                     scale=self.dose if self.mode == "add" else 1.0)
        return with_positions(self._base_payload, steer_positions)

    def unit(self) -> torch.Tensor:
        """The direction exactly as the worker computes it: the payload vector is cast to
        the model dtype (bf16) BEFORE normalisation (`vec = activations.to(target.dtype);
        u = (vec / vec.norm()).float()`), so the invariant is checked against the same u
        the clamp used, not an fp32 re-normalisation ~2^-8 away from it."""
        if self._unit is None:
            from safetensors.torch import load_file

            v = load_file(self.artifact)["vector"].float().reshape(-1).to(torch.bfloat16)
            self._unit = (v / v.norm()).float()
        return self._unit

    def provenance(self) -> dict:
        return {"artifact": self.artifact, "artifact_sha": self.sha, "mode": self.mode, "dose": self.dose,
                "layer": self.layer,
                "positions": "steer_positions = earlier assistant spans + generated branch (per turn)"}


# ------------------------------------------------------------------ steered positions

def span_markers(tokenizer) -> tuple[list[int], list[int], int]:
    """(header ids, think ids, im_end id) for `assistant_spans`, tokenized exactly as the proxy
    does through /tokenize (add_special_tokens=False)."""
    hdr = tokenizer("<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
    think = tokenizer("<think>\n", add_special_tokens=False)["input_ids"]
    end = tokenizer("<|im_end|>", add_special_tokens=False)["input_ids"]
    if len(end) != 1:
        raise RuntimeError(f"<|im_end|> is not a single token: {end}")
    return hdr, think, end[0]


def steered_positions(prompt_ids: list[int], n_branch: int, markers: tuple, recorded: list | None,
                      *, prefill_steered: bool = True) -> list[list[int]]:
    """The absolute [start, end) spans a model-tokens generation steered for this turn, for one
    replay prefill of prompt + branch: the spans of the model's earlier replies (prefill
    steering, via the proxy's own `assistant_spans`) plus the generated branch (decode steering).
    `recorded` = the spans the proxy sent at generation (sidecar); when present they must equal
    the recomputed ones — a different rendering would mean a different steered set.
    prefill_steered=False: generation's prefill was NOT steered on this turn (the proxy sent no
    positions), so only the branch is; `recorded` must then be None."""
    gs = len(prompt_ids)
    if n_branch <= 0:
        raise RuntimeError("a turn with no generated tokens has nothing to steer")
    if not prefill_steered:
        if recorded is not None:
            raise RuntimeError("prefill_steered=False but the sidecar recorded spans for this turn")
        return [[gs, gs + n_branch]]
    spans = assistant_spans(prompt_ids, *markers)
    if recorded is not None and [list(x) for x in recorded] != spans:
        raise RuntimeError(f"recomputed assistant spans {spans[:4]}… differ from the spans the proxy sent "
                           f"at generation {[list(x) for x in recorded][:4]}…")
    return spans + [[gs, gs + n_branch]]


# ----------------------------------------------------------------------- rendering

def with_reasoning_key(messages: list[dict]) -> list[dict]:
    """vLLM's chat parser (chat_utils `_parse_chat_message_content`) renders assistant reasoning
    from its own `reasoning` field and drops the deprecated `reasoning_content`; the server's
    /chat/completions maps `reasoning_content` onto `reasoning` before parsing, the offline
    `_preprocess_chat_one` (like /tokenize) does not. The arms' requests carry `reasoning_content`
    and the server rendered it (the sidecar's prompt hashes match the HF template WITH the
    reasoning on all 13,234 basetasks turns), so the replay fills `reasoning` in exactly as the
    proxy's `_with_reasoning` does for /tokenize (steering_reasoning.proxy, verified 2026-09-14:
    token-identical renders). Copies; input untouched."""
    out = []
    for m in messages:
        if m.get("role") == "assistant" and (m.get("reasoning_content") or m.get("reasoning")):
            m = dict(m); m["reasoning"] = m.get("reasoning") or m.get("reasoning_content")
        out.append(m)
    return out


def render_prompt_ids(llm, messages: list[dict], tools: list[dict] | None,
                      *, add_generation_prompt: bool = True) -> tuple[list[int], str]:
    """(prompt_token_ids, prompt_text) exactly as the vLLM server would render this request."""
    ei = llm._preprocess_chat_one(with_reasoning_key(messages), tools=tools or None,
                                  add_generation_prompt=add_generation_prompt)
    ids = list(ei["prompt_token_ids"])
    text = ei.get("prompt")
    if text is None:
        # re-rendering here would skip vLLM's chat_utils normalisation and could not
        # reproduce `ids`; message attribution needs the renderer's own text
        raise RuntimeError("vLLM renderer returned no prompt text alongside the token ids")
    return ids, text


def render_branch_ids(llm, tokenizer, messages: list[dict], response: dict, tools: list[dict] | None,
                      prompt_ids: list[int], prompt_text: str) -> tuple[list[int], str]:
    """Re-render the logged reply as the template's history form and return the ids that
    follow the prompt: `reasoning\\n</think>\\n\\n<reply>[<tool_call>…]<|im_end|>`.

    Used only when the log has no `token_ids`. Exact where vLLM's parse of the reply was
    lossless; the caller records the delta vs the logged completion_tokens."""
    (response,) = proxy_view([response])  # the proxy strips think tags from the reply too
    _, full_text = render_prompt_ids(llm, messages + [response], tools, add_generation_prompt=False)
    if not full_text.startswith(prompt_text):
        raise RuntimeError("assistant render is not concatenative with the prompt render")
    branch_text = full_text[len(prompt_text):]
    if branch_text.endswith("\n"):
        branch_text = branch_text[:-1]  # template glue after <|im_end|>, never generated
    if not branch_text.endswith("<|im_end|>"):
        raise RuntimeError(f"re-rendered reply does not end in <|im_end|>: {branch_text[-40:]!r}")
    full_ids = tokenizer(prompt_text + branch_text, add_special_tokens=False)["input_ids"]
    if full_ids[: len(prompt_ids)] != prompt_ids:
        # The reply's first token merged with the prompt's last (or the prompt re-tokenized) at
        # the boundary. Strict mode refuses; REPLAY_BOUNDARY_TOLERANT=1 keeps the recorded
        # prompt ids and takes the re-rendered ids past the longest common prefix as the branch
        # (a one- or two-token discrepancy at the boundary; every later generated token is the
        # logged text re-tokenized in context). Used for the 2026-09-25 push-replay of steered
        # transcripts without an ids sidecar.
        if os.environ.get("REPLAY_BOUNDARY_TOLERANT") != "1":
            raise RuntimeError("base tokenization shifted at the branch boundary")
        k = 0
        while k < min(len(full_ids), len(prompt_ids)) and full_ids[k] == prompt_ids[k]:
            k += 1
        print(f"  boundary shift tolerated: prompt {len(prompt_ids)} ids, common prefix {k}", flush=True)
        return full_ids[len(prompt_ids):] if len(full_ids) > len(prompt_ids) else [], branch_text
    return full_ids[len(prompt_ids):], branch_text


def check_branch_matches_reply(tokenizer, branch: list[int], response: dict, where: str) -> None:
    """The sidecar's generated ids must decode to the logged reply: its reasoning (think tags
    stripped, as the proxy does) and its content must both occur verbatim in the decoded stream
    (tool-call XML and the argument rename are not compared: the parser and the proxy edit those)."""
    decoded = tokenizer.decode(branch, skip_special_tokens=False)
    (resp,) = proxy_view([dict(response, role="assistant")])
    for k in ("reasoning", "reasoning_content", "content"):
        v = resp.get(k)
        if isinstance(v, str) and v.strip() and v.strip() not in decoded:
            raise RuntimeError(f"{where}: sidecar ids do not decode to the logged {k} (join collision?): "
                               f"{v.strip()[:60]!r} not in {decoded[:80]!r}…")


def build_turns(rec: dict, llm, tokenizer, *, max_model_len: int | None = None,
                on_branch_mismatch: str = "skip") -> tuple[list[ReplayTurn], list[dict]]:
    """Exact per-turn ids for one record; returns (turns, skipped) with every guard applied.

    on_branch_mismatch: what to do when a re-rendered branch's length differs from the
    server's logged completion_tokens (the parse round-trip lost or added tokens, so the
    rows after the divergence would be a different token stream than generation) —
    "skip" leaves the turn out of the dump (recorded in `skipped`), "fail" raises.
    Never "keep": an inexact turn must not sit in the dump next to exact ones."""
    if on_branch_mismatch not in ("skip", "fail"):
        raise ValueError(f"on_branch_mismatch must be 'skip' or 'fail', got {on_branch_mismatch!r}")
    rid = rec["rollout_id"]
    turns, skipped = [], []
    for t in rec["turns"]:
        mi = t["message_index"]
        if t.get("synthetic_length"):
            skipped.append({"message_index": mi, "reason": "synthetic_length"})
            continue
        bad = [k for k in RENDER_AFFECTING_KEYS if t["request"].get(k) is not None]
        if bad:
            raise RuntimeError(f"{rid} msg{mi}: request carries render-affecting keys {bad}; the "
                               "replay renderer does not reproduce them")
        messages = proxy_view(t["request"]["messages"])
        tools = t["request"].get("tools") or None
        ids, text = render_prompt_ids(llm, messages, tools)
        if t.get("prompt_ids_sha256") is None and not text.endswith(THINK_OPEN):
            # without the sidecar's byte-identity check, a thinking-on generation prompt
            # (the 27B arms) must end in "<think>\n": catches a template/enable_thinking
            # mismatch outright. A thinking-off template (Qwen3.5-2B default) ends in
            # "</think>\n\n" instead and is only accepted when the hash below vouches for it.
            raise RuntimeError(f"{rid} msg{mi}: rendered prompt does not end in {THINK_OPEN!r} — "
                               "template / enable_thinking mismatch with generation")
        logged = t["usage"]["prompt_tokens"]
        if logged is not None and len(ids) != logged:
            raise RuntimeError(f"{rid} msg{mi}: rendered {len(ids)} prompt tokens, server logged {logged}")
        if t.get("prompt_token_ids") is not None and list(t["prompt_token_ids"]) != ids:
            raise RuntimeError(f"{rid} msg{mi}: rendered prompt ids differ from the logged ids")
        if t.get("prompt_ids_sha256") is not None:
            # the proxy's ids sidecar hashed the exact ids the server prefilled: byte identity
            if t.get("n_prompt_ids") not in (None, len(ids)) or ids_sha(ids) != t["prompt_ids_sha256"]:
                raise RuntimeError(f"{rid} msg{mi}: rendered prompt ids differ from the server's "
                                   f"(sidecar sha {t['prompt_ids_sha256']}, n={t.get('n_prompt_ids')})")
        if t.get("token_ids") is not None:
            branch, src, delta = list(t["token_ids"]), "logged", 0
            ct = t["usage"]["completion_tokens"]
            if ct is not None and len(branch) != ct:
                raise RuntimeError(f"{rid} msg{mi}: sidecar has {len(branch)} generated ids, server logged {ct}")
            # the join key is a hash of the request, and every rollout's FIRST request in an arm
            # is identical (task content lives in the sandbox), so a collision can attach another
            # rollout's ids: the ids must decode to the reply this rollout logged. The one known
            # false alarm is a server-side reasoning-parser split (a tool call written inside the
            # think block moves the content/reasoning boundary; ids correct — basetasks findings,
            # base s42 lcbhard_9 msg70): under "skip" that turn is left out and recorded, so one
            # turn does not cost the rollout; under "fail" it aborts as before.
            try:
                check_branch_matches_reply(tokenizer, branch, t["response"], f"{rid} msg{mi}")
            except RuntimeError as e:
                if on_branch_mismatch == "fail":
                    raise
                skipped.append({"message_index": mi, "reason": "branch_decode_mismatch", "detail": str(e)[:200],
                                "finish_reason": t.get("finish_reason")})
                continue
        else:
            branch, _ = render_branch_ids(llm, tokenizer, messages, t["response"], tools, ids, text)
            src = "rendered"
            ct = t["usage"]["completion_tokens"]
            delta = len(branch) - ct if ct is not None else None
        if not branch:
            skipped.append({"message_index": mi, "reason": "empty_branch", "finish_reason": t.get("finish_reason")})
            continue
        if delta:
            msg = (f"{rid} msg{mi}: re-rendered branch has {len(branch)} tokens, server logged "
                   f"{t['usage']['completion_tokens']} (delta {delta:+d}) — not the generated stream")
            if on_branch_mismatch == "fail":
                raise RuntimeError(msg)
            skipped.append({"message_index": mi, "reason": "branch_delta", "delta": delta,
                            "finish_reason": t.get("finish_reason"),
                            "tool_calls_collapse_suspect": bool(t.get("tool_calls_collapse_suspect"))})
            continue
        # the capture prefill generates one token after the last branch row, so the engine
        # needs prompt + branch + 1 ≤ max_model_len (capture_rollouts.py builds it one larger
        # than the generation context for exactly this reason)
        if max_model_len is not None and len(ids) + len(branch) + 1 > max_model_len:
            raise RuntimeError(f"{rid} msg{mi}: {len(ids) + len(branch)} tokens (+1 for the capture step) "
                               f"exceed the engine's max_model_len {max_model_len}; the replay engine must "
                               "be built with the generation context length + 1")
        turns.append(ReplayTurn(mi, ids, text, branch, messages, token_ids_source=src,
                                branch_delta=delta or 0, steer_positions=t.get("steer_positions"),
                                ids_join=t.get("ids_join", "none"),
                                flags={"finish_reason": t.get("finish_reason"),
                                       "tool_calls_collapse_suspect": bool(t.get("tool_calls_collapse_suspect"))}))
    if not turns:
        raise RuntimeError(f"{rid}: no replayable turns")
    return turns, skipped


# ------------------------------------------------------------------------- capture

def clamp_check(rows: torch.Tensor, u: torch.Tensor, threshold: float) -> dict:
    """Clamp invariant on (n, d) post-steering residual rows: proj_u <= thr + CLAMP_TOL·‖h‖."""
    h = rows.float()
    proj = h @ u.to(h.device)
    tol = CLAMP_TOL * h.norm(dim=1)
    excess = proj - threshold - tol
    return {"n_rows": int(h.shape[0]), "max_excess": float(excess.max()) if h.shape[0] else 0.0,
            "n_violations": int((excess > 0).sum()), "max_proj": float(proj.max()) if h.shape[0] else None}


def layer_chunk_for(max_prompt: int, d_model: int, layer_chunk: int, host_margin_gib: int) -> int:
    """Layers per pass: payload cap on the longest prompt, and host RAM for the bf16 rows
    plus their fp32 working copies (3 bytes-multiples, as capture_all sizes it)."""
    import psutil

    cap = max(1, PAYLOAD_CAP_BYTES // (max_prompt * d_model * 2))
    usable = max(0, psutil.virtual_memory().available - host_margin_gib * (1 << 30))
    fit = max(1, usable // (max_prompt * d_model * 2 * 3))
    return int(max(1, min(layer_chunk, cap, fit)))


def capture_rollout(rec: dict, capturer, layers: list[int], layer_chunk: int, out_dir: Path,
                    s3_prefix: str | None, *, steering: SteeringSpec | None,
                    host_margin_gib: int = 128, on_branch_mismatch: str = "skip",
                    prefix_reuse: bool = False, log=print) -> dict:
    """Capture one rollout into the shard format; returns its manifest row. Meta is written
    (and uploaded) last, so its presence is the completion marker resume keys on.

    A steered replay needs the proxy's ids sidecar on EVERY turn: it is what says which spans
    generation steered — a turn the proxy sent without positions had an unsteered prefill,
    and only the sidecar can tell that apart from "no record".

    prefix_reuse=True (UNSTEERED records only): each turn is prefilled through the prefix cache
    (`ActivationCapturer.capture_suffix`) and only the computed suffix is captured; the tree is
    built with `share_branch=True`, so the keep range starts at the LCP with the previous
    turn — the positions the cache can serve are exactly the ones an earlier turn stored. A
    cache hit past the keep range (an older cached sequence, after a keep_start collapse)
    falls back to a full prefill of that turn and is counted in `n_prefix_reuse_fallbacks`."""
    llm, tokenizer = capturer.llm, capturer.llm.get_tokenizer()
    rid, task_id = rec["rollout_id"], rec["task_id"]
    mode, dose = rec["mode"], rec["dose"]
    if prefix_reuse and steering is not None:
        raise RuntimeError(f"{rid}: prefix reuse is for unsteered records only (a cached block's KV carries "
                           "the steering that produced it; steered replays recompute every turn)")
    if (mode is None) != (steering is None):
        raise RuntimeError(f"{rid}: record steering {mode} {dose} vs --mode/--dose {steering}")
    if steering is not None and (mode, dose) != (steering.mode, steering.dose):
        raise RuntimeError(f"{rid}: record steering {mode} {dose} != --mode {steering.mode} --dose {steering.dose}")

    max_len = int(llm.llm_engine.model_config.max_model_len)
    turns, skipped = build_turns(rec, llm, tokenizer, max_model_len=max_len,
                                 on_branch_mismatch=on_branch_mismatch)
    if steering is not None:
        bad = [t.message_index for t in turns if t.ids_join != "sidecar"]
        if bad:
            raise RuntimeError(f"{rid}: a steered replay needs a sidecar line on every turn; turns {bad[:6]} "
                               f"joined as {sorted({t.ids_join for t in turns if t.ids_join != 'sidecar'})} "
                               "(pass the proxy's --ids-log to export_eval_records.py)")
        if all(t.steer_positions is None for t in turns):
            # the proxy refuses a steered request whose spans fail, so null on EVERY turn means the
            # arm ran through a proxy that never sent positions (not this harness): refuse rather
            # than replay it branch-only
            raise RuntimeError(f"{rid}: the sidecar recorded no steer_positions on any turn — the arm's proxy "
                               "did not steer prefill positions; this replay reproduces only the current harness")
    tree = replay_tree(task_id, turns, tokenizer, share_branch=prefix_reuse)
    N, d_model = tree["n_tree"], capturer.d_model
    # the payload/RAM bound is on the longest PROMPT, not the delta: a cache hit is not known
    # ahead of time (eviction can drop it to 0), so a fallback turn returns every prompt row
    chunk = layer_chunk_for(tree["max_prompt"], d_model, layer_chunk, host_margin_gib)
    chunks = [layers[i:i + chunk] for i in range(0, len(layers), chunk)]
    log(f"  {rid}: {len(turns)} turns ({len(skipped)} skipped), N={N:,} tree rows, "
        f"max prompt {tree['max_prompt']:,}, {len(chunks)} pass(es) of ≤{chunk} layers"
        f"{', prefix reuse' if prefix_reuse else ''}")

    idx = index_tensors(tree)
    uploader = ThreadPoolExecutor(max_workers=6) if s3_prefix else None
    upload_futs: list = []
    try:
        return _capture_passes(rec, tree, turns, skipped, capturer, layers, chunks, chunk, idx, out_dir,
                               s3_prefix, steering, uploader, upload_futs, log, prefix_reuse=prefix_reuse)
    finally:
        if uploader:
            for f in upload_futs:
                try:
                    f.result()
                except Exception:  # noqa: BLE001 — the primary error is already propagating
                    pass
            uploader.shutdown()


def _capture_passes(rec, tree, turns, skipped, capturer, layers, chunks, chunk, idx, out_dir,
                    s3_prefix, steering, uploader, upload_futs, log, *, prefix_reuse: bool = False) -> dict:
    rid, task_id = rec["rollout_id"], rec["task_id"]
    N, d_model = tree["n_tree"], capturer.d_model
    turn_means_parts: list[torch.Tensor] = []
    turn_tokens: list[int] = []
    turn_meta: list[dict] = []
    n_nonfinite = 0
    n_fallbacks = 0
    clamp_stats: dict[int, dict] = {}
    u = steering.unit() if steering else None
    markers = span_markers(capturer.llm.get_tokenizer()) if steering else None
    steered: dict[int, list[list[int]] | None] = {}   # per turn: the absolute spans the replay steers

    for ci, chunk_layers in enumerate(chunks):
        if prefix_reuse:
            # start every pass cold: the previous pass (same prompts) or the previous rollout
            # (turn-0 prompts are identical across an arm's rollouts) would otherwise hit past
            # keep_start and force fallbacks
            capturer.reset_prefix_cache()
        mm = {li: create_acts_shard(out_dir / f"L{li:02d}" / f"{task_id}.safetensors", N, d_model, idx)
              for li in chunk_layers}
        t_means: list[torch.Tensor] = []
        row_ptr = 0
        for ti, pt in enumerate(tree["turns"]):
            t: ReplayTurn = pt["turn"]
            ids, ks, gs = pt["ids"], pt["keep_start"], pt["gen_start"]
            t0 = time.monotonic()
            if ti not in steered:
                # sidecar recorded null → the proxy sent no positions on this turn: generation's
                # prefill was unsteered, only decode was (capture_rollout required the sidecar)
                steered[ti] = (steered_positions(t.prompt_ids, len(t.branch_ids), markers, t.steer_positions,
                                                 prefill_steered=t.steer_positions is not None)
                               if steering else None)
            payload = steering.payload(steered[ti]) if steering else None
            fp = 0   # absolute position of act's first row
            if prefix_reuse:
                act, fp = capturer.capture_suffix([ids], chunk_layers, token_ids=True)[0]  # (L, n - fp, d)
                if fp > ks:
                    # the cache served rows this turn must store (a hit on an older cached sequence,
                    # possible after a keep_start collapse): recompute the turn from scratch
                    del act
                    act, fp = capturer.capture([ids], chunk_layers, token_ids=True)[0], 0
                    n_fallbacks += 1
            else:
                act = capturer.capture([ids], chunk_layers, token_ids=True, steering=payload)[0]  # (L, n, d)
            if act.shape[1] != len(ids) - fp:
                raise RuntimeError(f"{rid} msg{t.message_index}: captured {act.shape[1]} rows for {len(ids) - fp} "
                                   f"ids (first_pos {fp})")
            keep = act[:, ks - fp:, :]
            n_keep = keep.shape[1]
            for row, li in enumerate(chunk_layers):
                x = keep[row].to(torch.bfloat16)
                n_nonfinite += int((~torch.isfinite(x)).any(dim=-1).sum())
                mm[li][row_ptr:row_ptr + n_keep] = x.view(torch.uint16).cpu().numpy()
            if steering is not None and steering.mode == "clamp" and steering.layer in chunk_layers:
                # every row the replay steered (earlier assistant spans + branch), except the last
                # branch row: the final sampled token was never fed back at generation, so its
                # row is replay-only (tree["never_computed_rows"])
                layer_act = act[chunk_layers.index(steering.layer)]
                ranges = [(a, min(b, len(ids) - 1)) for a, b in steered[ti]]
                pieces = [layer_act[a:b] for a, b in ranges if b > a]
                if not pieces:   # a 1-token first turn: no computed steered row exists; nothing to check
                    clamp_stats[ti] = None
                else:
                    clamp_stats[ti] = cs = clamp_check(torch.cat(pieces), u, steering.threshold)
                if clamp_stats[ti] is not None and cs["n_violations"]:
                    raise RuntimeError(f"{rid} msg{t.message_index}: clamp invariant violated on "
                                       f"{cs['n_violations']}/{cs['n_rows']} steered rows at L{steering.layer} "
                                       f"(max excess {cs['max_excess']:.3f}) — steering hook did not apply")
            span = act[:, pt["reasoning_start"] - fp:pt["reasoning_end"] - fp, :]  # gen_start >= ks >= fp
            if span.shape[1]:
                t_means.append(span.mean(dim=1, dtype=torch.float32).cpu())
            else:  # the model closed </think> immediately: no reasoning rows this turn
                t_means.append(torch.zeros(len(chunk_layers), d_model))
            if ci == 0:
                turn_tokens.append(int(span.shape[1]))
                turn_meta.append({
                    "turn_row": ti, "row_start": pt["row_start"], "message_index": t.message_index,
                    "n_tokens": int(span.shape[1]), "reasoning_start": pt["reasoning_start"],
                    "reasoning_end": pt["reasoning_end"], "trunk_len": gs,
                    "prompt_n_tokens": len(ids), "base_n_tokens": pt["reasoning_end"] + 1,
                    "response_start": pt["resp_tok_start"], "tool_call_start": pt["tool_tok_start"],
                    "keep_start": ks, "first_pos": fp,
                    "gen_start": gs, "steer_positions": steered[ti], "ids_join": t.ids_join,
                    "prefill_unsteered_at_generation": bool(steering) and not t.steer_positions,
                    "prompt_tokens_rendered": gs, "branch_tokens": pt["n_branch"],
                    "branch_delta": t.branch_delta, "token_ids_source": t.token_ids_source,
                    **t.flags,
                })
            act = keep = span = None  # noqa: F841
            row_ptr += n_keep
            log(f"    pass {ci + 1}/{len(chunks)} turn {ti + 1}/{len(tree['turns'])} n={len(ids):,} "
                f"{time.monotonic() - t0:.1f}s")
        if row_ptr != N:
            raise RuntimeError(f"{rid}: wrote {row_ptr} rows, tree has {N}")
        turn_means_parts.append(torch.stack(t_means))
        for li in chunk_layers:
            mm[li].flush()
            del mm[li]
            path = out_dir / f"L{li:02d}" / f"{task_id}.safetensors"
            if uploader:
                upload_futs.append(uploader.submit(s3_cp_unlink, path, s3_prefix, f"L{li:02d}/{task_id}.safetensors"))
        del mm

    for ti, cs in clamp_stats.items():
        turn_meta[ti].update({"n_clamped_rows": cs["n_rows"] if cs else 0,
                              "clamp_max_excess": cs["max_excess"] if cs else None,
                              "clamp_max_proj": cs["max_proj"] if cs else None,
                              "clamp_invariant_ok": (cs["n_violations"] == 0) if cs else None})
    checked = [cs for cs in clamp_stats.values() if cs is not None]
    tok_path = write_tokens(out_dir, task_id, tree["tok_ids"], idx)
    seg_path = write_segmeans(out_dir, task_id, torch.cat(turn_means_parts, dim=1), turn_tokens)
    row = {
        "task_id": task_id, "rollout_id": rid, "arm": rec["arm"], "mode": rec["mode"], "dose": rec["dose"],
        "seed": rec["seed"], "split": rec["split"],
        "label": bool(rec.get("is_reward_hack")), "passed": rec.get("passed"),
        "is_reward_hack": rec.get("is_reward_hack"), "solved": rec.get("solved"),
        "has_gold_spans": False, "n_turns": len(turn_meta), "n_tree_tokens": N,
        "n_reasoning_tokens": tree["n_reasoning"], "n_segments": 0,
        "n_skipped_turns": len(skipped), "n_prefix_reuse_fallbacks": n_fallbacks,
        "prefix_reuse": prefix_reuse, "share_branch": bool(tree.get("share_branch")),
        "n_nonfinite_token_layer_rows": n_nonfinite, "layer_chunk_used": chunk,
        "n_turns_rendered_branch": sum(t.token_ids_source == "rendered" for t in turns),
        "n_turns_skipped_branch_delta": sum(s.get("reason") == "branch_delta" for s in skipped),
        # None = no clamp check ran (unsteered arm, add/ablate mode, or no computed steered row)
        "clamp_invariant_ok": all(cs["n_violations"] == 0 for cs in checked) if checked else None,
    }
    meta_path = write_meta(out_dir, task_id, row, {
        "skipped_turns": skipped, "keep_start_collapses": tree["keep_start_collapses"],
        "im_end_rows": tree["im_end_rows"],
        "never_computed_rows": tree["never_computed_rows"],
        "steering": steering.provenance() if steering else None,
        "generation": {"model_base_url": rec.get("model_base_url"), "eval_file": rec.get("eval_file"),
                       "sample_uuid": rec.get("sample_uuid"), "max_tokens_effective": rec.get("max_tokens_effective"),
                       "limit": rec.get("limit"), "error": rec.get("error")},
        "turns": turn_meta, "segments": [],
    })
    if s3_prefix:
        for f in upload_futs:  # every layer shard must land before the marker
            f.result()
        upload_futs.clear()
        s3_cp(tok_path, s3_prefix, f"tokens/{task_id}.safetensors")
        s3_cp(seg_path, s3_prefix, f"segmeans/{task_id}.safetensors")
        s3_cp(meta_path, s3_prefix, f"meta/{task_id}.json")  # completion marker
    return row


def generation_serving_block(records: list[dict]) -> dict:
    """What produced the rollouts, from the records + the arms' server flags (serving/bootstrap.sh)."""
    r0 = records[0]
    return {
        "model_base_urls": sorted({r.get("model_base_url") for r in records if r.get("model_base_url")}),
        "seed": r0["seed"], "arm": r0["arm"], "mode": r0["mode"], "dose": r0["dose"],
        "temperature": r0["turns"][0]["request"].get("temperature") if r0["turns"] else None,
        "top_p": r0["turns"][0]["request"].get("top_p") if r0["turns"] else None,
        "max_tokens_effective": r0.get("max_tokens_effective"),
        "decode_only_gate": "asserted by the launch procedure (scripts/steer_cache_probe.py), not recorded per request",
        "prefix_caching": True,
        "reasoning_parser": "qwen3", "tool_call_parser": "qwen3_coder",
        "note": "vLLM 0.24 bf16 enforce_eager max_model_len 131072, VLLM_LENS_STEER_DECODE_ONLY=1 "
                "(positions-aware gate, serving/patch_vllm_lens_gate.py): decode steers every generated "
                "token, prefill steers the proxy's steer_positions (the model's earlier replies); steered "
                "requests read the prefix cache (VLLM_LENS_STEER_READ_PREFIX_CACHE=1, cache_salt per arm); "
                "VLLM_ENFORCE_STRICT_TOOL_CALLING=0",
    }
