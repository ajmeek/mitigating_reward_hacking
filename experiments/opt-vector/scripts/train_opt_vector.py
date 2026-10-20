#!/usr/bin/env python3
"""Train a src-only suppression steering vector with Dunefsky-style optimization.

Bypasses experiments/transfer/train_optimization_vectors.py entirely: that CLI is welded
to a MODEL_REGISTRY, an 8-cell SCENARIOS table and a probes/meek/*.json layout that do not
exist in this checkout. We import optimize_completion directly instead.

v1/v3a are SRC-ONLY: src_completions = [the judge-labelled hack sentence], suppressed via
-log(1-p); dst_completions = [] (no promote term). v3b additionally carries
dst_completions (forward-looking refusal variants), promoted via -log p -- the batched HF
path in steering_opt_batched.py runs the src chunk with do_one_minus=True and the dst
chunk with do_one_minus=False, both backward into the same vector grad per iteration.
Rows without dst still train exactly as before.

Several upstream defects are worked around here; each is commented with WHY at its point
of use. Two of them (P1 log1mexp NaN, P2 prompt-length off-by-one) are silent-corruption
bugs that were only found by instrumenting a real run. Both produce an
artifact whose metadata reads like a converged run, so both are also asserted against:
check_trajectory() refuses to save a run with a dead iteration, and validate_alignment()
raises if the prompt/completion token boundary is ever wrong again.

--steer-mode clamp trains a vector THROUGH the eval's clamp operation instead of the add
op: at every steered position, u = v/||v||, proj = h.u, h <- h - max(0, proj - thr)*u --
the exact math of serving/patch_vllm_lens_modes.py (MODE_BODY, clamp branch), so what the
optimizer sees is what the 27B server will apply. Same objective (src suppress + optional
dst promote, satisfice hinge), same P1/P2 patches, same NaN guard and check_trajectory.
Sign convention flips: the clamp only ever REMOVES the component along +u, so suppression
training keeps u pointing TOWARD hacking (diffmean convention) -- the opposite polarity to
add-mode as-trained vectors. Default init is the diffmean_judge42_L31 direction, which
already points that way. --steer-mode add is byte-for-byte the old behavior.

Run with the /workspace/optvec venv (torch>=2.5 + transformers 5.x).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

# steering_opt_batched.py (Kyle's optimizer from steering-vectors-for-faithfulness, experiments/transfer)
# is vendored next to this script so the trainer is self-contained.
FAITHFULNESS_TRANSFER = str(Path(__file__).resolve().parent)

# Default clamp-mode init: the eval's best baseline direction (positive = TOWARD hacking,
# which is the polarity clamp training needs -- see the --steer-mode help text).
DIFFMEAN_INIT = str(Path(__file__).resolve().parents[3] / "src/steering_reasoning/canonical/u_L35_steering.safetensors")   # init of record for the clamp vector


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="HF repo id, e.g. Qwen/Qwen3.5-2B")
    p.add_argument("--layer", type=int, required=True, help="block index for the forward-PRE hook")
    p.add_argument("--examples", required=True, help="JSONL from build_opt_examples.py")
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--max-iters", type=int, default=20,
                   help="upstream breaks on `iters > max_iters` (STRICT >), so N here runs "
                        "N+1 iterations and yields an N+1-entry loss history")
    p.add_argument("--max-norm", type=float, default=5.0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--out", required=True, help="destination .pt")
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16",
                   help="bfloat16 for GPU; float32 for a CPU run (bf16 matmul on CPU is slow). "
                        "NB P1 (the _log1mexp NaN) is a bf16 rounding artefact -- it is rarer in "
                        "float32, but the clamp is applied either way.")
    p.add_argument("--limit", type=int, default=None, help="use only the first N examples")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--starting-norm", type=float, default=1.0)
    p.add_argument("--max-consecutive-nan", type=int, default=3,
                   help="abort after this many consecutive iterations that produce a "
                        "non-finite gradient OR no gradient at all")
    p.add_argument("--log1mexp-clamp", type=float, default=-1e-6,
                   help="upper clamp on log p before log(1-p); see patch_log1mexp")
    p.add_argument("--allow-no-progress", action="store_true",
                   help="downgrade the 'final loss did not beat the first' abort to a warning")
    p.add_argument("--exclude-task-ids", default="",
                   help="comma-separated meta.task_id values to drop (robustness refits)")
    # ---------------------------------------------------------------------------------
    # Satisficing. Without it the objective is unbounded: -log(1-p) is minimised by driving
    # p to 0 on every completion token, and the cheapest way to do that is to wreck the
    # output distribution outright. Measured on Qwen3.5-2B/L13: an unconstrained run reached
    # loss 0.098 with a norm-5.0 vector whose scale-1.0 generation is pure gibberish
    # ("End_ending_Aq_:蔚___天_蓝_出_Qim_"), while the model is fine at scale <= 0.4.
    # With --satisfice, steering_opt_batched.py:683 switches the src term to a hinge,
    #   ((src_losses - target_loss).clamp_min(0) ** 2).sum()
    # so a completion already suppressed below --target-loss contributes zero gradient and
    # the vector stops growing. Suppress enough, then stop.
    # ---------------------------------------------------------------------------------
    p.add_argument("--satisfice", action="store_true",
                   help="hinge the src loss at --target-loss instead of minimising to zero")
    p.add_argument("--target-loss", type=float, default=None,
                   help="per-completion loss floor for --satisfice (summed over completion tokens)")
    p.add_argument("--min-completion-tokens", type=int, default=0,
                   help="drop rows whose meta.n_completion_tokens is below this")
    p.add_argument("--debug", action="store_true", help="per-completion loss spam from the trainer")
    # ---------------------------------------------------------------------------------
    # Clamp mode. The eval's best baseline (diffmean_clamp) is not an add: the server
    # measures each position's projection onto the vector's unit direction and removes
    # the excess above a threshold. --steer-mode clamp trains the vector THROUGH that
    # exact op (see clamp_op / patch_clamp_steering), so the optimizer's regime matches
    # what run_prefill_opt.py will apply. Sign consequence: clamp only ever removes the
    # +u component, so the learned u points TOWARD hacking (diffmean convention), the
    # OPPOSITE of add-mode as-trained vectors -- recorded in the saved metadata.
    # ---------------------------------------------------------------------------------
    p.add_argument("--steer-mode", choices=["add", "clamp"], default="add",
                   help="add: h += v at steered positions (v1..v3b, unchanged). "
                        "clamp: u = v/||v||, h -= max(0, h.u - thr)*u, matching the "
                        "serving patch; --max-norm is ignored (v is renormalized to "
                        "unit after every step) and --clamp-thr sets thr")
    p.add_argument("--clamp-thr", type=float, default=-5.0,
                   help="clamp threshold on the h.u projection (steer-mode=clamp only); "
                        "-5.0 is the diffmean_clamp eval setting")
    p.add_argument("--init-vector", default=None,
                   help="repo .safetensors artifact to start from instead of a random "
                        "vector. In clamp mode defaults to the diffmean_judge42_L31 "
                        "direction (already toward-hacking); in add mode default stays "
                        "random (byte-for-byte old behavior)")
    p.add_argument("--validate-only", action="store_true",
                   help="load examples, run the P2 whitespace repair + alignment "
                        "validation with the tokenizer alone, print a summary and exit "
                        "before any model load")
    return p.parse_args()


# ---------------------------------------------------------------------------------------
# P1 (CRITICAL): NaN loss from _log1mexp applied before the completion mask
# ---------------------------------------------------------------------------------------
def patch_log1mexp(steering_opt, clamp_max):
    """steering_opt_batched.py:284 is

        token_loss = -_log1mexp(logp_next.float()) * comp_mask

    i.e. _log1mexp runs on the FULL (B,T-1) tensor and the completion mask is multiplied
    on afterwards. logp_next comes from log_softmax(logits * temperature) on **bf16**
    logits, so at a confident PROMPT position p rounds to exactly 1.0, logp_next == 0.0,
    _log1mexp(0) = log(-expm1(0)) = log(0) = -inf, and then -(-inf) * 0.0 = NaN.
    Masking does not kill an inf. One such position anywhere in the prompt NaNs the whole
    batch, upstream's chunk-level `if torch.isnan(src_term).any(): continue` (line 687)
    skips backward, vector.grad stays None, optimizer.step() is a no-op, and the
    prev_loss early-stop fires because loss stays 0.0. Observed: iter0 loss 94.2,
    iter1 loss 0.0, run "converges" after exactly one Adam step.

    Clamping log p just below 0 makes the -inf finite (~13.8 at clamp=-1e-6) and, since
    those positions are masked out anyway, changes nothing that is actually scored.

    CAVEAT, measured on a real run: torch.clamp passes ZERO gradient above the
    clamp, so if a *scored* completion token were ever confident enough to round to
    p == 1.0 in bf16 (needs p > 1 - 2^-9 ~ 0.998), the objective would stop pushing on
    exactly that token. Measured on the smoke set: see the "how often does the clamp
    bite" line. The alternative -- casting logits to float32 before
    log_softmax -- doubles the (B,T,248077) logits tensor, which is the single largest
    allocation in the run, so it is not free.
    """
    orig = steering_opt._log1mexp

    def clamped(log_p):
        return orig(log_p.float().clamp(max=clamp_max))

    steering_opt._log1mexp = clamped
    return orig


# ---------------------------------------------------------------------------------------
# P2 (CRITICAL): prompt_lens off-by-one drops the first token of the hack sentence
# ---------------------------------------------------------------------------------------
def shift_trailing_whitespace(prompt, completions):
    """_tokenize_pairs (steering_opt_batched.py:186) computes

        prompt_lens = [len(tokenizer(p).input_ids) for (p, _) in pairs]

    i.e. it tokenizes the prompt ALONE and assumes that is a token-prefix of
    tokenizer(p + c). build_opt_examples.py cuts every prompt exactly at a sentence
    boundary, so most prompts end in whitespace. Alone, that trailing space is its own
    token ('Ġ'); jointly it merges into the completion's first word ('ĠMaybe'). prompt_lens
    then over-counts by 1, _build_completion_mask starts at max(pl-1,0) one token late, and
    the single highest-value token to suppress is handed to the model as free context --
    the objective silently becomes "given '...different approach. Maybe', suppress
    ' I can use a global counter'", when the hack has already begun.

    Measured on the real dataset with the Qwen3.5-2B tokenizer: 106/190 rows (56%)
    violated the prefix property; 0/190 after this shift. p + c is unchanged, so the model
    sees exactly the same token sequence -- only the prompt/completion boundary moves.

    Applied CONDITIONALLY (see repair_alignment): on the other 84 rows the trailing
    whitespace does NOT merge (it is a standalone 'ĊĊ' token) and the prefix property
    already holds. Shifting those too would move a pure-newline token into the
    suppression target on all 84 -- i.e. train the vector to avoid paragraph breaks.
    Repair only what is broken.
    """
    stripped = prompt.rstrip()
    trail = prompt[len(stripped):]
    if not trail:
        return prompt, completions
    return stripped, [trail + c for c in completions]


def _is_token_prefix(tok, prompt, completions):
    pl = tok(prompt).input_ids
    return all(tok(prompt + c).input_ids[:len(pl)] == pl for c in completions)


def repair_alignment(tok, dps):
    """Apply the P2 shift only to the rows that actually need it.

    src and dst pairs share the same prompt in _tokenize_pairs, so the shift must be
    decided over src+dst JOINTLY and, when applied, must move the trailing whitespace
    onto EVERY completion -- shifting the prompt for src while leaving dst untouched
    would silently delete the whitespace between prompt and dst text.
    """
    import dataclasses
    out, n_shift, n_fail = [], 0, 0
    for dp in dps:
        prompt = dp.prompt
        src, dst = list(dp.src_completions), list(dp.dst_completions)
        comps = src + dst
        if not _is_token_prefix(tok, prompt, comps):
            p2, c2 = shift_trailing_whitespace(prompt, comps)
            if _is_token_prefix(tok, p2, c2):
                prompt, n_shift = p2, n_shift + 1
                src, dst = c2[:len(src)], c2[len(src):]
            else:
                n_fail += 1
        out.append(dataclasses.replace(dp, prompt=prompt, src_completions=src,
                                       dst_completions=dst))
    print(f"  P2 alignment repair: {n_shift}/{len(dps)} rows had their trailing whitespace "
          f"moved onto the completion; {len(dps) - n_shift - n_fail} were already correct "
          f"and were left alone; {n_fail} unrepairable", flush=True)
    return out


def validate_alignment(tok, dps, name="src"):
    """Assert the property upstream ASSUMES, so P2 can never regress silently."""
    bad = []
    for i, dp in enumerate(dps):
        pl = tok(dp.prompt).input_ids
        for c in list(dp.src_completions) + list(dp.dst_completions):
            full = tok(dp.prompt + c).input_ids
            if full[:len(pl)] != pl:
                bad.append((i, len(pl), repr(dp.prompt[-30:]), repr(c[:30])))
    if bad:
        for row in bad[:5]:
            print(f"  MISALIGNED row {row[0]} prompt_len={row[1]} tail={row[2]} comp={row[3]}")
        raise RuntimeError(
            f"{len(bad)}/{len(dps)} {name} pairs violate the token-prefix property that "
            "_tokenize_pairs assumes; the completion mask would be off by one on those rows"
        )
    print(f"  token alignment OK: tok(prompt) is an exact prefix of tok(prompt+completion) "
          f"on all {len(dps)} rows", flush=True)


def load_datapoints(steering_opt, path, limit=None, exclude_task_ids=(), min_completion_tokens=0):
    """JSONL -> TrainingDatapoint. Each row: {"prompt":..., "src_completions":[...], "meta":{...}}"""
    dps, metas = [], []
    n_read = n_excl = n_short = 0
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n_read += 1
            meta = row.pop("meta", None) or {}  # provenance only; the trainer must not see it
            if meta.get("task_id") in exclude_task_ids:
                n_excl += 1
                continue
            if min_completion_tokens and (meta.get("n_completion_tokens") or 0) < min_completion_tokens:
                n_short += 1
                continue
            src = row["src_completions"]
            if isinstance(src, str):
                src = [src]
            # v3b: dst_completions are threaded through to the promote term (-log p,
            # do_one_minus=False on the dst chunk in steering_opt_batched.py). A dst-less
            # row still gets [], i.e. src-only training is unchanged (v3a depends on that).
            dst = row.get("dst_completions") or []
            if isinstance(dst, str):
                dst = [dst]
            if row.get("is_negative"):
                raise ValueError(f"{path}: row has is_negative=True, unsupported in v1")
            # P2 alignment is repaired later, in repair_alignment(), because it needs the
            # model's tokenizer and must be applied only to the rows that need it.
            dps.append(steering_opt.TrainingDatapoint(
                prompt=row["prompt"],
                src_completions=src,
                dst_completions=dst,
                is_negative=False,
            ))
            metas.append(meta)
            if limit is not None and len(dps) >= limit:
                break
    print(f"  read {n_read} rows; excluded {n_excl} by task_id; dropped {n_short} short "
          f"completions; kept {len(dps)}", flush=True)
    return dps, metas


def load_tokenizer(args):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # FIX 2 (pad token): _tokenize_pairs calls tokenizer(..., padding=True) and only the
    # upstream CLI ever set pad_token, which we bypass. Every Qwen tokenizer in this
    # project already ships one (<|endoftext|>), so this branch is a safety net for models
    # that do not -- it is NOT load-bearing here. Set pad_token only and let pad_token_id
    # follow from it, so we never retarget an existing pad id.
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(args):
    from transformers import AutoModelForCausalLM

    tok = load_tokenizer(args)

    kwargs = dict(device_map={"": args.device}, trust_remote_code=True)
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
    else:
        # `torch_dtype=` is deprecated on transformers 5.x
        kwargs["dtype"] = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    # FIX 5: do NOT call model.resize_token_embeddings. We only aliased pad->eos, so the
    # vocab is unchanged; resizing reinitialises rows and is unsafe under 4-bit/device_map.
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tok.pad_token_id

    # FIX 1 (freeze): requires_grad_(False) appears nowhere in steering_opt_batched.py, yet
    # the Adam param list is only [vector]. Without freezing, every backward() allocates a
    # .grad for every model weight -> a second full copy of the model in fp32-ish memory.
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


# ---------------------------------------------------------------------------------------
# CLAMP MODE: train through the eval's clamp op instead of the add op
# ---------------------------------------------------------------------------------------
def clamp_op(x, v, thr, gate=None):
    """Training-time twin of the clamp branch in serving/patch_vllm_lens_modes.py:

        u = (vec / vec.norm()).float()
        proj = h.float() @ u
        h -= (max(0, proj - thr).unsqueeze(1) * u.unsqueeze(0)).to(h.dtype)

    x: (B,T,d) residual stream (bf16 or fp32); v: (d,) LEARNABLE parameter; gate:
    (B,T,1) 0/1 float marking steered positions (None = all). The normalization happens
    INSIDE the op so gradients flow through it: dL/dv = (I - uu^T)/||v|| dL/du, i.e. the
    optimizer turns the direction rather than scaling a frozen unit vector. All clamp
    math runs in float32 and the correction is cast to x.dtype exactly once, mirroring
    the serving patch, so bf16 activations cannot NaN the projection.

    Returns (steered x, proj.detach(), excess.detach()) -- the detached tensors feed the
    bind-fraction telemetry in patch_clamp_steering.
    """
    u = (v / v.norm()).float()                 # unit direction, grads flow through norm
    proj = x.float() @ u                       # (B,T) projection onto u
    excess = (proj - thr).clamp(min=0.0)       # only the component ABOVE thr is removed
    delta = excess.unsqueeze(-1) * u           # (B,T,d), fp32 like the serving patch
    if gate is not None:
        delta = delta * gate                   # zero on pad/unsteered positions
    return x - delta.to(x.dtype), proj.detach(), excess.detach()


def _flush_bind_stats(stats):
    """Print (and reset) the fraction of steered positions where the clamp bound."""
    if stats["steered"] == 0:
        return
    frac = stats["bound"] / stats["steered"]
    stats["iter_fracs"].append(round(frac, 4))
    line = (f"  clamp binds on {frac:.4f} of steered positions "
            f"({stats['bound']}/{stats['steered']} with proj > thr)")
    if frac < 0.01:
        line += "  !! ~0: the clamp almost never fires -- this run is DEAD (thr too high?)"
    print(line, flush=True)
    stats["bound"] = stats["steered"] = 0


def patch_clamp_steering(steering_opt, thr):
    """Swap the batched ADD hook for the serving clamp op.

    make_steering_hook_hf_mask is looked up on the module at call time inside
    _batched_losses_hf_with_hook, so replacing the module attribute reroutes every
    hooked forward. vector_getter() returns the raw learnable parameter (`_vector_ref`
    in optimize_completion with vector_clamp=None), which is exactly what clamp_op
    needs -- it normalizes internally.

    Telemetry: a clamp whose threshold sits above every projection is a measured no-op
    -- excess 0 everywhere, zero gradient, and the run would "converge" instantly at
    its init. That failure must be VISIBLE, so the hook counts positions with
    proj > thr and a wrapper around clip_grad_norm_ (which runs exactly once per
    iteration, line 761 upstream; the NaN guard is installed on top of this wrapper
    afterwards) prints the per-iteration bind fraction.
    """
    stats = {"bound": 0, "steered": 0, "iter_fracs": []}
    orig_make = steering_opt.make_steering_hook_hf_mask

    def make_clamp_hook(vector_getter, matrix_getter=None, token_mask_getter=None,
                        use_pre_add=True):
        def hook_fn(module, args):
            x = args[0]  # (B,T,d)
            if matrix_getter is not None and matrix_getter() is not None:
                raise RuntimeError("clamp mode does not support an affine/ablation matrix")
            if token_mask_getter is not None:
                full_mask = token_mask_getter()
                if full_mask.shape[0] != x.shape[0]:
                    # upstream slices the mask for DataParallel; we run single-device
                    raise RuntimeError("DataParallel batch split unsupported in clamp mode")
                # The mask carries +/-1 signs for is_negative datapoints, which have no
                # meaning for a projection cap; load_datapoints() already refuses
                # is_negative rows, so nonzero just means "steered position" here.
                gate = (full_mask != 0).to(torch.float32)
            else:
                gate = None
            out, proj, _ = clamp_op(x, vector_getter(), thr, gate)
            with torch.no_grad():
                on = gate.squeeze(-1) > 0 if gate is not None else torch.ones_like(proj, dtype=torch.bool)
                stats["bound"] += int(((proj > thr) & on).sum())
                stats["steered"] += int(on.sum())
            return out
        return hook_fn

    steering_opt.make_steering_hook_hf_mask = make_clamp_hook

    orig_clip = torch.nn.utils.clip_grad_norm_

    def flushing(parameters, *a, **kw):
        _flush_bind_stats(stats)
        return orig_clip(parameters, *a, **kw)

    torch.nn.utils.clip_grad_norm_ = flushing
    return stats, orig_make


def patch_unit_renorm_adam():
    """Replace the max-norm ball with the unit sphere (clamp mode only).

    clamp_op depends on v only through u = v/||v||, so ||v|| is a free gauge: max_norm
    projection is meaningless (every radius encodes the same steering op) and is
    disabled by passing max_norm=None. Left alone the gauge drifts -- the gradient
    through the normalization is orthogonal to v, so every step GROWS ||v||, silently
    shrinking the effective lr on u (dL/dv scales as 1/||v||). Renormalizing v to unit
    norm after each optimizer step pins the gauge for numerics without changing the op.
    optimize_completion instantiates torch.optim.Adam directly, so the swap happens at
    the torch attribute; the only Adam built while this is installed optimizes [vector].
    """
    orig_adam = torch.optim.Adam

    class UnitRenormAdam(orig_adam):
        @torch.no_grad()
        def step(self, closure=None):
            out = super().step(closure)
            for group in self.param_groups:
                for p in group["params"]:
                    n = torch.linalg.norm(p)
                    if torch.isfinite(n) and float(n) > 0:
                        p.mul_(1.0 / n)
            return out

    torch.optim.Adam = UnitRenormAdam
    return orig_adam


def load_init_vector(path, trainer_layer, steer_mode, starting_norm):
    """Load a repo safetensors artifact as the optimizer's starting vector.

    Artifacts store (n_layers, d) float32 with layer_indices in the .json sidecar,
    indexing block OUTPUTS; the trainer hooks resid_pre, so the row for artifact layer
    L belongs with --layer L+1 (same shift convert_artifact.py applies in reverse).
    A mismatch is a warning, not an error -- transplanting a direction across nearby
    layers is a legitimate (if dubious) experiment, but it must not happen silently.

    clamp mode: returned at unit norm (the gauge patch_unit_renorm_adam maintains).
    add mode: rescaled to --starting-norm so optimizer dynamics match a random init.
    """
    from safetensors.torch import load_file as st_load

    t = st_load(path)["vector"].float()
    meta = json.loads(Path(path).with_suffix(".json").read_text())
    layers = list(meta.get("layer_indices") or [])
    want = trainer_layer - 1
    if t.dim() == 2:
        if want in layers:
            row = layers.index(want)
        elif t.shape[0] == 1:
            row = 0
        else:
            raise SystemExit(f"{path}: layer_indices {layers} has no row for artifact "
                             f"layer {want} (trainer --layer {trainer_layer} - 1)")
        vec = t[row]
        got = layers[row] if row < len(layers) else None
    else:
        vec, got = t, layers[0] if layers else None
    if got is not None and got != want:
        print(f"  WARNING: --init-vector row is artifact layer {got}, but --layer "
              f"{trainer_layer} steers resid_pre[{trainer_layer}] == resid_post[{want}]. "
              "Cross-layer transplant -- make sure that is intentional.", flush=True)
    norm = float(vec.norm())
    target = 1.0 if steer_mode == "clamp" else starting_norm
    vec = vec * (target / norm)
    print(f"  init vector: {path}  artifact_layer={got}  norm {norm:.4f} -> {target:.4f} "
          f"({'unit gauge' if steer_mode == 'clamp' else 'starting_norm'})", flush=True)
    return vec


def patch_nan_guard(steering_opt, limit):
    """FIX 4 (NaN guard), corrected.

    Two upstream failure modes end an iteration without an optimizer step:
      (a) a non-finite GRADIENT -> `continue` at line 765;
      (b) a NaN LOSS -> the chunk-level `continue` at line 687 skips backward entirely,
          so p.grad is None and (a)'s check never fires.
    Both `continue`s sit BEFORE `iters += 1` (line 822), so a persistent failure spins
    forever. The first version of this guard only counted (a) AND reset its counter when
    p.grad was None -- i.e. it actively certified (b), the mode that actually happens, as
    healthy. Count both.

    Interception point: torch.nn.utils.clip_grad_norm_ (line 761) runs exactly once per
    iteration, immediately before upstream's own NaN check, and optimizer.zero_grad()
    at the top of the iteration sets grads to None -- so `p.grad is None` here means
    "no backward succeeded this iteration", not "start of training".
    """
    state = {"consecutive": 0, "iter_times": [], "no_grad_iters": 0, "nan_grad_iters": 0}
    orig = torch.nn.utils.clip_grad_norm_

    def guarded(parameters, *a, **kw):
        params = list(parameters) if not isinstance(parameters, torch.Tensor) else [parameters]
        state["iter_times"].append(time.time())
        no_grad = all(p.grad is None for p in params)
        bad = any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params)
        if bad or no_grad:
            state["consecutive"] += 1
            state["nan_grad_iters"] += int(bad)
            state["no_grad_iters"] += int(no_grad and not bad)
            why = "non-finite" if bad else "MISSING (NaN loss -> upstream skipped backward)"
            print(f"  !! {why} gradient ({state['consecutive']}/{limit} consecutive)", flush=True)
            if state["consecutive"] >= limit:
                raise RuntimeError(
                    f"aborting: {limit} consecutive iterations with no usable gradient. "
                    "The upstream `continue` does not increment iters, so this would "
                    "either loop forever or save an untrained vector."
                )
        else:
            state["consecutive"] = 0
        return orig(params, *a, **kw)

    torch.nn.utils.clip_grad_norm_ = guarded
    return state


def check_trajectory(history, allow_no_progress, guard, satisfice=False):
    """A run that produced no gradient must not silently write a .pt.

    The observed signature of the P1 bug was history == [94.2, 0.0]: loss_last=0.0 reads
    like perfect convergence in the artifact metadata, and the vector is one random Adam
    step away from its init.

    NB the 0.0 test is NOT valid under --satisfice: the hinge
    ((src_losses - target_loss).clamp_min(0) ** 2).sum() is exactly 0.0 when every
    completion is already below the target, which is precisely the success condition.
    The test that works in both regimes is the gradient census from the NaN guard: a
    healthy run produces a usable gradient on every single iteration.
    """
    if not history:
        raise RuntimeError("no loss history -- the training loop never completed an iteration")
    if any(not (l == l) or l in (float("inf"), float("-inf")) for l in history):
        raise RuntimeError(f"non-finite entry in loss history: {history}")
    dead = guard["no_grad_iters"] + guard["nan_grad_iters"]
    if dead:
        raise RuntimeError(
            f"{dead} of {len(guard['iter_times'])} iterations produced no usable gradient "
            f"({guard['no_grad_iters']} missing, {guard['nan_grad_iters']} non-finite). "
            "Those iterations were silent no-ops. Refusing to save."
        )
    if history[-1] == 0.0 and not satisfice:
        raise RuntimeError(
            f"final loss is exactly 0.0 (history={history}). Without --satisfice that is "
            "the NaN-loss signature (upstream sets loss=0.0 when every chunk is skipped), "
            "not convergence. Refusing to save."
        )
    if len(history) > 1 and history[-1] >= history[0]:
        msg = (f"final loss {history[-1]:.6f} did not beat the first {history[0]:.6f} "
               f"(history={history}). The optimizer made no progress.")
        if not allow_no_progress:
            raise RuntimeError(msg + " Refusing to save; pass --allow-no-progress to override.")
        print("WARNING: " + msg, flush=True)


def main():
    args = parse_args()
    sys.path.insert(0, FAITHFULNESS_TRANSFER)
    import steering_opt_batched as steering_opt

    excl = {t.strip() for t in args.exclude_task_ids.split(",") if t.strip()}
    dps, metas = load_datapoints(steering_opt, args.examples, args.limit,
                                 exclude_task_ids=excl,
                                 min_completion_tokens=args.min_completion_tokens)
    n_src = sum(len(d.src_completions) for d in dps)
    n_dst = sum(len(d.dst_completions) for d in dps)
    task_ids = sorted({m.get("task_id") for m in metas if m})
    contexts = len({(m.get("task_id"), m.get("message_index")) for m in metas if m})
    print(f"{len(dps)} datapoints  n_src={n_src}  n_dst={n_dst}  "
          f"{len(task_ids)} task_ids  {contexts} (task,msg) contexts  from {args.examples}",
          flush=True)
    if not dps:
        raise SystemExit("no datapoints loaded")

    if args.validate_only:
        # Data-plumbing preflight: everything up to (and including) the P2 repair and
        # its assertion runs on the tokenizer alone, so this works offline on a box
        # where the model weights have not landed yet.
        tok = load_tokenizer(args)
        dps = repair_alignment(tok, dps)                             # P2 fix
        validate_alignment(tok, dps)                                 # P2 assertion
        print(f"VALIDATE-ONLY OK: {len(dps)} datapoints (n_src={n_src} n_dst={n_dst}) "
              f"aligned for {args.model}; exiting before model load", flush=True)
        return

    model, tok = load_model(args)
    dps = repair_alignment(tok, dps)                                 # P2 fix
    validate_alignment(tok, dps)                                     # P2 assertion
    orig_log1mexp = patch_log1mexp(steering_opt, args.log1mexp_clamp)  # P1

    # Clamp mode: reroute the steering hook through the serving clamp op, drop the
    # (meaningless) max_norm ball, pin ||v||=1 after each step, and start from a
    # toward-hacking direction. Installed BEFORE patch_nan_guard so the guard's
    # clip_grad_norm_ wrapper calls the bind-fraction flusher once per iteration.
    starting_vec, clamp_stats, orig_adam, init_path = None, None, None, args.init_vector
    if args.steer_mode == "clamp" and init_path is None:
        init_path = DIFFMEAN_INIT
    if init_path is not None:
        starting_vec = load_init_vector(init_path, args.layer, args.steer_mode,
                                        args.starting_norm)
    if args.steer_mode == "clamp":
        print(f"clamp mode: thr={args.clamp_thr}; max_norm disabled (unit-sphere gauge)",
              flush=True)
        clamp_stats, _ = patch_clamp_steering(steering_opt, args.clamp_thr)
        orig_adam = patch_unit_renorm_adam()
    effective_max_norm = None if args.steer_mode == "clamp" else args.max_norm

    guard = patch_nan_guard(steering_opt, args.max_consecutive_nan)    # FIX 4, corrected

    on_cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    if on_cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        result = steering_opt.optimize_completion(
            model, dps, args.layer,
            tokenizer=tok,
            use_transformer_lens=False,   # plain HF path; no transformer_lens in this env
            lr=args.lr,
            max_iters=args.max_iters,
            batch_size=args.batch_size,
            # FIX 3 (temperature): line 278 is log_softmax(logits * temperature) -- a MULTIPLY.
            # The shipped 0.6/0.7 default FLATTENS the distribution. 1.0 is the identity.
            temperature=args.temperature,
            starting_norm=args.starting_norm,
            starting_vec=starting_vec,        # None (random init) unless --init-vector/clamp
            max_norm=effective_max_norm,      # None in clamp mode: see patch_unit_renorm_adam
            do_one_minus=True,            # src branch: -log(1 - p), i.e. suppress
            # Recorded, not defaulted: with only_hook_prompt=False and dp.token=None,
            # _build_steer_mask takes the `spec is None` branch (line 209) and adds the
            # vector at EVERY position, prompt and completion alike. The downstream
            # prefill eval can run decode-only (VLLM_LENS_STEER_DECODE_ONLY), which is a
            # DIFFERENT regime -- whoever configures it needs to know which was fitted.
            only_hook_prompt=False,
            return_loss=True,
            return_loss_history=True,
            do_target_loss_avg=False,     # with return_loss_history this puts the history in ['loss']
            satisfice=args.satisfice,
            target_loss=args.target_loss,
            debug=args.debug,
            device=args.device,
        )
    finally:
        steering_opt._log1mexp = orig_log1mexp
        if orig_adam is not None:
            torch.optim.Adam = orig_adam
    elapsed = time.time() - t0

    vector, info = result[0], result[-1]
    history = info["loss"] if isinstance(info.get("loss"), list) else []
    for i, l in enumerate(history):
        print(f"  iter {i:3d}  loss {l:.6f}", flush=True)

    if clamp_stats is not None:
        _flush_bind_stats(clamp_stats)  # the loop can break before the last clip_grad_norm_
        fracs = clamp_stats["iter_fracs"]
        if fracs and max(fracs) < 0.01:
            print("\n" + "!" * 72, flush=True)
            print(f"WARNING: the clamp NEVER bound (max per-iter bind fraction "
                  f"{max(fracs):.4f}). Every steered position already projects below "
                  f"thr={args.clamp_thr}, so the op was a measured no-op and the vector "
                  "is untrained. Lower --clamp-thr or check the init direction.", flush=True)
            print("!" * 72 + "\n", flush=True)

    iter_times = guard["iter_times"]
    per_iter = [round(b - a, 2) for a, b in zip(iter_times, iter_times[1:])]
    peak_mem_gb = (torch.cuda.max_memory_allocated() / 2**30) if on_cuda else None

    # refuses to save a dead run; satisfice-aware (a hinge legitimately reaches 0.0)
    check_trajectory(history, args.allow_no_progress, guard, satisfice=args.satisfice)

    raw_norm = float(vector.norm().item())
    unit = (vector / vector.norm()).detach().cpu().float()

    # Saturation check: all 48 shipped faithfulness vectors ended at exactly max_norm=5.0
    # in 21 steps. Hitting the clamp means the lr walked the vector into the wall; it is
    # NOT evidence of convergence. Meaningless in clamp mode, where ||v|| is pinned to
    # 1.0 by design (unit-sphere gauge), so it only runs for add.
    saturated = args.steer_mode == "add" and abs(raw_norm - args.max_norm) < 1e-3
    if saturated:
        print("\n" + "!" * 72, flush=True)
        print(f"WARNING: final norm {raw_norm:.4f} == max_norm {args.max_norm}. The vector is "
              "SATURATED at the clamp.\n  This means lr is too high, not that training "
              "converged. Lower --lr or raise --max-norm and re-run.", flush=True)
        print("!" * 72 + "\n", flush=True)

    if args.steer_mode == "clamp":
        method = "optimization_clamp"
    else:
        method = "optimization_src_dst" if n_dst else "optimization_src_only"

    payload = {
        "vectors": {args.layer: unit},
        "norms": {args.layer: raw_norm},
        "metadata": {
            "method": method,
            "model": args.model,
            # NB: `layer` is the trainer's forward-PRE hook index (see convert_artifact.py).
            "layer": args.layer,
            "lr": args.lr,
            "max_iters": args.max_iters,
            "max_iters_note": "upstream breaks on iters > max_iters, so n_iters == max_iters + 1",
            "max_norm": effective_max_norm,
            "temperature": args.temperature,
            "starting_norm": args.starting_norm,
            "batch_size": args.batch_size,
            "load_in_4bit": args.load_in_4bit,
            "dtype": args.dtype,
            "device": args.device,
            "examples_path": str(Path(args.examples).resolve()),
            "excluded_task_ids_at_train": sorted(excl),
            "min_completion_tokens": args.min_completion_tokens,
            "satisfice": args.satisfice,
            "target_loss": args.target_loss,
            "n_datapoints": len(dps),
            "n_src": n_src,
            "n_dst": n_dst,
            "n_task_ids": len(task_ids),
            "n_contexts": contexts,
            "task_ids": task_ids,
            "steer_positions": ("all tokens (prompt+completion); only_hook_prompt=False, "
                                "dp.token=None -> _build_steer_mask `spec is None` branch"),
            "patches": {
                "log1mexp_clamp": args.log1mexp_clamp,
                "prompt_trailing_whitespace_moved_to_completion": "conditional (only rows whose tok(prompt) was not a token-prefix of tok(prompt+completion))",
                "nan_guard_counts_missing_gradients": True,
                "temperature_passed": args.temperature,
                "model_params_frozen": True,
            },
            "loss_first": history[0] if history else None,
            "loss_last": history[-1] if history else None,
            "loss_history": history,
            "n_iters": info.get("iters"),
            "iters_with_no_gradient": guard["no_grad_iters"],
            "iters_with_nan_gradient": guard["nan_grad_iters"],
            "final_vector_norm": raw_norm,
            "saturated_at_max_norm": saturated,
            "elapsed_seconds": round(elapsed, 2),
            "seconds_per_iter": per_iter,
            "peak_gpu_mem_gb": round(peak_mem_gb, 2) if peak_mem_gb is not None else None,
        },
    }
    # New keys only when the new machinery ran, so a default add run's .pt is
    # byte-for-byte what the old trainer wrote.
    if init_path is not None:
        iv = starting_vec / starting_vec.norm()
        payload["metadata"]["init_vector"] = str(Path(init_path).resolve())
        payload["metadata"]["cos_to_init"] = round(float(unit @ iv), 6)
    if args.steer_mode == "clamp":
        payload["metadata"].update({
            "steer_mode": "clamp",
            "clamp_thr": args.clamp_thr,
            "clamp_op": ("u = v/||v||; proj = h.u; h -= max(0, proj - clamp_thr)*u "
                         "(matches serving/patch_vllm_lens_modes.py clamp branch)"),
            # The clamp only ever REMOVES the +u component, so suppression training
            # keeps u pointing at the behaviour it removes:
            "sign": ("positive = TOWARD hacking (diffmean convention) -- clamp at eval "
                     "with mode=clamp and a threshold; the OPPOSITE polarity to add-mode "
                     "as-trained vectors, which store positive = AWAY from hacking"),
            "unit_renorm_after_step": True,
            "clamp_bind_fraction_per_iter": clamp_stats["iter_fracs"],
        })
    elif init_path is not None:
        payload["metadata"]["steer_mode"] = "add"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(out)
    print(f"saved layer={args.layer} norm={raw_norm:.4f} iters={info.get('iters')} "
          f"elapsed={elapsed:.1f}s peak_mem={peak_mem_gb if peak_mem_gb is None else round(peak_mem_gb,2)}GiB "
          f"s/iter={per_iter} -> {out}", flush=True)


if __name__ == "__main__":
    main()
