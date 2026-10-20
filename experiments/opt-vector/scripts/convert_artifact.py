#!/usr/bin/env python3
"""Convert a train_opt_vector.py .pt into a steering_reasoning safetensors artifact.

Three things have to be right, and all three are silent-failure modes:

  (1) NORM  -- the trainer parks magnitude in a separate dict.
  (2) LAYER -- the trainer's hook is forward-PRE; vllm-lens indexes block OUTPUTS.
  (3) SIGN  -- we trained suppression, the repo convention stores promotion.

Each is argued at its point of use below. Uses steering_reasoning.steering.vectors.save_artifact
rather than hand-rolling safetensors.

Clamp-mode .pts (train_opt_vector.py --steer-mode clamp) change (1) and (3): the clamp
op only uses u = v/||v||, so the UNIT vector is stored (magnitude is gauge), and the
trained direction already points TOWARD hacking (diffmean convention), so there is
nothing to negate -- --sign repo is refused. (2) is identical. method/clamp_thr/sign
metadata are propagated so run_prefill_opt.py's guards can tell the polarities apart.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

SR_SRC = str(Path(__file__).resolve().parents[3] / "src")

# The prefill-intervention grid is 8 contexts: lcbhard_0, 2, 33, 41, 51, 63, 66, 72
# (the 4 that hack at baseline are 72/33/51/41). FOUR of the eight -- lcbhard_0, 2, 51, 66
# -- are TRAIN task_ids, so the whole grid is dropped from the fit rather than just the
# one that hacks. 24 train task_ids -> 20 kept, of which 14 actually carry judge-positive
# reasoning segments that fit the 4096-token prompt cap.
DEFAULT_EXCLUDED = ["lcbhard_0", "lcbhard_2", "lcbhard_33", "lcbhard_41",
                    "lcbhard_51", "lcbhard_63", "lcbhard_66", "lcbhard_72"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pt", required=True, help="input .pt from train_opt_vector.py")
    p.add_argument("--out", required=True, help="output .safetensors")
    p.add_argument("--sign", choices=["as-trained", "repo"], default="as-trained",
                   help="see the SIGN block below; default keeps scale=+1.0 == trained effect")
    p.add_argument("--extraction-model", default=None,
                   help="model the vector was fit on (defaults to the .pt's metadata['model'])")
    p.add_argument("--label-source", default="mistrace judge_label==True & kind=='reasoning'")
    p.add_argument("--split", default="train")
    p.add_argument("--excluded-task-ids", default=",".join(DEFAULT_EXCLUDED))
    p.add_argument("--mistrace-revision", default="7b4689723dd16fc992b116a8d6d3bb1a41627b62")
    return p.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, SR_SRC)
    from steering_reasoning.steering.vectors import load_artifact, save_artifact

    payload = torch.load(args.pt, map_location="cpu", weights_only=False)
    meta_in = payload["metadata"]
    (train_layer, unit), = payload["vectors"].items()
    raw_norm = float(payload["norms"][train_layer])
    train_layer = int(train_layer)

    steer_mode = meta_in.get("steer_mode", "add")

    # ---- (1) NORM ---------------------------------------------------------------
    # train_opt_vector.py stores vectors[L] as a UNIT vector and parks the magnitude in
    # norms[L] (matching the upstream faithfulness CLI). steering_reasoning / vllm-lens
    # applies raw rows: h += scale * v. So a unit row would mean scale=1.0 applies a
    # norm-1 nudge -- ~1/70th of a typical L35 residual -- i.e. nothing. Re-attach the
    # magnitude so scale=1.0 reproduces the strength the optimizer actually trained at.
    # EXCEPT clamp mode: the serving clamp branch normalizes whatever it is given
    # (u = vec/|vec|), so magnitude is pure gauge -- store the unit vector so the
    # artifact says exactly what is applied. (The trainer pinned ||v||=1 anyway.)
    if steer_mode == "clamp":
        vec = unit.float()
    else:
        vec = unit.float() * raw_norm
        if abs(float(vec.norm()) - raw_norm) > 1e-3:
            raise SystemExit(f"norm re-attach failed: {float(vec.norm())} != {raw_norm}")

    # ---- (2) LAYER --------------------------------------------------------------
    # VERIFIED, not assumed. steering_opt_batched.py:84 --
    #     hooks = [layers[cur_layer].register_forward_pre_hook(hook_fn) for ...]
    # register_forward_PRE_hook, and make_steering_hook_hf_mask (line ~141) mutates
    # args[0], the block's INPUT. The TransformerLens twin at line 435 uses
    # 'blocks.{L}.hook_resid_pre', which agrees. So the trained vector lives at
    # resid_pre[L] == resid_post[L-1].
    # vllm-lens / steering_reasoning artifacts index block OUTPUTS, so the same point in
    # the network is layer L-1 there.
    if train_layer <= 0:
        raise SystemExit(
            "layer 0: resid_pre[0] is the embedding output, which has no block-output "
            "index. Cannot express this hook point as a steering_reasoning layer_index."
        )
    artifact_layer = train_layer - 1

    # ---- (3) SIGN ---------------------------------------------------------------
    # We trained SRC-ONLY suppression: the src branch loss is -log(1 - p(hack sentence))
    # (steering_opt_batched.py:284), so the optimizer pushes the vector in whatever
    # direction makes the hack sentence LESS likely. The trained +v therefore points
    # AWAY from hacking -- the opposite polarity to this repo's diffmean vectors, whose
    # sidecars say "positive = toward hacking; steer with negative scale to suppress".
    #   --sign as-trained (default): store +v. scale=+1.0 == the trained suppression.
    #                                Honest about what was optimized, but INVERTED
    #                                relative to diffmean_judge42_*.
    #   --sign repo:                 store -v, restoring the repo convention so the same
    #                                negative scales that suppress the diffmean vector
    #                                also suppress this one (drop-in for existing sweeps).
    # CLAMP-MODE EXCEPTION: the clamp op h -= max(0, h.u - thr)*u only ever REMOVES the
    # +u component, so suppression training keeps u pointing AT the behaviour it removes.
    # The trained vector therefore already stores positive = toward hacking -- the repo /
    # diffmean convention -- and negating it under --sign repo would produce a vector the
    # clamp branch strips the ANTI-hacking component from. Refuse rather than flip.
    if steer_mode == "clamp":
        if args.sign == "repo":
            raise SystemExit("--sign repo is meaningless for a clamp vector: as-trained "
                             "already IS the repo convention (positive = toward hacking); "
                             "negating it would clamp away the anti-hacking component.")
        sign_words = ("positive = TOWARD hacking (diffmean convention, as trained). Apply "
                      "with mode=clamp and a threshold on the h.u projection; scale is "
                      "unused. The OPPOSITE polarity to add-mode as-trained opt vectors.")
    elif args.sign == "repo":
        vec = -vec
        sign_words = ("positive = toward hacking; steer with a NEGATIVE scale to suppress "
                      "(repo convention; this is the NEGATION of the trained vector, which "
                      "was optimized src-only to suppress at positive scale)")
    else:
        sign_words = ("positive = AWAY from hacking; steer with a POSITIVE scale to suppress. "
                      "This is the vector as trained (src-only -log(1-p) suppression) and is "
                      "the OPPOSITE polarity to diffmean_judge42_*, which stores "
                      "positive = toward hacking. Do not reuse a diffmean scale sign here.")

    excluded = [t for t in args.excluded_task_ids.split(",") if t.strip()]
    meta = {
        # apply_steering_smoke.py:48 does meta['method'] unguarded -> KeyErrors without this.
        # Propagated from the trainer: optimization_src_dst when dst rows exist (v3b),
        # optimization_src_only otherwise (v1/v2/v3a).
        "method": meta_in.get("method", "optimization_src_only"),
        "extraction_model": args.extraction_model or meta_in.get("model"),
        "label_source": args.label_source,
        "split": args.split,
        "excluded_task_ids": excluded,
        "excluded_reason": ("all 8 prefill-intervention grid contexts are dropped from the "
                            "fit; 4 of them (lcbhard_0/2/51/66) are TRAIN task_ids, so "
                            "fitting on them would leak into the eval this vector is "
                            "judged by"),
        "mistrace_revision": args.mistrace_revision,
        "sign": sign_words,
        "sign_mode": args.sign,
        "trainer_hook": "register_forward_pre_hook (steering_opt_batched.py:84) -> resid_pre",
        "trainer_layer": train_layer,
        "layer_offset_note": (f"trainer hooked resid_pre[{train_layer}] == "
                              f"resid_post[{artifact_layer}]; artifact layer_indices "
                              f"= [{artifact_layer}]"),
        "vector_norm": round(float(vec.norm()), 4),
        "source_pt": str(Path(args.pt).resolve()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        # Fitted regime, propagated so whoever configures the eval knows whether to turn
        # on the decode-only gate (VLLM_LENS_STEER_DECODE_ONLY). The vector was trained
        # with the steering added at EVERY position, prompt included.
        "steer_positions": meta_in.get("steer_positions"),
        "trainer_patches": meta_in.get("patches"),
        "train": {k: meta_in.get(k) for k in
                  ("lr", "max_iters", "max_norm", "temperature", "batch_size",
                   "n_datapoints", "n_src", "n_dst", "n_task_ids", "n_contexts",
                   "loss_first", "loss_last", "loss_history", "n_iters",
                   "iters_with_no_gradient", "iters_with_nan_gradient",
                   "saturated_at_max_norm", "peak_gpu_mem_gb", "elapsed_seconds",
                   "examples_path")},
    }
    if steer_mode == "clamp":
        # method (e.g. optimization_clamp) is already propagated above; these keys make
        # the clamp regime reproducible from the artifact alone, and let the eval
        # runner's polarity guards tell clamp vectors from add-mode ones.
        meta.update({
            "steer_mode": "clamp",
            "clamp_thr": meta_in.get("clamp_thr"),
            "clamp_op": meta_in.get("clamp_op"),
            "init_vector": meta_in.get("init_vector"),
            "cos_to_init": meta_in.get("cos_to_init"),
            "clamp_bind_fraction_per_iter": meta_in.get("clamp_bind_fraction_per_iter"),
        })

    save_artifact(args.out, vec.unsqueeze(0), [artifact_layer], meta)

    # ---- round-trip -------------------------------------------------------------
    back, layer_indices, meta_back = load_artifact(args.out)
    assert layer_indices == [artifact_layer], (layer_indices, artifact_layer)
    assert back.shape == (1, vec.shape[0]), back.shape
    assert torch.equal(back[0], vec), "round-trip tensor mismatch"
    assert meta_back["method"] == meta_in.get("method", "optimization_src_only")

    print(f"wrote {args.out}")
    print(f"  trainer layer (resid_pre) {train_layer} -> artifact layer_index {artifact_layer}")
    if steer_mode == "clamp":
        print(f"  norm {float(back[0].norm()):.4f}  (UNIT vector stored; clamp mode -- "
              f"magnitude is gauge; clamp_thr={meta_in.get('clamp_thr')})")
    else:
        print(f"  norm {float(back[0].norm()):.4f}  (unit x {raw_norm:.4f})")
    print(f"  sign_mode={args.sign}: {sign_words}")
    print("  round-trip OK")
    if meta_in.get("saturated_at_max_norm"):
        print("  WARNING: source vector was SATURATED at max_norm -- lr too high, not converged.")


if __name__ == "__main__":
    main()
