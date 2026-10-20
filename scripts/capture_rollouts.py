#!/usr/bin/env python
"""Replay steering-arm rollouts and capture their residual streams to the shard format.

Serving env (`cd serving && uv run python ../scripts/capture_rollouts.py …`) on a box that
mirrors the generation config. Input is records.jsonl from scripts/export_eval_records.py; all records in one
run must share an arm. Steered arms need --steer-artifact + --mode/--dose (asserted against
the records) and run under the positions-aware gate (VLLM_LENS_STEER_DECODE_ONLY=1 with
serving/patch_vllm_lens_gate.py applied in this env), reproducing how generation steered:
prefill steered the model's earlier replies (steer_positions), decode every generated token.
Baseline records forbid the steering flags. Resume-safe: rollouts whose meta/<task>.json is
already in --s3-prefix are skipped. Manifest is written last.

    VLLM_LENS_STEER_DECODE_ONLY=1 uv run python ../scripts/capture_rollouts.py --records records_cn8_s42.jsonl \\
        --select lcbhard_4 --model Qwen/Qwen3.5-27B --layers all \\
        --steer-artifact ../src/steering_reasoning/canonical/u_L35_steering.safetensors --mode clamp --dose -8 \\
        --out-dir /workspace/dumps --s3-prefix \\
        s3://steering-reasoning-artifacts/derived/<experiment>/dumps/cn8/conflicting/s42
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from steering_reasoning.steering.activations import ActivationCapturer, apply_no_nvcc_workarounds  # noqa: E402
from steering_reasoning.steering.replay import SteeringSpec, capture_rollout, generation_serving_block  # noqa: E402
from steering_reasoning.steering.shards import s3_cp, s3_existing_metas, serving_provenance  # noqa: E402


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


def _s3_read_json(s3_prefix: str, rel: str, out_dir: Path) -> dict | None:
    import subprocess

    local = out_dir / "_s3" / rel
    local.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["aws", "s3", "cp", "--quiet", f"{s3_prefix.rstrip('/')}/{rel}", str(local)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None
    return json.loads(local.read_text())


def load_prior_manifest(s3_prefix: str, out_dir: Path) -> dict:
    return _s3_read_json(s3_prefix, "manifest.json", out_dir) or {}


def load_meta_row(s3_prefix: str, out_dir: Path, task_id: str) -> dict:
    """The manifest row embedded in meta/<task>.json (everything except the per-turn detail)."""
    meta = _s3_read_json(s3_prefix, f"meta/{task_id}.json", out_dir)
    if meta is None:
        raise RuntimeError(f"meta/{task_id}.json listed in S3 but not readable")
    return {k: v for k, v in meta.items()
            if k not in ("turns", "segments", "skipped_turns", "keep_start_collapses", "im_end_rows",
                         "never_computed_rows", "steering", "generation")}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--records", required=True)
    p.add_argument("--select", default="", help="comma-separated task ids (default: all records)")
    p.add_argument("--model", default="Qwen/Qwen3.5-27B")
    p.add_argument("--layers", default="all", help="'all' or comma-separated layer indices")
    p.add_argument("--layer-chunk", type=int, default=64, help="max layers per pass (payload/RAM may lower it)")
    p.add_argument("--max-model-len", type=int, default=131072,
                   help="the GENERATION context length (the engine is built one token larger)")
    p.add_argument("--tensor-parallel", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    p.add_argument("--host-mem-margin-gib", type=int, default=128)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--s3-prefix", default=None)
    p.add_argument("--steer-artifact", default=None)
    p.add_argument("--mode", choices=("clamp", "add", "ablate"), default=None, help="steering mode of the arm (default clamp when --dose is given)")
    p.add_argument("--dose", type=float, default=None, help="clamp threshold / add scale of the arm")
    p.add_argument("--expect-template-sha", default=None,
                   help="sha256 prefix of the generation-time chat template (hard-fail on mismatch)")
    p.add_argument("--expect-tp", type=int, default=None, help="generation TP (warning only)")
    p.add_argument("--prefix-reuse", action="store_true",
                   help="unsteered records only: prefill each turn through the prefix cache and capture only the "
                        "computed suffix (needs serving/patch_vllm_lens_prefix_reuse.py in this env); the tree "
                        "stores a re-read reply once (share_branch), as the 103 dump does")
    p.add_argument("--on-branch-mismatch", choices=("skip", "fail"), default="skip",
                   help="baseline arms only (steered arms need the sidecar): a re-rendered branch whose "
                        "length differs from the logged completion_tokens is not the generated stream — "
                        "leave that turn out of the dump (skip) or abort the rollout (fail)")
    args = p.parse_args()

    records = [json.loads(line) for line in open(args.records) if line.strip()]
    select = {t for t in args.select.split(",") if t}
    if select:
        records = [r for r in records if r["task_id"] in select]
        missing = select - {r["task_id"] for r in records}
        if missing:
            raise SystemExit(f"--select ids not in records: {sorted(missing)}")
    if not records:
        raise SystemExit("no records selected")
    arms = {(r["arm"], r["mode"], r["dose"], r["split"], r["seed"]) for r in records}
    if len(arms) != 1:
        raise SystemExit(f"records span several arms/splits/seeds: {sorted(arms, key=str)} — one run per arm")
    (arm, mode, dose, split, seed), = arms
    if mode is None and (args.steer_artifact or args.dose is not None or args.mode is not None):
        raise SystemExit(f"arm {arm} is unsteered: do not pass --steer-artifact/--mode/--dose")
    args.mode = args.mode or "clamp"
    if mode is not None and not (args.steer_artifact and (args.dose is not None or args.mode == "ablate")):
        raise SystemExit(f"arm {arm} is steered ({mode} {dose}): pass --steer-artifact, --mode {mode} and --dose {dose}")
    steering = SteeringSpec.from_artifact(args.steer_artifact, args.dose, mode=args.mode) if mode is not None else None
    if args.prefix_reuse and steering is not None:
        raise SystemExit("--prefix-reuse is for unsteered arms only")
    if steering and (steering.mode, steering.dose) != (mode, dose):
        raise SystemExit(f"--mode {args.mode} --dose {args.dose} != records' steering {mode} {dose}")

    done = s3_existing_metas(args.s3_prefix) if args.s3_prefix else set()
    todo = [r for r in records if r["task_id"] not in done]
    log(f"{arm} {mode or 'unsteered'} dose={dose} {split} s{seed}: {len(todo)}/{len(records)} rollouts to capture "
        f"({len(records) - len(todo)} already in {args.s3_prefix})")

    gen_models = {r["model"] for r in records}
    if len(gen_models) != 1:
        raise SystemExit(f"records name several generation models: {sorted(gen_models)}")
    (gen_model,) = gen_models

    apply_no_nvcc_workarounds()
    # Single-pass prefills: each turn is ONE forward pass, so the gate's position bookkeeping
    # cannot depend on chunking (it is correct under chunked prefill too, tests/gpu; this
    # removes the dependence entirely and is recorded in serving{}).
    # +1: a turn may fill the generation context to the last token, and the capture
    # prefill emits one token past the branch (SamplingParams(max_tokens=1))
    engine_len = args.max_model_len + 1
    if args.prefix_reuse:
        # read by capture_suffix (driver) and by the patched plugin; must precede the engine build
        import os
        os.environ["VLLM_LENS_CAPTURE_PREFIX_REUSE"] = "1"
    # Single-pass prefills are a STEERED-replay requirement (the gate's bookkeeping); prefix reuse
    # takes vLLM's scheduler defaults instead, as the all-103 capture did: with prefix caching on,
    # this hybrid (Mamba) model refuses to build without chunked prefill ("Chunked prefill is
    # required for mamba cache mode 'align'"), and capture_suffix's row accounting holds under it.
    sched = ({} if args.prefix_reuse else
             {"enable_chunked_prefill": False, "max_num_batched_tokens": engine_len})
    capturer = ActivationCapturer(args.model, max_model_len=engine_len,
                                  gpu_memory_utilization=args.gpu_memory_utilization,
                                  tensor_parallel_size=args.tensor_parallel,
                                  enable_prefix_caching=args.prefix_reuse, **sched)
    warns = capturer.assert_serving_matches({
        "model": gen_model, "dtype": "bfloat16", "template_sha256": args.expect_template_sha,
        "require_patches": (["modes", "seq_lens", "positions_gate"] if steering else [])
                           + (["prefix_reuse"] if args.prefix_reuse else []),
        "gate": steering is not None,
        "tensor_parallel_size": args.expect_tp,
    })
    if args.expect_template_sha is None:
        log("WARNING: --expect-template-sha not given; the chat template is recorded in the manifest "
            "but not checked against generation")
    for w in warns:
        log(f"WARNING: {w}")
    layers = (list(range(capturer.n_layers)) if args.layers == "all"
              else sorted({int(x) for x in args.layers.split(",")}))
    if steering and steering.mode == "clamp" and steering.layer not in layers:
        raise SystemExit(f"--layers must include the steering layer L{steering.layer} (clamp invariant check)")
    if steering and steering.mode != "clamp":
        log(f"WARNING: {steering.mode} mode has no row-wise invariant; a swallowed steering-hook exception "
            "would go unnoticed (the manifest records this)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, errors = [], []
    t_start = time.monotonic()
    for i, rec in enumerate(todo):
        t0 = time.monotonic()
        try:
            row = capture_rollout(rec, capturer, layers, args.layer_chunk, out_dir, args.s3_prefix,
                                  steering=steering, host_margin_gib=args.host_mem_margin_gib,
                                  on_branch_mismatch=args.on_branch_mismatch,
                                  prefix_reuse=args.prefix_reuse, log=log)
        except Exception as e:  # noqa: BLE001 — batch runner: record the rollout's failure, keep going
            log(f"ERROR {rec['rollout_id']}: {type(e).__name__}: {e}")
            errors.append({"rollout_id": rec["rollout_id"], "error": f"{type(e).__name__}: {e}"})
            if "EngineDead" in type(e).__name__ or "EngineDead" in str(e) or "reset_prefix_cache failed" in str(e):
                # the engine is gone (worker OOM / transport failure) or its cache cannot be cleared:
                # every later rollout would fail in seconds; stop and let a relaunch resume
                log("=== ENGINE_DEAD: aborting the run; rollouts captured so far are in S3 ===")
                errors[-1]["aborted_run"] = True
                break
            continue
        rows.append(row)
        log(f"=== PROGRESS {i + 1}/{len(todo)} {rec['task_id']} N={row['n_tree_tokens']:,} "
            f"clamp_ok={row['clamp_invariant_ok']} fallbacks={row['n_prefix_reuse_fallbacks']} "
            f"{(time.monotonic() - t0) / 60:.1f}m "
            f"elapsed={(time.monotonic() - t_start) / 60:.1f}m ===")

    # resume: rollouts captured by earlier runs stay in the manifest (their meta rows are the
    # source of truth; a fresh manifest with only this run's rows would drop them)
    prior_rows = []
    if args.s3_prefix:
        prior = load_prior_manifest(args.s3_prefix, out_dir)
        prior_rows = [r for r in prior.get("transcripts", []) if r["task_id"] in done]
        missing = done - {r["task_id"] for r in prior_rows}
        for tid in sorted(missing):  # meta present but not in the old manifest: rebuild the row
            prior_rows.append(load_meta_row(args.s3_prefix, out_dir, tid))
    all_rows = sorted(prior_rows + rows, key=lambda r: r["task_id"])
    manifest = {
        "extraction_model": args.model, "scope": "replay of steering-arm rollouts (all tree tokens)",
        "arm": arm, "mode": mode, "dose": dose, "split": split, "seed": seed,
        "layers": layers, "n_model_layers": capturer.n_layers, "d_model": capturer.d_model,
        "layers_note": (None if len(layers) == capturer.n_layers else
                        f"layer subset: segmeans/turn_means have {len(layers)} layer columns in this "
                        "order, not one per model layer"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "steering": steering.provenance() if steering else None,
        "serving": serving_provenance(capturer, layer_chunk_cap=args.layer_chunk, prefix_reuse=args.prefix_reuse,
                                      tokenizer=capturer.llm.get_tokenizer(),
                                      extra={"replay": True, "serving_warnings": warns,
                                             "gate_env": capturer.gate_env}),
        "generation_serving": generation_serving_block(records),
        "exactness": {
            "prompt_ids": "vLLM renderer on the pre-proxy wire request (+strip_think); length asserted "
                          "against the server's logged prompt_tokens every turn; sha-identical to the "
                          "server's prompt when the sidecar is present",
            "branch_ids": "sidecar token_ids when present (exact), else template re-render with branch_delta recorded",
            "steering": (None if not steering else
                         "steer_positions = earlier assistant spans (asserted equal to the sidecar's) + branch, "
                         "under the positions-aware gate, inside one prefill"),
            "invariant": ("proj_u <= threshold + 2^-8·‖h‖ on every steered row at the steering layer"
                          if steering and steering.mode == "clamp" else
                          None if not steering else f"none for {steering.mode} mode"),
        },
        "transcripts": all_rows, "errors": errors,
    }
    mpath = out_dir / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=1) + "\n")
    if args.s3_prefix:
        s3_cp(mpath, args.s3_prefix, "manifest.json")
    log(f"done: {len(rows)} captured, {len(errors)} errors -> {args.s3_prefix or out_dir}")
    if errors:
        sys.exit(2 if any(e.get("aborted_run") for e in errors) else 1)


if __name__ == "__main__":
    main()
