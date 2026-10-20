"""Dump per-turn and per-segment mean activations, and per-token activations, for MisTrace rollouts.

Runs in the SERVING env (vllm + vllm-lens). The paper's per-token capture of the 42 hacked rollouts at
layers 31 and 35:

    cd serving && uv run python ../scripts/dump_activations.py --mistrace-model qwen3_5_27b \
        --model Qwen/Qwen3.5-27B --layers 31,35 --per-token-layers 31,35 --spans-only \
        --max-model-len 131072 --tensor-parallel 4 --out-dir ../data/captures/qwen35_27b_bf16_token_L31_35

    # smoke: --limit 2 --max-turns 3 --gpu-memory-utilization 0.5

Transcripts come from the published MisTrace dataset. Output per rollout: <task>.safetensors (turn and
segment means per layer; token_acts [n_tok, n_layers, d] when --per-token-layers is set; token_turn,
token_pos) and <task>.json (turns, segments with text and offsets), plus manifest.json. Read with
steering_reasoning.activations.load_dump.
"""

import argparse
import hashlib
import json
import sys
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
    THINK_OPEN,
    _part_text,
    iter_reasoning_turns,
    load_transcripts,
)


def resolve_layers(model: str, layers_arg: str, error) -> tuple[list[int], int]:
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model)
    n_layers = getattr(cfg, "num_hidden_layers", None) or getattr(
        getattr(cfg, "text_config", None), "num_hidden_layers", None
    )
    if n_layers is None:
        error(f"cannot determine layer count for {model}")
    if layers_arg == "auto":
        layers = sorted({round(n_layers * f) for f in (0.25, 0.35, 0.45, 0.55, 0.65, 0.75)})
    else:
        layers = sorted(int(x) for x in layers_arg.split(","))
        bad = [li for li in layers if not 0 <= li < n_layers]
        if bad:
            error(f"layers {bad} out of range ({n_layers} layers)")
    return layers, n_layers


def serving_provenance(args) -> dict:
    """Capture-time config that changes numerics — what distinguishes two runs.

    Reduction order depends on tensor-parallel degree and on how chunked prefill splits a
    sequence, so two runs of the same model over the same tokens can differ by ~1% on some
    turns while matching bit-for-bit on others. None of that is recoverable from the
    tensors, so it has to be written down at capture time.
    """
    import torch
    import vllm

    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "tensor_parallel_size": args.tensor_parallel,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": "auto (follows the checkpoint)",
        "vllm_version": vllm.__version__,
        "torch_version": torch.__version__,
        "gpu": dev,
        "n_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mistrace-model", default=None,
                   help="load transcripts from the published MisTrace dataset (preferred): "
                        "qwen3_5_27b | deepseek_r1 | ...")
    p.add_argument("--mistrace-config", default="reward-hacking")
    p.add_argument("--mistrace-revision", default=None, help="pin a dataset revision")
    p.add_argument("--transcripts", default=None,
                   help="legacy: path to an in-repo transcript JSON (stale; no judge labels)")
    p.add_argument("--model", required=True)
    p.add_argument("--layers", default="auto")
    p.add_argument("--label-field", default="is_reward_hack")
    p.add_argument("--limit", type=int, default=None, help="max transcripts (smoke)")
    p.add_argument("--max-turns", type=int, default=None)
    p.add_argument("--max-model-len", type=int, default=14336)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--gold-only", action="store_true",
                   help="only transcripts with gold (human) span labels (fast segment-arm run)")
    p.add_argument("--spans-only", action="store_true",
                   help="only transcripts carrying any span labels (all hacked rollouts in MisTrace)")
    p.add_argument("--tensor-parallel", type=int, default=1)
    p.add_argument("--per-token-layers", default=None,
                   help="comma-separated subset of --layers to also store PER TOKEN (fp16), "
                        "for span-labelled rollouts only. Lets future label revisions "
                        "re-pool offline instead of re-running the model. Must be a subset "
                        "of --layers. ~10GB per layer over 42 hacked rollouts.")
    p.add_argument("--skip-existing", action="store_true",
                   help="skip transcripts whose output pair already exists (spot-interruption resume)")
    args = p.parse_args()

    import torch
    from safetensors.torch import save_file
    from transformers import AutoTokenizer

    from steering_reasoning.steering.activations import (
        ActivationCapturer,
        apply_no_nvcc_workarounds,
    )

    layers, n_layers = resolve_layers(args.model, args.layers, p.error)
    per_token_layers = (
        sorted(int(x) for x in args.per_token_layers.split(",")) if args.per_token_layers else []
    )
    missing = [li for li in per_token_layers if li not in layers]
    if missing:
        p.error(f"--per-token-layers {missing} not in --layers {layers}")
    print(f"model has {n_layers} layers; capturing {layers}"
          + (f"; per-token storage for {per_token_layers}" if per_token_layers else ""))

    if bool(args.mistrace_model) == bool(args.transcripts):
        p.error("pass exactly one of --mistrace-model (preferred) or --transcripts")
    if args.mistrace_model:
        from steering_reasoning.data.mistrace import load_mistrace

        records, source_provenance = load_mistrace(
            args.mistrace_model,
            config=args.mistrace_config,
            revision=args.mistrace_revision,
        )
    else:
        records = load_transcripts(args.transcripts)
        source_provenance = {"legacy_transcripts": str(args.transcripts)}

    records = [r for r in records if r.get(args.label_field) in (True, False)]
    if args.gold_only:
        records = [r for r in records if r.get("has_gold_spans")]
    if args.spans_only:
        records = [r for r in records if r.get("span_labels")]
    if args.limit:
        records = records[: args.limit]
    print(f"{len(records)} transcripts ({sum(r[args.label_field] for r in records)} pos, "
          f"{sum(bool(r.get('span_labels')) for r in records)} with span labels, "
          f"{sum(bool(r.get('has_gold_spans')) for r in records)} with human gold)")

    apply_no_nvcc_workarounds()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if not tokenizer.is_fast:
        p.error("segment mapping needs a fast tokenizer (offset_mapping)")
    capturer = ActivationCapturer(
        args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for ri, rec in enumerate(records):
        if args.skip_existing and (out_dir / f"{rec['task_id']}.json").exists():
            meta = json.loads((out_dir / f"{rec['task_id']}.json").read_text())
            manifest_rows.append({
                "task_id": rec["task_id"], "label": meta["label"],
                "has_gold_spans": meta["has_gold_spans"],
                "n_turns": len(meta["turns"]),
                "n_tokens": sum(t["n_tokens"] for t in meta["turns"]),
                "n_segments": len(meta["segments"]),
            })
            print(f"[{ri + 1}/{len(records)}] {rec['task_id']}: exists, skipped")
            continue
        turns = iter_reasoning_turns(
            rec, tokenizer, max_tokens=args.max_model_len - 8, max_turns=args.max_turns
        )
        if not turns:
            print(f"[{ri + 1}/{len(records)}] {rec['task_id']}: no usable turns, skipped")
            continue

        # any span-labelled rollout gets segment activations — MisTrace carries judge labels
        # on every hacked rollout, so this is 42 rollouts rather than the 4 human-gold ones
        seg_by_msg = reasoning_segments_by_message(rec) if rec.get("span_labels") else {}

        turn_means, turn_tokens, turn_meta = [], [], []
        seg_means, seg_tokens, seg_meta = [], [], []
        tok_acts, tok_turn, tok_pos = [], [], []
        # per-token rows index into `layers` — record which layers they correspond to
        pt_rows = [layers.index(li) for li in per_token_layers] if per_token_layers else []
        store_per_token = bool(pt_rows) and bool(seg_by_msg)

        for bi in range(0, len(turns), args.batch_size):
            batch = turns[bi : bi + args.batch_size]
            acts = capturer.capture([t.prompt for t in batch], layers)
            for turn, act in zip(batch, acts):
                span = act[:, turn.reasoning_start : turn.reasoning_end, :]
                if span.shape[1] == 0:
                    continue
                turn_means.append(span.float().mean(dim=1).cpu())
                turn_tokens.append(span.shape[1])
                turn_meta.append({
                    "message_index": turn.turn_index,
                    "n_tokens": span.shape[1],
                    # absolute prompt-token index of the first reasoning token, so segment
                    # spans recorded in prompt coordinates can be re-mapped onto tok_pos
                    "reasoning_start": turn.reasoning_start,
                })

                if store_per_token:
                    ti = len(turn_means) - 1
                    tok_acts.append(span[pt_rows].permute(1, 0, 2).half().cpu())  # (n_tok, L_pt, D)
                    tok_turn.append(torch.full((span.shape[1],), ti, dtype=torch.int32))
                    tok_pos.append(torch.arange(span.shape[1], dtype=torch.int32))

                if turn.turn_index in seg_by_msg:
                    reasoning = _part_text(
                        rec["messages"][turn.turn_index].get("content"), "reasoning"
                    ).strip()
                    prefix_chars = len(turn.prompt) - len(reasoning) - len("\n</think>")
                    assert turn.prompt[prefix_chars : prefix_chars + len(reasoning)] == reasoning
                    located = locate_segments(reasoning, seg_by_msg[turn.turn_index])
                    for seg, cs, ce, ts, te in char_to_token_spans(
                        turn.prompt, prefix_chars, located, tokenizer
                    ):
                        sspan = act[:, ts:te, :]
                        if sspan.shape[1] == 0:
                            continue
                        seg_means.append(sspan.float().mean(dim=1).cpu())
                        seg_tokens.append(sspan.shape[1])
                        seg_meta.append({
                            "message_index": turn.turn_index,
                            "segment_idx": seg["segment_idx"],
                            "position": seg["position"],
                            "gold_label": bool(seg["gold_label"]),
                            "judge_label": seg.get("judge_label"),
                            "n_tokens": sspan.shape[1],
                            # prompt-absolute; subtract the turn's reasoning_start to index
                            # token_pos, so per-token data can reproduce this segment's mean
                            "token_start": ts,
                            "token_end": te,
                            "text_sha1": hashlib.sha1(seg["text"].encode()).hexdigest()[:10],
                            "text": seg["text"],
                        })

        tensors = {
            "turn_means": torch.stack(turn_means),
            "turn_tokens": torch.tensor(turn_tokens, dtype=torch.int32),
        }
        if seg_means:
            # fp16: pooling happens in fp32, only the result is stored narrow (~1e-3
            # relative error, irrelevant to projections) — halves a multi-GB artifact
            tensors["segment_means"] = torch.stack(seg_means).half()
            tensors["segment_tokens"] = torch.tensor(seg_tokens, dtype=torch.int32)
        if tok_acts:
            tensors["token_acts"] = torch.cat(tok_acts)  # (N, L_pt, D) fp16
            tensors["token_turn"] = torch.cat(tok_turn)  # row index into turn_means
            tensors["token_pos"] = torch.cat(tok_pos)  # offset from the turn's reasoning_start

        save_file(tensors, str(out_dir / f"{rec['task_id']}.safetensors"))
        (out_dir / f"{rec['task_id']}.json").write_text(json.dumps({
            "task_id": rec["task_id"],
            "label": bool(rec[args.label_field]),
            "has_gold_spans": bool(rec.get("has_gold_spans")),
            "turns": turn_meta,
            "segments": seg_meta,
        }, indent=1) + "\n")

        n_gold_pos = sum(s["gold_label"] for s in seg_meta)
        print(f"[{ri + 1}/{len(records)}] {rec['task_id']}: {len(turn_means)} turns, "
              f"{sum(turn_tokens)} tokens"
              + (f", {len(seg_means)} segments ({n_gold_pos} gold-pos)" if seg_means else ""))
        manifest_rows.append({
            "task_id": rec["task_id"],
            "label": bool(rec[args.label_field]),
            "has_gold_spans": bool(rec.get("has_gold_spans")),
            "n_turns": len(turn_means),
            "n_tokens": int(sum(turn_tokens)),
            "n_segments": len(seg_means),
        })

    (out_dir / "manifest.json").write_text(json.dumps({
        "extraction_model": args.model,
        "transcript_source": source_provenance,
        "transcript_model": records[0].get("target_model", "unknown") if records else None,
        "label_field": args.label_field,
        "layers": layers,
        "n_model_layers": n_layers,
        "created_at": datetime.now(timezone.utc).isoformat(),
        # Serving config: two captures of the same layer on bit-identical inputs diverged
        # on ~27% of segments (turn-aligned, ~1% median) with no way to attribute it after
        # the fact. Record enough to tell two runs apart. 
        "serving": serving_provenance(args),
        "transcripts": manifest_rows,
    }, indent=1) + "\n")
    print(f"\nDumped {len(manifest_rows)} transcripts to {out_dir}")


if __name__ == "__main__":
    main()
