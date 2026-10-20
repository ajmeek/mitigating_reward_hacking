#!/usr/bin/env python
"""Read the reward-hack DiffMean direction through the Jacobian lens, not the logit lens.

The logit lens decodes a mid-stack residual vector by unembedding it directly, which
assumes the remaining layers act as the identity. At layer 35 of 64 that assumption is
weak. The
Jacobian lens replaces the identity with the model's averaged input-output Jacobian,

    J_L = E[d h_final / d h_L]        (Anthropic, "Verbalizable Representations Form a
                                       Global Workspace in Language Models", 2026)

so the readout becomes

    scores(v) = W_U @ final_norm(J_L @ v)      vs.     W_U @ final_norm(v)

Both are ranked here side by side, for +v and -v, so the "+v is crisp / -v is noise"
asymmetry can be re-tested after correcting for depth.

The fitted lens is Neuronpedia's, for Qwen/Qwen3.5-27B (bf16 weights, wikitext corpus).

    uv run --with torch --with safetensors --with transformers --with huggingface_hub \
        python experiments/j-lens-readout/scripts/jlens_readout.py \
        --vector src/steering_reasoning/canonical/u_L35_steering.safetensors \
        --layer 35 \
        --out experiments/j-lens-readout/artifacts/jlens_vs_logitlens_L35.md
"""

import argparse
import json
import sys
from pathlib import Path

import torch

EXP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from logit_lens_vectors import load_unembedding  # noqa: E402

LENS_REPO = "neuronpedia/jacobian-lens"
LENS_FILE = "qwen3.5-27b/jlens/Salesforce-wikitext/Qwen3.5-27B_jacobian_lens.pt"


def load_lens(path: str | None) -> dict:
    """Return the fitted lens dict: {'J': {layer: [d, d]}, 'source_layers', 'd_model', ...}."""
    if path is None:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(LENS_REPO, LENS_FILE)
    blob = torch.load(path, map_location="cpu", weights_only=True)
    return blob


def load_vector(path: Path, layer: int) -> torch.Tensor:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as f:
        keys = list(f.keys())
        t = f.get_tensor(keys[0]) if len(keys) == 1 else f.get_tensor(f"layer_{layer}")
    t = t.to(torch.float32).squeeze()
    if t.ndim != 1:
        raise ValueError(f"expected a 1-D direction, got {tuple(t.shape)} from {keys}")
    return t


def rms_norm(v: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Qwen RMSNorm. The 1/rms factor is a positive scalar, so it cannot change token
    ranking for a single direction; gamma can, and is what the existing logit-lens
    artifact means by 'final-norm gain applied'."""
    return gamma * v / (v.pow(2).mean().sqrt() + eps)


def top_tokens(scores: torch.Tensor, tok, k: int) -> list[tuple[str, float]]:
    vals, idx = scores.topk(k)
    return [(tok.decode([int(i)]), float(s)) for s, i in zip(vals, idx)]


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--vector", required=True, help="safetensors direction at --layer")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--lens", default=None, help="local lens .pt (default: fetch from hub)")
    p.add_argument("--model", default="Qwen/Qwen3.5-27B", help="source of W_U / final norm")
    p.add_argument("--tokenizer", default="Qwen/Qwen3.5-27B")
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    W_U, gamma = load_unembedding(args.model, None)
    W_U, gamma = W_U.to(torch.float32), gamma.to(torch.float32)

    lens = load_lens(args.lens)
    J_all = lens["J"]
    if args.layer not in J_all:
        raise SystemExit(
            f"layer {args.layer} not fitted; available: {sorted(J_all)[:8]}..."
        )
    J = J_all[args.layer].to(torch.float32)

    v = load_vector(Path(args.vector), args.layer)
    if v.shape[0] != J.shape[0]:
        raise SystemExit(f"dim mismatch: vector {v.shape[0]} vs lens {J.shape[0]}")

    Jv = J @ v
    cos = torch.nn.functional.cosine_similarity(Jv, v, dim=0).item()
    ident = {
        li: float(
            (Jl.to(torch.float32) - torch.eye(Jl.shape[0])).norm()
            / torch.eye(Jl.shape[0]).norm()
        )
        for li in sorted(J_all)
        if li in (args.layer, max(J_all), min(J_all))
        for Jl in [J_all[li]]
    }

    readouts = {}
    for lens_name, vec in (("logit lens", v), ("J-lens", Jv)):
        for sign, s in (("+v", 1.0), ("-v", -1.0)):
            h = rms_norm(s * vec, gamma)
            readouts[(lens_name, sign)] = top_tokens(W_U @ h, tok, args.topk)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write(f"# J-lens vs logit lens on `{Path(args.vector).name}` (layer {args.layer})\n\n")
        f.write(f"- lens: `{LENS_REPO}` / `{LENS_FILE}`, fitted on `Qwen/Qwen3.5-27B`\n")
        f.write(f"- prompts fitted: {lens.get('n_prompts', 'n/a')}, d_model {lens.get('d_model')}\n")
        f.write(f"- unembedding + final norm from `{args.model}`\n")
        f.write(f"- cos(J@v, v) = **{cos:.4f}**, |J@v|/|v| = {Jv.norm() / v.norm():.3f}\n")
        f.write(f"- relative distance of J from identity, ||J - I||/||I||: "
                + ", ".join(f"L{li} {d:.3f}" for li, d in sorted(ident.items())) + "\n\n")
        for sign in ("+v", "-v"):
            f.write(f"## {sign}\n\n| # | logit lens | score | J-lens | score |\n|---|---|---|---|---|\n")
            for i in range(args.topk):
                lt, ls = readouts[("logit lens", sign)][i]
                jt, js = readouts[("J-lens", sign)][i]
                f.write(f"| {i + 1} | `{lt}` | {ls:.2f} | `{jt}` | {js:.2f} |\n")
            f.write("\n")

    summary = {
        "vector": str(args.vector),
        "layer": args.layer,
        "cos_Jv_v": cos,
        "norm_ratio": float(Jv.norm() / v.norm()),
        "identity_distance": ident,
        "readouts": {f"{k[0]}|{k[1]}": v_ for k, v_ in readouts.items()},
    }
    Path(str(out).replace(".md", ".json")).write_text(json.dumps(summary, indent=2))
    print(f"cos(J@v, v) = {cos:.4f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
