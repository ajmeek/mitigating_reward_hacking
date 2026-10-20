"""Unembedding loader for the J-lens readout."""

def load_unembedding(model: str, weights_path: str | None):
    """Return (W_U, final_norm_gamma) as float32, resolving shards if needed.

    Prefers `lm_head.weight` over `embed_tokens.weight`: Qwen3.5-27B unties them, so the
    embedding matrix is NOT the unembedding and using it silently yields the wrong tokens.
    Tensors may live in different shards, so each is opened from its own file.
    """
    import json

    import torch
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    def resolve(names: list[str]) -> tuple[str, str]:
        """(file, tensor_name) for the first name present."""
        if weights_path:
            with safe_open(weights_path, framework="pt") as f:
                keys = set(f.keys())
            for n in names:
                if n in keys:
                    return weights_path, n
            raise KeyError(f"none of {names} in {weights_path}")
        index = json.load(open(hf_hub_download(model, "model.safetensors.index.json")))
        wm = index["weight_map"]
        for n in names:
            if n in wm:
                return hf_hub_download(model, wm[n]), n
        raise KeyError(f"none of {names} in {model} index")

    out = []
    for names in (
        ["lm_head.weight", "model.language_model.embed_tokens.weight", "model.embed_tokens.weight"],
        ["model.language_model.norm.weight", "model.norm.weight"],
    ):
        path, name = resolve(names)
        with safe_open(path, framework="pt") as f:
            t = f.get_tensor(name)
        out.append(t.to(torch.float32))
    print(f"unembedding: {out[0].shape}, norm: {out[1].shape}")
    return out[0], out[1]
