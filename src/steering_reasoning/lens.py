"""Read a residual-stream direction out as tokens, through the logit lens or the J-lens.

Two decoders for the same question ("which tokens does this direction promote?"):

    logit lens:  scores = W_U @ rms_norm(v, gamma)
    J-lens:      scores = W_U @ rms_norm(J_L @ v, gamma)

The logit lens unembeds a mid-stack vector directly, i.e. it assumes every layer above L
acts as the identity. The Jacobian lens replaces that identity with the model's fitted
average input-output Jacobian J_L = E[d h_final / d h_L] (Anthropic, "Verbalizable
Representations Form a Global Workspace in Language Models", 2026), which is a much better
approximation at mid depth.

Two ranking modes for either lens. Writing e_t = J^T (gamma * W_U[t]) for token t's
pullback direction (with J = I for the logit lens), the score above is exactly <e_t, v> up
to a positive scalar. So:

    mode="raw"     ranks by <e_t, v>          -- how hard v pushes that token's logit
    mode="cosine"  ranks by <e_t, v> / |e_t|  -- how well v is *aligned* with the token

Raw is dominated by tokens with large-norm pullbacks; cosine asks the alignment question
instead. Both are useful and both are supported.

The fitted lens is Neuronpedia's, for `Qwen/Qwen3.5-27B` on wikitext. Not every layer is fitted -- see
`available_layers()`.

Loading is lazy and memoised per process: W_U is [248320, 5120] float32 (~5 GB) and the
lens file is ~3 GB, so importing this module costs nothing and each artifact is read at
most once.

Example:

    from steering_reasoning.lens import top_tokens, render_markdown_table

    cols = top_tokens("src/steering_reasoning/canonical/u_L35_steering.safetensors", layer=35, lens="both", k=20)
    print(render_markdown_table(cols))
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, NamedTuple, Sequence

if TYPE_CHECKING:  # heavy imports stay inside functions at runtime
    import torch

DEFAULT_MODEL = "Qwen/Qwen3.5-27B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_FILE = "qwen3.5-27b/jlens/Salesforce-wikitext/Qwen3.5-27B_jacobian_lens.pt"

LENSES = ("logit", "jacobian")
MODES = ("raw", "cosine")

RMS_EPS = 1e-6
CHUNK = 8192  # rows of W_U per pullback-norm matmul


class TokenScore(NamedTuple):
    """A (token_string, score) pair; unpacks as a plain tuple."""

    token: str
    score: float


class ProbeScore(NamedTuple):
    """One hand-specified probe token, scored and ranked against the full vocabulary."""

    group: str
    token: str
    token_id: int
    score: float
    rank: int  # 1-based rank among all vocabulary tokens under the same lens/mode/sign


# --------------------------------------------------------------------------------------
# cached artifact loading
# --------------------------------------------------------------------------------------

_UNEMBED_CACHE: dict[tuple[str, str | None], tuple[Any, Any]] = {}
_LENS_CACHE: dict[str, dict] = {}
_TOKENIZER_CACHE: dict[str, Any] = {}
_PULLBACK_NORM_CACHE: dict[tuple, Any] = {}


def load_unembedding(
    model: str = DEFAULT_MODEL, weights_path: str | None = None
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Return (W_U, final_norm_gamma) as float32, resolving shards if needed. Memoised.

    Prefers `lm_head.weight` over `embed_tokens.weight`: Qwen3.5-27B unties them, so the
    embedding matrix is NOT the unembedding and using it silently yields the wrong tokens.
    The two tensors may live in different shards, so each is opened from its own file.
    """
    key = (model, weights_path)
    if key in _UNEMBED_CACHE:
        return _UNEMBED_CACHE[key]

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
        weight_map = index["weight_map"]
        for n in names:
            if n in weight_map:
                return hf_hub_download(model, weight_map[n]), n
        raise KeyError(f"none of {names} in {model} index")

    out = []
    for names in (
        ["lm_head.weight", "model.language_model.embed_tokens.weight", "model.embed_tokens.weight"],
        ["model.language_model.norm.weight", "model.norm.weight"],
    ):
        path, name = resolve(names)
        with safe_open(path, framework="pt") as f:
            out.append(f.get_tensor(name).to(torch.float32))

    _UNEMBED_CACHE[key] = (out[0], out[1])
    return _UNEMBED_CACHE[key]


def load_lens(lens_path: str | Path | None = None) -> dict:
    """Return the fitted lens dict, memoised by resolved path.

    Keys: 'J' (layer int -> [d, d] tensor), 'source_layers', 'd_model', 'n_prompts'.
    With `lens_path=None` the file is fetched from the HF hub (cached there too).
    """
    import torch

    if lens_path is None:
        from huggingface_hub import hf_hub_download

        lens_path = hf_hub_download(LENS_REPO, LENS_FILE)
    lens_path = str(lens_path)
    if lens_path not in _LENS_CACHE:
        _LENS_CACHE[lens_path] = torch.load(lens_path, map_location="cpu", weights_only=True)
    return _LENS_CACHE[lens_path]


def available_layers(lens_path: str | Path | None = None) -> list[int]:
    """Layers for which a Jacobian was fitted (not every layer is)."""
    return sorted(load_lens(lens_path)["J"])


def jacobian(layer: int, lens_path: str | Path | None = None) -> "torch.Tensor":
    """The fitted average Jacobian J_L = E[d h_final / d h_L] at `layer`, as float32."""
    import torch

    lens = load_lens(lens_path)
    J_all = lens["J"]
    if layer not in J_all:
        raise KeyError(
            f"layer {layer} is not fitted in this lens; available layers: {sorted(J_all)}"
        )
    return J_all[layer].to(torch.float32)


def load_tokenizer(tokenizer: str = DEFAULT_MODEL):
    """Memoised `AutoTokenizer.from_pretrained`."""
    if tokenizer not in _TOKENIZER_CACHE:
        from transformers import AutoTokenizer

        _TOKENIZER_CACHE[tokenizer] = AutoTokenizer.from_pretrained(tokenizer)
    return _TOKENIZER_CACHE[tokenizer]


def clear_caches() -> None:
    """Drop the cached unembedding / lens / tokenizer / pullback norms (frees ~8 GB)."""
    for cache in (_UNEMBED_CACHE, _LENS_CACHE, _TOKENIZER_CACHE, _PULLBACK_NORM_CACHE):
        cache.clear()


# --------------------------------------------------------------------------------------
# direction handling
# --------------------------------------------------------------------------------------


def load_vector(path: str | Path, layer: int | None = None) -> "torch.Tensor":
    """Load a 1-D float32 direction from a .safetensors artifact.

    Single-key files (this repo's convention: key `vector`, shape [1, d]) are squeezed;
    multi-key files are indexed by `layer_{layer}`.
    """
    import torch
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as f:
        keys = list(f.keys())
        if len(keys) == 1:
            t = f.get_tensor(keys[0])
        elif layer is None:
            raise ValueError(f"{path} has keys {keys}; pass `layer` to select one")
        else:
            t = f.get_tensor(f"layer_{layer}")
    t = t.to(torch.float32).squeeze()
    if t.ndim != 1:
        raise ValueError(f"expected a 1-D direction, got {tuple(t.shape)} from {keys}")
    return t


def as_direction(vector: "torch.Tensor | str | Path", layer: int | None = None) -> "torch.Tensor":
    """Coerce a tensor or a .safetensors path into a 1-D float32 direction."""
    import torch

    if isinstance(vector, (str, Path)):
        return load_vector(vector, layer)
    v = vector.detach().to(torch.float32).squeeze()
    if v.ndim != 1:
        raise ValueError(f"expected a 1-D direction, got {tuple(v.shape)}")
    return v


def rms_norm(v: "torch.Tensor", gamma: "torch.Tensor", eps: float = RMS_EPS) -> "torch.Tensor":
    """Qwen RMSNorm applied to a lone vector.

    The 1/rms factor is a positive scalar, so it cannot change token ranking for a single
    direction; gamma can, and is what "final-norm gain applied" means in this repo.
    """
    return gamma * v / (v.pow(2).mean().sqrt() + eps)


def _as_sign(sign: int | float | str) -> float:
    if isinstance(sign, str):
        if sign in ("+", "+v", "promote"):
            return 1.0
        if sign in ("-", "-v", "suppress"):
            return -1.0
        raise ValueError(f"unrecognised sign {sign!r}")
    if sign not in (1, -1, 1.0, -1.0):
        raise ValueError(f"sign must be +1 or -1, got {sign!r}")
    return float(sign)


def _check(name: str, value: str, allowed: tuple[str, ...]) -> str:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
    return value


# --------------------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------------------


def _pullback_norms(
    lens: str, layer: int, model: str, weights_path: str | None, lens_path: str | Path | None
) -> "torch.Tensor":
    """|e_t| for every token, where e_t = J^T (gamma * W_U[t]). Chunked and memoised."""
    import torch

    W_U, gamma = load_unembedding(model, weights_path)
    if lens == "logit":
        key = ("logit", model, weights_path)
        if key not in _PULLBACK_NORM_CACHE:
            _PULLBACK_NORM_CACHE[key] = (gamma * W_U).norm(dim=1)
        return _PULLBACK_NORM_CACHE[key]

    key = ("jacobian", model, weights_path, str(lens_path), layer)
    if key not in _PULLBACK_NORM_CACHE:
        J = jacobian(layer, lens_path)
        V = W_U.shape[0]
        norms = torch.empty(V)
        for i in range(0, V, CHUNK):
            P = (gamma * W_U[i : i + CHUNK]) @ J  # (chunk, d): pullback directions e_t
            norms[i : i + CHUNK] = P.norm(dim=1)
        _PULLBACK_NORM_CACHE[key] = norms
    return _PULLBACK_NORM_CACHE[key]


def score_vocab(
    vector: "torch.Tensor | str | Path",
    layer: int,
    *,
    lens: str = "jacobian",
    mode: str = "raw",
    sign: int | float | str = 1,
    model: str = DEFAULT_MODEL,
    weights_path: str | None = None,
    lens_path: str | Path | None = None,
) -> "torch.Tensor":
    """Score every vocabulary token against `vector` read at `layer`.

    Returns a [vocab_size] float32 tensor. `lens` is "logit" or "jacobian" (no "both"
    here -- use `top_tokens` for that). `mode` is "raw" (logit push, in logit units) or
    "cosine" (alignment of v with the token's pullback direction, in [-1, 1]).
    """
    _check("lens", lens, LENSES)
    _check("mode", mode, MODES)
    s = _as_sign(sign)

    W_U, gamma = load_unembedding(model, weights_path)
    v = s * as_direction(vector, layer)
    if mode == "cosine":
        # normalise BEFORE the pullback: the cosine is <e_t, v>/(|e_t| |v|), so the
        # denominator is |v|, not |J @ v| (they differ by ~1% at layer 35)
        v = v / v.norm()
    if lens == "jacobian":
        J = jacobian(layer, lens_path)
        if v.shape[0] != J.shape[0]:
            raise ValueError(f"dim mismatch: vector {v.shape[0]} vs lens {J.shape[0]}")
        v = J @ v
    if v.shape[0] != W_U.shape[1]:
        raise ValueError(f"dim mismatch: vector {v.shape[0]} vs unembedding {W_U.shape[1]}")

    if mode == "raw":
        return W_U @ rms_norm(v, gamma)
    # cosine: <e_t, v_hat> / |e_t|, using (gamma * W_U) @ x == W_U @ (gamma * x)
    overlap = W_U @ (gamma * v)
    return overlap / _pullback_norms(lens, layer, model, weights_path, lens_path).clamp_min(1e-9)


def top_tokens(
    vector: "torch.Tensor | str | Path",
    layer: int,
    *,
    lens: str = "both",
    mode: str = "raw",
    k: int = 20,
    sign: int | float | str = 1,
    model: str = DEFAULT_MODEL,
    tokenizer: str | None = None,
    weights_path: str | None = None,
    lens_path: str | Path | None = None,
) -> dict[str, list[TokenScore]]:
    """Top-k most promoted tokens for `sign * vector`, per lens.

    Always returns a dict keyed by lens name ("logit", "jacobian"), so callers do not have
    to branch on whether `lens="both"` was asked for. Values are lists of TokenScore, which
    unpack as (token_string, score). Pass `sign=-1` for the suppressed end of the axis.
    """
    _check("lens", lens, LENSES + ("both",))
    names = LENSES if lens == "both" else (lens,)
    tok = load_tokenizer(tokenizer or model)

    out: dict[str, list[TokenScore]] = {}
    for name in names:
        scores = score_vocab(
            vector,
            layer,
            lens=name,
            mode=mode,
            sign=sign,
            model=model,
            weights_path=weights_path,
            lens_path=lens_path,
        )
        vals, idx = scores.topk(k)
        out[name] = [TokenScore(tok.decode([int(i)]), float(s)) for s, i in zip(vals, idx)]
    return out


def probe_scores(
    vector: "torch.Tensor | str | Path",
    layer: int,
    probes: Mapping[str, Sequence[str]] | Iterable[str],
    *,
    lens: str = "jacobian",
    mode: str = "raw",
    sign: int | float | str = 1,
    model: str = DEFAULT_MODEL,
    tokenizer: str | None = None,
    weights_path: str | None = None,
    lens_path: str | Path | None = None,
    skip_multi_token: bool = True,
) -> list[ProbeScore]:
    """Score an explicit list of probe tokens and report where each ranks in the vocabulary.

    `probes` is either a flat iterable of token strings or a {group: [tokens]} mapping (the
    grouped form documents which hypothesis each probe tests). Strings that do not resolve
    to exactly one token are skipped unless `skip_multi_token=False`, which raises instead.
    """
    import torch

    groups: Mapping[str, Sequence[str]] = (
        probes if isinstance(probes, Mapping) else {"": list(probes)}
    )
    tok = load_tokenizer(tokenizer or model)
    scores = score_vocab(
        vector,
        layer,
        lens=lens,
        mode=mode,
        sign=sign,
        model=model,
        weights_path=weights_path,
        lens_path=lens_path,
    )
    V = scores.shape[0]
    order = scores.argsort(descending=True)
    rank_of = torch.empty(V, dtype=torch.long)
    rank_of[order] = torch.arange(V)

    rows: list[ProbeScore] = []
    for group, toks in groups.items():
        for t in toks:
            tid = _token_id(tok, t, V)
            if tid is None:
                if skip_multi_token:
                    continue
                raise ValueError(f"{t!r} is not a single token in {tokenizer or model}")
            rows.append(
                ProbeScore(group, t, tid, float(scores[tid]), int(rank_of[tid]) + 1)
            )
    return rows


def _token_id(tok, text: str, vocab_size: int) -> int | None:
    """Resolve a probe string to a single token id, or None if it isn't one token."""
    tid = tok.convert_tokens_to_ids(text)
    unk = getattr(tok, "unk_token_id", None)
    if tid is not None and tid != unk and tid < vocab_size:
        return int(tid)
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) == 1 and ids[0] < vocab_size:
        return int(ids[0])
    return None


def jacobian_diagnostics(
    vector: "torch.Tensor | str | Path",
    layer: int,
    *,
    lens_path: str | Path | None = None,
) -> dict[str, float]:
    """How far the J-lens moves this direction: cos(J@v, v), |J@v|/|v|, ||J - I||/||I||."""
    import torch

    v = as_direction(vector, layer)
    J = jacobian(layer, lens_path)
    Jv = J @ v
    eye = torch.eye(J.shape[0])
    return {
        "cos_Jv_v": float(torch.nn.functional.cosine_similarity(Jv, v, dim=0)),
        "norm_ratio": float(Jv.norm() / v.norm()),
        "identity_distance": float((J - eye).norm() / eye.norm()),
    }


# --------------------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------------------

_LENS_TITLES = {"logit": "logit lens", "jacobian": "J-lens"}


def render_markdown_table(
    columns: Mapping[str, Sequence[TokenScore]],
    *,
    score_fmt: str = ".2f",
    index: bool = True,
) -> str:
    """Render one or more token lists side by side as a markdown table.

    `columns` maps a column title (e.g. the lens name from `top_tokens`) to its list of
    (token, score) pairs. Lists of unequal length are padded with blanks.
    """
    titles = [_LENS_TITLES.get(name, name) for name in columns]
    rows_n = max((len(v) for v in columns.values()), default=0)

    header = (["#"] if index else []) + [c for t in titles for c in (t, "score")]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for i in range(rows_n):
        cells = [str(i + 1)] if index else []
        for values in columns.values():
            if i < len(values):
                token, score = values[i]
                cells += [f"`{token}`", format(score, score_fmt)]
            else:
                cells += ["", ""]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def render_probe_table(
    rows: Sequence[ProbeScore], *, score_fmt: str = "+.3f", score_label: str = "score"
) -> str:
    """Render `probe_scores` output as a markdown table (group, token, score, rank)."""
    grouped = any(r.group for r in rows)
    header = (["group"] if grouped else []) + ["token", score_label, "rank"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        cells = ([r.group] if grouped else []) + [
            f"`{r.token}`",
            format(r.score, score_fmt),
            f"#{r.rank:,}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
