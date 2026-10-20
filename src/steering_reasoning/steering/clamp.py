"""Clamp-steering request payload and reasoning-tag hygiene for steering proxies.

`build_clamp_payload` serializes a single-layer steering-vector artifact into the JSON
string that vllm-lens expects under vllm_xargs["apply_steering_vectors"]; `strip_think`
removes the literal <think> tags the reasoning parser leaves in replayed reasoning.

Heavy deps (numpy, zstandard, safetensors) load lazily inside build_clamp_payload, so
this module — and the steering package — stay importable from the root eval env.
"""

import base64
import json


def build_steer_payload(artifact: str, mode: str = "clamp", threshold: float | None = None,
                        scale: float = 1.0, steer_positions: list[list[int]] | None = None) -> str:
    """Serialize a steering artifact exactly as vllm_lens._helpers._serialize.serialize_tensor does.

    artifact: path to a .safetensors file holding "vector", with a .json sidecar carrying
    "layer_indices". Returns the JSON string for vllm_xargs["apply_steering_vectors"].
    mode: "clamp" (h -= max(0, h.u - threshold) u; needs threshold), "add" (h += scale * v,
    vllm-lens upstream), or "ablate" (h -= (h.u) u). The server applies whichever mode the
    bootstrap-patched vllm-lens supports (serving/patch_vllm_lens_modes.py). Generated tokens are
    always steered; during prefill only positions inside steer_positions ([start, end) absolute
    token ranges the model itself wrote) are, so prompts and tool outputs are never touched
    (serving/patch_vllm_lens_gate.py). The proxy fills steer_positions per request
    (assistant_spans); a replay (steering/replay.py) fills them with the spans generation steered
    plus the generated branch, so one prefill reproduces a generation turn.
    """
    if mode not in ("clamp", "add", "ablate"):
        raise ValueError(f"mode must be clamp|add|ablate, got {mode!r}")
    if mode == "clamp" and threshold is None:
        raise ValueError("clamp mode needs a threshold")
    import numpy as np
    import zstandard as zstd
    from safetensors.numpy import load_file

    vector = load_file(artifact)["vector"].astype(np.float32)
    meta = json.loads(open(artifact.replace(".safetensors", ".json")).read())
    compressed = zstd.ZstdCompressor(level=1).compress(vector.tobytes())
    sv = {
        "activations": {
            "data": base64.b64encode(compressed).decode("ascii"),
            "dtype": "float32",
            "original_dtype": "torch.float32",
            "shape": list(vector.shape),
            "compression": "zstd",
        },
        "layer_indices": meta["layer_indices"],
        "scale": scale,
        "norm_match": False,
        "position_indices": None,
        "mode": mode,
        "threshold": threshold,
        "steer_positions": steer_positions,
    }
    return json.dumps([sv])


def with_positions(payload: str, steer_positions: list[list[int]] | None) -> str:
    """Return the payload JSON with steer_positions set on every vector (per-request, cheap)."""
    svs = json.loads(payload)
    for sv in svs:
        sv["steer_positions"] = steer_positions
    return json.dumps(svs)


def assistant_spans(ids: list[int], header: list[int], think: list[int], end_id: int) -> list[list[int]]:
    """[start, end) token ranges of every completed assistant message in a rendered prompt.

    header = ids of '<|im_start|>assistant\\n', think = ids of '<think>\\n' (the '<think>' token and its
    newline are skipped when they directly follow the header: the generation prompt supplies them, the
    model did not write them), end_id =
    '<|im_end|>' (excluded: the model emitted it as EOS, which never ran a forward pass). The
    trailing generation prompt has no end token and yields no span. Rendering decides what counts:
    the Qwen template drops reasoning from assistant messages before the last user message, so
    spans cover exactly the tokens the server will see.
    """
    spans, n, h = [], len(ids), len(header); i = 0
    while i + h <= n:
        if ids[i:i + h] == header:
            s = i + h
            if think and s < n and ids[s] == think[0]:        # '<think>' special token, then its newline if rendered as one token
                s += 1
                if len(think) > 1 and s < n and ids[s] == think[1]:
                    s += 1
            j = s
            while j < n and ids[j] != end_id:
                j += 1
            if j >= n:
                break
            if j > s:
                spans.append([s, j])
            i = j + 1
        else:
            i += 1
    return spans


def history_boundary(ids: list[int], header: list[int], k: int) -> int:
    """Token index where the (k+1)-th '<|im_start|>assistant\\n' header starts in a rendered prompt:
    the end of a recorded history holding k assistant messages. Every assistant message renders a
    header (an empty body still gets one; the trailing generation prompt is a header too), so the
    count is exact whatever the template dropped from the bodies (reasoning of earlier turns, which
    changes when a later user message appears). Raises if the prompt has fewer than k+1 headers."""
    h, seen = len(header), 0
    for i in range(len(ids) - h + 1):
        if ids[i:i + h] == header:
            if seen == k:
                return i
            seen += 1
    raise ValueError(f"prompt renders {seen} assistant headers, need {k + 1} (history has {k} assistant messages)")


def build_clamp_payload(artifact: str, threshold: float, mode: str = "clamp", scale: float = 1.0) -> str:
    """Back-compatible wrapper: clamp payload (see build_steer_payload)."""
    return build_steer_payload(artifact, mode=mode, threshold=threshold, scale=scale)


def artifact_sha(artifact: str) -> str:
    """sha256 prefix of a steering artifact's tensor file (provenance)."""
    import hashlib

    return hashlib.sha256(open(artifact, "rb").read()).hexdigest()[:16]


def proxy_view(messages: list[dict]) -> list[dict]:
    """The chat request as the arm proxy forwards it: think tags stripped from prior
    assistant reasoning (steering_reasoning.proxy). The proxy's other stamps — max_tokens,
    the steering payload, steer_positions — do not change the rendered prompt. Pure; input
    untouched."""
    import copy

    out = copy.deepcopy(messages)
    for m in out:
        if m.get("role") == "assistant":
            for k in ("reasoning", "reasoning_content"):
                if m.get(k):
                    m[k] = strip_think(m[k])
    return out


def request_sha(messages: list[dict], tools: list[dict] | None) -> str:
    """Canonical hash of a forwarded chat request (post-proxy messages + tools): the join
    key between the proxy's ids sidecar and a record rebuilt from the eval log. Both sides
    hash the same JSON (sorted keys, no whitespace), so the key is renderer-independent."""
    import hashlib

    blob = json.dumps({"messages": messages, "tools": tools or []}, sort_keys=True,
                      separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def ids_sha(ids: list[int]) -> str:
    """sha256 prefix of a token-id list (prompt-render byte-identity check)."""
    import hashlib

    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()[:24]


def strip_think(s):
    """Remove literal <think>/</think> tags the reasoning parser leaves in the reasoning field.

    The Qwen chat template wraps replayed reasoning in its own <think> block, so leaving
    them in produces nested tags on every prior turn (fidelity fix, 2026-08-27).
    """
    if not isinstance(s, str):
        return s
    s2 = s.strip()
    if s2.startswith("<think>"):
        s2 = s2[len("<think>"):]
    if s2.rstrip().endswith("</think>"):
        s2 = s2.rstrip()[: -len("</think>")]
    return s2.strip("\n")
