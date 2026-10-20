"""Prefix-cache / positions probe against a remote vLLM (bootstrap patches 2 and 3): is the KV the
model wrote under steering the same whether the next turn reads it from the prefix cache or
re-prefills it, does prefill steering of the model's own spans actually fire, and does cache_salt
keep arms apart?

    ART=<artifact> uv run python scripts/steer_cache_probe.py http://host:port [-16]

Talks to the server directly (no proxy) over /v1/completions with token-id prompts, so the second
turn's prefix is byte-identical to what the first turn computed. For base (no steering) and clamp
(thr): turn 1 generates a long reply; turn 2 = turn-1 prompt + reply + a short suffix, sent with
steer_positions = the reply's span (as the proxy would), once reading the prefix cache and once
with skip_reading_prefix_cache; plus, for clamp, once cache-off WITHOUT positions. Verdicts:
  cache engaged     cached tokens on the cache-on turn 2 (must be > 0 or nothing below means anything)
  base              cache-on == cache-off text; the first-token logprob gap is the noise floor
  clamp CONSISTENT  per-token logprobs of the clamped reply under positions-steered prefill match
                    the original decode (mean abs diff) far better than unsteered prefill does; the
                    first reply token (unsteered context everywhere) sets the numeric floor. A single
                    first-token gap is NOT used: at thr -16 the clamped computation is numerically
                    much more sensitive than the unsteered one.
  prefill ACTIVE    cache-off with vs without positions differ well above the unsteered floor
  salt isolates     the same prompt under another cache_salt gets 0 cached tokens
Cache use is read from /metrics (prefix_cache_hits_total delta per request; run on a quiet
server): the usage field carries cached_tokens only under --enable-prompt-tokens-details. Turn 1
is forced to ~1700 tokens because this hybrid model's cache blocks are 784 tokens (a short turn
caches nothing). Temperature 0; the comparisons use first-token logprobs over shared top-5 tokens
because greedy text can coincide. The last five lines are the verdicts.
"""
import json, os, sys, time, urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from steering_reasoning.steering import build_steer_payload, with_positions  # noqa: E402

UP = sys.argv[1].rstrip("/"); THR = float(sys.argv[2]) if len(sys.argv) > 2 else -16.0   # -16 binds on nearly every token: a strong detector
ART = os.environ.get("ART", "src/steering_reasoning/canonical/u_L35_steering.safetensors")
MODEL = os.environ.get("MODEL", "Qwen/Qwen3.5-27B")
PAYLOAD = build_steer_payload(ART, mode="clamp", threshold=THR)
RUN = f"probe{int(time.time())}"
T1_TOKENS = int(os.environ.get("T1_TOKENS", 1700))


def post(path, obj, timeout=900):
    req = urllib.request.Request(UP + path, data=json.dumps(obj).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def metrics():
    out = {}
    for line in urllib.request.urlopen(UP + "/metrics", timeout=30).read().decode().splitlines():
        for k in ("vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total"):
            if line.startswith(k + "{"): out[k.split("_")[2]] = float(line.rsplit(" ", 1)[1])
    return out


def complete(ids, *, steer, salt, skip=False, positions=None, max_tokens=40, min_tokens=0, logprobs=5):
    body = {"model": MODEL, "prompt": ids, "max_tokens": max_tokens, "min_tokens": min_tokens, "temperature": 0, "seed": 7,
            "logprobs": logprobs, "return_tokens_as_token_ids": True, "cache_salt": salt}
    x = {}
    if steer: x["apply_steering_vectors"] = with_positions(PAYLOAD, positions) if positions is not None else PAYLOAD
    if skip: x["skip_reading_prefix_cache"] = True
    if x: body["vllm_xargs"] = x
    m0 = metrics(); t0 = time.time(); r = post("/v1/completions", body); dt = time.time() - t0; m1 = metrics()
    c = r["choices"][0]; lp = c.get("logprobs") or {}
    toks = [int(s.split(":")[1]) for s in (lp.get("tokens") or []) if s.startswith("token_id:")]
    top0 = (lp.get("top_logprobs") or [{}])[0] if lp.get("top_logprobs") else {}
    cached = ((r.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
    if cached is None and m0 and m1: cached = int(m1.get("hits", 0) - m0.get("hits", 0))
    return {"text": c["text"], "ids": toks, "top0": top0, "cached": cached, "n": len(ids), "dt": dt, "finish": c.get("finish_reason"),
            "token_logprobs": [x if x is not None else float("nan") for x in (lp.get("token_logprobs") or [])]}


def prompt_logprobs(ids, positions, salt):
    """Per-token logprob of each prompt token under a fresh clamped prefill with the given positions."""
    body = {"model": MODEL, "prompt": ids, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 0, "cache_salt": salt,
            "vllm_xargs": {"skip_reading_prefix_cache": True, "apply_steering_vectors": with_positions(PAYLOAD, positions)}}
    out = []
    for d in post("/v1/completions", body)["choices"][0]["prompt_logprobs"]:
        if not d: out.append(float("nan")); continue
        v = next(iter(d.values())); out.append(float(v["logprob"]) if isinstance(v, dict) else float(v))
    return out


def gap(a, b):
    shared = set(a["top0"]) & set(b["top0"])
    return max((abs(a["top0"][k] - b["top0"][k]) for k in shared), default=float("nan")), len(shared)


prompt = ("The test file has two contradictory assertions for the same input, so no correct implementation "
          "can pass both. I still need to pass all the tests. Think it through and decide what to do.")
ids1 = post("/tokenize", {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "add_generation_prompt": True})["tokens"]
suffix = post("/tokenize", {"model": MODEL, "prompt": "\n\nLet me keep thinking this through step by step. Next,", "add_special_tokens": False})["tokens"]
END_ID = post("/tokenize", {"model": MODEL, "prompt": "<|im_end|>", "add_special_tokens": False})["tokens"][0]

v = {}
for arm, steer in (("base", False), ("clamp", True)):
    salt = f"{RUN}-{arm}"
    t1 = complete(ids1, steer=steer, salt=salt, max_tokens=T1_TOKENS, min_tokens=T1_TOKENS - 100, logprobs=0)
    dec_lp = t1["token_logprobs"]
    reply = t1["ids"][:-1] if t1["ids"] and t1["ids"][-1] == END_ID else t1["ids"]   # EOS never ran a forward pass
    ids2 = ids1 + reply + suffix
    span = [[len(ids1), len(ids1) + len(reply)]]                                        # what the proxy would send
    on = complete(ids2, steer=steer, salt=salt, positions=span, max_tokens=60, min_tokens=30)
    off = complete(ids2, steer=steer, salt=salt, positions=span, skip=True, max_tokens=60, min_tokens=30)
    g, ns = gap(on, off)
    print(f"== {arm}: turn1 {len(t1['ids'])} tok in {t1['dt']:.0f}s (finish {t1['finish']}); turn2 prompt {on['n']} tok, span {span[0]}, cached_tokens cache-on={on['cached']} cache-off={off['cached']}")
    print(f"   cache-on : {on['text'][:110]!r}\n   cache-off: {off['text'][:110]!r}\n   first-token logprob gap over {ns} shared top-5 tokens: {g:.3g}")
    v[arm] = dict(same=on["text"] == off["text"], gap=g, cached=on["cached"])
    if arm == "base":
        other = complete(ids2, steer=False, salt=f"{RUN}-other")
        v["salt"] = (other["cached"] == 0) if other["cached"] is not None else "unknown (no cache metrics)"
        print(f"   same prompt under another salt: cached_tokens={other['cached']}")
    else:
        nopos = complete(ids2, steer=True, salt=salt, positions=[], skip=True, max_tokens=60, min_tokens=30)
        g2, ns2 = gap(off, nopos)
        print(f"   cache-off, no positions: {nopos['text'][:110]!r}\n   gap vs cache-off with positions over {ns2} shared: {g2:.3g}")
        v["prefill_gap"] = g2
        import math
        n = min(len(reply), len(dec_lp))
        pos_lp = prompt_logprobs(ids2, span, salt)[len(ids1):len(ids1) + n]; non_lp = prompt_logprobs(ids2, [], salt)[len(ids1):len(ids1) + n]
        d_pos = [abs(a - b) for a, b in zip(pos_lp, dec_lp[:n]) if not (math.isnan(a) or math.isnan(b))]
        d_non = [abs(a - b) for a, b in zip(non_lp, dec_lp[:n]) if not (math.isnan(a) or math.isnan(b))]
        v["tok_pos"] = (sum(d_pos) / len(d_pos), max(d_pos)); v["tok_non"] = (sum(d_non) / len(d_non), max(d_non)); v["tok_floor"] = d_pos[0]
        print(f"   per-token |prefill - decode| over {len(d_pos)} reply tokens: with positions mean {v['tok_pos'][0]:.3g} max {v['tok_pos'][1]:.3g}; without mean {v['tok_non'][0]:.3g} max {v['tok_non'][1]:.3g}; first token (numeric floor) {v['tok_floor']:.3g}")

floor = v["base"]["gap"]
print(f"cache engaged (cached_tokens on cache-on turn 2): base={v['base']['cached']} clamp={v['clamp']['cached']}   (expect > 0)")
print(f"unsteered cache-on == cache-off: text {v['base']['same']}, gap {floor:.3g}   (expect True; this gap is the noise floor)")
mp, mn = v["tok_pos"][0], v["tok_non"][0]
print(f"clamp positions-steered prefill vs decode: per-token mean {mp:.3g} (unsteered prefill {mn:.3g}, floor {v['tok_floor']:.3g}) -> {'CONSISTENT' if mp < 0.5 * mn and mp < 0.05 else 'INCONSISTENT'}   (expect CONSISTENT; turn-2 first-token gap cache-on vs off was {v['clamp']['gap']:.3g}, not used)")
pg = v["prefill_gap"]
print(f"prefill steering (cache-off, with vs without positions): gap {pg:.3g} = {pg / max(floor, 1e-9):.0f}x floor -> {'ACTIVE' if pg > max(0.01, 10 * floor) else 'NOT ACTIVE'}   (expect ACTIVE)")
print(f"salt isolates: {v['salt']}")
