#!/usr/bin/env python
"""Label which sentences of the spec-arm rollouts CITE the integrity principles (plan.md Step 1).

One judge call (the model configured in .env through steering_reasoning.judge; the paper used GLM-5 at
temperature 0, 200 output tokens) per CANDIDATE sentence, with the system prompt of
../judge_prompt.md and ~500 tokens of same-turn, same-source context on each side.

Sentences: each assistant message's reasoning (`reasoning_content`, source "reasoning") and visible
text (`content`, source "visible") is split on lines, then on sentence ends (the MisTrace tagger's
rule, reimplemented here — the tagger itself is not in the repo), then regrouped with
steering_reasoning.steering.segments.regroup_segments (fences and enumerations), the same repair
the u segments got. Every unit keeps its character span in the ORIGINAL field, so the fit maps
labels to tokens with segments.char_to_token_spans. Candidates = units matching a principle ID
(\\b(ARH[1-5]|GP[1-4])\\b) or one of the words principle(s) / integrity / spec / faithful
(match_kind "id" | "word"). The base arm is labelled with the same net: it never saw the spec, so a
CITES label there is a judge false positive.

    uv run python experiments/spec-vector/scripts/judge_cites.py --prompt v1 --smoke 50     # spec arm, 50 candidates, prints them
    uv run python experiments/spec-vector/scripts/judge_cites.py --prompt v2 --arms spec base   # full pass, resumable
Two prompts (judge_prompt.md): v1 = two-way CITES/NOT (the original), v2 = three-way
CITES/RESTATES/NOT (inclusive). Both are run over everything; labels are chosen at fit time.
Output: ../artifacts/cite_labels_<prompt>.json (or --out), one row per candidate:
  rollout_id, arm, task_id, seed, message_index, source, unit_idx, char_start, char_end, text,
  match_kind, ids_in_text, label (CITES | RESTATES | NOT | EXCLUDED), principle_ids, confidence, reason.
"""
import argparse, json, random, re, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = EXP.parents[1]
sys.path.insert(0, str(REPO / "src"))
from steering_reasoning.steering.segments import regroup_segments  # noqa: E402
from steering_reasoning import judge  # noqa: E402

RECORDS = REPO / "experiments/spec-prompting-basetasks/data/records"
ID_RE = re.compile(r"\b(ARH[1-5]|GP[1-4])\b")
WORD_RE = re.compile(r"\b(principles?|integrity|spec|faithful(?:ly|ness)?)\b", re.I)
SENT_END = re.compile(r"(?<=[.!?])\s+(?=\S)")
CONTEXT_CHARS = 2000          # ≈ 500 tokens per side (the anti-hack judge used 1000 tokens total)
MAX_TARGET_CHARS = 4000
MAX_WORKERS_DEFAULT = 8


def system_prompt(version: str) -> str:
    """The fenced block under '## System prompt v<version>' in judge_prompt.md."""
    md = (EXP / "judge_prompt.md").read_text()
    m = re.search(r"## System prompt " + re.escape(version) + r"\b[^\n]*\n\s*```\n(.*?)\n```", md, re.S)
    if not m:
        raise SystemExit(f"judge_prompt.md: no fenced system prompt {version}")
    return m.group(1)


def split_units(text: str) -> list[dict]:
    """Line-then-sentence split with char offsets, then regroup. Units keep original offsets."""
    segs = []
    pos = 0
    for line in text.split("\n"):
        s = 0
        for m in SENT_END.finditer(line):
            e = m.start()
            if line[s:e].strip():
                segs.append({"text": line[s:e], "start": pos + s, "end": pos + e})
            s = m.end()
        if line[s:].strip():
            segs.append({"text": line[s:], "start": pos + s, "end": pos + len(line)})
        pos += len(line) + 1
    units = []
    for group in regroup_segments(segs):
        a, b = segs[group[0]]["start"], segs[group[-1]]["end"]
        units.append({"start": a, "end": b, "text": text[a:b]})
    return units


def candidates(records: list[dict], sources: list[str]) -> list[dict]:
    out = []
    for r in records:
        for mi, m in enumerate(r["messages"]):
            if m.get("role") != "assistant":
                continue
            fields = []
            if "reasoning" in sources:
                fields.append(("reasoning", m.get("reasoning_content") or ""))
            if "visible" in sources:
                c = m.get("content")
                fields.append(("visible", c if isinstance(c, str) else ""))
            for source, text in fields:
                if not text.strip():
                    continue
                units = split_units(text)
                for ui, u in enumerate(units):
                    ids = ID_RE.findall(u["text"])
                    if ids:
                        kind = "id"
                    elif WORD_RE.search(u["text"]):
                        kind = "word"
                    else:
                        continue
                    pre = "\n".join(x["text"] for x in units[:ui])[-CONTEXT_CHARS:]
                    post = "\n".join(x["text"] for x in units[ui + 1:])[:CONTEXT_CHARS]
                    out.append({
                        "rollout_id": r["rollout_id"], "arm": r["arm"], "task_id": r["task_id"],
                        "seed": r["seed"], "message_index": mi, "source": source, "unit_idx": ui,
                        "char_start": u["start"], "char_end": u["end"], "text": u["text"],
                        "match_kind": kind, "ids_in_text": sorted(set(ids)),
                        "_pre": pre, "_post": post,
                    })
    return out


def key(c: dict) -> str:
    return f"{c['rollout_id']}|{c['message_index']}|{c['source']}|{c['unit_idx']}"


def user_message(c: dict) -> str:
    src = "private reasoning" if c["source"] == "reasoning" else "visible reply"
    tgt = c["text"][:MAX_TARGET_CHARS]
    return (f"SOURCE: {src}\n\nCONTEXT BEFORE:\n{c['_pre']}\n\n>>> {tgt} <<<\n\n"
            f"CONTEXT AFTER:\n{c['_post']}")


LABELS = {"v1": ("CITES", "NOT"), "v2": ("CITES", "RESTATES", "NOT")}


def _first_object(text: str) -> dict | None:
    """First JSON object in the reply that carries a label (fences, trailing prose tolerated)."""
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            o, _ = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            continue
        if isinstance(o, dict) and "label" in o:
            return o
    return None


def parse(text: str, allowed: tuple) -> dict | None:
    try:
        o = _first_object(text or "")
    except Exception:
        return None
    if o is None:
        return None
    lab = str(o.get("label", "")).upper()
    if lab not in allowed:
        return None
    ids = o.get("principle_ids")
    ids = ids if isinstance(ids, list) else ([ids] if isinstance(ids, str) else [])
    ids = [str(x).upper() for x in ids if ID_RE.fullmatch(str(x).upper())]
    try:
        conf = int(o.get("confidence"))
    except (TypeError, ValueError):
        conf = None
    return {"label": lab, "principle_ids": sorted(set(ids)), "confidence": conf,
            "reason": str(o.get("reason", ""))[:300]}


def judge_one(J, model: str, system: str, c: dict, allowed: tuple) -> dict:
    """Two parse attempts (the second at temperature 0.3), each with transport retries."""
    raw = None
    for attempt, temp in enumerate((0.0, 0.3)):
        for tries in range(6):
            try:
                raw = judge.complete(system, user_message(c), model=model, max_tokens=200, temperature=temp, cfg=J).text
                break
            except judge.JudgeHTTPError as e:  # throttling, transient 5xx
                if e.fatal: raise
                if tries == 5:
                    return {**strip(c), "label": "EXCLUDED", "principle_ids": [], "confidence": None,
                            "reason": f"transport: {type(e).__name__}: {str(e)[:120]}", "raw": None}
                time.sleep(min(2 ** tries, 16) + random.random())
        p = parse(raw, allowed)
        if p:
            return {**strip(c), **p, "raw": raw}
    return {**strip(c), "label": "EXCLUDED", "principle_ids": [], "confidence": None,
            "reason": "parse failure", "raw": raw}


def strip(c: dict) -> dict:
    return {k: v for k, v in c.items() if not k.startswith("_")}


def load_records(arms: list[str], seeds: list[int]) -> list[dict]:
    recs = []
    for arm in arms:
        for seed in seeds:
            p = RECORDS / f"{arm}_conflicting_s{seed}.jsonl"
            with open(p) as f:
                for line in f:
                    recs.append(json.loads(line))
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["spec"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    ap.add_argument("--sources", nargs="+", default=["reasoning", "visible"])
    ap.add_argument("--prompt", default="v1", choices=["v1", "v2"], help="v1 = two-way CITES/NOT; v2 = three-way CITES/RESTATES/NOT")
    ap.add_argument("--out", default=None, help="default artifacts/cite_labels_<prompt>.json")
    ap.add_argument("--smoke", type=int, default=0, help="label N sampled candidates (half id, half word) into artifacts/smoke_labels.json and print them")
    ap.add_argument("--model", default=None, help="default: JUDGE_MODEL from .env (the paper used zai.glm-5)")
    ap.add_argument("--workers", type=int, default=MAX_WORKERS_DEFAULT)
    ap.add_argument("--dry-run", action="store_true", help="count candidates only")
    ap.add_argument("--sample-seed", type=int, default=0)
    A = ap.parse_args()

    recs = load_records(A.arms, A.seeds)
    cands = candidates(recs, A.sources)
    cnt = Counter((c["arm"], c["source"], c["match_kind"]) for c in cands)
    print(f"{len(recs)} rollouts, {len(cands)} candidates:")
    for k in sorted(cnt):
        print(f"  {k[0]:5s} {k[1]:9s} {k[2]:4s} {cnt[k]:6d}")
    if A.dry_run:
        return

    out_path = Path(A.out) if A.out else EXP / f"artifacts/cite_labels_{A.prompt}.json"
    if A.smoke:
        rng = random.Random(A.sample_seed)
        ids_ = [c for c in cands if c["match_kind"] == "id"]
        words = [c for c in cands if c["match_kind"] == "word"]
        k_id = min(A.smoke // 2, len(ids_))
        pick = rng.sample(ids_, k_id) + rng.sample(words, min(A.smoke - k_id, len(words)))
        cands, out_path = pick, EXP / f"artifacts/smoke_labels_{A.prompt}.json"

    done = {}
    if out_path.exists() and not A.smoke:
        cur = {key(c): c for c in cands}
        n_old = n_retry = n_stale = 0
        for o in json.loads(out_path.read_text()):
            n_old += 1
            k = key(o); c = cur.get(k)
            if c is None or (c["text"], c["char_start"], c["char_end"]) != (o["text"], o["char_start"], o["char_end"]):
                n_stale += 1; continue          # splitter changed: the key points at another sentence now
            if o["label"] == "EXCLUDED":
                n_retry += 1; continue          # transport or parse failure: try again
            done[k] = o
        print(f"resuming: {n_old} rows on disk, {len(done)} kept, {n_retry} EXCLUDED to retry, {n_stale} stale dropped")
    todo = [c for c in cands if key(c) not in done]
    print(f"{len(todo)} to judge with {A.workers} workers -> {out_path}")

    J = judge.settings(A.model); A.model = J.model
    system = system_prompt(A.prompt)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def flush():
        rows = list(done.values())
        tmp = out_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rows, indent=0, ensure_ascii=False))
        tmp.replace(out_path)

    t0, n = time.time(), 0
    allowed = LABELS[A.prompt]
    ex = ThreadPoolExecutor(max_workers=A.workers)
    try:
        futs = {ex.submit(judge_one, J, A.model, system, c, allowed): c for c in todo}
        for fut in as_completed(futs):
            o = fut.result()
            done[key(o)] = o
            n += 1
            if n % 100 == 0 or n == len(todo):
                flush()
                lab = Counter(x["label"] for x in done.values())
                print(f"  {n}/{len(todo)} {time.time() - t0:.0f}s  {dict(lab)}", flush=True)
    except BaseException:
        ex.shutdown(wait=False, cancel_futures=True)
        flush()
        raise
    ex.shutdown(wait=True)
    flush()

    rows = list(done.values())
    lab = Counter((o["arm"], o["source"], o["match_kind"], o["label"]) for o in rows)
    print("labels by arm / source / match_kind:")
    for k in sorted(lab):
        print(f"  {k[0]:5s} {k[1]:9s} {k[2]:4s} {k[3]:8s} {lab[k]:6d}")
    if A.smoke:
        for o in rows:
            print(f"\n[{o['label']} conf={o['confidence']} ids={o['principle_ids']} | {o['source']} {o['match_kind']} {o['rollout_id']} msg{o['message_index']}]")
            print("  " + o["text"][:400].replace("\n", "\n  "))
            print(f"  -> {o['reason']}")


if __name__ == "__main__":
    main()
