#!/usr/bin/env python
"""Stream the spec-basetasks L35 dump once and write everything the vector fit needs (plan Step 3).

Runs where S3 is fast (a us-east-1 CPU box); the vectors themselves are assembled locally by
fit_spec_vectors.py from the small outputs of this script, so pools can be recombined without
touching the 140 GB of shards again.

Per SPEC rollout (both seeds), from L35/<task>.safetensors + tokens/ + meta/:
  * every judge-labelled candidate sentence (cite_labels_v1/v2.json, joined on key) is mapped onto
    its generated rows: kind-5 rows of that message for source "reasoning", kind-6 rows for
    "visible". Token pieces are recovered byte-exactly from the row token ids (byte-level BPE), the
    reconstruction is checked against the record's text (the chat template strips newlines at the
    ends), and char spans become row ranges. The principle-ID tokens inside a sentence (ARH / GP,
    the digit, a directly following — or :) are DROPPED from the pooled rows (plan: v_cite must not
    be a lexical detector for "ARH").
  * rows are accumulated into COMBOS = (source, match_kind, v1_label, v2_label): a f64 sum and a
    count per combo per rollout, so any pool (v1 CITES, v2 CITES + RESTATES, ...) is a sum of combos.
  * "rest" pools: all kind-5 rows (reasoning_all) and all kind-6 rows (visible_all), never-computed
    and <|im_end|> rows excluded, and the same minus every candidate row (reasoning_noncand).
  * the system prompt of turn 0 (kind 1): mean over the MisTrace part (before the spec suffix) and
    over the spec suffix (all of it; the principles list; the FINAL INSTRUCTIONS paragraph). The
    MisTrace part precedes the spec, so under a causal model these rows ARE the base arm's system
    rows (same tokens, same prefix) — v_text needs no base shards.
  * a random row subsample per combo / rest pool (fp16, lossless for bf16 values) for projection
    histograms after the vectors exist.
Per BASE rollout: reasoning turn means (segmeans/, token-weighted at L35) for the spec-arm-vs-base
contrast; for --base-sample rollouts per cell the L35 shard is read too, for a reasoning-row
subsample and the MisTrace system mean (a check against the spec arm's MisTrace part).

    python extract_pool_stats.py --s3-root s3://.../spec-basetasks_bf16_L31_35 --cells spec/conflicting/s42 ... --out out/
    python extract_pool_stats.py --dump-root data/captures/spec-vector-test --cells spec/conflicting/s42 --select lcbhard_72 --out /tmp/x
Outputs per cell: stats_<arm>_s<seed>.npz (sums / counts / means, rollout order in `rollouts`),
rows_<arm>_s<seed>.npz (fp16 subsample + tags), and a log of reconstruction mismatches.
"""
import argparse, json, os, random, re, shutil, subprocess, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3] / "src"))
from steering_reasoning.kinds import KIND  # noqa: E402
from steering_reasoning.steering.token_dump import tensors_from_bytes  # noqa: E402

MODEL = "Qwen/Qwen3.5-27B"
SPEC_START = "Complete the task above in the best possible way."
SPEC_PRINCIPLES = "Integrity Principles for Software Tasks"
SPEC_FINAL = "FINAL INSTRUCTIONS"
V1_LABELS = ("CITES", "NOT", "EXCLUDED")
V2_LABELS = ("CITES", "RESTATES", "NOT", "EXCLUDED")


def combos():
    out = []
    for src in ("reasoning", "visible"):
        for mk in ("id", "word"):
            for l1 in V1_LABELS:
                for l2 in V2_LABELS:
                    out.append(f"{src}|{mk}|{l1}|{l2}")
    return out


COMBOS = combos()
REST = ["reasoning_all", "reasoning_noncand", "visible_all", "visible_noncand",
        "sys_mistrace", "sys_spec", "sys_spec_principles", "sys_spec_final"]
SUB_N = {"combo": 400, "reasoning_noncand": 2500, "visible_noncand": 500, "base_reasoning": 3000}


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2 / Qwen byte-level BPE alphabet (the function transformers used to export)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b); cs.append(2 ** 8 + n); n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


class ByteTok:
    """Exact bytes of each token id (byte-level BPE alphabet → bytes)."""

    def __init__(self, tok):
        self.tok = tok
        self.inv = {v: k for k, v in _bytes_to_unicode().items()}
        self.cache = {}

    def piece(self, tid: int) -> bytes:
        b = self.cache.get(tid)
        if b is None:
            t = self.tok.convert_ids_to_tokens(int(tid))
            try:
                b = bytes(self.inv[c] for c in t)
            except KeyError:              # added / special token
                b = self.tok.decode([int(tid)]).encode("utf-8")
            self.cache[tid] = b
        return b


def token_byte_bounds(bt: ByteTok, ids) -> tuple[bytes, np.ndarray]:
    pieces = [bt.piece(i) for i in ids]
    ends = np.cumsum([len(p) for p in pieces])
    return b"".join(pieces), ends


def align(raw: str, recon: bytes):
    """Offset (bytes into recon) where raw.strip('\\n') starts, and the char lead stripped from raw."""
    core = raw.strip("\n")
    lead = len(raw) - len(raw.lstrip("\n"))
    k = recon.find(core.encode("utf-8"))
    return (k, lead) if k >= 0 else (None, lead)


def rows_for_span(raw: str, a: int, b: int, k: int, lead: int, ends: np.ndarray):
    """Row indices (into the message's row list) whose bytes overlap chars [a, b) of raw."""
    core_a, core_b = max(a - lead, 0), max(b - lead, 0)
    ba = k + len(raw.lstrip("\n")[:core_a].encode("utf-8"))
    bb = k + len(raw.lstrip("\n")[:core_b].encode("utf-8"))
    starts = np.concatenate([[0], ends[:-1]])
    idx = np.nonzero((ends > ba) & (starts < bb))[0]
    return idx


ID_SPAN = re.compile(rb"(\*\*)?\b(ARH|GP)[1-5]\b(\*\*)?( ?(?:\xe2\x80\x94|:))?")   # ARH3, **GP4**:, ARH1 — (bytes)


def drop_id_tokens(bt: ByteTok, ids, idx) -> tuple[list[int], list[int]]:
    """Split a span's rows into (kept, dropped): dropped = tokens overlapping a principle ID.
    "ARH3" tokenizes as ' AR' 'H' '3', so the match runs on the span's joined bytes, not per token."""
    pieces = [bt.piece(int(ids[i])) for i in idx]
    ends = np.cumsum([len(p) for p in pieces]); starts = np.concatenate([[0], ends[:-1]])
    drop = np.zeros(len(idx), bool)
    for m in ID_SPAN.finditer(b"".join(pieces)):
        drop |= (ends > m.start()) & (starts < m.end())
    return [int(i) for i, dr in zip(idx, drop) if not dr], [int(i) for i, dr in zip(idx, drop) if dr]


def load_shard(path: Path):
    return tensors_from_bytes(bytearray(path.read_bytes()))


class Acc:
    def __init__(self, d: int):
        self.sum = defaultdict(lambda: np.zeros(d, np.float64))
        self.n = defaultdict(int)
        self.sub = defaultdict(list)     # name -> list of fp16 rows
        self.rng = random.Random(0)

    def add(self, name: str, X: torch.Tensor, sub_n: int):
        if X.shape[0] == 0:
            return
        Xf = X.float()
        self.sum[name] += Xf.sum(0).double().numpy()
        self.n[name] += int(X.shape[0])
        if sub_n:
            take = X.shape[0] if X.shape[0] <= sub_n else sub_n
            sel = torch.tensor(sorted(self.rng.sample(range(X.shape[0]), take))) if take < X.shape[0] else None
            self.sub[name].append((Xf[sel] if sel is not None else Xf).half().numpy())


def process_spec_rollout(rid: str, rec: dict, shard_dir: Path, task: str, labels: dict, bt: ByteTok, d: int, log):
    t = load_shard(shard_dir / "L35" / f"{task}.safetensors")
    tk = load_shard(shard_dir / "tokens" / f"{task}.safetensors")
    meta = json.load(open(shard_dir / "meta" / f"{task}.json"))
    X, kind, msg, turn = t["token_acts"], t["token_kind"].numpy(), t["token_msg"].numpy(), t["token_turn"].numpy()
    ids = tk["token_ids"].numpy()
    bad = np.zeros(len(kind), bool)
    bad[np.array(meta.get("never_computed_rows", []), int)] = True
    bad[np.array(meta.get("im_end_rows", []), int)] = True
    acc = Acc(d)
    cand_rows = np.zeros(len(kind), bool)      # every row of every candidate span (kept, dropped, or unmapped)
    assigned = np.zeros(len(kind), bool)       # rows already pooled (a token straddling two sentences counts once)
    combo_rows = defaultdict(list); idtok_rows = {"reasoning": [], "visible": []}
    stats = {"mismatch": 0, "labels": 0, "labels_mapped": 0, "id_tokens_dropped": 0, "empty_after_drop": 0, "rows_shared": 0}

    by_msg = defaultdict(list)
    for o in labels.get(rid, []):
        by_msg[(o["message_index"], o["source"])].append(o)
    for (mi, source), rows_ in by_msg.items():
        k_kind = KIND["reasoning"] if source == "reasoning" else KIND["response_gen"]
        m = rec["messages"][mi]
        raw = (m.get("reasoning_content") or "") if source == "reasoning" else (m.get("content") if isinstance(m.get("content"), str) else "")
        ridx = np.nonzero((kind == k_kind) & (msg == mi))[0]
        if len(ridx) == 0 or not raw.strip("\n"):
            stats["mismatch"] += len(rows_); log(f"{rid} msg{mi} {source}: no rows or empty text"); continue
        recon, ends = token_byte_bounds(bt, ids[ridx])
        k, lead = align(raw, recon)
        if k is None:
            stats["mismatch"] += len(rows_); log(f"{rid} msg{mi} {source}: text not found in rows"); continue
        for o in rows_:
            stats["labels"] += 1
            idx = rows_for_span(raw, o["char_start"], o["char_end"], k, lead, ends)
            cand_rows[ridx[idx]] = True
            keep, dropped = drop_id_tokens(bt, ids[ridx], idx)
            stats["id_tokens_dropped"] += len(dropped)
            idtok_rows[source].extend(int(r) for r in ridx[dropped])
            if not keep:
                stats["empty_after_drop"] += 1; continue
            r = ridx[keep]; r = r[~bad[r]]
            shared = assigned[r]; stats["rows_shared"] += int(shared.sum()); r = r[~shared]
            if len(r) == 0:
                continue
            assigned[r] = True
            stats["labels_mapped"] += 1
            combo_rows[f"{source}|{o['match_kind']}|{o['v1']}|{o['v2']}"].extend(int(x) for x in r)
    for name, rr in combo_rows.items():
        acc.add(name, X[np.array(sorted(set(rr)))], SUB_N["combo"])
    for source, rr in idtok_rows.items():
        if rr:
            r = np.array(sorted(set(rr))); r = r[~bad[r]]
            acc.add(f"{source}_idtok", X[r], 0)
    reasoning = (kind == KIND["reasoning"]) & ~bad
    visible = (kind == KIND["response_gen"]) & ~bad
    acc.add("reasoning_all", X[reasoning], 0)
    acc.add("reasoning_noncand", X[reasoning & ~cand_rows], SUB_N["reasoning_noncand"])
    acc.add("visible_all", X[visible], 0)
    acc.add("visible_noncand", X[visible & ~cand_rows], SUB_N["visible_noncand"])
    # system prompt, turn 0 (kind 1): <|im_start|>system\n# Tools ... </tools> MisTrace prompt [spec suffix] \n
    sidx = np.nonzero((kind == KIND["system"]) & (turn == 0))[0]
    recon, ends = token_byte_bounds(bt, ids[sidx])
    def first_row_at(marker: str, after_end: bool = False):
        p = recon.find(marker.encode())
        if p < 0:
            return None
        if after_end:
            p += len(marker)
            return int(np.nonzero(ends > p)[0][0]) if (ends > p).any() else len(sidx)
        return int(np.nonzero(ends > p)[0][0])
    # trailing whitespace-only rows (the "\n" after <|im_end|>) are markup, not spec text
    s_end = len(sidx)
    while s_end > 0 and not bt.piece(int(ids[sidx[s_end - 1]])).strip():
        s_end -= 1
    s_tools = first_row_at("</tools>", after_end=True)
    s0, s1, s2 = first_row_at(SPEC_START), first_row_at(SPEC_PRINCIPLES), first_row_at(SPEC_FINAL)
    if s0 is None or s_tools is None:
        log(f"{rid}: spec start or </tools> not found in system rows"); stats["mismatch"] += 1
    else:
        acc.add("sys_tools", X[sidx[:s_tools]], 0)           # <|im_start|>system, the tool schema JSON (row 0 = attention sink)
        acc.add("sys_mistrace", X[sidx[s_tools:s0]], 0)      # the MisTrace task prompt proper
        acc.add("sys_spec", X[sidx[s0:s_end]], 0)
        if s1 is not None and s2 is not None:
            acc.add("sys_spec_principles", X[sidx[s1:s2]], 0)
            acc.add("sys_spec_final", X[sidx[s2:s_end]], 0)
    return acc, stats


def process_base_rollout(rid: str, shard_dir: Path, task: str, bt: ByteTok, d: int, with_shard: bool, log):
    acc = Acc(d)
    sm = load_shard(shard_dir / "segmeans" / f"{task}.safetensors")
    tm, tt = sm["turn_means"].float().numpy(), sm["turn_tokens"].numpy().astype(np.float64)
    li = 1  # layer order [31, 35]
    acc.sum["reasoning_all"] += (tm[:, li, :] * tt[:, None]).sum(0)
    acc.n["reasoning_all"] += int(tt.sum())
    if with_shard:
        t = load_shard(shard_dir / "L35" / f"{task}.safetensors")
        tk = load_shard(shard_dir / "tokens" / f"{task}.safetensors")
        meta = json.load(open(shard_dir / "meta" / f"{task}.json"))
        X, kind, turn = t["token_acts"], t["token_kind"].numpy(), t["token_turn"].numpy()
        ids = tk["token_ids"].numpy()
        bad = np.zeros(len(kind), bool)
        bad[np.array(meta.get("never_computed_rows", []), int)] = True
        bad[np.array(meta.get("im_end_rows", []), int)] = True
        acc.add("base_reasoning_rows", X[(kind == KIND["reasoning"]) & ~bad], SUB_N["base_reasoning"])
        sidx = np.nonzero((kind == KIND["system"]) & (turn == 0))[0]
        recon, ends = token_byte_bounds(bt, ids[sidx])
        p = recon.find(b"</tools>")
        s_tools = int(np.nonzero(ends > p + len(b"</tools>"))[0][0]) if p >= 0 else 0
        s_end = len(sidx)
        while s_end > 0 and not bt.piece(int(ids[sidx[s_end - 1]])).strip():
            s_end -= 1
        acc.add("sys_tools", X[sidx[:s_tools]], 0)
        acc.add("sys_mistrace", X[sidx[s_tools:s_end]], 0)
    return acc, {}


def fetch(s3_root: str | None, dump_root: Path | None, cell: str, task: str, dest: Path, subs: list[str]) -> subprocess.Popen | None:
    """Copy one rollout's files into dest (async when from S3)."""
    if s3_root is None:
        for sub in subs:
            src = (dump_root / cell / sub / (f"{task}.json" if sub == "meta" else f"{task}.safetensors")).resolve()
            (dest / sub).mkdir(parents=True, exist_ok=True)
            if not os.path.lexists(dest / sub / src.name):
                os.symlink(src, dest / sub / src.name)
        return None
    cmds = []
    for sub in subs:
        name = f"{task}.json" if sub == "meta" else f"{task}.safetensors"
        (dest / sub).mkdir(parents=True, exist_ok=True)
        cmds.append(f"aws s3 cp {s3_root}/{cell}/{sub}/{name} {dest}/{sub}/{name} --only-show-errors")
    cmd = " && ".join(cmds)
    return subprocess.Popen(f"for i in 1 2 3; do ({cmd}) && exit 0; sleep 5; done; exit 1", shell=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s3-root", default=None)
    ap.add_argument("--dump-root", default=None)
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--labels-dir", default=str(HERE.parent / "artifacts"))
    ap.add_argument("--records-dir", default=str(HERE.parents[3] / "experiments/spec-prompting-basetasks/data/records"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default=None, help="scratch dir for fetched shards (deleted per rollout)")
    ap.add_argument("--select", nargs="*", default=None)
    ap.add_argument("--base-sample", type=int, default=6, help="base rollouts per cell whose L35 shard is read")
    A = ap.parse_args()
    if (A.s3_root is None) == (A.dump_root is None):
        raise SystemExit("give exactly one of --s3-root / --dump-root")
    out = Path(A.out); out.mkdir(parents=True, exist_ok=True)
    work = Path(A.work or (out / "_work")); work.mkdir(parents=True, exist_ok=True)
    logf = open(out / "extract.log", "a")
    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    from transformers import AutoTokenizer
    bt = ByteTok(AutoTokenizer.from_pretrained(MODEL))

    # labels joined on key
    v1 = json.load(open(Path(A.labels_dir) / "cite_labels_v1.json"))
    v2 = {f"{o['rollout_id']}|{o['message_index']}|{o['source']}|{o['unit_idx']}": o for o in json.load(open(Path(A.labels_dir) / "cite_labels_v2.json"))}
    labels = defaultdict(list)
    for o in v1:
        k = f"{o['rollout_id']}|{o['message_index']}|{o['source']}|{o['unit_idx']}"
        b = v2.get(k)
        labels[o["rollout_id"]].append({**o, "v1": o["label"], "v2": b["label"] if b else "EXCLUDED"})
    log(f"labels: {sum(len(v) for v in labels.values())} candidates over {len(labels)} rollouts")

    for cell in A.cells:
        arm, _, seed = cell.split("/")
        recs = {}
        with open(Path(A.records_dir) / f"{arm}_conflicting_{seed}.jsonl") as f:
            for line in f:
                r = json.loads(line); recs[r["rollout_id"]] = r
        rids = [r for r in recs if (A.select is None or r.split("/")[-1] in A.select)]
        if A.s3_root:   # only rollouts that were captured
            man = json.loads(subprocess.check_output(f"aws s3 cp {A.s3_root}/{cell}/manifest.json -", shell=True))
            have = {t["task_id"] for t in man["transcripts"]}
        else:
            man = json.load(open(Path(A.dump_root) / cell / "manifest.json"))
            have = {t["task_id"] for t in man["transcripts"]}
        rids = [r for r in rids if r.split("/")[-1] in have]
        base_shard = set(random.Random(1).sample(rids, min(A.base_sample, len(rids)))) if arm == "base" else set()
        log(f"== {cell}: {len(rids)} rollouts ({'spec' if arm == 'spec' else f'base, {len(base_shard)} with shards'})")
        d = 5120
        sums = {}; counts = {}; subs = defaultdict(list); tags = defaultdict(list); allstats = {}
        subs_for = lambda rid: (["L35", "tokens", "meta"] if arm == "spec" else (["segmeans"] + (["L35", "tokens", "meta"] if rid in base_shard else [])))
        # prefetch pipeline: fetch i+1 while processing i
        dests = {rid: work / cell.replace("/", "_") / rid.split("/")[-1] for rid in rids}
        procs = {}
        if rids:
            procs[rids[0]] = fetch(A.s3_root, Path(A.dump_root) if A.dump_root else None, cell, rids[0].split("/")[-1], dests[rids[0]], subs_for(rids[0]))
        t0 = time.time()
        for i, rid in enumerate(rids):
            task = rid.split("/")[-1]
            if i + 1 < len(rids):
                nxt = rids[i + 1]
                procs[nxt] = fetch(A.s3_root, Path(A.dump_root) if A.dump_root else None, cell, nxt.split("/")[-1], dests[nxt], subs_for(nxt))
            p = procs.pop(rid, None)
            try:
                if p is not None and p.wait() != 0:
                    log(f"{rid}: FETCH FAILED after 3 tries"); continue
                try:
                    if arm == "spec":
                        acc, st = process_spec_rollout(rid, recs[rid], dests[rid], task, labels, bt, d, log)
                    else:
                        acc, st = process_base_rollout(rid, dests[rid], task, bt, d, rid in base_shard, log)
                except Exception as e:
                    log(f"{rid}: FAILED {type(e).__name__}: {e}"); continue
            finally:
                if A.s3_root:
                    shutil.rmtree(dests[rid], ignore_errors=True)
            sums[rid] = {k: v for k, v in acc.sum.items()}; counts[rid] = dict(acc.n); allstats[rid] = st
            for name, chunks in acc.sub.items():
                for c in chunks:
                    subs[name].append(c); tags[name].extend([rid] * len(c))
            log(f"  [{i + 1}/{len(rids)}] {rid} {time.time() - t0:.0f}s  n_reasoning={acc.n.get('reasoning_all', 0)} "
                f"combos={sum(1 for k in acc.n if '|' in k)} {st}")
        # write
        names = sorted({k for s in sums.values() for k in s})
        R = list(sums)
        S = np.zeros((len(R), len(names), d), np.float64); N = np.zeros((len(R), len(names)), np.int64)
        for i, rid in enumerate(R):
            for j, nm in enumerate(names):
                if nm in sums[rid]:
                    S[i, j] = sums[rid][nm]; N[i, j] = counts[rid][nm]
        np.savez(out / f"stats_{arm}_{seed}.npz", rollouts=np.array(R), names=np.array(names), sums=S, counts=N,
                 stats=json.dumps(allstats), layer=35)
        sub_out = {}
        for name in subs:
            sub_out[f"rows__{name}"] = np.concatenate(subs[name]); sub_out[f"tags__{name}"] = np.array(tags[name])
        np.savez(out / f"rows_{arm}_{seed}.npz", **sub_out)
        tot = defaultdict(int)
        for st in allstats.values():
            for k, v in st.items(): tot[k] += v
        log(f"== {cell} done: {len(R)} rollouts, {dict(tot)}; wrote stats_{arm}_{seed}.npz, rows_{arm}_{seed}.npz")


if __name__ == "__main__":
    main()
