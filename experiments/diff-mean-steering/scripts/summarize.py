#!/usr/bin/env python
"""Inspect .eval logs -> artifacts/results.csv (one row per rollout) and artifacts/summary.csv
(per arm x variant: rate, Wilson 90% interval, n, per-group counts). Aggregates every seed
present, so it is valid at any resampling depth. Run from the repo root (eval env):
    uv run python experiments/diff-mean-steering/scripts/summarize.py [--logs experiments/diff-mean-steering/data/logs] [--ext m120]
Columns of results.csv: arm, thr, variant, seed, task, group, score (C/I), outcome (1 = hacked on
impossible / solved on original), turns, out_tokens, at_limit (1 = ended on the message limit),
log. 'group' is test_groups A/B/C from splits.json.
--ext m<limit> also reads s<seed>/m<limit>/*.eval (the authors' rollouts that were continued under a
larger message limit from a shorter first pass); an extended record replaces the base record of the
same rollout. A run generated at 120 messages directly (MSG_LIMIT in scripts/run_arm.sh) needs no --ext.
"""
import argparse, csv, glob, json, math, os, re
from collections import defaultdict
from inspect_ai.log import read_eval_log

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
E = os.path.join(ROOT, "experiments/diff-mean-steering")
ap = argparse.ArgumentParser(); ap.add_argument("--logs", default=os.path.join(E, "data/logs"))
ap.add_argument("--arms", nargs="+", default=[os.path.join(E, "arms.tsv"), os.path.join(E, "arms_opt.tsv")], help="arm tables; the arm column of results.csv is their name, thr their dose")
ap.add_argument("--ext", default=None, help="m<limit>: merge extended runs from s<seed>/m<limit>/"); args = ap.parse_args()
ARMS = {}
for t in args.arms:
    for line in open(t):
        f = line.split("#", 1)[0].split()
        if f: ARMS[f[0]] = None if f[2] == "-" else float(f[2])
splits = json.load(open(os.path.join(ROOT, "src/steering_reasoning/data/splits.json")))
group_of = {t: g[0] for g, ids in splits["test_groups"].items() for t in ids}     # A / B / C

rows = []
files = [(f, False) for f in sorted(glob.glob(os.path.join(args.logs, "*", "*", "s*", "*.eval")))]
if args.ext:   # extended records come last so they override base records in `uniq` below
    files += [(f, True) for f in sorted(glob.glob(os.path.join(args.logs, "*", "*", "s*", args.ext, "*.eval")))]
for f, ext in files:
    arm, variant, sdir = f.split(os.sep)[-5:-2] if ext else f.split(os.sep)[-4:-1]; seed = int(sdir[1:])
    if arm not in ARMS: continue
    try: log = read_eval_log(f)
    except Exception as ex: print(f"skip {f}: {ex}"); continue
    for s in (log.samples or []):
        if not s.scores: continue
        v = next(iter(s.scores.values())).value
        if v in (None, ""): continue
        score = "C" if str(v).upper().startswith("C") or v == 1 or v == 1.0 else "I"
        n_asst = sum(1 for m in (s.messages or []) if getattr(m, "role", "") == "assistant")
        out_tok = None
        if getattr(s, "model_usage", None):
            out_tok = sum(getattr(u, "output_tokens", 0) or 0 for u in s.model_usage.values())
        at_limit = int(s.limit is not None and s.limit.type == "message")
        rows.append(dict(arm=arm, thr=ARMS[arm], variant=variant, seed=seed, task=s.id, group=group_of.get(s.id, "?"),
                         score=score, outcome=int(score == "C"), turns=n_asst, out_tokens=out_tok, at_limit=at_limit,
                         log=os.path.relpath(f, ROOT)))
# keep one row per (arm, variant, seed, task): the latest log wins (resumed runs re-score)
uniq = {}
for r in rows: uniq[(r["arm"], r["variant"], r["seed"], r["task"])] = r
rows = sorted(uniq.values(), key=lambda r: (r["variant"], ARMS[r["arm"]] if ARMS[r["arm"]] is not None else 99, r["seed"], r["task"]))
os.makedirs(os.path.join(E, "artifacts"), exist_ok=True)
with open(os.path.join(E, "artifacts/results.csv"), "w") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["arm"]); w.writeheader(); w.writerows(rows)

def wilson(k, n, z=1.645):
    if n == 0: return (float("nan"), float("nan"))
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h
summary = []
by = defaultdict(list)
for r in rows: by[(r["arm"], r["variant"])].append(r)
for (arm, variant), rs in sorted(by.items(), key=lambda kv: (kv[0][1], ARMS[kv[0][0]] if ARMS[kv[0][0]] is not None else 99)):
    k = sum(r["outcome"] for r in rs); n = len(rs); lo, hi = wilson(k, n)
    per_task = defaultdict(int)
    for r in rs: per_task[r["task"]] += 1
    grp = {g: (sum(r["outcome"] for r in rs if r["group"] == g), sum(1 for r in rs if r["group"] == g)) for g in "ABC"}
    summary.append(dict(arm=arm, thr=ARMS[arm], variant=variant, metric="hack rate" if variant == "conflicting" else "solve rate",
                        count=k, n=n, rate=round(k / n, 3) if n else "", ci90_lo=round(lo, 3), ci90_hi=round(hi, 3),
                        at_limit=sum(r["at_limit"] for r in rs),
                        tasks=len(per_task), samples_per_task=f"{min(per_task.values())}-{max(per_task.values())}" if per_task else "",
                        **{f"{g}": f"{grp[g][0]}/{grp[g][1]}" for g in "ABC"},
                        median_turns=sorted(r["turns"] for r in rs)[n // 2] if n else "",
                        median_out_tokens=sorted(r["out_tokens"] or 0 for r in rs)[n // 2] if n else ""))
with open(os.path.join(E, "artifacts/summary.csv"), "w") as fh:
    w = csv.DictWriter(fh, fieldnames=list(summary[0]) if summary else ["arm"]); w.writeheader(); w.writerows(summary)
for s in summary: print(s)
print(f"{len(rows)} rollouts -> artifacts/results.csv, {len(summary)} arm x variant rows -> artifacts/summary.csv")
