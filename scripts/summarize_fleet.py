#!/usr/bin/env python
"""Per arm x variant rates from fleet runs at the message limit (120).

    uv run python scripts/summarize_fleet.py --runs data/fleet/<run> ... [--out artifacts/summary_fleet.csv]

A run whose plan has message_limit == --limit (the default in fleet.toml) is counted
directly: its rows carry native=1 and fill the at-limit (*_ext) columns, which the figure and table
scripts read. The remaining options reproduce the authors' two-pass runs (a shorter first pass that
was continued to 120):

    uv run python scripts/summarize_fleet.py \
        --runs data/fleet/<shorter run> ... [--extend data/extend/m120] [--native-runs data/fleet/<120 run> ...] [--out artifacts/summary_fleet.csv] \
        [--exp experiments/diff-mean-steering]

Relative --runs/--extend/--out paths resolve against the cwd if they exist there, else against the
experiment dir (--exp, repo-relative; default experiments/diff-mean-steering), so
the default output is <exp>/artifacts/summary_fleet.csv. Needs inspect_ai only for --extend (base env).

Reads each run's plan/state (completed jobs only; a completed record wins over a placeholder when
runs overlap). A --native-runs run was generated at the --limit itself: its records fill only the at-limit columns and its rows carry native=1 (the at-60
columns are empty). With --extend, a rollout whose record ended on the message limit and that has a
scored sample in <extend>/<arm>/<variant>/s<seed>/m<limit>/*.eval takes that sample's outcome
and at-limit flag. Prints and writes one row per (arm, variant) with
n, hack/solve count and rate, Wilson 90% interval, at-limit count, and for the extended view how
many of the cell's limit-hit rollouts have been extended so far.
"""
import argparse, csv, glob, json, os
from collections import defaultdict
from pathlib import Path

from steering_reasoning.stats import wilson

REPO = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="*", default=[]); ap.add_argument("--extend", default=None)
ap.add_argument("--out", default=None); ap.add_argument("--limit", type=int, default=120); ap.add_argument("--native-runs", nargs="*", default=[])
ap.add_argument("--exp", default="experiments/diff-mean-steering", help="experiment dir (repo-relative) that relative paths and the default --out resolve against")
A = ap.parse_args()
if not A.runs and not A.native_runs: ap.error("give --runs and/or --native-runs")
EXP = Path(A.exp) if Path(A.exp).is_absolute() else REPO / A.exp
rel = lambda p: Path(p) if (Path(p).is_absolute() or Path(p).exists()) else EXP / p

best = {}; plan = None; A.native_runs = list(A.native_runs)
for rd in A.runs:
    rd = rel(rd); plan_r = json.load(open(rd / "plan.json"))
    if plan_r["work"]["message_limit"] == A.limit: A.native_runs.append(rd); continue   # generated at the limit itself: counted below as native
    if plan_r["work"]["message_limit"] > A.limit: raise SystemExit(f"{rd} ran at {plan_r['work']['message_limit']} messages; pass --limit {plan_r['work']['message_limit']} (its at_limit flags do not describe a {A.limit}-message run)")
    plan = plan_r; state = json.load(open(rd / "state.json")); jobs = {j["id"]: j for j in plan["jobs"]}
    for jid, rec in state["jobs"].items():
        j = jobs[jid]; key = (j["arm"], j["variant"], j["seed"], j["task"])
        if key not in best or (rec["status"] == "complete" and best[key][1]["status"] != "complete"): best[key] = (j, rec)
ARM_SPEC = {a["name"]: a for a in plan["arms"]} if plan else {}
native = {}
for rd in A.native_runs:
    rd = rel(rd); plan_n = json.load(open(rd / "plan.json")); state = json.load(open(rd / "state.json")); jobs = {j["id"]: j for j in plan_n["jobs"]}
    assert plan_n["work"]["message_limit"] == A.limit, f"{rd} is not a native {A.limit}-message run"
    ARM_SPEC.update({a["name"]: a for a in plan_n["arms"] if a["name"] not in ARM_SPEC})
    for jid, rec in state["jobs"].items():
        j = jobs[jid]; key = (j["arm"], j["variant"], j["seed"], j["task"])
        if rec["status"] == "complete" and key not in native: native[key] = (j, rec)
both = {k[:2] for k in best} & {k[:2] for k in native}
if both: raise SystemExit(f"{len(both)} arm x variant cells have rollouts both from a shorter-limit run and from a {A.limit}-message run (e.g. {sorted(both)[0]}); summarize one view at a time")

extended = {}   # (arm, variant, seed, task) -> (outcome, at_limit)
if A.extend:
    from inspect_ai.log import read_eval_log_sample_summaries
    for f in glob.glob(str(rel(A.extend) / "*" / "*" / "s*" / f"m{A.limit}" / "*.eval")):
        arm, variant, sdir = f.split(os.sep)[-5:-2]; seed = int(sdir[1:])
        try: summaries = read_eval_log_sample_summaries(f)
        except Exception: continue
        for s in summaries:
            if s.error or not s.scores: continue
            v = next(iter(s.scores.values())).value
            if v in (None, ""): continue
            extended[(arm, variant, seed, str(s.id))] = (1 if (str(v).upper().startswith("C") or v in (1, 1.0)) else 0, bool(s.limit and (getattr(s.limit, "type", s.limit) == "message")))   # summaries carry the limit as a string, full logs as an object

rows = []
cells = defaultdict(lambda: dict(n=0, k=0, lim=0, k_ext=0, lim_ext=0, ext_done=0))
for (arm, variant, seed, task), (j, rec) in best.items():
    if rec["status"] != "complete": continue
    v = next(iter(rec["result"]["scores"].values())); ok = 1 if (str(v).upper().startswith("C") or v in (1, 1.0)) else 0; lim = bool(rec["result"]["at_limit"])
    c = cells[(arm, variant)]; c["n"] += 1; c["k"] += ok; c["lim"] += lim
    ok2, lim2 = ok, lim
    if lim and (arm, variant, seed, task) in extended:
        ok2, lim2 = extended[(arm, variant, seed, task)]; c["ext_done"] += 1
    c["k_ext"] += ok2; c["lim_ext"] += lim2
for (arm, variant, seed, task), (j, rec) in native.items():
    v = next(iter(rec["result"]["scores"].values())); ok = 1 if (str(v).upper().startswith("C") or v in (1, 1.0)) else 0
    c = cells[(arm, variant)]; c["n"] += 1; c["k_ext"] += ok; c["lim_ext"] += bool(rec["result"]["at_limit"]); c["native"] = True
MODE_RANK = {"none": 0, "clamp": 1, "add": 2, "ablate": 3}
ARMS = sorted(ARM_SPEC, key=lambda n: (MODE_RANK.get(ARM_SPEC[n].get("mode"), 9), abs(ARM_SPEC[n].get("dose") or 0)))   # base, c0..cn16, addn2..addn8
hdr = f"{'arm':6s} {'variant':11s} {'n':>3s} | {'at short lim':>14s} {'CI90':>11s} {'limit':>5s} | {'at ' + str(A.limit):>14s} {'CI90':>11s} {'limit':>5s} {'extended':>8s}"
print(hdr); print("-" * len(hdr))
for variant in ("conflicting", "original"):
    for arm in ARMS:
        c = cells[(arm, variant)]
        if not c["n"]: continue
        lo, hi = wilson(c["k"], c["n"], z=1.645); lo2, hi2 = wilson(c["k_ext"], c["n"], z=1.645); nat = c.get("native", False)   # z=1.645 as the original script, so the tracked CSV is unchanged (stats.Z90 differs in the 5th decimal)
        left = f"{'native':>14s} {'':>11s} {'':>5s}" if nat else f"{c['k']:3d}/{c['n']:<3d} = {c['k']/c['n']:.2f} {lo:.2f}-{hi:.2f} {c['lim']:5d}"
        print(f"{arm:6s} {variant:11s} {c['n']:3d} | {left} | {c['k_ext']:3d}/{c['n']:<3d} = {c['k_ext']/c['n']:.2f} {lo2:.2f}-{hi2:.2f} {c['lim_ext']:5d} {c['ext_done']:3d}/{c['lim']:<3d}")
        rows.append(dict(arm=arm, variant=variant, metric="hack rate" if variant == "conflicting" else "solve rate", n=c["n"], native=int(nat),
                         count_60="" if nat else c["k"], rate_60="" if nat else round(c["k"] / c["n"], 3), ci90_lo_60="" if nat else round(lo, 3), ci90_hi_60="" if nat else round(hi, 3), at_limit_60="" if nat else c["lim"],
                         count_ext=c["k_ext"], rate_ext=round(c["k_ext"] / c["n"], 3), ci90_lo_ext=round(lo2, 3), ci90_hi_ext=round(hi2, 3), at_limit_ext=c["lim_ext"], extended="" if nat else c["ext_done"], extend_limit=A.limit))
if not rows: raise SystemExit("no completed rollouts in the given runs")
out = rel(A.out) if A.out else EXP / "artifacts" / "summary_fleet.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("wrote", out)
