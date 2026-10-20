#!/usr/bin/env python
"""Measured off-manifold push per arm from replayed steered transcripts (data/replay_push/push_<arm>.npz:
per generated token, the pre-intervention layer-35 projection onto the arm's own direction and the
whitened coordinate along it). Push per token = |c| * kappa with c = -max(0, proj - thr) (cap) or the
add scale; 'outside' = share of tokens whose post-intervention whitened coordinate leaves the unsteered
central 99% band (band from whitened_L35.npz, u-coordinate). Compares with the counterfactual push of
final_figure.py (unsteered population) in artifacts/measured_push.md and a figure.
    uv run python experiments/intervention-magnitude/scripts/measured_push.py
"""
import glob, os, csv, sys
from pathlib import Path
import numpy as np
EXP = Path(__file__).resolve().parents[1]
TASKS = {"lcbhard_4", "lcbhard_13", "lcbhard_33", "lcbhard_35", "lcbhard_48"}   # the five tasks every arm was replayed on
DOSE = {"c0": ("clamp", 0), "cn4": ("clamp", -4), "cn8": ("clamp", -8), "cn12": ("clamp", -12), "cn16": ("clamp", -16),
        "addn2": ("add", -2), "addn4": ("add", -4), "addn6": ("add", -6), "addn8": ("add", -8),
        "optc_r4": ("clamp", -4), "optc_r8": ("clamp", -8), "optc_r12": ("clamp", -12),
        "opta_k12": ("add", 12), "opta_k16": ("add", 16), "opta_k20": ("add", 20)}
W = np.load(EXP / "artifacts/whitened_L35.npz"); S = np.load(EXP / "artifacts/stats_L35.npz")
kappa_u = float(np.sqrt(S["u"] @ W["w"])); mu = S["mu"]
z0 = (W["wh"].astype(np.float64) - float(W["w"] @ mu)) / kappa_u
zlo, zhi = np.quantile(z0, [0.005, 0.995])
cf = {r["arm"]: r for r in csv.DictReader(open(EXP / "artifacts/off_manifold_table.csv"))}
rows = []
for f in sorted(glob.glob(str(EXP / "data/replay_push/push_*.npz"))):
    arm = os.path.basename(f)[5:-4]; d = np.load(f); kind, dose = DOSE[arm]
    keep = np.array([r.split("/")[-1] in TASKS for r in d["rollouts"]])[d["rollout"]]   # the same 5 tasks for every arm
    proj, wh, kappa = d["proj"].astype(np.float64)[keep], d["wh"].astype(np.float64)[keep], float(d["kappa"])
    n_keep = int(np.array([r.split("/")[-1] in TASKS for r in d["rollouts"]]).sum())
    c = -np.maximum(0.0, proj - dose) if kind == "clamp" else np.full(len(proj), float(dose))
    push = np.abs(c) * kappa
    z_post = (wh + c * kappa * kappa) / kappa            # whitened coordinate after the shift (w.v = kappa^2)
    outside = float(((z_post < zlo) | (z_post > zhi)).mean()) if str(d["direction"]) == "u" else float("nan")
    n_roll = n_keep
    rows.append(dict(arm=arm, rollouts=n_roll, tokens=len(proj), measured_mean=float(push.mean()), measured_p50=float(np.median(push)),
                     measured_p95=float(np.quantile(push, 0.95)), moved=float((c != 0).mean()), outside=outside,
                     counterfactual_mean=float(cf[arm]["push_mean"]) if arm in cf else float("nan"),
                     proj_mean=float(proj.mean())))
L = ["# Measured vs counterfactual off-manifold push (replayed steered transcripts, the same 5 impossible-variant tasks per arm, seed 42)\n",
     "Measured: the intervention's own displacement on the pre-intervention layer-35 activations of the steered rollouts' own tokens",
     "(unsteered replay of each steered transcript). Counterfactual: the same formula on the 1.7M unsteered reference tokens (final_figure.py).",
     "Both in whitened-coordinate sd units. 'outside' = share of post-intervention tokens beyond the unsteered central 99% band (u arms only).\n",
     "| arm | rollouts | gen tokens | mean proj (pre) | measured mean push | median | p95 | tokens moved | outside | counterfactual mean push |",
     "|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    L.append(f"| {r['arm']} | {r['rollouts']} | {r['tokens']:,} | {r['proj_mean']:.2f} | {r['measured_mean']:.2f} | {r['measured_p50']:.2f} | {r['measured_p95']:.2f} | "
             f"{100*r['moved']:.0f}% | {'n/a' if r['outside'] != r['outside'] else f'{100*r['outside']:.1f}%'} | {r['counterfactual_mean']:.2f} |")
(EXP / "artifacts/measured_push.md").write_text("\n".join(L) + "\n")
with open(EXP / "artifacts/measured_push.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("\n".join(L))
