#!/usr/bin/env python
"""How far off manifold each intervention pushes the model, and the behaviour that buys.

    uv run python experiments/intervention-magnitude/scripts/final_figure.py            # from the capture statistics
    PAPER_STYLE=1 uv run python experiments/intervention-magnitude/scripts/final_figure.py --x measured --from-table
                                                                                        # the paper's figure, redrawn from
                                                                                        # the tracked off_manifold_table_measured.csv

Off-manifold is measured on the axis the interventions actually act on: the u-coordinate of an
activation given the rest of that activation,

    z = w.(h - mu) / kappa,   w = Sigma_a^-1 u,   kappa = sqrt(u' Sigma_a^-1 u)

which has unit variance across unsteered tokens by construction, so z is already in standard
deviations. A displacement c*u moves it by exactly c*kappa, so the average push an arm applies is
mean|c|*kappa over every reasoning token, counting the ones a clamp leaves alone as zero.

Writes off_manifold_table.{csv,md} (one row per arm) and off_manifold_final.png: hack rate and
solve rate at the 120-message budget against that average push, one series per intervention family,
as point estimates; the per-arm counts and their two-sided Wilson 90% intervals stay in
off_manifold_table.csv.
Geometry from artifacts/{stats,whitened}_L35.npz (gitignored, 2 GB; regenerate with capture_stats.py and
whiten_pass.py) and data/proj_{opt,spec}_L35.npz; behaviour from the sweep's summary_fleet.csv. With
--from-table the per-arm rows are read back from the tracked off_manifold_table[_measured].csv instead, so
the figure redraws without the capture statistics.
Design: steering_reasoning.plot_style; PAPER_STYLE=1 uses paper-artifacts/paper_style.py (CM Sans) instead.
"""
import argparse
import csv
import os
import sys
from math import sqrt
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]

ap = argparse.ArgumentParser()
ap.add_argument("--stats", default=str(EXP / "artifacts/stats_L35.npz"))
ap.add_argument("--whitened", default=str(EXP / "artifacts/whitened_L35.npz"))
ap.add_argument("--summary", default=str(REPO / "experiments/diff-mean-steering/artifacts/summary_fleet.csv"))
ap.add_argument("--out", default=str(EXP / "artifacts/off_manifold_final.png"))
ap.add_argument("--table", default=str(EXP / "artifacts/off_manifold_table"))
ap.add_argument("--opt", default=str(EXP / "data/proj_opt_L35.npz"),
                help="per-token projections onto the optimized vectors (capture_proj_opt.py); '' to omit the opt arms")
ap.add_argument("--kappa-opt", default=str(EXP / "artifacts/kappa_opt_L35.npz"))
ap.add_argument("--x", choices=("counterfactual", "measured"), default="counterfactual",
                help="x axis: push on the unsteered reference (counterfactual) or, where artifacts/measured_push.csv has the arm, "
                     "the push measured on the arm's own replayed steered rollouts (arms without a measurement keep the counterfactual and are drawn hollow)")
ap.add_argument("--from-table", action="store_true",
                help="skip the capture statistics and read the per-arm rows back from <table>[_measured].csv (tracked)")
A = ap.parse_args()
D = 5120

if A.from_table:
    # Read the rows this script wrote on the machine with the capture statistics; the types round-trip.
    def _row(r):
        out = {}
        for k, v in r.items():
            if k in ("arm", "family"): out[k] = v
            elif k == "measured": out[k] = v == "True"
            elif k in ("hack_k", "hack_n", "solve_k", "solve_n"): out[k] = int(v)
            else: out[k] = float(v)
        return out
    if A.x == "measured":
        A.out = str(A.out).replace(".png", "_measured.png"); A.table = str(A.table) + "_measured"
    _rows = [_row(r) for r in csv.DictReader(open(A.table + ".csv"))]
    base, rows = _rows[0], _rows[1:]
    assert base["family"] == "none", base
else:
    S = np.load(A.stats)
    W = np.load(A.whitened)
    mu, Sigma, u = S["mu"], S["Sigma"], S["u"]
    alpha = float(W["alpha"])
    Sig_a = (1 - alpha) * Sigma + alpha * (np.trace(Sigma) / D) * np.eye(D)
    w = W["w"]
    kappa = float(np.sqrt(u @ w))

    proj_all = S["proj"].astype(np.float64)                 # every reasoning token, for the average push
    idx = W["sub_idx"]
    proj = proj_all[idx]
    g = W["wh"].astype(np.float64)[idx] - float(w @ mu)
    z0 = g / kappa
    zlo, zhi = np.quantile(z0, [0.005, 0.995])              # the family's own central 99% band

    CLAMP = [0, -4, -8, -12, -16]
    ADD = [-2, -4, -6, -8]
    ARM = {("clamp", 0): "c0", ("clamp", -4): "cn4", ("clamp", -8): "cn8", ("clamp", -12): "cn12",
           ("clamp", -16): "cn16", ("add", -2): "addn2", ("add", -4): "addn4", ("add", -6): "addn6",
           ("add", -8): "addn8"}

    rate = {}
    for r in csv.DictReader(open(A.summary)):
        rate[(r["arm"], r["variant"])] = (int(r["count_ext"]), int(r["n"]))


    def wilson(k, n, zc=1.645):
        """Wilson score interval, TWO-SIDED 90% (zc = z_0.95 = 1.645; 95% would be 1.96)."""
        p = k / n
        den = 1 + zc * zc / n
        c = (p + zc * zc / (2 * n)) / den
        h = zc * sqrt(p * (1 - p) / n + zc * zc / (4 * n * n)) / den
        return p, max(0.0, c - h), min(1.0, c + h)


    def displacement(x, kind, dose):
        return -np.maximum(0.0, x - dose) if kind == "clamp" else np.full(len(x), float(dose))


    rows = []
    for kind, doses in (("clamp", CLAMP), ("add", ADD)):
        for dose in doses:
            c_all = displacement(proj_all, kind, dose)
            push = np.abs(c_all) * kappa                    # off-manifold push per token, in sd
            z = z0 + displacement(proj, kind, dose) * kappa
            arm = ARM[(kind, dose)]
            hk, hn = rate[(arm, "conflicting")]
            sv, sn = rate[(arm, "original")]
            hp, hlo, hhi = wilson(hk, hn)
            spr, slo, shi = wilson(sv, sn)
            rows.append(dict(arm=arm, family=kind, dose=dose,
                             push_mean=float(push.mean()), push_p50=float(np.median(push)),
                             push_p95=float(np.quantile(push, 0.95)),
                             moved=float((c_all != 0).mean()),
                             outside=float(((z < zlo) | (z > zhi)).mean()),
                             hack=hp, hack_lo=hlo, hack_hi=hhi, hack_k=hk, hack_n=hn,
                             solve=spr, solve_lo=slo, solve_hi=shi, solve_k=sv, solve_n=sn))

    # Optimized-vector arms (2026-09-25): same population, projections onto each vector's own unit
    # direction, kappa from the same shrunk covariance. Cap thresholds are raw (the vector's own basis);
    # add doses are Matthew's units along the unit vector (scale k/24 on the norm-24 vector = k units).
    # The whitened v-coordinate of the subsample is not stored, so 'outside' is left NaN for these rows.
    OPT = []
    if A.opt and Path(A.opt).exists():
        P = np.load(A.opt); K = np.load(A.kappa_opt)
        assert len(P["proj_u"]) == len(proj_all) and np.allclose(P["proj_u"], proj_all, atol=2e-2), "opt projections do not align with stats_L35"
        OPT = [("opt cap", "optc_r4", -4, P["proj_optc"].astype(np.float64), float(K["kappa_optc"])),
               ("opt cap", "optc_r8", -8, P["proj_optc"].astype(np.float64), float(K["kappa_optc"])),
               ("opt cap", "optc_r12", -12, P["proj_optc"].astype(np.float64), float(K["kappa_optc"])),
               ("opt add", "opta_k12", 12, P["proj_opta"].astype(np.float64), float(K["kappa_opta"])),
               ("opt add", "opta_k16", 16, P["proj_opta"].astype(np.float64), float(K["kappa_opta"])),
               ("opt add", "opta_k20", 20, P["proj_opta"].astype(np.float64), float(K["kappa_opta"]))]
        for fam, arm, dose, pv, kv in OPT:
            if (arm, "conflicting") not in rate: continue
            c_all = displacement(pv, "clamp" if fam == "opt cap" else "add", dose)
            push = np.abs(c_all) * kv
            hk, hn = rate[(arm, "conflicting")]; sv, sn = rate[(arm, "original")]
            hp, hlo, hhi = wilson(hk, hn); spr, slo, shi = wilson(sv, sn)
            rows.append(dict(arm=arm, family=fam, dose=dose,
                             push_mean=float(push.mean()), push_p50=float(np.median(push)),
                             push_p95=float(np.quantile(push, 0.95)), moved=float((c_all != 0).mean()),
                             outside=float("nan"),
                             hack=hp, hack_lo=hlo, hack_hi=hhi, hack_k=hk, hack_n=hn,
                             solve=spr, solve_lo=slo, solve_hi=shi, solve_k=sv, solve_n=sn))

    # Austin's spec-citation direction v1_cites (experiments/spec-vector on worktree-model-spec-expm): his arms are
    # FLOORS at +5 / +10 / +16.5 (clamp on the negated unit vector at -t: tokens with h.v < t are lifted to t) and
    # additions of +8 / +16 along the unit vector. Push is computed the same way; the rows go to the table only,
    # and to the figure once their hack/solve rates exist in the summary under these arm names.
    SPEC = EXP / "data/proj_spec_L35.npz"
    if SPEC.exists():
        Q = np.load(SPEC); pv, kv = Q["proj_spec"].astype(np.float64), float(Q["kappa_spec"])
        for fam, arm, dose in (("spec floor", "v1floor5", 5), ("spec floor", "v1floor10", 10), ("spec floor", "v1floor16", 16.5),
                               ("spec add", "v1add8", 8), ("spec add", "v1add16", 16)):
            c_all = np.maximum(0.0, dose - pv) if fam == "spec floor" else np.full(len(pv), float(dose))
            push = np.abs(c_all) * kv
            if (arm, "conflicting") in rate:
                hk, hn = rate[(arm, "conflicting")]; sv, sn = rate[(arm, "original")]
                hp, hlo, hhi = wilson(hk, hn); spr, slo, shi = wilson(sv, sn)
            else:
                hk = hn = sv = sn = 0; hp = hlo = hhi = spr = slo = shi = float("nan")
            rows.append(dict(arm=arm, family=fam, dose=dose, push_mean=float(push.mean()), push_p50=float(np.median(push)),
                             push_p95=float(np.quantile(push, 0.95)), moved=float((c_all != 0).mean()), outside=float("nan"),
                             hack=hp, hack_lo=hlo, hack_hi=hhi, hack_k=hk, hack_n=hn, solve=spr, solve_lo=slo, solve_hi=shi, solve_k=sv, solve_n=sn))

    if A.x == "measured":
        M = {r["arm"]: r for r in csv.DictReader(open(EXP / "artifacts/measured_push.csv"))} if (EXP / "artifacts/measured_push.csv").exists() else {}
        for r in rows:
            r["counterfactual_mean"] = r["push_mean"]
            if r["arm"] in M:
                r["push_mean"] = float(M[r["arm"]]["measured_mean"]); r["push_p50"] = float(M[r["arm"]]["measured_p50"]); r["push_p95"] = float(M[r["arm"]]["measured_p95"])
                r["moved"] = float(M[r["arm"]]["moved"]); r["measured"] = True
            else:
                r["measured"] = False
        A.out = str(A.out).replace(".png", "_measured.png"); A.table = str(A.table) + "_measured"

    bk, bn = rate[("base", "conflicting")]
    bs, bsn = rate[("base", "original")]
    bh, bh_lo, bh_hi = wilson(bk, bn)
    bs_p, bs_lo, bs_hi = wilson(bs, bsn)
    base = dict(arm="base (unsteered)", family="none", dose=0.0, push_mean=0.0, push_p50=0.0, push_p95=0.0,
                moved=0.0, outside=float(((z0 < zlo) | (z0 > zhi)).mean()),
                hack=bh, hack_lo=bh_lo, hack_hi=bh_hi, hack_k=bk, hack_n=bn,
                solve=bs_p, solve_lo=bs_lo, solve_hi=bs_hi, solve_k=bs, solve_n=bsn)

    if A.x == "measured":
        base["counterfactual_mean"] = 0.0; base["measured"] = True
    with open(A.table + ".csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else list(base))
        wr.writeheader()
        wr.writerows([base] + rows)

    L = ["# How far off manifold each intervention pushes the model\n",
         "Push is measured on the u-coordinate given the rest of the activation, which has unit variance",
         "across unsteered tokens, so the numbers are standard deviations. The mean is over every reasoning",
         f"token ({len(proj_all):,} of them), counting tokens a cap leaves untouched as zero.",
         "'outside natural range' is the share of tokens whose steered coordinate leaves the band holding",
         "the central 99% of unsteered tokens. Rates are at the 120-message budget, seeds 42+43 pooled.\n",
         "| arm | mean push (sd) | median | 95th pct | tokens moved | outside natural range | hack rate | solve rate |",
         "|---|---|---|---|---|---|---|---|"]
    for r in [base] + rows:
        L.append(f"| {r['arm']} | {r['push_mean']:.2f} | {r['push_p50']:.2f} | {r['push_p95']:.2f} | "
                 f"{100 * r['moved']:.0f}% | {('n/a' if r['outside'] != r['outside'] else f"{100 * r['outside']:.1f}%")} | "
                 + ("pending | pending |" if r['hack'] != r['hack'] else f"{r['hack_k']}/{r['hack_n']} = {r['hack']:.2f} | {r['solve_k']}/{r['solve_n']} = {r['solve']:.2f} |"))
    Path(A.table + ".md").write_text("\n".join(L) + "\n")
    print("\n".join(L))

if os.environ.get("PAPER_STYLE") == "1":   # the paper's typography (CM Sans), for paper/artifacts/
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "paper-artifacts"))
    import paper_style as plot_style  # noqa: E402
else:
    from steering_reasoning import plot_style  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

plot_style.use()
VERM, TEAL, MUTED, INK = "#b8442b", "#3a7d85", "#4a4a4a", "#111111"
# One marker shape everywhere: colour carries the series, never shape. Error bars are plain
# vertical ink lines (no caps), and every dot takes an ink edge.
PT = dict(marker="o", ms=5.5, mec=INK, mew=0.7)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

# hacking, symlog so that a true zero has a place on the axis: linear within +-1, log above.
ax = axes[0]
# Colours follow the paper's dose-response figure (paper-artifacts/dose_response/dose_response.py): u teal,
# u addition a paper-lightened teal, optimized vectors the categorical yellow (lightened for addition),
# spec-citation the categorical green, baseline vermillion.
def _mix(c1, c2, t):
    a = tuple(int(c1[i:i + 2], 16) for i in (1, 3, 5)); b = tuple(int(c2[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % tuple(round(x + (y - x) * t) for x, y in zip(a, b))
PAPER = getattr(plot_style, "PAPER", "#faf7f2"); CAT = getattr(plot_style, "CATEGORICAL", [TEAL, "#c9a227", "#5b8c5a"])
YEL, GRN = CAT[1], CAT[2]
FAMS = [("clamp", TEAL, "-", "$u$ cap"), ("add", _mix(TEAL, PAPER, 0.45), "-", "$u$ addition"),
        ("opt cap", YEL, "-", "optimized cap"), ("opt add", _mix(YEL, PAPER, 0.45), "-", "optimized addition"),
        ("spec floor", GRN, "-", "spec-citation floor"), ("spec add", _mix(GRN, PAPER, 0.45), "-", "spec-citation addition")]
for fam, col, ls, name in FAMS:
    rs = sorted([r for r in rows if r["family"] == fam and r["hack"] == r["hack"]], key=lambda r: r["push_mean"])
    if not rs: continue
    ax.plot([r["push_mean"] for r in rs], [100 * r["hack"] for r in rs],
            ls=ls, color=col, lw=1.5, label=name, **PT)
    if A.x == "measured":
        hollow = [r for r in rs if not r.get("measured")]
        ax.plot([r["push_mean"] for r in hollow], [100 * r["hack"] for r in hollow], ls="none", marker="o", ms=5.5, mfc="white", mec=col, mew=1.0, zorder=4)
ax.plot([0], [100 * base["hack"]], ls="none", color=VERM, label="unsteered", **PT)
ax.set_yscale("symlog", linthresh=1, linscale=0.45)
ax.set_ylim(-0.45, 45)
ax.set_yticks([0, 1, 2, 5, 10, 20, 40])
ax.get_yaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
ax.get_yaxis().set_minor_locator(plt.matplotlib.ticker.NullLocator())
ax.set_ylabel("hack rate, % of rollouts (log)")
ax.set_title("hacking", loc="left")

# capability, linear: the whole point is that it does not move
ax = axes[1]
for fam, col, ls, name in FAMS:
    rs = sorted([r for r in rows if r["family"] == fam and r["solve"] == r["solve"]], key=lambda r: r["push_mean"])
    if not rs: continue
    ax.plot([r["push_mean"] for r in rs], [100 * r["solve"] for r in rs],
            ls=ls, color=col, lw=1.5, label=name, **PT)
    if A.x == "measured":
        hollow = [r for r in rs if not r.get("measured")]
        ax.plot([r["push_mean"] for r in hollow], [100 * r["solve"] for r in hollow], ls="none", marker="o", ms=5.5, mfc="white", mec=col, mew=1.0, zorder=4)
ax.plot([0], [100 * base["solve"]], ls="none", color=VERM, label="unsteered", **PT)
ax.set_ylim(0, 100)
ax.set_ylabel("solve rate, % of rollouts")
ax.set_title("capability", loc="left")

has_opt = any(r["family"].startswith("opt") for r in rows)
for ax in axes:
    ax.set_xlabel(("measured off-manifold push (sd" if A.x == "measured" else "average off-manifold push (sd") + (", symlog)" if has_opt else ")"))
    if has_opt:   # the optimized arms sit an order of magnitude further out; keep u's ladder readable
        ax.set_xscale("symlog", linthresh=1, linscale=0.6)
        ax.set_xlim(-0.15, max(r["push_mean"] for r in rows if r["hack"] == r["hack"]) * 1.6)
        ax.set_xticks([0, 0.5, 1, 2, 5, 10, 20, 50, 100]); ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
        ax.get_xaxis().set_minor_locator(plt.matplotlib.ticker.NullLocator())
    else:
        ax.set_xlim(-0.3, max(r["push_mean"] for r in rows if r["hack"] == r["hack"]) + 0.4)
axes[0].legend(frameon=False, fontsize=7, loc="upper right")
TITLE = ("Hacking and capability against the off-manifold push MEASURED on each arm's own steered rollouts (hollow = counterfactual only)\n"
         if A.x == "measured" else "Hacking and capability against how far off the L35 manifold each intervention pushes the model\n")
if os.environ.get("PAPER_STYLE") != "1":   # the paper figure carries no header; its caption says what is plotted
    fig.suptitle(TITLE + "(120-message budget, seeds 42+43; counts and Wilson 90% intervals in off_manifold_table.csv)", x=0.01, ha="left", fontsize=9)
if os.environ.get("PAPER_STYLE") != "1":
    plot_style.mono(*axes)   # monospace labels are the sandbox look; the paper uses CM Sans
fig.tight_layout(w_pad=2.0)
fig.savefig(A.out, dpi=180)
if os.environ.get("PAPER_STYLE") == "1":
    fig.savefig(str(A.out).replace(".png", "") + ".pdf")
print(A.out)
