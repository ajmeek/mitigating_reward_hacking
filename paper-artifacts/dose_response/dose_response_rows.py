"""Figure 4 (120-message budget, n = 60 per arm): honest-submission rate and hack rate on the impossible tests, and solve
rate on the original tests, one marker per arm, joined within each steering family, with the unsteered baseline as a dotted
reference line per series. The paper's figure is fig4_oneplane_bands.pdf (all three series on one axis, 90% Wilson bands);
--all also draws the other five variants (three stacked rows, and error bars or no intervals). Data as in dose_response_honest.py: summary_fleet.csv (u / optimized arms), spec-prompting
summary_m120.csv (spec prompt), spec-vector results_120msg/summary.csv (v_spec arms), honest_submit/honest_counts.csv.
    uv run python paper-artifacts/dose_response/dose_response_rows.py [--all]
"""
import argparse, math, os, sys
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE)); import paper_style; paper_style.use()

# ---- data: hack_k, honest_k, solve_k per arm key, n = 60 -------------------------------------------------
fleet = pd.read_csv(f"{ROOT}/experiments/diff-mean-steering/artifacts/summary_fleet.csv")
D = {}
for _, r in fleet.iterrows():
    D.setdefault(r.arm, {})["hack" if r.metric == "hack rate" else "solve"] = int(r.count_ext)
sp2 = pd.read_csv(f"{ROOT}/experiments/spec-prompting/artifacts/summary_m120.csv"); sp2 = sp2[(sp2.arm == "spec") & (sp2.group == "all")]
D["specprompt"] = {"hack": int(sp2[sp2.variant == "conflicting"]["count"].iloc[0]), "solve": int(sp2[sp2.variant == "original"]["count"].iloc[0])}
sv = pd.read_csv(f"{ROOT}/experiments/spec-vector/artifacts/results_120msg/summary.csv"); sv = sv[sv.group == "all"]
for a in ("v1floor5", "v1floor10", "v1floor16", "v1add8", "v1add16"):
    D[a] = {"hack": int(sv[(sv.arm == a) & (sv.variant == "conflicting")]["count"].iloc[0]), "solve": int(sv[(sv.arm == a) & (sv.variant == "original")]["count"].iloc[0])}
hc = pd.read_csv(f"{ROOT}/paper-artifacts/honest_submit/honest_counts.csv").set_index("arm")
HMAP = {"base": "baseline", "c0": "u cap 0", "cn4": "u cap -4", "cn8": "u cap -8", "cn12": "u cap -12", "cn16": "u cap -16",
        "addn2": "u add -2", "addn4": "u add -4", "addn6": "u add -6", "addn8": "u add -8",
        "optc_r4": "opt cap -4", "optc_r8": "opt cap -8", "optc_r12": "opt cap -12", "opta_k12": "opt add 12", "opta_k16": "opt add 16", "opta_k20": "opt add 20",
        "specprompt": "spec prompt", "v1floor5": "v_spec floor +5", "v1floor10": "v_spec floor +10", "v1floor16": "v_spec floor +16.5", "v1add8": "v_spec add +8", "v1add16": "v_spec add +16"}
for a, h in HMAP.items():
    D[a]["honest"] = int(hc.loc[h, "honest"]); assert int(hc.loc[h, "hacked"]) == D[a]["hack"], a
N = 60
GROUPS = [("no steering", ["base", "specprompt"], ["base", "spec"]),
          ("$u$ cap", ["c0", "cn4", "cn8", "cn12", "cn16"], ["0", "$-4$", "$-8$", "$-12$", "$-16$"]),
          ("$u$ add", ["addn2", "addn4", "addn6", "addn8"], ["$-2$", "$-4$", "$-6$", "$-8$"]),
          ("opt. cap", ["optc_r4", "optc_r8", "optc_r12"], ["$-4$", "$-8$", "$-12$"]),
          ("opt. add", ["opta_k12", "opta_k16", "opta_k20"], ["12", "16", "20"]),
          ("spec floor", ["v1floor5", "v1floor10", "v1floor16"], ["5", "10", "16.5"]),
          ("spec add", ["v1add8", "v1add16"], ["8", "16"])]
GAP = 1.4; xpos, labels, span = {}, {}, {}; x = 0.0
for name, arms, labs in GROUPS:
    x0 = x
    for a, l in zip(arms, labs): xpos[a] = x; labels[a] = l; x += 1
    span[name] = (x0, x - 1); x += GAP
ARMS = list(xpos); X = np.array([xpos[a] for a in ARMS])

def wilson(k, n, z=1.645):
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h
def rate(key, arms): return np.array([D[a][key] for a in arms]) / N
def ci(key, arms): return np.array([wilson(D[a][key], N) for a in arms]).T

SERIES = [("honest", paper_style.TEAL, "honest\nsubmission\n(impossible)", "honest submission"),
          ("hack", paper_style.VERMILLION, "hack\n(impossible)", "hack"),
          ("solve", paper_style.NEUTRAL, "solve\n(original)", "solve (original tests)")]
ROW_LIM = {"honest": (0.3, 1), "hack": (0, 0.5), "solve": (0.5, 1)}   # honest reaches 0.50 with a lower bound near 0.40 (spec floor 5)
ROW_TICKS = {"honest": [0.5, 0.75, 1], "hack": [0, 0.25, 0.5], "solve": [0.5, 0.75, 1]}

def draw_series(ax, key, col, style):
    """One series over every group: markers joined within a group (unsteered arms stay single), interval per `style`."""
    for name, arms, _ in GROUPS:
        xs = np.array([xpos[a] for a in arms]); p = rate(key, arms); lo, hi = ci(key, arms)
        if name != "no steering":
            ax.plot(xs, p, color=col, lw=1.0, zorder=3)
        if style == "bands":
            if name == "no steering":   # unjoined arms: a narrow vertical band per point
                for xi, l, h in zip(xs, lo, hi):
                    ax.fill_between([xi - 0.08, xi + 0.08], [l, l], [h, h], color=col, alpha=0.18, lw=0, zorder=1)
            else:
                ax.fill_between(xs, lo, hi, color=col, alpha=0.18, lw=0, zorder=1)
        elif style == "errorbars":
            ax.errorbar(xs, p, yerr=[p - lo, hi - p], fmt="none", ecolor=paper_style.INK, elinewidth=0.6, zorder=2)
        ax.scatter(xs, p, s=18, c=col, edgecolors=paper_style.INK, linewidths=0.5, zorder=4)
    ax.axhline(D["base"][key] / N, color=col, lw=0.6, ls=(0, (1, 2)), zorder=1)

def xaxis(ax, ticklabels=True):
    ax.set_xticks(X); ax.set_xticklabels([labels[a] for a in ARMS] if ticklabels else [], fontsize=6.5); ax.set_xlim(-0.6, X.max() + 0.6)
    ax.tick_params(axis="x", length=0)
    ax.spines["bottom"].set_visible(False)

def group_labels(ax, y=-0.17):
    for name, (x0, x1) in span.items():
        ax.plot([x0 - 0.36, x1 + 0.36], [y, y], color=paper_style.MUTED, lw=0.8, transform=ax.get_xaxis_transform(), clip_on=False)
        ax.text((x0 + x1) / 2, y - 0.04, name, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7.5, color=paper_style.MUTED)

def trim_left(ax, lo, hi, ticks):
    ax.set_ylim(lo - 0.02 * (hi - lo), hi + 0.02 * (hi - lo)); ax.set_yticks(ticks); ax.spines["left"].set_bounds(lo, hi)
    ax.yaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks])); ax.tick_params(axis="y", labelsize=7)

def rows3(style):
    fig, axes = plt.subplots(3, 1, figsize=(paper_style.FULL, 2.8), sharex=True, gridspec_kw={"hspace": 0.35})
    for ax, (key, col, lab, _) in zip(axes, SERIES):
        draw_series(ax, key, col, style); lo, hi = ROW_LIM[key]; trim_left(ax, lo, hi, ROW_TICKS[key])
        ax.set_ylabel(lab, fontsize=7, labelpad=6); xaxis(ax, ticklabels=ax is axes[-1])
    group_labels(axes[-1], y=-0.42)
    fig.subplots_adjust(left=0.1, right=0.99, top=0.98, bottom=0.2)
    print(paper_style.save(fig, f"fig4_rows3_{style}"))

def oneplane(style):
    fig, ax = plt.subplots(figsize=(paper_style.FULL, 2.8))
    for key, col, _, name in SERIES:
        draw_series(ax, key, col, style)
        ax.text(X.max() + 0.9, D[ARMS[-1]][key] / N, name, color=col, fontsize=7.5, va="center", ha="left")
    trim_left(ax, 0, 1, [0, 0.25, 0.5, 0.75, 1]); ax.set_ylabel("share of rollouts", fontsize=8); xaxis(ax); group_labels(ax, y=-0.12)
    for t in ax.get_xticklabels(): t.set_fontsize(6)
    fig.subplots_adjust(left=0.08, right=0.87, top=0.98, bottom=0.2)
    print(paper_style.save(fig, f"fig4_oneplane_{style}"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Figure 4; the paper's variant by default")
    ap.add_argument("--all", action="store_true", help="also draw the three-row layout and the error-bar / no-interval styles")
    A = ap.parse_args()
    if A.all:
        for style in ("bands", "errorbars", "nobars"): rows3(style); oneplane(style)
    else:
        oneplane("bands")
