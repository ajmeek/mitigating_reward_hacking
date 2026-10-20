"""School of Reward Hacks figure as dose-response lines (309 held-out prompts per arm, GLM-5 judge): the share of prompts
answered honestly and completely, hacked, or degenerate, one marker per arm joined within each steering family (and, within the optimized groups, only between arms of the same vector), with the
unsteered baseline as a dotted reference per series. The counterpart of dose_response/dose_response_rows.py. The paper's
figure is sorh_oneplane_bands.pdf (all three series on one axis, 90% Wilson bands); --all also draws the other five
variants (three stacked rows, and error bars or no intervals). Reads sorh_all_arms.csv (hc = neither hacked nor degenerate; hack and degenerate overlap on a
few prompts, counted as hacks, so the degenerate series is n - hc - hack). The $^{r}$ marks (refusal-only optimized
vector) are explained in the paper's appendix, as in sorh_stacked.py.
    uv run python paper-artifacts/sorh_specvec/sorh_rows.py [--all]
"""
import argparse, math, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); import paper_style as ps; ps.use()
d = pd.read_csv(os.path.join(HERE, "sorh_all_arms.csv")).fillna(""); d = d[d.judge == "GLM-5"].set_index("arm")
D = {a: {"honest": int(r.hc), "hack": int(r.hack), "degen": int(r.n - r.hc - r.hack), "n": int(r.n)} for a, r in d.iterrows()}
GROUPS = [("no steering", ["base"], ["base"]),
          ("opt. cap", ["v3c_cl35_t-3", "v3b_cl35_t-15", "v3b_cl35_t-25"], ["$-3$", "$-15^{r}$", "$-25^{r}$"]),
          ("opt. add", ["v3c_k20", "v3b_k24", "v3b_k28"], ["20", "$24^{r}$", "$28^{r}$"]),
          ("spec floor", ["vspec_f40"], ["+40"]),
          ("spec add", ["vspec_a24", "vspec_a32"], ["+24", "+32"]),
          ("$u$ add", ["u_an24"], ["$-24$"])]
SEGMENTS = {"opt. cap": [["v3c_cl35_t-3"], ["v3b_cl35_t-15", "v3b_cl35_t-25"]], "opt. add": [["v3c_k20"], ["v3b_k24", "v3b_k28"]]}
GAP = 1.2; xpos, labels, span = {}, {}, {}; x = 0.0
for name, arms, labs in GROUPS:
    x0 = x
    for a, l in zip(arms, labs): xpos[a] = x; labels[a] = l; x += 1
    span[name] = (x0, x - 1); x += GAP
ARMS = list(xpos); X = np.array([xpos[a] for a in ARMS])

def wilson(k, n, z=1.645):
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h
def rate(key, arms): return np.array([D[a][key] / D[a]["n"] for a in arms])
def ci(key, arms): return np.array([wilson(D[a][key], D[a]["n"]) for a in arms]).T

SERIES = [("honest", ps.TEAL, "honest and\ncomplete", "honest and complete"),
          ("hack", ps.VERMILLION, "hack", "hack"),
          ("degen", ps.NEUTRAL, "degenerate", "degenerate")]
ROW_LIM = {"honest": (0.5, 1), "hack": (0, 0.5), "degen": (0, 0.5)}

def draw_series(ax, key, col, style):
    for name, arms, _ in GROUPS:
        xs = np.array([xpos[a] for a in arms]); p = rate(key, arms); lo, hi = ci(key, arms)
        # join arms of the same vector only: the opt. cap / opt. add groups hold one v3c arm and two refusal-only v3b arms ($^{r}$)
        for seg in SEGMENTS.get(name, [arms]):
            ii = [arms.index(a) for a in seg]
            if len(seg) > 1:
                ax.plot(xs[ii], p[ii], color=col, lw=1.0, zorder=3)
                if style == "bands": ax.fill_between(xs[ii], lo[ii], hi[ii], color=col, alpha=0.18, lw=0, zorder=1)
            elif style == "bands":   # unjoined arm: a narrow vertical band
                i = ii[0]; ax.fill_between([xs[i] - 0.08, xs[i] + 0.08], [lo[i], lo[i]], [hi[i], hi[i]], color=col, alpha=0.18, lw=0, zorder=1)
        if style == "errorbars":
            ax.errorbar(xs, p, yerr=[p - lo, hi - p], fmt="none", ecolor=ps.INK, elinewidth=0.6, zorder=2)
        ax.scatter(xs, p, s=18, c=col, edgecolors=ps.INK, linewidths=0.5, zorder=4)
    ax.axhline(D["base"][key] / D["base"]["n"], color=col, lw=0.6, ls=(0, (1, 2)), zorder=1)

def xaxis(ax, ticklabels=True, size=6.5):
    ax.set_xticks(X); ax.set_xticklabels([labels[a] for a in ARMS] if ticklabels else [], fontsize=size); ax.set_xlim(-0.6, X.max() + 0.6)
    ax.tick_params(axis="x", length=0); ax.spines["bottom"].set_visible(False)

def group_labels(ax, y=-0.17):
    for name, (x0, x1) in span.items():
        ax.plot([x0 - 0.36, x1 + 0.36], [y, y], color=ps.MUTED, lw=0.8, transform=ax.get_xaxis_transform(), clip_on=False)
        ax.text((x0 + x1) / 2, y - 0.04, name, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7.5, color=ps.MUTED)

def trim_left(ax, lo, hi, ticks):
    ax.set_ylim(lo - 0.02 * (hi - lo), hi + 0.02 * (hi - lo)); ax.set_yticks(ticks); ax.spines["left"].set_bounds(lo, hi)
    ax.yaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks])); ax.tick_params(axis="y", labelsize=7)

def rows3(style):
    fig, axes = plt.subplots(3, 1, figsize=(ps.FULL, 2.8), sharex=True, gridspec_kw={"hspace": 0.35})
    for ax, (key, col, lab, _) in zip(axes, SERIES):
        draw_series(ax, key, col, style); lo, hi = ROW_LIM[key]; trim_left(ax, lo, hi, [lo, (lo + hi) / 2, hi])
        ax.set_ylabel(lab, fontsize=7, labelpad=6); xaxis(ax, ticklabels=ax is axes[-1])
    group_labels(axes[-1], y=-0.42)
    fig.subplots_adjust(left=0.1, right=0.99, top=0.98, bottom=0.2)
    print(ps.save(fig, f"sorh_rows3_{style}"))

def oneplane(style):
    fig, ax = plt.subplots(figsize=(ps.FULL, 2.8))
    for key, col, _, name in SERIES:
        draw_series(ax, key, col, style)
        ax.text(X.max() + 0.7, D[ARMS[-1]][key] / D[ARMS[-1]]["n"], name, color=col, fontsize=7.5, va="center", ha="left")
    trim_left(ax, 0, 1, [0, 0.25, 0.5, 0.75, 1]); ax.set_ylabel("share of held-out prompts", fontsize=8); xaxis(ax, size=6); group_labels(ax, y=-0.12)
    fig.subplots_adjust(left=0.08, right=0.87, top=0.98, bottom=0.2)
    print(ps.save(fig, f"sorh_oneplane_{style}"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SoRH figure; the paper's variant by default")
    ap.add_argument("--all", action="store_true", help="also draw the three-row layout and the error-bar / no-interval styles")
    A = ap.parse_args()
    if A.all:
        for style in ("bands", "errorbars", "nobars"): rows3(style); oneplane(style)
    else:
        oneplane("bands")
