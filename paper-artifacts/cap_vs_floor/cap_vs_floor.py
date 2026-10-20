"""Cap on u vs floor on v_cite: same one-sided operator, opposite tail. Two panels, both PER TOKEN, the
same held-out tokens on both axes.

Left:  held-out rollouts (val + test-A of the 42-rollout bf16 capture), per-token projection on u by
       judge segment class (other / anti-hack / hack); dotted rules = cap thresholds tau in
       {0, -4, -8, -12, -16}; the cap moves tokens RIGHT of a rule leftwards.
Right: per-token projection on v_cite of spec-citing reasoning and of other reasoning from the spec-
       prompted fit rollouts (in sample), and of the SAME held-out tokens as the left panel, which never
       saw the spec (all classes pooled); dotted rules = floors t in {5, 10, 16.5}; the floor moves
       tokens LEFT of a rule rightwards. Ticks mark the held-out hack and anti-hack medians on v.
Reads experiments/spec-vector/artifacts/cap_floor/{heldout_token_hist.csv, token_hist.csv, thresholds.json}.
    uv run --with matplotlib --with pandas python paper-artifacts/cap_vs_floor/cap_vs_floor.py
"""
import json, os, sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(ROOT, "experiments/spec-vector/artifacts/cap_floor")
sys.path.insert(0, os.path.dirname(HERE))
try:
    import paper_style as ps
    ps.use(); INK, TEAL, VERM, FAINT, COL = ps.INK, ps.TEAL, ps.VERMILLION, ps.FAINT, ps.COL
    save = ps.save
except Exception:   # design-language dependency missing: inline the same rules
    INK, TEAL, VERM, FAINT, COL = "#1a1a1a", "#1b8a8f", "#d55e00", "#9a9a9a", 3.3
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 7.5, "savefig.dpi": 300, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.unicode_minus": False})
    def save(fig, stem, where=HERE):
        for ext in ("pdf", "png"): fig.savefig(os.path.join(where, f"{stem}.{ext}"))
        return os.path.join(where, f"{stem}.png")
MUTED = "#4a4a4a"

meta = json.load(open(os.path.join(DATA, "thresholds.json")))
held = pd.read_csv(os.path.join(DATA, "heldout_token_hist.csv")); held = held[held.split.isin(["val", "test"])]
tok = pd.read_csv(os.path.join(DATA, "token_hist.csv"))
hsum = json.load(open(os.path.join(DATA, "heldout_token_hist.json")))["groups"]
W = 0.5


def agg(d):
    return d.groupby("bin_left", as_index=False)["count"].sum()


def step_density(ax, d, color, label, fill, lo, hi):
    d = d[(d.bin_left >= lo) & (d.bin_left < hi)].sort_values("bin_left")
    x = np.r_[d.bin_left.values, d.bin_left.values[-1] + W]; y = np.r_[d["count"].values, 0] / (d["count"].sum() * W)
    if fill: ax.fill_between(x, y, step="post", color=color, alpha=0.45, lw=0, label=label, zorder=1)
    else: ax.step(x, y, where="post", color=color, lw=1.1, label=label, zorder=3)


def rules(ax, ts, labels, ymax):
    for t, lab in zip(ts, labels):
        ax.axvline(t, color=INK, ls=":", lw=0.7, zorder=2); ax.text(t, ymax * 1.02, lab, ha="center", va="bottom", fontsize=7, color=INK)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2 * COL, 0.72 * COL))
# --- left: cap on u, held-out tokens by class
lo, hi = -36, 18
hu = held[held.vector == "u"]
for cls, color, label, fill in (("other", FAINT, "other", True), ("decline", TEAL, "anti-hack", False), ("hack", VERM, "hack", False)):
    step_density(ax1, agg(hu[hu.cls == cls]), color, label, fill, lo, hi)
ymax = ax1.get_ylim()[1]
rules(ax1, meta["u"]["cap_grid"], [str(t) for t in meta["u"]["cap_grid"]], ymax)
ax1.annotate("", xy=(-13.5, ymax * 0.62), xytext=(-2.5, ymax * 0.62), arrowprops=dict(arrowstyle="->", color=INK, lw=0.8))
ax1.text(-8, ymax * 0.65, "cap", ha="center", fontsize=7, color=INK)
ax1.set_xlim(lo, hi); ax1.set_ylim(0, ymax); ax1.set_yticks([]); ax1.spines["left"].set_visible(False)
ax1.set_xlabel(r"projection on $u$, per token (held out)"); ax1.set_ylabel("density"); ax1.legend(loc="upper left", frameon=False)
# --- right: floor on v_cite: in-sample citing / other spec-arm reasoning, held-out tokens (no spec)
lo, hi = -10, 45
hv = held[held.vector == "v_cite"]
step_density(ax2, agg(tok[(tok.vector == "v_cite") & (tok.cls == "other_spec_reasoning")]), FAINT, "other reasoning, spec in prompt", True, lo, hi)
step_density(ax2, agg(hv), MUTED, "held-out reasoning, no spec", False, lo, hi)
step_density(ax2, agg(tok[(tok.vector == "v_cite") & (tok.cls == "citing_reasoning")]), TEAL, "spec-citing reasoning", False, lo, hi)
ymax = ax2.get_ylim()[1]
rules(ax2, meta["v_cite"]["floors"], [f"+{t:g}" for t in meta["v_cite"]["floors"]], ymax)
for cls, lab, color in (("hack", "hack", VERM), ("decline", "anti-hack", TEAL)):
    m = np.median([hsum[f"{sp}/v_cite/{cls}"]["median"] for sp in ("val", "test") if f"{sp}/v_cite/{cls}" in hsum])
    ax2.plot([m], [ymax * 0.035], marker="v", color=color, ms=5, mec=INK, mew=0.5, ls="none", zorder=4)
    ax2.text(m, ymax * 0.085, f"held-out\n{lab}", ha="center", fontsize=6, color=INK)
ax2.annotate("", xy=(14.5, ymax * 0.62), xytext=(3.5, ymax * 0.62), arrowprops=dict(arrowstyle="->", color=INK, lw=0.8))
ax2.text(9, ymax * 0.65, "floor", ha="center", fontsize=7, color=INK)
ax2.set_xlim(lo, hi); ax2.set_ylim(0, ymax); ax2.set_yticks([]); ax2.spines["left"].set_visible(False)
ax2.set_xlabel(r"projection on $v_{\mathrm{spec}}$, per token"); ax2.legend(loc="upper right", frameon=False)
fig.tight_layout()
print(save(fig, "cap_vs_floor"))
