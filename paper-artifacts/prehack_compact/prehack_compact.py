"""Compact version of the pre-hack appendix figures (candidate; option 1).

prehack_ramps.pdf     1x2: projection onto u per turn aligned on the first hack edit, pooled by mean
                      (left) and max (right) over the turn's segments, z within rollout, 95% bootstrap
                      (reads prehack-u-profiles/artifacts/turn_onset_ramps.csv).
prehack_window_auc.pdf 1x3: trailing 20-token mean and max of h.u over the last 128 tokens before the
                      call for hack-edit / honest-submit / other turns (as paper-artifacts/prehack_window),
                      and leave-one-rollout-out AUC of the hazard monitor by window K
                      (reads prehack-u-profiles/artifacts/hazard_mean_vs_max_gap0.csv).

    uv run python paper-artifacts/prehack_compact/prehack_compact.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paper_style  # noqa: E402

paper_style.use()
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); PU = f"{ROOT}/experiments/prehack-u-profiles"; HA = f"{ROOT}/experiments/hack-action-anchor"
W, K, B = 128, 20, 2000

def tidy(ax):
    ax.tick_params(length=2.5, width=0.6)
    for sp in ("left", "bottom"): ax.spines[sp].set_linewidth(0.7)

# --- row 1: turn ramps, first hack edit only ----------------------------------------------------
R = pd.read_csv(f"{PU}/artifacts/turn_onset_ramps.csv"); R = R[(R.anchor == "edit") & (R.direction == "u")]
fig, axes = plt.subplots(1, 2, figsize=(paper_style.FULL, 1.9), sharey=True)
for ax, pool in zip(axes, ("mean", "max")):
    S = R[R.pool == pool].sort_values("offset")
    ax.fill_between(S.offset, S.z_lo, S.z_hi, color=paper_style.VERMILLION, alpha=0.18, lw=0)
    ax.plot(S.offset, S.z_mean, color=paper_style.VERMILLION, lw=1.2, marker="o", ms=2.8, markeredgecolor=paper_style.INK, markeredgewidth=0.4)
    ax.axhline(0, color=paper_style.RULE, lw=0.8, zorder=0); ax.axvline(0, color=paper_style.RULE, lw=0.8, zorder=0)
    ax.set_title(f"{pool} over the turn", loc="left", fontsize=8, color=paper_style.MUTED, pad=3)
    ax.set_xlabel("turns relative to the first hack edit"); tidy(ax)
axes[0].set_ylabel("$u$-projection (z within rollout)")
fig.tight_layout(w_pad=1.2)
print(paper_style.save(fig, "prehack_ramps"))

# --- row 2: trailing mean | trailing max | hazard AUC ------------------------------------------
Z = np.load(f"{HA}/artifacts/turn_paths_last512_bf16.npz", allow_pickle=True)
U = Z["proj"] @ Z["u_xy"]; tid, turn = Z["task_id"], Z["turn"]
sys.path.insert(0, f"{PU}/scripts")
from edits import hack_edit_turns, first_hack_turn  # noqa: E402
hack_edit, first_hack = hack_edit_turns(), first_hack_turn()
is_he = np.array([(t, k) in hack_edit for t, k in zip(tid, turn)])
honest = np.array(["submit" in str(t) for t in Z["tool_detail"]]) & np.array([k < first_hack.get(t, 10 ** 9) for t, k in zip(tid, turn)])
ok = (Z["split"] != "train") & (Z["n_tokens"] >= W + K - 1)
sel = {"hack": ok & is_he, "submit": ok & honest, "other": ok & ~is_he & ~honest & (Z["n_hack_segments"] == 0) & (Z["n_decline_segments"] == 0)}
LAB = {"hack": "hack edit", "submit": "honest submit", "other": "other"}
COL = {"hack": paper_style.VERMILLION, "submit": paper_style.TEAL, "other": paper_style.NEUTRAL}
win = np.lib.stride_tricks.sliding_window_view(U[:, -(W + K - 1):], K, axis=1)
x = np.arange(-W + 1, 1); rng = np.random.default_rng(0)

fig, axes = plt.subplots(1, 3, figsize=(paper_style.FULL, 2.0), gridspec_kw=dict(width_ratios=[1, 1, 0.9]))
for ax, (name, S) in zip(axes[:2], (("mean", win.mean(-1)), ("max", win.max(-1)))):
    for g in ("other", "submit", "hack"):
        m = sel[g]; ids = np.unique(tid[m]); Rr = np.stack([S[m & (tid == r)].mean(0) for r in ids])
        bs = np.stack([Rr[rng.integers(0, len(Rr), len(Rr))].mean(0) for _ in range(B)])
        ax.fill_between(x, np.percentile(bs, 5, 0), np.percentile(bs, 95, 0), color=COL[g], alpha=0.18, lw=0)
        ax.plot(x, Rr.mean(0), color=COL[g], lw=1.1)
    ax.axhline(0, color=paper_style.RULE, lw=0.8, zorder=0); ax.set_xlim(x[0], x[-1]); ax.set_xticks([-120, -80, -40, 0])
    ax.set_title(f"{name} of the last {K} tokens", loc="left", fontsize=8, color=paper_style.MUTED, pad=3)
    ax.set_xlabel("tokens before the call"); tidy(ax)
axes[0].set_ylabel("$h \\cdot u$")
axes[0].legend(handles=[Line2D([], [], color=COL[g], lw=1.4, label=LAB[g]) for g in ("hack", "submit", "other")],
               frameon=False, fontsize=7, loc="lower left", handlelength=1.2, borderaxespad=0.2)
H = pd.read_csv(f"{PU}/artifacts/hazard_mean_vs_max_gap0.csv"); ctrl = H[H.model == "controls"].loro_auc.iloc[0]; H = H[H.K != "-"]
ax = axes[2]
# direct labels at the right end of each line instead of a legend
for name, col in (("max", paper_style.TEAL), ("mean", paper_style.NEUTRAL)):
    S = H[H.model == f"controls + {name}"]; xi = np.arange(len(S)); y = S.loro_auc.to_numpy()
    ax.plot(xi, y, color=col, marker="o", ms=2.8, lw=1.1, markeredgecolor=paper_style.INK, markeredgewidth=0.4)
    ax.text(xi[-1] + 0.25, y[-1], name, color=col, fontsize=7.5, va="center", ha="left")
ax.hlines(ctrl, -0.4, xi[-1], color=paper_style.VERMILLION, lw=0.9, ls="--")   # stop short of the label
ax.text(xi[-1] + 0.25, ctrl, "controls", color=paper_style.VERMILLION, fontsize=7.5, va="center", ha="left")
ax.set_xlim(-0.4, xi[-1] + 1.6); ax.set_ylim(0.77, 1.0); ax.set_yticks([0.8, 0.9, 1.0])
ax.set_xticks(xi); ax.set_xticklabels([("all" if int(k) >= 492 else k) for k in S.K])
ax.set_xlabel("window K (tokens)"); ax.set_ylabel("out-of-fold AUC")
ax.set_title("predicting the first hack edit", loc="left", fontsize=8, color=paper_style.MUTED, pad=3)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
tidy(ax)
fig.tight_layout(w_pad=1.0)
print(paper_style.save(fig, "prehack_window_auc"))
