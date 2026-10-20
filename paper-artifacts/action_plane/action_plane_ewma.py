"""Paper Figure 5 (right): mean path through the (e, u_perp) plane over the last W = 128 reasoning
tokens before a tool call, smoothed per token instead of binned (replaces the d-binned
action_plane_hackedit.py --decline=submit --ci=none --bins=1.5:9 on 2026-09-25).

Data and selection are those of action_plane_hackedit.py --decline=submit: held-out val + test
rollouts, turns >= W tokens; hack = turns whose tool call is a HACK-judged edit, decline = honest
submits before the rollout's first hack edit, other = neither, one drawn per rollout and averaged
over N_DRAWS draws. Mean within rollout, then across rollouts. No intervals drawn.

Smoothing, with d = tokens remaining before the call (d = 1 is the last token): a causal EWMA with
half-life 1.5 log2(d) tokens (floor 1), applied twice. It runs over all the turn's available context
(up to 512 tokens before the call), so the window's first point is already warmed up. Two passes give
a smooth one-sided kernel with about the lag of one pass at 3 log2(d). Dots mark d in a log-spaced
set, opacity rising with -log2(d).

Output: action_plane_ewma.{pdf,png} beside this script, copied to paper/artifacts/action_plane_turn_paths.pdf.

    uv run python paper-artifacts/action_plane/action_plane_ewma.py
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paper_style  # noqa: E402

paper_style.use()
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); HA = f"{ROOT}/experiments/hack-action-anchor"
W, N_DRAWS, PASSES, C = 128, 20, 2, 1.5

Z = np.load(f"{HA}/artifacts/turn_paths_last512_bf16.npz", allow_pickle=True)
PF, tid, n = Z["proj"], Z["task_id"], Z["n_tokens"]          # proj [turns, 512, 2], NaN-padded in front
sys.path.insert(0, f"{ROOT}/experiments/prehack-u-profiles/scripts")
from edits import hack_edit_turns, first_hack_turn  # noqa: E402   (edit judge with the two hand adjudications applied)
hack_edit, first_hack = hack_edit_turns(), first_hack_turn()
is_he = np.array([(t, k) in hack_edit for t, k in zip(tid, Z["turn"])])
honest = np.array(["submit" in str(t) for t in Z["tool_detail"]]) & np.array([k < first_hack.get(t, 10 ** 9) for t, k in zip(tid, Z["turn"])])
ok = (Z["split"] != "train") & (n >= W)
sel = {"hack": ok & is_he, "decline": ok & honest,
       "other": ok & ~is_he & ~honest & (Z["n_decline_segments"] == 0) & (Z["n_hack_segments"] == 0)}

L = PF.shape[1]; dfull = L - np.arange(L)                    # tokens remaining per column
HL = np.maximum(1.0, C * np.log2(dfull))

def ewma(T):                                                 # causal, half-life HL per column; starts at a turn's first real token
    lam = 1 - 0.5 ** (1 / HL); E = np.full_like(T, np.nan); run = np.full(T[:, 0].shape, np.nan)
    for j in range(L):
        x = T[:, j]; fresh = np.isnan(run) & ~np.isnan(x)
        run = np.where(fresh, x, np.where(np.isnan(x), run, (1 - lam[j]) * run + lam[j] * x)); E[:, j] = run
    return E

E = PF
for _ in range(PASSES): E = ewma(E)
E = E[:, -W:]

def roll_mean(X, ids): return np.stack([X[ids == r].mean(0) for r in np.unique(ids)])
rng = np.random.default_rng(0); P = {g: roll_mean(E[sel[g]], tid[sel[g]]).mean(0) for g in ("hack", "decline")}
idx = np.where(sel["other"])[0]; roll = tid[idx]
P["other"] = np.mean([E[[rng.choice(idx[roll == r]) for r in np.unique(roll)]].mean(0) for _ in range(N_DRAWS)], 0)

COL = {"hack": paper_style.VERMILLION, "decline": paper_style.TEAL, "other": paper_style.NEUTRAL}
MARKS = [128, 96, 64, 48, 32, 24, 16, 12, 8, 6, 4, 3, 2, 1]   # d values dotted along each path
KEYS = [128, 64, 32, 16, 8, 4, 1]
def alpha(dd): return 0.12 + 0.88 * (1 - np.log2(dd) / np.log2(W)) ** 1.3

fig, ax = plt.subplots(figsize=(paper_style.COL + 0.4, paper_style.COL + 0.2))
for g in ("other", "decline", "hack"):
    m = P[g]; ax.plot(m[:, 0], m[:, 1], color=COL[g], lw=0.9, alpha=0.45, solid_joinstyle="round", zorder=2)
    xy = np.stack([m[W - dd] for dd in MARKS])
    ax.scatter(xy[:, 0], xy[:, 1], s=[34 if dd == 1 else 14 for dd in MARKS], c=[to_rgba(COL[g], alpha(dd)) for dd in MARKS],
               edgecolors=[to_rgba(paper_style.INK, 0.6 * alpha(dd)) for dd in MARKS], linewidths=0.4, zorder=3)
    print(f"{g:7s} n_rollouts={len(set(tid[sel[g]])):3d} start ({m[0,0]:6.2f},{m[0,1]:6.2f}) end ({m[-1,0]:6.2f},{m[-1,1]:6.2f})")
u_xy = Z["u_xy"]; R_ = 4
ax.annotate("", xy=tuple(-R_ * u_xy), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=paper_style.INK, lw=0.9, shrinkA=0, shrinkB=0), zorder=4)
ax.text(*(-R_ * 1.25 * u_xy), r"$-u$", ha="center", va="center", fontsize=9, color=paper_style.INK, zorder=4)
ax.axhline(0, color=paper_style.RULE, lw=0.8, zorder=0); ax.axvline(0, color=paper_style.RULE, lw=0.8, zorder=0)
ax.set_xlabel(r"approach to action, $e$"); ax.set_ylabel(r"hack vs. decline, $u_{\perp e}$")
ax.set_ylim(top=12); ax.set_yticks(np.arange(-8, 13, 4)); ax.set_xlim(-3, 9); ax.set_xticks(np.arange(-2, 9, 2))   # headroom so both keys sit inside the axes
leg = ax.legend(handles=[Line2D([], [], ls="none", marker="o", ms=(6 if dd == 1 else 4), mew=0.4, mec=to_rgba(paper_style.INK, 0.6 * alpha(dd)),
                                mfc=to_rgba(paper_style.INK, alpha(dd)), label=str(dd)) for dd in KEYS],
                title="tokens before the tool call", title_fontsize=8, fontsize=8, loc="upper left", bbox_to_anchor=(0, 1.04), ncol=len(KEYS),
                handlelength=0.8, handletextpad=0.3, columnspacing=0.9, borderaxespad=0.4, frameon=False, alignment="left")
ax.add_artist(leg)
ax.legend(handles=[Line2D([], [], color=COL[g], lw=1.2, marker="o", ms=4, mew=0, label=g) for g in ("hack", "decline", "other")],
          loc="upper left", bbox_to_anchor=(0, 0.86), handlelength=1.4)
fig.subplots_adjust(left=0.17, right=0.97, bottom=0.13, top=0.97)
print(paper_style.save(fig, "action_plane_ewma"))
