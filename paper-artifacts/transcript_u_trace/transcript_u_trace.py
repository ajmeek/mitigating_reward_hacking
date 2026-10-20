"""Per-token projection onto u (canonical bf16 u, layer 35, raw h.u) over one held-out hacked
transcript, with the GLM-5 judge's hack segments shaded and the edit judge's hack edits marked.
Top: the whole rollout; below: a window around the first hack edit. Each panel draws the trailing
mean of the last 100 tokens with a band of +-1 sd over the same window (only tokens up to and
including the current one). Adjacent hack segments are merged into one band. x is the reasoning-token index across the
rollout's turns; faint vertical lines are turn boundaries.

Sources: data/captures/qwen35_27b_bf16_token_L35/<task>.safetensors (activations),
data/derived/token_series.parquet (per-token judge labels),
experiments/hack-action-anchor/data/edit_judgments.json (edit judge; an edit sits at the end of
its turn's reasoning).

    uv run python paper-artifacts/transcript_u_trace/transcript_u_trace.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from safetensors import safe_open

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/src")
import paper_style  # noqa: E402
from steering_reasoning.canonical import load  # noqa: E402

FLAGS = {a.split("=")[0][2:]: (a.split("=") + [""])[1] for a in sys.argv[1:] if a.startswith("--")}
POS = [a for a in sys.argv[1:] if not a.startswith("--")]
TASK = POS[0] if POS else "lcbhard_22"         # validation split
OUTDIR = POS[1] if len(POS) > 1 else None      # default: beside this script
# Paper figure (default): each panel is the trailing mean of the last n tokens with a +-1 sd band over
# the same window, n = 100 for both panels (--ntop / --nzoom override). --lines gives the earlier
# EWMA / trailing-mean lines without a band, with the per-task settings below.
BAND_MODE = "lines" not in FLAGS
LINE = paper_style.TEAL if "teal" in FLAGS else None   # --teal: series in teal instead of ink
# zoom window in reasoning tokens; (None, None) = the reasoning turn that ends in the first hack edit
# (at most its last 2,500 tokens) through 150 tokens past the edit
ZOOM = {"lcbhard_72": (2500, None), "lcbhard_22": (28200, None)}.get(TASK, (None, None))
N_TOP, N_ZOOM = {"lcbhard_22": (50, 20)}.get(TASK, (20, 10))   # trailing window lengths (top panel, zoom panels)
if BAND_MODE:
    N_TOP, N_ZOOM = 100, 100
N_TOP, N_ZOOM = int(FLAGS.get("ntop") or N_TOP), int(FLAGS.get("nzoom") or N_ZOOM)

EWMA_HL = {"lcbhard_22": 20}.get(TASK)   # if set, the first zoom panel is an EWMA with this half-life instead of a trailing mean
TOP_EWMA_HL = {"lcbhard_22": 50}.get(TASK)   # if set, the top panel is an EWMA with this half-life instead of a trailing mean

paper_style.use()
V, G, INK = paper_style.VERMILLION, paper_style.NEUTRAL, paper_style.INK
# opaque tints rather than alpha, so the figure reads the same on any background
BAND, RULE = paper_style._mix(V, "#ffffff", 0.8), paper_style._mix(G, "#ffffff", 0.6)

CAP = f"{ROOT}/data/captures/qwen35_27b_bf16_token_L35"; LI = json.load(open(f"{CAP}/manifest.json"))["layers"].index(35)
with safe_open(f"{CAP}/{TASK}.safetensors", "pt") as f:
    acts = f.get_tensor("token_acts")[:, LI].float()
    turn = f.get_tensor("token_turn").numpy()
p = (acts @ load("u").float()).numpy()
tok = pd.read_parquet(f"{ROOT}/data/derived/token_series.parquet")
tok = tok[tok.task_id == TASK].sort_values("tok")
assert len(tok) == len(p) and (tok.turn.to_numpy() == turn).all()
hack_tok = tok.label.to_numpy() == 1
turn_end = {k: int(np.where(turn == k)[0][-1]) for k in np.unique(turn)}
edits = [turn_end[int(e["capture_turn"])] for e in json.load(open(f"{ROOT}/experiments/hack-action-anchor/data/edit_judgments.json"))
         if e["task_id"] == TASK and e["label"] == "HACK"]
first = min(edits)
hack_turn_start = int(np.where(turn == turn[first])[0][0])
lo = ZOOM[0] if ZOOM[0] is not None else max(hack_turn_start - 100, first - 2500, 0)
hi = ZOOM[1] or min(len(p), first + 150)
x = np.arange(len(p))
CENTRED = "centred" in FLAGS   # --centred: windows centred on the token (n/2 either side) instead of trailing
trail = lambda n, fn: getattr(pd.Series(p).rolling(n, min_periods=1, center=CENTRED), fn)().to_numpy()
# --iqr: band is the 25th-75th percentile of the window; --q90: the 5th-95th; otherwise mean +- sd
# paper default: the central 90% band (5th-95th percentile); --sd for mean +- sd, --iqr for 25th-75th
IQR = "sd" not in FLAGS and "maxline" not in FLAGS
MAXLINE = "maxline" in FLAGS   # --maxline: no band; a second line for the max over the same window
QLO, QHI = (0.25, 0.75) if "iqr" in FLAGS else (0.05, 0.95)
quant = lambda n, q: pd.Series(p).rolling(n, min_periods=1, center=CENTRED).quantile(q).to_numpy()


def spans(mask):
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return zip(np.where(d == 1)[0], np.where(d == -1)[0])


def merged_spans(mask, gap=5):
    """spans() with runs separated by at most `gap` tokens joined into one."""
    out = []
    for s, e in spans(mask):
        if out and s - out[-1][1] <= gap:
            out[-1][1] = e
        else:
            out.append([s, e])
    return out


def decorate(ax, a, b, top):
    for s, e in merged_spans(hack_tok):
        if e > a and s < b:
            ax.axvspan(s, e, color=BAND, lw=0, zorder=0)
    for pos in edits:
        if a <= pos < b:
            ax.plot(pos, top, marker="v", ms=5, mfc=V, mec=INK, mew=0.6, clip_on=False, zorder=5)
    ax.set_xlim(a, b)


ewma = lambda hl: pd.Series(p).ewm(halflife=hl).mean().to_numpy()
top = (ewma(TOP_EWMA_HL), f"EWMA, half-life {TOP_EWMA_HL}") if TOP_EWMA_HL else (trail(N_TOP, "mean"), f"mean of last {N_TOP}")
PANELS = [(*top, (-25, 5), (0, len(p))),
          (*((ewma(EWMA_HL), f"EWMA, half-life {EWMA_HL}") if EWMA_HL else (trail(N_ZOOM, "mean"), f"mean of last {N_ZOOM}")), (-20, 5), (lo, hi)),
          ]
fig, axes = plt.subplots(len(PANELS), 1, figsize=(paper_style.FULL, 1.2 * len(PANELS)), gridspec_kw=dict(hspace=0.6))
if BAND_MODE:
    stat = "mean and max" if MAXLINE else ("mean and central 90%" if QLO == 0.05 else "mean and IQR") if IQR else "mean $\\pm$ sd"
    wlab = (lambda n: f"{stat} over {n} tokens centred on the token") if CENTRED else (lambda n: f"{stat} of last {n} tokens")
    PANELS = [(trail(N_TOP, "mean"), wlab(N_TOP), (-27, 20) if MAXLINE else (-27, 6), (0, len(p))),
              (trail(N_ZOOM, "mean"), "" if N_ZOOM == N_TOP else wlab(N_ZOOM), (-25, 20) if MAXLINE else (-25, 10), (lo, hi))]   # label once when both panels use the same window
    SD = {0: trail(N_TOP, "std"), 1: trail(N_ZOOM, "std")}
    if IQR:
        LOHI = {0: (quant(N_TOP, QLO), quant(N_TOP, QHI)), 1: (quant(N_ZOOM, QLO), quant(N_ZOOM, QHI))}
for i, (ax, (y, label, ylim, (a, b))) in enumerate(zip(axes, PANELS)):
    if BAND_MODE:
        sd = np.nan_to_num(SD[i])
        blo, bhi = LOHI[i] if IQR else (y - sd, y + sd)
        if not MAXLINE:   # y range from the band itself, so no panel clips it
            pad = 0.15 * (np.nanmax(bhi[a:b]) - np.nanmin(blo[a:b]))
            ylim = (np.nanmin(blo[a:b]) - pad, np.nanmax(bhi[a:b]) + pad)
        if MAXLINE:
            mx = trail(N_TOP if i == 0 else N_ZOOM, "max")
            ax.plot(x[a:b], mx[a:b], color=paper_style.TEAL, lw=0.8, zorder=2)
        else:
          ax.fill_between(x[a:b], blo[a:b], bhi[a:b], color=paper_style._mix(LINE or INK, "#ffffff", 0.75 if LINE else 0.82), lw=0, zorder=1)
    ax.plot(x[a:b], y[a:b], color=LINE or INK, lw=0.8, zorder=2)
    ax.set_ylim(*ylim)
    decorate(ax, a, b, ylim[1])
    ax.text(0.005, 1.0, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=7, color=paper_style.MUTED)
axes[0].axvspan(lo, hi, facecolor="none", edgecolor=paper_style.MUTED, lw=0.8, ls=(0, (2, 2)))
axes[-1].set_xlabel("reasoning token")
fig.supylabel("projection onto $u$", x=0.02, fontsize=9)
axes[0].legend(handles=([Line2D([], [], color=INK, lw=0.8, label="mean"), Line2D([], [], color=paper_style.TEAL, lw=0.8, label="max")] if MAXLINE else []) + [Patch(color=BAND, lw=0, label="hack verbalization"),
                        Line2D([], [], ls="none", marker="v", ms=5, mfc=V, mec=INK, mew=0.6, label="hack action")],
               frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=4 if MAXLINE else 2, fontsize=7,
               borderaxespad=0.1, handlelength=1.2, columnspacing=1.2)
fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.14)
stem = "transcript_u_trace" if OUTDIR is None else f"transcript_u_trace_{TASK}" + (f"_band_n{N_TOP}_{N_ZOOM}" if BAND_MODE else "") + ("_teal" if LINE else "") + ("_centred" if CENTRED else "") + (("_q90" if QLO == 0.05 else "_iqr") if IQR else "") + ("_maxline" if MAXLINE else "")
print("wrote", paper_style.save(fig, stem, where=OUTDIR))
