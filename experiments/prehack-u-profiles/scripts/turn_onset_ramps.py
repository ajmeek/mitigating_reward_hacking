"""Turn-level projection onto u (L35, bf16, canonical) aligned at two onset anchors:
  verbalization  first assistant message with a judge-labelled hack segment (GLM-5 sentence judge)
  edit           first HACK-judged code edit (hack-action-anchor/artifacts/action_anchor_table.csv)
Per turn, the bf16 segment means (data/captures/qwen35_27b_bf16_segmeans_L35), centred on the bf16 mu,
are projected onto u and pooled two ways: token-weighted mean over the turn's segments (= mean over
its reasoning tokens) and max over segments. Each rollout's per-turn series is z-scored within the
rollout (removes per-rollout offsets; the scale prior ramp results used). Rollouts need >= 8 turns before the anchor.
Ramp = z(-1) - z(-8), 95% bootstrap over rollouts; held-out (val + test) reported separately,
since u was fit on train rollouts.

Outputs (artifacts/): turn_onset_ramps.{png,pdf}, turn_onset_ramps.csv (mean curve per anchor x
pooling x offset, z and raw h.u), turn_onset_ramps_stats.csv.

    uv run --with torch python experiments/prehack-u-profiles/scripts/turn_onset_ramps.py
"""
import glob, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from safetensors import safe_open
import matplotlib.pyplot as plt

REPO = str(Path(__file__).resolve().parents[3]); sys.path.insert(0, f"{REPO}/src"); sys.path.insert(0, f"{REPO}/paper-artifacts")
from steering_reasoning.canonical import load  # noqa: E402
import paper_style  # noqa: E402
paper_style.use()
OUT = f"{REPO}/experiments/prehack-u-profiles/artifacts"; CAP = f"{REPO}/data/captures/qwen35_27b_bf16_segmeans_L35"
PRE, POST, B = 8, 3, 2000
split = {t: s for s, ts in json.load(open(f"{REPO}/src/steering_reasoning/data/splits.json"))["impossible"].items() for t in ts}
with safe_open(f"{REPO}/experiments/probe-refit-bf16/artifacts/mu_bf16_L35.safetensors", "pt") as f: mu = f.get_tensor("mu").float().numpy()
DIRS = load("u").numpy()[None].astype(np.float32); DNAME = ["u"]
EDGE = pd.read_csv(f"{REPO}/experiments/hack-action-anchor/artifacts/action_anchor_table.csv").set_index("task_id")["first_hack_edit_msg"]

series = {}   # task_id -> dict(msgs, mean [turns, dirs], max [turns, dirs], anchors)
for mf in sorted(glob.glob(f"{CAP}/*.meta.json")):
    M = json.load(open(mf)); tid = M["task_id"]; X = np.load(mf.replace(".meta.json", ".npy")).astype(np.float32)
    P = (X - mu) @ DIRS.T                                    # [segments, dirs]
    seg = pd.DataFrame(M["segments"]); seg[[f"p{j}" for j in range(len(DIRS))]] = P
    msgs = sorted(seg.message_index.unique()); g = seg.groupby("message_index")
    pc = [f"p{j}" for j in range(len(DIRS))]
    mean = np.stack([np.average(g.get_group(m)[pc].to_numpy(), axis=0, weights=g.get_group(m).n_tokens.to_numpy()) for m in msgs])
    mx = np.stack([g.get_group(m)[pc].to_numpy().max(0) for m in msgs])
    verb = seg.loc[seg.judge_label.astype(bool), "message_index"]
    series[tid] = dict(msgs=np.array(msgs), mean=mean, max=mx,
                       verbalization=int(verb.min()) if len(verb) else None,
                       edit=int(EDGE[tid]) if tid in EDGE and pd.notna(EDGE[tid]) else None)

def aligned(anchor, pool, z=True):
    rows, ids = [], []
    for tid, S in series.items():
        a = S[anchor]
        if a is None: continue
        if a not in S["msgs"]: a = S["msgs"][S["msgs"] <= a].max()      # edit message without its own segments: last turn before it
        i = int(np.where(S["msgs"] == a)[0][0])
        if i < PRE: continue
        Y = S[pool]; Y = (Y - Y.mean(0)) / Y.std(0) if z else Y
        W = np.full((PRE + POST + 1, len(DIRS)), np.nan); hi = min(len(Y), i + POST + 1); W[: hi - (i - PRE)] = Y[i - PRE: hi]
        rows.append(W); ids.append(tid)
    return np.stack(rows), np.array(ids)

off = np.arange(-PRE, POST + 1); rng = np.random.default_rng(0); recs, stats = [], []
fig, axes = plt.subplots(2, 2, figsize=(paper_style.FULL, 4.6), sharex=True, sharey="row")
for r, anchor in enumerate(("verbalization", "edit")):
    for c, pool in enumerate(("mean", "max")):
        A, ids = aligned(anchor, pool); Araw, _ = aligned(anchor, pool, z=False); ax = axes[r, c]
        for j, dn in enumerate(DNAME):
            m = np.nanmean(A[:, :, j], 0); bs = np.stack([np.nanmean(A[rng.integers(0, len(A), len(A)), :, j], 0) for _ in range(B)])
            lo, hi = np.nanpercentile(bs, 2.5, 0), np.nanpercentile(bs, 97.5, 0)
            ax.fill_between(off, lo, hi, color=paper_style.VERMILLION, alpha=0.18, lw=0)
            ax.plot(off, m, color=paper_style.VERMILLION, lw=1.4, marker="o", ms=3)
            for k, o in enumerate(off):
                recs.append(dict(anchor=anchor, pool=pool, direction=dn, offset=o, z_mean=m[k], z_lo=lo[k], z_hi=hi[k],
                                 raw_mean=np.nanmean(Araw[:, k, j]), n=int(np.isfinite(A[:, k, j]).sum())))
            for sub, keep in (("all", np.ones(len(ids), bool)), ("held-out", np.array([split[t] != "train" for t in ids]))):
                d = A[keep, PRE - 1, j] - A[keep, 0, j]
                bsd = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]
                stats.append(dict(anchor=anchor, pool=pool, direction=dn, subset=sub, n=int(keep.sum()),
                                  ramp=d.mean(), lo=np.percentile(bsd, 2.5), hi=np.percentile(bsd, 97.5),
                                  onset_minus_m8=np.nanmean(A[keep, PRE, j] - A[keep, 0, j])))
        ax.axhline(0, color=paper_style.RULE, lw=0.8, zorder=0); ax.axvline(0, color=paper_style.RULE, lw=0.8, zorder=0)
        ax.set_title(f"first hack {'verbalization' if anchor == 'verbalization' else 'edit'}, {pool} over turn (n = {len(ids)})", loc="left", fontsize=8.5)
        if c == 0: ax.set_ylabel("$h \\cdot u$, z within rollout")
        if r == 1: ax.set_xlabel("turns relative to anchor")
fig.tight_layout()
for ext in ("png", "pdf"): fig.savefig(f"{OUT}/turn_onset_ramps.{ext}")
pd.DataFrame(recs).to_csv(f"{OUT}/turn_onset_ramps.csv", index=False)
S = pd.DataFrame(stats); S.to_csv(f"{OUT}/turn_onset_ramps_stats.csv", index=False)
pd.set_option("display.width", 200)
print(S.round(2).to_string(index=False))
