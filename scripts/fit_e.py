#!/usr/bin/env python
"""Paper-style e: DiffMean of L35 token activations 20-60 tokens before the end of a reasoning
turn vs >200 tokens before (band dropped), matched on position-so-far in 12 log bins, on
turns with NO hack or anti-hack segments, per-rollout equal weighting, fit on train rollouts,
evaluated on val rollouts. Variants: any tool-call end (primary) and submit-call end only.
Reads data/derived/e_token_sample/tokens_{train,val}.safetensors (the
20-60 / >200 token sample built by scripts/build_e_token_sample.py). Also: position-only baseline,
logistic probe, tricky-negative check, g-mean^2, cosines with s and u.
Writes experiments/wrapup-axis-paperstyle/artifacts/e_paper_bf16_L<layer>.safetensors and friends.

The canonical e (src/steering_reasoning/canonical/INDEX.md) is `--layer 35`:
    uv run --with torch --with scikit-learn python scripts/fit_e.py --layer 35
"""
import json, re
from pathlib import Path
import numpy as np, pandas as pd, torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
REPO = Path(__file__).resolve().parents[1]; EXP = REPO/"experiments"; HERE = EXP/"wrapup-axis-paperstyle"; ART = HERE/"artifacts"
TPP = REPO/"data/derived/e_token_sample"; AH = EXP/"anti-hack-labeling/artifacts"; HA = EXP/"hack-action-anchor/artifacts"
from steering_reasoning.data.mistrace import load_mistrace
from steering_reasoning.canonical import load as canon
import argparse
ap = argparse.ArgumentParser(); ap.add_argument("--layer", type=int, default=35, choices=[31, 35]); A = ap.parse_args()
rng = np.random.default_rng(20260913); N_BINS = 12; L = A.layer; SUF = ""; TAG = f"bf16_L{L}"

meta = json.load(open(TPP/"dataset_meta.json"))
tok = pd.read_parquet(REPO/"data/derived/token_series.parquet")
turn_lab = tok.groupby(["task_id", "turn"]).agg(has_hack=("label", lambda x: (x == 1).any()), has_dec=("label", lambda x: (x == 2).any()), msg=("message_index", "first")).reset_index()
T = pd.read_csv(HA/"action_anchor_table.csv"); hack_turn = {r.task_id: int(r.first_hack_edit_turn) for r in T.itertuples() if pd.notna(r.first_hack_edit_turn)}
recs = {r["task_id"]: r for r in load_mistrace("qwen3_5_27b")[0]}
def ends_in_submit(tid, msg):
    return any((tc.get("function") or tc.get("name")) == "submit" for tc in (recs[tid]["messages"][int(msg)].get("tool_calls") or []))
turn_lab["control"] = ~turn_lab.has_hack & ~turn_lab.has_dec & ~turn_lab.apply(lambda r: hack_turn.get(r.task_id) == r.turn, axis=1)
turn_lab["submit"] = turn_lab.apply(lambda r: ends_in_submit(r.task_id, r.msg), axis=1)
flags = {(r.task_id, int(r.turn)): (bool(r.control), bool(r.submit)) for r in turn_lab.itertuples()}
segs_text = {}
def seg_text(tid, seg):
    if tid not in segs_text: segs_text[tid] = [s.get("text") or "" for s in json.load(open(REPO/f"data/captures/qwen35_27b_bf16_token_L35/{tid}.json"))["segments"]]
    return segs_text[tid][seg] if 0 <= seg < len(segs_text[tid]) else ""
CONCL = re.compile(r"\b(submit|finali[sz]e|wrap(ping)? up|i'?m done|that'?s (it|everything)|final (answer|version|solution)|go with|hand (it )?in|move on|let me (just )?submit)\b", re.I)

def load(split):
    d = load_file(TPP/f"tokens_{SUF}{split}.safetensors"); rid = meta["splits"][split]["rollouts"]
    X = d[f"acts_l{L}"].float(); df = pd.DataFrame({k: d[k].numpy() for k in ("label", "rollout_id", "turn_index", "pos_from_start", "dist_to_end", "token_pos")})
    df["task_id"] = [rid[i] for i in df.rollout_id]; df["control"] = [flags.get((t, int(k)), (False, False))[0] for t, k in zip(df.task_id, df.turn_index)]
    df["submit"] = [flags.get((t, int(k)), (False, False))[1] for t, k in zip(df.task_id, df.turn_index)]
    # segment of each token, for the tricky-negative check: dataset token_pos is the offset within the turn;
    # token_series.tok is rollout-wide, so add the turn's first rollout-wide index
    key = tok.set_index(["task_id", "tok"]).seg; start = tok.groupby(["task_id", "turn"]).tok.min()
    df["seg"] = [key.get((t, int(start.get((t, int(k)), 0)) + int(p)), -1) for t, k, p in zip(df.task_id, df.turn_index, df.pos_from_start)]
    return X, df
Xtr, dtr = load("train"); Xva, dva = load("val")
print(f"train {len(dtr)} tokens, control-turn {dtr.control.sum()}, of which submit-ending {(dtr.control & dtr.submit).sum()}; val {len(dva)}, control {dva.control.sum()}, submit {(dva.control & dva.submit).sum()}")

def match(df, edges):
    p = df.pos_from_start.values; b = np.clip(np.digitize(p, edges) - 1, 0, len(edges) - 2); keep = []
    for bi in range(len(edges) - 1):
        pos = np.where((b == bi) & (df.label.values == 1))[0]; neg = np.where((b == bi) & (df.label.values == 0))[0]; n = min(len(pos), len(neg))
        if n: keep += list(rng.choice(pos, n, replace=False)) + list(rng.choice(neg, n, replace=False))
    return np.sort(np.array(keep))
def diffmean(X, df):
    vs = []
    for r in df.rollout_id.unique():
        m = df.rollout_id.values == r; y = df.label.values[m]
        if y.sum() and (1 - y).sum(): vs.append(X[m][y == 1].mean(0) - X[m][y == 0].mean(0))
    v = torch.stack(vs).mean(0); return v / v.norm()
def gmean2(y, s, thr): return float(((s[y == 1] > thr).mean()) * ((s[y == 0] <= thr).mean()))
def best_thr(y, s):
    qs = np.quantile(s, np.linspace(0.01, 0.99, 99)); g = [gmean2(y, s, q) for q in qs]; return qs[int(np.argmax(g))], max(g)
def boot_auc(y, s, groups, B=1000, level=90):
    ug = np.unique(groups); idx = {g: np.where(groups == g)[0] for g in ug}; vals = []
    for _ in range(B):
        ii = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))]);
        if len(np.unique(y[ii])) == 2: vals.append(roc_auc_score(y[ii], s[ii]))
    a = (100 - level) / 2; return float(np.percentile(vals, a)), float(np.percentile(vals, 100 - a))

g = lambda p, k: safe_open(str(p), "pt").get_tensor(k).float().flatten()
# reference directions for the cosine table: the canonical u and s
refs = {"u": canon("u"), "s": canon("s")}
refs = {k: v / v.norm() for k, v in refs.items()}
results, dirs = [], {}
for variant, sel in [("any_call", lambda d: d.control.values), ("submit_only", lambda d: (d.control & d.submit).values)]:
    tr = dtr[sel(dtr)].reset_index(drop=True); va = dva[sel(dva)].reset_index(drop=True); Xt, Xv = Xtr[sel(dtr)], Xva[sel(dva)]
    edges = np.quantile(np.log1p(tr.pos_from_start), np.linspace(0, 1, N_BINS + 1)); edges = np.expm1(edges); edges[0] = 0; edges[-1] = np.inf
    itr, iva = match(tr, edges), match(va, edges); mtr, mva = tr.iloc[itr].reset_index(drop=True), va.iloc[iva].reset_index(drop=True); Xmt, Xmv = Xt[itr], Xv[iva]
    print(f"[{variant}] matched train {len(mtr)} ({mtr.rollout_id.nunique()} rollouts, {mtr.groupby(['rollout_id','turn_index']).ngroups} turns), val {len(mva)} ({mva.rollout_id.nunique()} rollouts, {mva.groupby(['rollout_id','turn_index']).ngroups} turns)")
    if len(mva) == 0 or mva.label.nunique() < 2: print("  skipped: no matched val"); continue
    e = diffmean(Xmt, mtr); dirs[f"e_paper_{variant}"] = e
    # within-turn paired variant: each turn contributes (mean of its positives - mean of its nearest eligible negatives,
    # i.e. those within 260..201 tokens of the end); turn diffs averaged within rollout, then across rollouts.
    def within_turn(X, d, nearest=True):
        per_roll = []
        for r in d.rollout_id.unique():
            diffs = []
            for k in d[d.rollout_id == r].turn_index.unique():
                m = (d.rollout_id.values == r) & (d.turn_index.values == k); y = d.label.values[m]; dist = d.dist_to_end.values[m]
                neg = (y == 0) & ((dist <= 260) if nearest else True)
                if (y == 1).sum() and neg.sum(): diffs.append(X[m][y == 1].mean(0) - X[m][neg].mean(0))
            if diffs: per_roll.append(torch.stack(diffs).mean(0))
        v = torch.stack(per_roll).mean(0); return v / v.norm()
    e_wt = within_turn(Xt, tr, nearest=True); e_wa = within_turn(Xt, tr, nearest=False)
    dirs[f"e_withinturn_nearest_{variant}"] = e_wt; dirs[f"e_withinturn_all_{variant}"] = e_wa
    yv, gv = mva.label.values, mva.rollout_id.values
    models = {"e_paper (diffmean)": (Xmv @ e).numpy(), "e_withinturn_nearest": (Xmv @ e_wt).numpy(), "e_withinturn_all": (Xmv @ e_wa).numpy(), **{k: (Xmv @ refs[k]).numpy() for k in ("e_ctrl_drift", "term_diffmean") if k in refs}, "s": (Xmv @ refs["s"]).numpy(), "-u": (-(Xmv @ refs["u"])).numpy()}
    P = lambda d: np.column_stack([np.log1p(d.pos_from_start), np.log1p(d.token_pos), d.turn_index])
    models["position baseline (logreg)"] = LogisticRegression(max_iter=2000).fit(P(mtr), mtr.label).decision_function(P(mva))
    mu_, sd_ = Xmt.mean(0), Xmt.std(0) + 1e-6
    lr = LogisticRegression(C=1e-4, max_iter=3000).fit(((Xmt - mu_) / sd_).numpy(), mtr.label); models["logreg probe (C=1e-4)"] = lr.decision_function(((Xmv - mu_) / sd_).numpy())
    dirs[f"logreg_{variant}"] = torch.tensor(lr.coef_[0] / sd_.numpy(), dtype=torch.float32)
    # tricky negatives: val negatives whose segment text sounds like a conclusion
    tricky = (yv == 0) & np.array([bool(CONCL.search(seg_text(t, int(sg)))) for t, sg in zip(mva.task_id, mva.seg)])
    for name, sv in models.items():
        st = {"e_paper (diffmean)": (Xmt @ e).numpy()}.get(name); thr, _ = best_thr(yv, sv)
        auc = roc_auc_score(yv, sv); lo, hi = boot_auc(yv, sv, gv)
        auc_tricky = roc_auc_score(np.r_[np.ones((yv == 1).sum()), np.zeros(tricky.sum())], np.r_[sv[yv == 1], sv[tricky]]) if tricky.sum() >= 20 else np.nan
        results.append(dict(variant=variant, model=name, val_auroc=auc, ci90_lo=lo, ci90_hi=hi, val_gmean2_at_best_thr=gmean2(yv, sv, thr), fpr_tricky_at_thr=float((sv[tricky] > thr).mean()) if tricky.sum() else np.nan, fpr_all_neg_at_thr=float((sv[yv == 0] > thr).mean()), auroc_vs_tricky=auc_tricky, n_tricky=int(tricky.sum())))
        print(f"  {name:30s} AUROC {auc:.3f} [{lo:.3f},{hi:.3f}]  gmean2 {results[-1]['val_gmean2_at_best_thr']:.3f}  tricky FPR {results[-1]['fpr_tricky_at_thr']:.2f} vs all-neg FPR {results[-1]['fpr_all_neg_at_thr']:.2f} (n_tricky {tricky.sum()})")
df = pd.DataFrame(results); df.to_csv(ART/f"results_val_{TAG}.csv", index=False)
cos = {f"cos({a},{b})": float(dirs[a] @ (dirs[b] if b in dirs else refs[b])) for a in dirs if a.startswith("e_") for b in list(refs) + [k for k in dirs if k != a and k.startswith("e_")]}
for k in list(dirs):
    if k.startswith("logreg"): cos[f"cos({k},e_paper_any_call)"] = float(dirs[k] / dirs[k].norm() @ dirs["e_paper_any_call"])
print({k: round(v, 3) for k, v in cos.items()})
save_file(dirs, str(ART/f"e_paper_{TAG}.safetensors"), metadata={"recipe": f"DiffMean 20-60 vs >200 tokens before turn end, band dropped, 12-bin position match, control turns only, per-rollout weighting, train rollouts; bf16 capture L{L}"})
with open(ART/f"results_val_{TAG}.md", "w") as f:
    f.write(f"# Paper-style e: val results, {TAG} (8 held-out rollouts, position-matched)\n\n| variant | model | AUROC | 90% CI | g-mean² | FPR tricky | FPR all neg | AUROC vs tricky | n tricky |\n|---|---|---|---|---|---|---|---|---|\n")
    for r in results: f.write(f"| {r['variant']} | {r['model']} | {r['val_auroc']:.3f} | [{r['ci90_lo']:.3f}, {r['ci90_hi']:.3f}] | {r['val_gmean2_at_best_thr']:.3f} | {r['fpr_tricky_at_thr']:.2f} | {r['fpr_all_neg_at_thr']:.2f} | {r['auroc_vs_tricky']:.3f} | {r['n_tricky']} |\n")
    f.write("\n| cosine | value |\n|---|---|\n" + "".join(f"| {k} | {v:+.3f} |\n" for k, v in cos.items()))
json.dump(cos, open(ART/f"cosines_{TAG}.json", "w"), indent=1); print("wrote", ART/f"results_val_{TAG}.md")
