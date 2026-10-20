"""Discrete-time hazard monitor for the first hack edit, comparing trailing-window summaries of h.u.

Unit: every reasoning turn that ends in a tool call (turn_paths_last512_bf16.npz: bf16 L35 capture,
canonical u, last 512 reasoning tokens per turn). Event: the turn whose tool call is the rollout's
first HACK-judged edit (hack-action-anchor edit judge, adjudications applied; scripts/edits.py); turns after it leave the risk set; the 2
rollouts without a hack edit are censored at their last turn.
Features, from the last K reasoning tokens before the call (all available tokens if the turn is
shorter), K in KS: mean_K and max_K of h.u. Variant --gap=G drops the last G tokens first
(features from tokens [-G-K, -G)), to ask whether the signal is only the moment of commitment.
Controls: turn index, log(1 + reasoning tokens in earlier turns), submit calls in earlier turns.
Models (logistic, standardised in-fold): controls; controls + mean_K; controls + max_K;
controls + both. Evaluation: leave-one-rollout-out (42 folds) pooled out-of-fold AUC and Brier,
out-of-fold log-likelihood; cluster bootstrap over rollouts for AUC(max) - AUC(mean); in-sample
likelihood-ratio tests of each summary over the other. Also train -> val+test (u is fit on train).

    uv run --with torch --with statsmodels --with scikit-learn python experiments/prehack-u-profiles/scripts/hazard_mean_vs_max.py [--gap=0]
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

REPO = str(Path(__file__).resolve().parents[3]); HA = f"{REPO}/experiments/hack-action-anchor"; OUT = f"{REPO}/experiments/prehack-u-profiles/artifacts"
GAP = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--gap=")), 0))
KS = (5, 10, 20, 50, 128, 512 - GAP); NB = 2000

Z = np.load(f"{HA}/artifacts/turn_paths_last512_bf16.npz", allow_pickle=True)
U = Z["proj"] @ Z["u_xy"]; U = U[:, : U.shape[1] - GAP] if GAP else U        # [turns, 512 - GAP], NaN-padded at the front
sys.path.insert(0, f"{REPO}/experiments/prehack-u-profiles/scripts")
from edits import first_hack_turn  # noqa: E402
first = first_hack_turn()
D = pd.DataFrame({"task_id": Z["task_id"], "split": Z["split"], "turn": Z["turn"], "n_tokens": Z["n_tokens"],
                  "submit": ["submit" in str(t) for t in Z["tool_detail"]]})
D["row"] = np.arange(len(D)); D = D.sort_values(["task_id", "turn"])
D["event"] = [int(first.get(t, -1) == k) for t, k in zip(D.task_id, D.turn)]
D = D[[k <= first.get(t, 10 ** 9) for t, k in zip(D.task_id, D.turn)]].copy()            # risk set
D = D[D.n_tokens - GAP >= 1].copy()                                                       # need at least one token left after the gap
g = D.groupby("task_id")
D["clock"] = D.turn; D["log_prior_tokens"] = np.log1p(g.n_tokens.cumsum() - D.n_tokens); D["prior_submits"] = g.submit.cumsum() - D.submit
for K in KS:
    W = U[D.row.to_numpy(), -K:]
    D[f"mean_{K}"] = np.nanmean(W, 1); D[f"max_{K}"] = np.nanmax(W, 1)
CTRL = ["clock", "log_prior_tokens", "prior_submits"]
print(f"gap={GAP}: {len(D)} turns at risk, {D.event.sum()} events, {D.task_id.nunique()} rollouts "
      f"(held-out {D[D.split != 'train'].event.sum()} events)")

def fit_predict(tr, te, cols):
    Xtr, Xte = tr[cols].to_numpy(float), te[cols].to_numpy(float); m, s = Xtr.mean(0), Xtr.std(0) + 1e-9
    clf = LogisticRegression(C=1e6, max_iter=5000).fit((Xtr - m) / s, tr.event)
    return clf.predict_proba((Xte - m) / s)[:, 1]

def loro(cols):
    p = np.empty(len(D))
    for t in D.task_id.unique():
        te = (D.task_id == t).to_numpy(); p[te] = fit_predict(D[~te], D[te], cols)
    return p

def boot_auc_diff(pa, pb):
    ids = D.task_id.to_numpy(); u = np.unique(ids); idx = {t: np.where(ids == t)[0] for t in u}; rng = np.random.default_rng(0); out = []
    for _ in range(NB):
        rows = np.concatenate([idx[t] for t in rng.choice(u, len(u))]); y = D.event.to_numpy()[rows]
        if y.min() == y.max(): continue
        out.append(roc_auc_score(y, pb[rows]) - roc_auc_score(y, pa[rows]))
    return np.percentile(out, [2.5, 97.5]), np.mean(np.array(out) <= 0)

def lrt(small, big):
    y = D.event.to_numpy(); f = lambda c: sm.Logit(y, sm.add_constant((D[c] - D[c].mean()) / D[c].std())).fit(disp=0).llf
    return 2 * (f(big) - f(small))

y = D.event.to_numpy(); rows = []; tr, ho = D.split == "train", D.split != "train"
p_ctrl = loro(CTRL)
rows.append(dict(K="-", model="controls", loro_auc=roc_auc_score(y, p_ctrl), loro_brier=brier_score_loss(y, p_ctrl),
                 loro_ll=np.sum(y * np.log(p_ctrl) + (1 - y) * np.log(1 - p_ctrl)),
                 holdout_auc=roc_auc_score(D[ho].event, fit_predict(D[tr], D[ho], CTRL))))
for K in KS:
    P = {}
    for name, cols in (("mean", CTRL + [f"mean_{K}"]), ("max", CTRL + [f"max_{K}"]), ("both", CTRL + [f"mean_{K}", f"max_{K}"])):
        p = P[name] = loro(cols); ph = fit_predict(D[tr], D[ho], cols)
        rows.append(dict(K=K, model=f"controls + {name}", loro_auc=roc_auc_score(y, p), loro_brier=brier_score_loss(y, p),
                         loro_ll=np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)), holdout_auc=roc_auc_score(D[ho].event, ph)))
    ci, pneg = boot_auc_diff(P["mean"], P["max"])
    rows[-1].update(auc_max_minus_mean=roc_auc_score(y, P["max"]) - roc_auc_score(y, P["mean"]), diff_lo=ci[0], diff_hi=ci[1], p_diff_le0=pneg,
                    lrt_max_over_mean=lrt(CTRL + [f"mean_{K}"], CTRL + [f"mean_{K}", f"max_{K}"]),
                    lrt_mean_over_max=lrt(CTRL + [f"max_{K}"], CTRL + [f"mean_{K}", f"max_{K}"]))
R = pd.DataFrame(rows); R.to_csv(f"{OUT}/hazard_mean_vs_max_gap{GAP}.csv", index=False)
pd.set_option("display.width", 250); print(R.round(3).to_string(index=False))
