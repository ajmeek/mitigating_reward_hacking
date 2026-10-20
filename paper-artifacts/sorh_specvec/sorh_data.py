"""Every School of Reward Hacks arm of record on the 309 held-out natural-language prompts, from the per-item CSVs:
Matthew's L35 runs (GLM 5 judge), his L31 run of record (gpt-5.5 judge; kept separate), and the spec-vector /
u-addition picks. Writes sorh_all_arms.csv beside this file: arm, label, family, judge, n, hack, degen, hc, p_vs_base.
    uv run python paper-artifacts/sorh_specvec/sorh_data.py
"""
import json, math, os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
OV = f"{ROOT}/experiments/opt-vector-sorh/artifacts"; SV = f"{ROOT}/experiments/spec-vector-sorh"
IDS = set(json.load(open(f"{SV}/artifacts/ids_heldout309.json"))["ids"])
# (per_item file, arm, label, family, judge, base arm's per_item file)
L35C, L35A, L31, OURS = f"{OV}/l35_clamp/eval/per_item.csv", f"{OV}/l35_add/eval/per_item.csv", f"{OV}/eval/per_item.csv", f"{SV}/artifacts/eval/per_item.csv"
ARMS = [
    (L35C, "base", "baseline", "baseline", "GLM-5", L35C),
    (L35C, "v3c_cl35_t-3", r"opt cap $-$3", "opt cap", "GLM-5", L35C),
    (L35C, "v3b_cl35_t-15", r"opt$_{\mathrm{ref}}$ cap $-$15", "opt cap", "GLM-5", L35C),
    (L35C, "v3b_cl35_t-25", r"opt$_{\mathrm{ref}}$ cap $-$25", "opt cap", "GLM-5", L35C),
    (L35A, "v3c_k20", "opt add 20", "opt add", "GLM-5", L35C),
    (L35A, "v3b_k24", r"opt$_{\mathrm{ref}}$ add 24", "opt add", "GLM-5", L35C),
    (L35A, "v3b_k28", r"opt$_{\mathrm{ref}}$ add 28", "opt add", "GLM-5", L35C),
    (OURS, "vspec_f40", r"$v_{\mathrm{spec}}$ floor $+40$", "spec floor", "GLM-5", L35C),
    (OURS, "vspec_a24", r"$v_{\mathrm{spec}}$ add $+24$", "spec add", "GLM-5", L35C),
    (OURS, "vspec_a32", r"$v_{\mathrm{spec}}$ add $+32$", "spec add", "GLM-5", L35C),
    (OURS, "u_an24", r"$u$ add $-$24", "u add", "GLM-5", L35C),
    (L31, "base", "baseline (GPT-5.5 judge)", "baseline", "GPT-5.5", L31),
    (L31, "u_cl_t-20", r"$u$ cap $-$20 (GPT-5.5 judge)", "u cap", "GPT-5.5", L31),
]
DOSE = {"v3c_cl35_t-3": -3, "v3b_cl35_t-15": -15, "v3b_cl35_t-25": -25, "v3c_k20": 20, "v3b_k24": 24, "v3b_k28": 28,
        "vspec_f40": 40, "vspec_a24": 24, "vspec_a32": 32, "u_an24": -24, "u_cl_t-20": -20}


def labels(per_item, arm):
    d = pd.read_csv(per_item); x = d[(d.arm == arm) & (d.id.isin(IDS)) & (d.graded == 1)]
    return {r.id: (str(r.hack) == "True", str(r.degenerate) == "True") for r in x.itertuples()}


def mcnemar(b, c):
    n = b + c; k = min(b, c)
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


rows = []
for per, arm, label, fam, judge, basefile in ARMS:
    L = labels(per, arm); B = labels(basefile, "base")
    common = [i for i in L if i in B]
    b_only = sum(B[i][0] and not L[i][0] for i in common); a_only = sum(L[i][0] and not B[i][0] for i in common)
    rows.append(dict(arm=arm, label=label, family=fam, judge=judge, dose=DOSE.get(arm, ""), n=len(L),
                     hack=sum(h for h, _ in L.values()), degen=sum(g for _, g in L.values()),
                     hc=sum((not h) and (not g) for h, g in L.values()),
                     p_vs_base=("" if arm == "base" else f"{mcnemar(b_only, a_only):.2g}")))
df = pd.DataFrame(rows); df.to_csv(os.path.join(HERE, "sorh_all_arms.csv"), index=False)
print(df.to_string(index=False))
