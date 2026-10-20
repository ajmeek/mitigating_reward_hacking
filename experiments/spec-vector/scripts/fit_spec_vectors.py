#!/usr/bin/env python
"""Assemble the spec vectors from extract_pool_stats.py's outputs and run the geometry checks (plan Steps 3–4).

Every vector is a per-rollout DiffMean at L35 (each rollout one vote, then the mean over rollouts,
then unit-normalised), the recipe that produced u. Positive pole = toward the spec.

Pools are sums of COMBOS (source|match_kind|v1_label|v2_label) over the spec-arm rollouts:
  v1_cites          reasoning, id, v1 CITES                 vs rest of the rollout's reasoning   (PRIMARY, the simple experiment)
  v2_cites          reasoning, id, v2 CITES (applied only)  vs rest
  v2_restates       reasoning, id, v2 RESTATES              vs rest
  v2_cites_restates reasoning, id, v2 CITES + RESTATES      vs rest
  v1_cites_vs_not   reasoning, id, v1 CITES                 vs reasoning, id, v1 NOT        (within the regex net)
  v2_cites_vs_restates                                       applied vs restated
  v1_cites_idword   reasoning, id + word, v1 CITES          vs rest
  v1_cites_reasvis  reasoning + visible, id, v1 CITES       vs rest of reasoning + visible
  specarm_vs_base   spec-arm reasoning mean − base-arm reasoning mean, paired by (task, seed)
  text_spec         spec suffix tokens (system prompt, turn 0) − MisTrace prompt tokens, per spec rollout
  text_principles   the principles list only − MisTrace;  text_final: the FINAL INSTRUCTIONS paragraph − MisTrace
  text_vs_reasoning spec suffix tokens − the rollout's reasoning mean
"rest" = the rollout's reasoning rows minus the pool's rows minus the dropped principle-ID tokens
(never-computed / <|im_end|> rows excluded). text_*: the MisTrace part excludes the tool-schema JSON
and the <|im_start|>system markup (pooled separately as sys_tools).

Checks written to artifacts/fit_report.md: n rollouts / tokens per pool, raw diff norms, cosines with
u / s / e and between vectors, leave-one-task-out cosine, and projection tables (mean, sd,
percentiles) of the row subsamples — citing rows per pool, other spec-arm reasoning, base-arm
reasoning, spec-text rows — on each vector and on u. Threshold calibration against u's grid is a
separate step (calibrate_thresholds.py on the −v artifacts, needs the val capture locally).

    uv run --with torch --with safetensors python experiments/spec-vector/scripts/fit_spec_vectors.py --fit-dir experiments/spec-vector/data/fit_out
Artifacts: artifacts/<name>_L35.safetensors (+ .json; proxy format, unit), artifacts/<name>_neg_L35.safetensors
(−v, what the clamp arms load), artifacts/directions_L35.safetensors (all vectors, unit, one tensor each),
artifacts/per_rollout_diffs.npz.
"""
import argparse, hashlib, json, sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = EXP.parents[1]
sys.path.insert(0, str(REPO / "src"))
from steering_reasoning.canonical import load as canon  # noqa: E402
from steering_reasoning.steering.vectors import save_artifact  # noqa: E402

LAYER = 35
MIN_TOKENS = 20   # a rollout votes for a pool only with at least this many pooled rows


def load_cells(fit_dir: Path, arm: str):
    """{rollout_id: {name: (sum, n)}} over both seeds of one arm."""
    out = {}
    for seed in ("s42", "s43"):
        p = fit_dir / f"stats_{arm}_{seed}.npz"
        if not p.exists():
            continue
        z = np.load(p, allow_pickle=False)
        names = list(z["names"]); S, N = z["sums"], z["counts"]
        for i, rid in enumerate(z["rollouts"]):
            out[str(rid)] = {nm: (S[i, j], int(N[i, j])) for j, nm in enumerate(names) if N[i, j] > 0}
    return out


def load_rows(fit_dir: Path, arm: str):
    rows, tags = {}, {}
    for seed in ("s42", "s43"):
        p = fit_dir / f"rows_{arm}_{seed}.npz"
        if not p.exists():
            continue
        z = np.load(p, allow_pickle=False)
        for k in z.files:
            if k.startswith("rows__"):
                nm = k[len("rows__"):]
                rows.setdefault(nm, []).append(z[k]); tags.setdefault(nm, []).append(z["tags__" + nm])
    return {k: np.concatenate(v) for k, v in rows.items()}, {k: np.concatenate(v) for k, v in tags.items()}


def combo_sel(source=("reasoning",), kind=("id",), v1=None, v2=None):
    def f(name):
        if "|" not in name:
            return False
        s, k, a, b = name.split("|")
        return s in source and k in kind and (v1 is None or a in v1) and (v2 is None or b in v2)
    return f


def pool(stats: dict, sel) -> tuple[np.ndarray, int]:
    s = np.zeros(5120); n = 0
    for nm, (v, c) in stats.items():
        if sel(nm):
            s += v; n += c
    return s, n


def diff_pool_vs_rest(stats, sel, rest_names=("reasoning_all",)):
    """pool mean − mean of the rollout's other rows of the same kind(s): all rows of rest_names minus
    the pool's rows minus the principle-ID tokens dropped from candidate spans (they must be on
    neither side, or v turns into an "ARH" detector). NOT / RESTATES sentences stay in rest."""
    ps, pn = pool(stats, sel)
    rs, rn = pool(stats, lambda nm: nm in rest_names)
    idtok = tuple(n.split("_")[0] + "_idtok" for n in rest_names)
    ts, tn = pool(stats, lambda nm: nm in idtok)
    rs, rn = rs - ps - ts, rn - pn - tn
    if pn < MIN_TOKENS or rn < MIN_TOKENS:
        return None, pn
    return ps / pn - rs / rn, pn


def diff_pool_vs_pool(stats, sel_a, sel_b):
    a, na = pool(stats, sel_a); b, nb = pool(stats, sel_b)
    if na < MIN_TOKENS or nb < MIN_TOKENS:
        return None, na
    return a / na - b / nb, na


def unit(v):
    return v / np.linalg.norm(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-dir", default=str(EXP / "data/fit_out"))
    ap.add_argument("--out", default=str(EXP / "artifacts"))
    A = ap.parse_args()
    fit_dir, out = Path(A.fit_dir), Path(A.out); out.mkdir(parents=True, exist_ok=True)
    spec = load_cells(fit_dir, "spec"); base = load_cells(fit_dir, "base")
    rows, tags = load_rows(fit_dir, "spec"); brows, btags = load_rows(fit_dir, "base")
    print(f"spec rollouts {len(spec)}, base rollouts {len(base)}")

    defs = {
        "v1_cites":            lambda st: diff_pool_vs_rest(st, combo_sel(v1=("CITES",))),
        "v2_cites":            lambda st: diff_pool_vs_rest(st, combo_sel(v2=("CITES",))),
        "v2_restates":         lambda st: diff_pool_vs_rest(st, combo_sel(v2=("RESTATES",))),
        "v2_cites_restates":   lambda st: diff_pool_vs_rest(st, combo_sel(v2=("CITES", "RESTATES"))),
        "v1_cites_vs_not":     lambda st: diff_pool_vs_pool(st, combo_sel(v1=("CITES",)), combo_sel(v1=("NOT",))),
        "v2_cites_vs_restates": lambda st: diff_pool_vs_pool(st, combo_sel(v2=("CITES",)), combo_sel(v2=("RESTATES",))),
        "v1_cites_idword":     lambda st: diff_pool_vs_rest(st, combo_sel(kind=("id", "word"), v1=("CITES",))),
        "v1_cites_reasvis":    lambda st: diff_pool_vs_rest(st, combo_sel(source=("reasoning", "visible"), v1=("CITES",)), ("reasoning_all", "visible_all")),
        "text_spec":           lambda st: diff_pool_vs_pool(st, lambda n: n == "sys_spec", lambda n: n == "sys_mistrace"),
        "text_principles":     lambda st: diff_pool_vs_pool(st, lambda n: n == "sys_spec_principles", lambda n: n == "sys_mistrace"),
        "text_final":          lambda st: diff_pool_vs_pool(st, lambda n: n == "sys_spec_final", lambda n: n == "sys_mistrace"),
        "text_vs_reasoning":   lambda st: diff_pool_vs_pool(st, lambda n: n == "sys_spec", lambda n: n == "reasoning_all"),
    }
    per, ntok = {}, {}      # name -> {rid: diff}; name -> pooled tokens over voting rollouts
    for name, fn in defs.items():
        per[name] = {}; ntok[name] = 0
        for rid, st in spec.items():
            d, n = fn(st)
            if d is not None:
                per[name][rid] = d; ntok[name] += n
    # spec arm vs base arm, paired
    per["specarm_vs_base"] = {}
    for rid, st in spec.items():
        brid = rid.replace("spec/", "base/")
        if brid in base and "reasoning_all" in base[brid]:
            a, na = st["reasoning_all"]; b, nb = base[brid]["reasoning_all"]
            per["specarm_vs_base"][rid] = a / na - b / nb; ntok["specarm_vs_base"] = ntok.get("specarm_vs_base", 0) + na

    per = {k: v for k, v in per.items() if v}   # drop pools with no voting rollout (e.g. no base cells)
    vecs, raw = {}, {}
    for name, d in per.items():
        M = np.stack(list(d.values()))
        raw[name] = M.mean(0); vecs[name] = unit(raw[name])
    u, s, e = (canon(k).float().numpy() for k in ("u", "s", "e"))
    names = list(vecs)

    lines = ["# spec-vector fit report\n", f"Layer {LAYER}, per-rollout DiffMean, unit vectors, positive = toward the spec. Spec rollouts {len(spec)}, base rollouts {len(base)}.\n"]
    lines.append("\n## Vectors\n\n| vector | rollouts | pooled tokens (sum over rollouts) | raw diff norm | cos u | cos s | cos e | LOTO min cos |\n|---|---|---|---|---|---|---|---|")
    for name in names:
        d = per[name]; n_t = ntok[name]
        # leave-one-task-out
        tasks = {r.split("/")[-1] for r in d}
        M = np.stack(list(d.values())); rids = list(d)
        loto = []
        for t in tasks:
            keep = np.array([not r.endswith("/" + t) for r in rids])
            if keep.sum() >= 2:
                loto.append(float(unit(M[keep].mean(0)) @ vecs[name]))
        lines.append(f"| {name} | {len(d)} | {n_t:,} | {np.linalg.norm(raw[name]):.2f} | {vecs[name] @ u:+.3f} | {vecs[name] @ s:+.3f} | {vecs[name] @ e:+.3f} | {(min(loto) if loto else float('nan')):.4f} |")
    lines.append("\n## Cosines between vectors\n\n| | " + " | ".join(names) + " |\n|---|" + "---|" * len(names))
    for a in names:
        lines.append(f"| {a} | " + " | ".join(f"{vecs[a] @ vecs[b]:+.2f}" for b in names) + " |")

    # projection tables on the row subsamples
    groups = {
        "v1 CITES rows (reasoning, id)": [k for k in rows if combo_sel(v1=("CITES",))(k)],
        "v2 CITES rows (reasoning, id)": [k for k in rows if combo_sel(v2=("CITES",))(k)],
        "v2 RESTATES rows (reasoning, id)": [k for k in rows if combo_sel(v2=("RESTATES",))(k)],
        "v1 NOT rows (reasoning, id)": [k for k in rows if combo_sel(v1=("NOT",))(k)],
        "word-only CITES rows (reasoning)": [k for k in rows if combo_sel(kind=("word",), v1=("CITES",))(k)],
        "visible CITES rows (id)": [k for k in rows if combo_sel(source=("visible",), v1=("CITES",))(k)],
        "other spec-arm reasoning": ["reasoning_noncand"],
        "other spec-arm visible": ["visible_noncand"],
    }
    G = {g: np.concatenate([rows[k] for k in ks]).astype(np.float32) for g, ks in groups.items() if ks}
    G = {g: X for g, X in G.items() if len(X)}
    if "base_reasoning_rows" in brows:
        G["base-arm reasoning"] = brows["base_reasoning_rows"].astype(np.float32)
    lines.append("\n## Projections of token rows (subsample per rollout; raw units of each unit vector; IN-SAMPLE — the same rollouts the vectors were fit on)\n")
    lines.append("Columns: mean ± sd [p05, p50, p95]. Rows in each group: " + ", ".join(f"{g} {len(X):,}" for g, X in G.items()) + "\n")
    hdr = "| vector | " + " | ".join(G) + " |"
    lines.append(hdr + "\n|---|" + "---|" * len(G))
    def fmt(p):
        return f"{p.mean():+.1f} ± {p.std():.1f} [{np.percentile(p, 5):+.1f}, {np.percentile(p, 50):+.1f}, {np.percentile(p, 95):+.1f}]"
    for name in names + ["u (canonical)"]:
        v = u if name.startswith("u ") else vecs[name]
        lines.append(f"| {name} | " + " | ".join(fmt(X @ v) for X in G.values()) + " |")
    # separation: AUROC-like, citing rows vs other reasoning, per vector
    lines.append("\n## Separation of citing rows from other spec-arm reasoning (AUROC of the projection, in-sample)\n\n| vector | v1 CITES vs other | v2 CITES vs other | v2 RESTATES vs other | base reasoning vs spec other |\n|---|---|---|---|---|")
    def auroc(pos, neg):
        pos, neg = np.sort(pos), np.sort(neg)
        return float((np.searchsorted(neg, pos, side="left").sum() + 0.5 * (np.searchsorted(neg, pos, side="right") - np.searchsorted(neg, pos, side="left")).sum()) / (len(pos) * len(neg)))
    for name in names + ["u (canonical)"]:
        v = u if name.startswith("u ") else vecs[name]
        if "other spec-arm reasoning" not in G:
            break
        other = G["other spec-arm reasoning"] @ v
        cells = [auroc(G[g] @ v, other) if g in G else float("nan") for g in ("v1 CITES rows (reasoning, id)", "v2 CITES rows (reasoning, id)", "v2 RESTATES rows (reasoning, id)")]
        cells.append(auroc(G["base-arm reasoning"] @ v, other) if "base-arm reasoning" in G else float("nan"))
        lines.append(f"| {name} | " + " | ".join(f"{c:.3f}" for c in cells) + " |")

    # artifacts
    lab_sha = hashlib.sha256((EXP / "artifacts/cite_labels_v1.json").read_bytes() + (EXP / "artifacts/cite_labels_v2.json").read_bytes()).hexdigest()[:16]
    tensors = {}
    for name in names:
        v = torch.tensor(vecs[name], dtype=torch.float32)
        tensors[name] = v
        meta = {"recipe": f"{name}: see experiments/spec-vector/scripts/fit_spec_vectors.py (docstring); per-rollout DiffMean at L{LAYER} on the spec-basetasks bf16 replay (prefix-reuse), unit",
                "n_rollouts": len(per[name]), "pooled_tokens": int(ntok[name]), "labels_sha": lab_sha,
                "sign": "positive = toward the spec (spec-citing reasoning / spec text)", "cos_u": float(vecs[name] @ u), "cos_s": float(vecs[name] @ s), "cos_e": float(vecs[name] @ e),
                "experiment": "experiments/spec-vector", "capture": "s3://steering-reasoning-artifacts/activations/spec-basetasks_bf16_L31_35"}
        save_artifact(out / f"{name}_L{LAYER}.safetensors", v[None], [LAYER], meta)
        save_artifact(out / f"{name}_neg_L{LAYER}.safetensors", -v[None], [LAYER], {**meta, "sign": "NEGATED: positive = away from the spec; the artifact the clamp arms load (a floor on h·v)"})
    from safetensors.torch import save_file
    save_file(tensors, out / f"directions_L{LAYER}.safetensors", metadata={"labels_sha": lab_sha, "layer": str(LAYER)})
    np.savez(out / "per_rollout_diffs.npz", **{f"{n}__{r.replace('/', '_')}": d for n, dd in per.items() for r, d in dd.items()})
    (out / "fit_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
