#!/usr/bin/env python
"""Projection distributions along the spec vectors, to choose a cleavage point for steering
(the paper's u_token_hist figure, redone for each spec vector).

Figure 1 (`artifacts/figures/token_hist_<vector>.pdf/.png`): per-token projection densities of
  - spec-citing reasoning tokens (v1 CITES, ID-match, reasoning, ID tokens removed),
  - restating tokens (v2 RESTATES, ID-match, reasoning),
  - other spec-arm reasoning (no candidate sentence),
  - base-arm reasoning (no spec in context),
  - citing tokens in the visible reply,
from the fit's row subsamples (in-sample: the same rollouts the vectors were fit on). Dotted rules
mark candidate cleavage points: the equal-error threshold t* (P[citing < t*] = P[other > t*]) and
the thresholds where 90 % / 75 % / 50 % of citing tokens lie above.
Figure 2 (`token_hist_val_<vector>.pdf`): the val split's judge-segment means (hack / anti-hack /
other, the classes u was fit on) projected on the same vector — held-out from the spec fit, and
what calibrate_thresholds.py binds on.
Table (`artifacts/cleavage_table.csv` + printed): for each vector and a grid of thresholds, the
fraction of each class above the threshold, plus t* and the 90/75/50 % points.

    uv run --with torch --with safetensors --with matplotlib --with pandas python experiments/spec-vector/scripts/make_projection_figures.py
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = EXP.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))
from steering_reasoning.canonical import load as canon, paths  # noqa: E402
from fit_spec_vectors import load_rows, combo_sel  # noqa: E402

INK, TEAL, VERMILLION, FAINT, MUTED, PURPLE, GOLD = "#1a1a1a", "#1b8a8f", "#d55e00", "#9a9a9a", "#4a4a4a", "#7b5ea7", "#c9a227"
COL = 3.3
VECTORS = ["v1_cites", "v2_cites", "v2_restates", "v2_cites_restates", "text_spec", "text_final"]


def style():
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 7.5, "savefig.dpi": 300, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.unicode_minus": False})


def density_step(ax, x, color, label, fill, width, lo, hi):
    edges = np.arange(lo, hi + width, width)
    c, _ = np.histogram(np.clip(x, lo, hi - 1e-9), bins=edges)
    y = np.r_[c, 0] / (c.sum() * width)
    if fill:
        ax.fill_between(edges, y, step="post", color=color, alpha=0.4, lw=0, label=label, zorder=1)
    else:
        ax.step(edges, y, where="post", color=color, lw=1.1, label=label, zorder=3)


def equal_error(pos, neg):
    """t with P[pos < t] = P[neg > t]."""
    lo, hi = min(pos.min(), neg.min()), max(pos.max(), neg.max())
    ts = np.linspace(lo, hi, 2000)
    f = np.array([(pos < t).mean() - (neg > t).mean() for t in ts])
    i = int(np.argmin(np.abs(f)))
    return float(ts[i])


def val_segments(v_by_name: dict):
    """val judge-segment means (hack / decline / other) projected on each vector; the calibrate_thresholds recipe."""
    from steering_reasoning.activations import load_dump, join_judge_labels   # noqa: E402
    ART = REPO / "experiments/anti-hack-labeling/artifacts"
    labels = json.load(open(ART / "labels_full.json"))
    kinds = {(o["task_id"], o["message_index"], o["segment_idx"]): o.get("kind") for o in json.load(open(ART / "decline_unique.json"))}
    JUNK = {"no letters", "code / arithmetic / list item", "prompt quotation", "empty"}
    DEC, EXC = set(), set()
    for o in labels:
        k = (o["task_id"], o["message_index"], o["segment_idx"])
        if o["label"] == "EXCLUDED": EXC.add(k)
        elif o["label"] == "DECLINE" and kinds.get(k) not in JUNK: DEC.add(k)
    sp = json.load(open(paths["splits"])); ids = set(sp["impossible"]["val"])
    man, rows = load_dump(Path(paths["token_capture_bf16"]), layers=[35]); join_judge_labels(rows); li = man["layers"].index(35)
    out = {name: {"hack": [], "other": [], "decline": []} for name in v_by_name}
    for r in rows:
        if r["task_id"] not in ids or r.get("segment_means") is None: continue
        Z = r["segment_means"][:, li].float().numpy(); seen = {c: set() for c in ("hack", "other", "decline")}
        for i, sg in enumerate(r["segment_meta"]):
            key = (r["task_id"], sg["message_index"], sg["segment_idx"])
            if key in EXC: continue
            t = (sg.get("text") or "").strip().lower(); c = "hack" if sg.get("judge_label") else ("decline" if key in DEC else "other")
            if t in seen[c]: continue
            seen[c].add(t)
            for name, v in v_by_name.items():
                out[name][c].append(float(Z[i] @ v))
    return {n: {c: np.array(x) for c, x in d.items()} for n, d in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-dir", default=str(EXP / "data/fit_out"))
    ap.add_argument("--vectors", nargs="+", default=VECTORS)
    ap.add_argument("--no-val", action="store_true")
    A = ap.parse_args()
    style()
    out = EXP / "artifacts/figures"; out.mkdir(parents=True, exist_ok=True)
    from safetensors.numpy import load_file
    D = load_file(str(EXP / "artifacts/directions_L35.safetensors"))
    rows, _ = load_rows(Path(A.fit_dir), "spec"); brows, _ = load_rows(Path(A.fit_dir), "base")
    groups = {
        "citing (reasoning)": np.concatenate([rows[k] for k in rows if combo_sel(v1=("CITES",))(k)]),
        "restating (v2 RESTATES)": np.concatenate([rows[k] for k in rows if combo_sel(v2=("RESTATES",))(k)]),
        "other spec-arm reasoning": rows["reasoning_noncand"],
        "base-arm reasoning": brows["base_reasoning_rows"],
        "citing (visible reply)": np.concatenate([rows[k] for k in rows if combo_sel(source=("visible",), v1=("CITES",))(k)]),
    }
    groups = {g: X.astype(np.float32) for g, X in groups.items()}
    styles = {"citing (reasoning)": (TEAL, False), "restating (v2 RESTATES)": (PURPLE, False),
              "other spec-arm reasoning": (FAINT, True), "base-arm reasoning": (VERMILLION, False), "citing (visible reply)": (GOLD, False)}
    table = ["vector,threshold,citing_above,restating_above,other_above,base_above,visible_citing_above"]
    summary = [["vector", "t* (equal error)", "citing 90 % above", "75 % above", "50 % above", "other above t*", "base above t*"] + ([] if A.no_val else ["val medians hack / anti-hack / other"])]
    vals = None if A.no_val else val_segments({n: D[n] for n in A.vectors})
    for name in A.vectors:
        v = D[name]
        P = {g: X @ v for g, X in groups.items()}
        pos, other, base = P["citing (reasoning)"], P["other spec-arm reasoning"], P["base-arm reasoning"]
        t_star = equal_error(pos, other)
        q = {p: float(np.percentile(pos, 100 - p)) for p in (90, 75, 50)}
        summary.append([name, f"{t_star:+.1f}", f"{q[90]:+.1f}", f"{q[75]:+.1f}", f"{q[50]:+.1f}", f"{(other > t_star).mean():.2f}", f"{(base > t_star).mean():.2f}"])
        lo = np.floor(min(np.percentile(x, 0.2) for x in P.values()) / 5) * 5; hi = np.ceil(max(np.percentile(x, 99.8) for x in P.values()) / 5) * 5
        for t in np.arange(lo, hi + 1, 2.5 if hi - lo > 40 else 1.0):
            table.append(f"{name},{t:.1f}," + ",".join(f"{(P[g] > t).mean():.3f}" for g in ("citing (reasoning)", "restating (v2 RESTATES)", "other spec-arm reasoning", "base-arm reasoning", "citing (visible reply)")))
        width = 1.0 if hi - lo > 30 else 0.5
        fig, ax = plt.subplots(figsize=(COL * 1.35, 0.72 * COL * 1.1))
        for g, x in P.items():
            color, fill = styles[g]
            density_step(ax, x, color, g, fill, width, lo, hi)
        ymax = ax.get_ylim()[1]
        marks = [(t_star, "t*"), (q[90], "90%"), (q[75], "75%"), (q[50], "50%")]
        merged = []
        for t, lab in sorted(marks):
            if merged and abs(t - merged[-1][0]) < 0.04 * (hi - lo):
                merged[-1] = ((merged[-1][0] + t) / 2, merged[-1][1] + "=" + lab)
            else:
                merged.append((t, lab))
        for t, lab in merged:
            ax.axvline(t, color=INK, ls=":", lw=0.7, zorder=2)
            ax.text(t, ymax * 1.02, f"{lab}\n{t:+.0f}", ha="center", va="bottom", fontsize=6.5, color=INK)
        ax.set_ylim(0, ymax); ax.set_xlim(lo, hi); ax.set_yticks([]); ax.spines["left"].set_visible(False)
        ax.set_xlabel(f"per-token projection on {name.replace('_', ' ')} (L35, unit vector)"); ax.set_ylabel("density")
        ax.legend(loc="upper left", frameon=False)
        fig.tight_layout(); fig.savefig(out / f"token_hist_{name}.pdf"); fig.savefig(out / f"token_hist_{name}.png"); plt.close(fig)
        if vals is not None:
            V = vals[name]
            fig, ax = plt.subplots(figsize=(COL * 1.35, 0.72 * COL * 1.1))
            allv = np.concatenate(list(V.values())); lo2 = np.floor(np.percentile(allv, 0.5) / 5) * 5; hi2 = np.ceil(np.percentile(allv, 99.5) / 5) * 5
            w2 = 1.0 if hi2 - lo2 > 30 else 0.5
            for c, color, lab, fill in (("other", FAINT, "other (val segments)", True), ("decline", TEAL, "anti-hack (val)", False), ("hack", VERMILLION, "hack (val)", False)):
                if len(V[c]): density_step(ax, V[c], color, f"{lab}, n={len(V[c])}", fill, w2, lo2, hi2)
            ymax = ax.get_ylim()[1]
            for t, lab in [m for m in merged if m[1] != "75%"]:
                ax.axvline(t, color=INK, ls=":", lw=0.7); ax.text(t, ymax * 1.02, f"{lab}\n{t:+.0f}", ha="center", va="bottom", fontsize=6.5, color=INK)
            ax.set_ylim(0, ymax); ax.set_xlim(min(lo2, lo), max(hi2, hi)); ax.set_yticks([]); ax.spines["left"].set_visible(False)
            ax.set_xlabel(f"segment-mean projection on {name.replace('_', ' ')} (val, held out)"); ax.set_ylabel("density")
            ax.legend(loc="upper left", frameon=False)
            fig.tight_layout(); fig.savefig(out / f"token_hist_val_{name}.pdf"); fig.savefig(out / f"token_hist_val_{name}.png"); plt.close(fig)
            summary[-1].append(f"{np.median(V['hack']):+.1f} / {np.median(V['decline']):+.1f} / {np.median(V['other']):+.1f}")
    (EXP / "artifacts/cleavage_table.csv").write_text("\n".join(table) + "\n")
    md = ["| " + " | ".join(summary[0]) + " |", "|" + "---|" * len(summary[0])] + ["| " + " | ".join(r) + " |" for r in summary[1:]]
    (EXP / "artifacts/cleavage_summary.md").write_text("# Cleavage points (per-token, in-sample rows; val segment medians held out)\n\n" + "\n".join(md) + "\n")
    print("\n".join(md)); print("figures in", out)


if __name__ == "__main__":
    main()
