#!/usr/bin/env python
"""Export the data behind the cap-vs-floor figures to artifacts/cap_floor/ (paper-artifacts scripts
read from experiments/*/artifacts, never from raw data).

Projections are raw units on unit vectors at L35: u (canonical, cap side, positive = hacking) and
v = v1_cites (floor side, positive = spec-citing). Every row set is projected on BOTH so the same
tokens can be drawn on either axis or in the (u, v) plane.

Files:
  token_hist.csv        vector, cls, bin_left (0.5 wide), count — per-token densities of the spec-arm
                        classes (citing_reasoning = v1 CITES id-match, restating = v2 RESTATES,
                        other_spec_reasoning, base_reasoning, citing_visible), in-sample subsample
                        (a random ≤400 rows per candidate sentence combo per rollout, ≤2500 other rows
                        per rollout, ≤3000 base rows per sampled base rollout).
  val_segment_hist.csv  vector, cls (hack | decline | other), bin_left, count — HELD-OUT val judge
                        segment means (the classes u was fit on; rollouts that never saw the spec).
  token_joint_sample.csv cls, proj_u, proj_v — ≤4000 rows per class, for scatter / 2-D density.
  val_segment_joint.csv  cls, proj_u, proj_v — every val segment mean.
  thresholds.json       the u cap grid, the v floors and add doses, class medians on each axis,
                        bind fractions (share of each class the cap / floor touches).
Run: uv run --with torch --with safetensors python experiments/spec-vector/scripts/export_cap_floor_data.py
"""
import json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = EXP.parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(HERE))
from steering_reasoning.canonical import load as canon  # noqa: E402
from fit_spec_vectors import load_rows, combo_sel  # noqa: E402
from make_projection_figures import val_segments  # noqa: E402

OUT = EXP / "artifacts/cap_floor"; OUT.mkdir(parents=True, exist_ok=True)
FIT = EXP / "data/fit_out"
WIDTH = 0.5
U_GRID = [0, -4, -8, -12, -16]
V_FLOORS = [5, 10, 16.5]
V_ADDS = [8, 16]
rng = np.random.default_rng(0)


def hist_rows(name_vec, cls, x):
    lo = np.floor(x.min() / WIDTH) * WIDTH; hi = np.ceil(x.max() / WIDTH) * WIDTH + WIDTH
    edges = np.arange(lo, hi + WIDTH / 2, WIDTH)
    c, _ = np.histogram(x, bins=edges)
    return [(name_vec, cls, f"{e:.2f}", int(n)) for e, n in zip(edges[:-1], c)]


def main():
    from safetensors.numpy import load_file
    u = canon("u").float().numpy(); v = load_file(str(EXP / "artifacts/v1_cites_L35.safetensors"))["vector"]
    rows, _ = load_rows(FIT, "spec"); brows, _ = load_rows(FIT, "base")
    classes = {
        "citing_reasoning": np.concatenate([rows[k] for k in rows if combo_sel(v1=("CITES",))(k)]),
        "restating": np.concatenate([rows[k] for k in rows if combo_sel(v2=("RESTATES",))(k)]),
        "other_spec_reasoning": rows["reasoning_noncand"],
        "base_reasoning": brows["base_reasoning_rows"],
        "citing_visible": np.concatenate([rows[k] for k in rows if combo_sel(source=("visible",), v1=("CITES",))(k)]),
    }
    tok_hist, joint, med = [], [], {"u": {}, "v": {}}
    for cls, X in classes.items():
        X = X.astype(np.float32); pu, pv = X @ u, X @ v
        tok_hist += hist_rows("u", cls, pu) + hist_rows("v_cite", cls, pv)
        med["u"][cls] = float(np.median(pu)); med["v"][cls] = float(np.median(pv))
        idx = rng.choice(len(X), min(4000, len(X)), replace=False)
        joint += [(cls, f"{pu[i]:.3f}", f"{pv[i]:.3f}") for i in idx]
    with open(OUT / "token_hist.csv", "w") as f:
        f.write("vector,cls,bin_left,count\n"); f.writelines(",".join(map(str, r)) + "\n" for r in tok_hist)
    with open(OUT / "token_joint_sample.csv", "w") as f:
        f.write("cls,proj_u,proj_v\n"); f.writelines(",".join(r) + "\n" for r in joint)
    # held-out val segments on both axes
    V = val_segments({"u": u, "v_cite": v})
    seg_hist, seg_joint, seg_med = [], [], {"u": {}, "v": {}}
    for cls in ("hack", "decline", "other"):
        pu, pv = V["u"][cls], V["v_cite"][cls]
        seg_hist += hist_rows("u", cls, pu) + hist_rows("v_cite", cls, pv)
        seg_joint += [(cls, f"{a:.3f}", f"{b:.3f}") for a, b in zip(pu, pv)]
        seg_med["u"][cls] = float(np.median(pu)); seg_med["v"][cls] = float(np.median(pv))
    with open(OUT / "val_segment_hist.csv", "w") as f:
        f.write("vector,cls,bin_left,count\n"); f.writelines(",".join(map(str, r)) + "\n" for r in seg_hist)
    with open(OUT / "val_segment_joint.csv", "w") as f:
        f.write("cls,proj_u,proj_v\n"); f.writelines(",".join(r) + "\n" for r in seg_joint)
    # bind fractions: cap on u touches tokens ABOVE tau; floor on v touches tokens BELOW t
    bind = {"u_cap": {}, "v_floor": {}}
    for tau in U_GRID:
        bind["u_cap"][str(tau)] = {cls: float((V["u"][cls] > tau).mean()) for cls in ("hack", "decline", "other")}
        bind["u_cap"][str(tau)].update({f"token_{cls}": float(((X.astype(np.float32) @ u) > tau).mean()) for cls, X in classes.items()})
    for t in V_FLOORS:
        bind["v_floor"][str(t)] = {cls: float((V["v_cite"][cls] < t).mean()) for cls in ("hack", "decline", "other")}
        bind["v_floor"][str(t)].update({f"token_{cls}": float(((X.astype(np.float32) @ v) < t).mean()) for cls, X in classes.items()})
    meta = {
        "layer": 35, "bin_width": WIDTH, "units": "raw projection on the unit vector",
        "u": {"artifact": "src/steering_reasoning/canonical/u_L35_steering.safetensors", "positive_pole": "hacking",
              "cap_grid": U_GRID, "operator": "h <- h - max(0, u.h - tau) u  (cap: touches tokens above tau)"},
        "v_cite": {"artifact": "experiments/spec-vector/artifacts/v1_cites_L35.safetensors", "positive_pole": "spec-citing",
                   "floors": V_FLOORS, "add_doses": V_ADDS, "cleavage_90pct_citing": 16.5, "equal_error": 16.7,
                   "operator": "h <- h + max(0, t - v.h) v  (floor: touches tokens below t; = cap on -v at -t)",
                   "cos_with_u": float(v @ u)},
        "token_class_medians": med, "val_segment_medians": seg_med, "bind_fractions": bind,
        "token_class_n": {cls: int(len(X)) for cls, X in classes.items()},
        "val_segment_n": {cls: int(len(V["u"][cls])) for cls in ("hack", "decline", "other")},
        "sources": {"tokens": "experiments/spec-vector/data/fit_out (in-sample subsample of the fit rollouts; see extract_pool_stats.py)",
                    "val_segments": "data/captures/qwen35_27b_bf16_token_L31_35 val split, judge labels from experiments/anti-hack-labeling/artifacts"},
    }
    (OUT / "thresholds.json").write_text(json.dumps(meta, indent=1))
    print("wrote", sorted(p.name for p in OUT.iterdir()))
    print(json.dumps({"token_class_medians": med, "val_segment_medians": seg_med}, indent=1))


if __name__ == "__main__":
    main()
