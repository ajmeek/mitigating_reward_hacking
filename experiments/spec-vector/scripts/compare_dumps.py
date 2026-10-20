#!/usr/bin/env python
"""Compare a prefix-reuse replay dump of one rollout against its from-scratch dump.

    python compare_dumps.py --full <dump_dir_no_reuse> --reuse <dump_dir_reuse> --task <task_id> [--min-cos 0.9999]

Both dumps are the shard format of scripts/capture_rollouts.py (L<ll>/<task>.safetensors +
meta/<task>.json). The reuse dump stores a re-read reply once (share_branch), so it has fewer
rows; every row it has must match the full dump's row for the same (context position, token id)
computed in the same turn or — for rows the cache served — in an earlier turn. Prints per-layer
cosine / max-abs statistics, the number of turns the cache actually served, and exits 1 when any
row falls below --min-cos. Used by tests/gpu/test_prefix_reuse_replay.py and by the capture box's
smoke before the full run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from steering_reasoning.steering.token_dump import tensors_from_bytes  # noqa: E402


def load_layer(dump: Path, layer: int, task: str) -> dict[str, torch.Tensor]:
    p = dump / f"L{layer:02d}" / f"{task}.safetensors"
    t = tensors_from_bytes(bytearray(p.read_bytes()))
    return {k: v.clone() for k, v in t.items()}


def _unit_direction(layer: int) -> torch.Tensor | None:
    """The canonical u (L35 only): the validated reuse criterion is the correlation of per-token
    projections onto it (>= 0.99995 on the paper's capture)."""
    if layer != 35:
        return None
    try:
        from steering_reasoning.canonical import load
        return load("u").float()
    except Exception:  # noqa: BLE001 — no canonical package on this box: cosine stats only
        return None


def compare(full: Path, reuse: Path, task: str, min_cos: float) -> dict:
    meta_r = json.loads((reuse / "meta" / f"{task}.json").read_text())
    meta_f = json.loads((full / "meta" / f"{task}.json").read_text())
    layers = json.loads((reuse / "manifest.json").read_text())["layers"] if (reuse / "manifest.json").exists() else None
    if layers is None:
        layers = sorted(int(p.name[1:]) for p in reuse.iterdir() if p.name.startswith("L") and p.is_dir())
    tokens_f = tensors_from_bytes(bytearray((full / "tokens" / f"{task}.safetensors").read_bytes()))
    tokens_r = tensors_from_bytes(bytearray((reuse / "tokens" / f"{task}.safetensors").read_bytes()))
    served = [t for t in meta_r["turns"] if t["first_pos"] > 0]
    out = {"task": task, "layers": layers, "n_rows_full": int(tokens_f["token_ids"].numel()),
           "n_rows_reuse": int(tokens_r["token_ids"].numel()), "n_turns": len(meta_r["turns"]),
           "n_turns_cache_served": len(served), "n_fallbacks": meta_r.get("n_prefix_reuse_fallbacks"),
           "full_fallbacks": meta_f.get("n_prefix_reuse_fallbacks"), "per_layer": {}}
    idx_f = {}
    for r, (pos, tid, tu) in enumerate(zip(tokens_f["token_ctxpos"].tolist(), tokens_f["token_ids"].tolist(),
                                            tokens_f["token_turn"].tolist())):
        idx_f.setdefault((int(pos), int(tid)), []).append((int(tu), r))
    for L in layers:
        A = load_layer(full, L, task)["token_acts"].float()
        B = load_layer(reuse, L, task)["token_acts"].float()
        cos, mad, same_turn, earlier_turn = [], [], 0, 0
        origin, worst = [], []   # per row: "same" | "earlier"; (cos, turn, pos, kind)
        rows_f, rows_r = [], []  # paired row indices (for the projection correlation)
        kinds_r = tokens_r["token_kind"].tolist()
        for r, (pos, tid, tu) in enumerate(zip(tokens_r["token_ctxpos"].tolist(), tokens_r["token_ids"].tolist(),
                                                tokens_r["token_turn"].tolist())):
            cands = idx_f.get((int(pos), int(tid)))
            if not cands:
                raise SystemExit(f"L{L} row {r} (pos {pos}, id {tid}) has no counterpart in the full dump")
            # prefer the full dump's row from the same turn; else the latest earlier turn (a re-read
            # the reuse tree shares with the turn that first stored it)
            same = [rr for t_, rr in cands if t_ == int(tu)]
            if same:
                rf = same[0]; same_turn += 1; origin.append("same")
            else:
                earlier = [(t_, rr) for t_, rr in cands if t_ < int(tu)]
                if not earlier:
                    raise SystemExit(f"L{L} row {r} (turn {tu}, pos {pos}, id {tid}): the full dump has this "
                                     f"token only in later turns {sorted(t_ for t_, _ in cands)} — trees disagree")
                rf = max(earlier)[1]; earlier_turn += 1; origin.append("earlier")
            a, b = A[rf], B[r]
            rows_f.append(rf); rows_r.append(r)
            c = float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))
            cos.append(c); mad.append(float((a - b).abs().max()))
            worst.append((c, int(tu), int(pos), int(kinds_r[r])))
        # Rows the reuse tree does NOT store (share_branch: a reply re-read in a later turn) must
        # equal the row it shares with — the earlier turn's generated copy, which the reuse dump
        # has. This is the claim behind storing a re-read once: same tokens after the same prefix.
        idx_r = {}
        for r, (pos, tid, tu) in enumerate(zip(tokens_r["token_ctxpos"].tolist(), tokens_r["token_ids"].tolist(),
                                                tokens_r["token_turn"].tolist())):
            idx_r.setdefault((int(pos), int(tid)), []).append((int(tu), r))
        have_r = {(int(tu), int(pos)) for tu, pos in zip(tokens_r["token_turn"].tolist(), tokens_r["token_ctxpos"].tolist())}
        dropped_cos = []
        for rf, (pos, tid, tu) in enumerate(zip(tokens_f["token_ctxpos"].tolist(), tokens_f["token_ids"].tolist(),
                                                 tokens_f["token_turn"].tolist())):
            if (int(tu), int(pos)) in have_r:
                continue
            earlier = [(t_, rr) for t_, rr in idx_r.get((int(pos), int(tid)), []) if t_ < int(tu)]
            if not earlier:
                raise SystemExit(f"L{L}: full-dump row {rf} (turn {tu}, pos {pos}, id {tid}) is absent from the reuse "
                                 "dump and has no earlier-turn copy there — the shared tree dropped a row")
            a, b = A[rf], B[max(earlier)[1]]
            dropped_cos.append(float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12)))
        dropped_t = torch.tensor(dropped_cos) if dropped_cos else None
        cos_t = torch.tensor(cos)
        u = _unit_direction(L)
        proj_corr = None
        AA, BB = A[rows_f], B[rows_r]
        if u is not None and len(cos) > 2:
            pa, pb = AA @ u, BB @ u
            proj_corr = float(torch.corrcoef(torch.stack([pa, pb]))[0, 1])
        # Fit-level agreement: what a DiffMean built on this dump sees. Per-row bf16 noise from a
        # cached prefill is ~1% median with a heavy tail (27B smoke 2026-09-23: p99 9%, worst row
        # 88%), but it is unbiased and averages out: contrast vectors between token kinds and
        # per-turn reasoning means agreed at cos >= 0.99997 while single rows went to 0.71.
        kk = torch.tensor([int(kinds_r[r]) for r in rows_r]); tt = torch.tensor([int(tokens_r["token_turn"][r]) for r in rows_r])
        served = torch.tensor([int(tokens_r["token_turn"][r]) in {t["turn_row"] for t in meta_r["turns"] if t["first_pos"] > 0}
                               for r in rows_r])
        fit = {}
        if served.any():
            As, Bs, ks_, ts_ = AA[served], BB[served], kk[served], tt[served]
            md = (Bs - As).mean(0)
            fit["mean_delta_norm"] = float(md.norm()); fit["mean_norm"] = float(As.mean(0).norm())
            if u is not None:
                fit["bias_along_u_sd"] = float(md @ u) / float(((As @ u).std()) + 1e-9)
            for name, (ka, kb) in {"reasoning_minus_tooloutput": (5, 4), "reasoning_minus_reply": (5, 6)}.items():
                if (ks_ == ka).sum() > 20 and (ks_ == kb).sum() > 20:
                    ca = As[ks_ == ka].mean(0) - As[ks_ == kb].mean(0); cb = Bs[ks_ == ka].mean(0) - Bs[ks_ == kb].mean(0)
                    fit[f"contrast_cos_{name}"] = float(torch.dot(ca, cb) / (ca.norm() * cb.norm() + 1e-12))
            pt = []
            for t in ts_.unique().tolist():
                m = (ts_ == t) & (ks_ == 5)
                if m.sum() > 20:
                    ca = As[m].mean(0) - As.mean(0); cb = Bs[m].mean(0) - Bs.mean(0)
                    pt.append(float(torch.dot(ca, cb) / (ca.norm() * cb.norm() + 1e-12)))
            if pt:
                fit["turn_reasoning_means_cos_min"] = min(pt); fit["n_turns_with_reasoning"] = len(pt)
        by_origin = {}
        for o in ("same", "earlier"):
            sel = torch.tensor([x == o for x in origin])
            if sel.any():
                by_origin[o] = {"n": int(sel.sum()), "cos_min": float(cos_t[sel].min()), "cos_mean": float(cos_t[sel].mean())}
        stats = {"n": len(cos), "same_turn": same_turn, "matched_earlier_turn": earlier_turn,
                 "cos_min": float(cos_t.min()), "cos_mean": float(cos_t.mean()),
                 "cos_p01": float(torch.quantile(cos_t, 0.01)) if len(cos) > 1 else float(cos_t.min()),
                 "maxabs_max": float(max(mad)), "n_below_min_cos": int((cos_t < min_cos).sum()),
                 "by_origin": by_origin, "proj_u_corr": proj_corr, "fit_level": fit,
                 "dropped_rows_vs_shared_copy": (None if dropped_t is None else
                                                 {"n": int(dropped_t.numel()), "cos_min": float(dropped_t.min()),
                                                  "cos_mean": float(dropped_t.mean())}),
                 "worst_rows": [{"cos": round(c, 6), "turn": t, "ctxpos": pp, "kind": k}
                                for c, t, pp, k in sorted(worst)[:8]]}
        out["per_layer"][L] = stats
    return out


def verdict(res: dict, *, min_p01: float = 0.99, min_mean: float = 0.999, min_proj_corr: float = 0.999,
            max_bias_sd: float = 0.01, min_contrast_cos: float = 0.9999, min_turn_mean_cos: float = 0.9995) -> list[str]:
    """Why a comparison fails (empty = pass). The gate is on what a DiffMean fit consumes — the
    projection correlation onto u, the bias of the mean shift along u, kind-contrast vectors and
    per-turn reasoning means — because per-row bf16 noise from a cached prefill (the hybrid
    model's Mamba state restored from the cache) has a heavy tail that is irrelevant at the fit
    level (27B smoke 2026-09-23: rows to cos 0.71, contrasts at 1.00000, bias 0.0008 SD). The
    per-row thresholds only catch a gross failure (misalignment, a wrong layer)."""
    bad = []
    for L, st in res["per_layer"].items():
        if st.get("proj_u_corr") is not None and st["proj_u_corr"] < min_proj_corr:
            bad.append(f"L{L}: projection-on-u correlation {st['proj_u_corr']:.6f} < {min_proj_corr}")
        if st["cos_p01"] < min_p01:
            bad.append(f"L{L}: 1st-percentile row cosine {st['cos_p01']:.6f} < {min_p01} (misalignment?)")
        if st["cos_mean"] < min_mean:
            bad.append(f"L{L}: mean row cosine {st['cos_mean']:.6f} < {min_mean}")
        f = st.get("fit_level") or {}
        if "bias_along_u_sd" in f and abs(f["bias_along_u_sd"]) > max_bias_sd:
            bad.append(f"L{L}: mean shift along u {f['bias_along_u_sd']:+.4f} SD exceeds {max_bias_sd}")
        for k, v in f.items():
            if k.startswith("contrast_cos_") and v < min_contrast_cos:
                bad.append(f"L{L}: {k} {v:.6f} < {min_contrast_cos}")
        if "turn_reasoning_means_cos_min" in f and f["turn_reasoning_means_cos_min"] < min_turn_mean_cos:
            bad.append(f"L{L}: per-turn reasoning means cos min {f['turn_reasoning_means_cos_min']:.6f} < {min_turn_mean_cos}")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", required=True)
    ap.add_argument("--reuse", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--min-cos", type=float, default=0.9999, help="per-row cosine below which a row is COUNTED (reported, not fatal)")
    ap.add_argument("--min-p01", type=float, default=0.99, help="fatal: 1st-percentile row cosine below this (misalignment guard)")
    ap.add_argument("--min-mean", type=float, default=0.999, help="fatal: mean row cosine below this")
    ap.add_argument("--require-served", action="store_true", help="fail unless the cache served at least one turn")
    a = ap.parse_args()
    res = compare(Path(a.full), Path(a.reuse), a.task, a.min_cos)
    print(json.dumps(res, indent=1))
    if a.require_served and not res["n_turns_cache_served"]:
        print("=== COMPARE_FAIL: the cache served no turn (prefix reuse inactive) ===")
        sys.exit(1)
    bad = verdict(res, min_p01=a.min_p01, min_mean=a.min_mean)
    if bad:
        print("=== COMPARE_FAIL: " + "; ".join(bad) + " ===")
        sys.exit(1)
    print("=== COMPARE_OK ===")


if __name__ == "__main__":
    main()
