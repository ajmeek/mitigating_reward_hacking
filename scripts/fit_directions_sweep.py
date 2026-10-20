#!/usr/bin/env python
"""Per-layer DiffMean probe sweep on the bf16 capture, streamed from S3.

This is the layer sweep that produced the canonical u and s: it writes
experiments/probe-refit-bf16/artifacts/{directions_bf16_all_layers.safetensors, sweep_bf16.json}
(the vectors in src/steering_reasoning/canonical/ are cut from the L35 row; see INDEX.md).

Runs on
s3://steering-reasoning-artifacts/activations/qwen35_27b_bf16_all103/segmeans/ (segment
means, all 64 layers, f16). Files are pulled one at a time into data/cache/bf16_all103 and
deleted after use (70 GB total; BUDGET bounds what is resident at once). Needs AWS read
access to the bucket (the `aws` CLI with credentials in the environment).

Pass 1 (train rollouts): per-rollout class means per layer -> a, b, u, c per layer.
Pass 2 (val + test rollouts): per-segment projections onto the four directions, all
layers -> AUROCs. val is the selection set (as in phase2); test is reported but held out.
SKIP_TRAIN=1 reuses the saved directions and runs pass 2 only.

Run: uv run --with torch python scripts/fit_directions_sweep.py
"""
import json, os, collections
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file

from steering_reasoning.canonical import paths
from steering_reasoning.labels import load_anti_hack_labels, classify_segments
from steering_reasoning.stats import auroc_ranks as auroc
from steering_reasoning.s3stream import s3_sizes, s3_cp, Prefetch, log
from steering_reasoning.activations import memmap_tensor

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments"
OUT = EXP / "probe-refit-bf16/artifacts"
CACHE = REPO / "data/cache/bf16_all103"
S3 = "s3://steering-reasoning-artifacts/activations/qwen35_27b_bf16_all103"
(CACHE / "segmeans").mkdir(parents=True, exist_ok=True); (CACHE / "meta").mkdir(exist_ok=True)

splits = json.loads(paths["splits"].read_text())
TRAIN, VAL, TEST = list(splits["train"]), list(splits["val"]), list(splits["test"])

DECLINE, EXCLUDED = load_anti_hack_labels(paths["decline_labels"], paths["decline_kinds"])

def fetch_meta(tid):
    dst = CACHE / "meta" / f"{tid}.json"
    if not dst.exists():
        s3_cp(f"{S3}/meta/{tid}.json", dst)
    return dst

BUDGET = 20e9   # bytes of segmeans files allowed on disk at once (26 GB free)
sizes = s3_sizes(f"{S3}/segmeans/")
pf = Prefetch(f"{S3}/segmeans", CACHE / "segmeans", sizes, budget=BUDGET)

def open_segmeans(path):
    return memmap_tensor(path, "segment_means", dtype_expect="F16")

def classes(tid, segs):
    return classify_segments({"task_id": tid, "segment_meta": segs}, DECLINE, EXCLUDED, dedup_text=True)

CH = 192  # rows per chunk: 192 x 64 x 5120 x 4 bytes = 250 MB in f32

def class_sums(mm, idx_sets):
    """Sum of rows per class per layer -> {name: (sum [64,5120] f64, n)}; one sweep over rows."""
    S = mm.shape[0]; owner = np.full(S, -1, dtype=np.int64)
    names = list(idx_sets)
    for ci, name in enumerate(names):
        owner[idx_sets[name]] = ci
    sums = {n: np.zeros((64, 5120), dtype=np.float64) for n in names}; counts = {n: 0 for n in names}
    for s0 in range(0, S, CH):
        X = np.asarray(mm[s0:s0 + CH], dtype=np.float32); ow = owner[s0:s0 + CH]
        for ci, name in enumerate(names):
            m = ow == ci
            if m.any(): sums[name] += X[m].sum(0, dtype=np.float64); counts[name] += int(m.sum())
    return sums, counts

# ---------------- pass 1: train ----------------
DIRF = OUT / "directions_bf16_all_layers.safetensors"
if os.environ.get("SKIP_TRAIN") == "1" and DIRF.exists():
    from safetensors.torch import load_file
    D = load_file(DIRF); a, b, c, u = (D[k].numpy().astype(np.float64) for k in ("a", "b", "c", "u"))
    norm = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)
    a_hat, b_hat, c_hat = norm(a), norm(b), norm(c); cos_ab = (a_hat * b_hat).sum(-1)
    pa = pb = pc = []; log("pass 1 skipped: loaded saved directions")
else:
  log(f"pass 1: {len(TRAIN)} train rollouts")
  pa, na, pb, nb, pc, nc = [], [], [], [], [], []
  pf.start(TRAIN[0])
  for i, tid in enumerate(TRAIN):
      meta = json.load(open(fetch_meta(tid))); path = pf.wait()
      if i + 1 < len(TRAIN): pf.start(TRAIN[i + 1], resident=tid)
      mm = open_segmeans(path); h, dc, o = classes(tid, meta["segments"])
      assert mm.shape[0] == len(meta["segments"]), (mm.shape, len(meta["segments"]))
      sums, cnt = class_sums(mm, {"h": h, "d": dc, "o": o})
      mean = lambda ks: sum(sums[k] for k in ks) / max(1, sum(cnt[k] for k in ks))
      if cnt["h"] and (cnt["o"] + cnt["d"]): pa.append(mean("h")); na.append(mean("od"))
      if cnt["d"] and (cnt["o"] + cnt["h"]): pb.append(mean("d")); nb.append(mean("oh"))
      if cnt["h"] and cnt["d"]: pc.append(mean("h")); nc.append(mean("d"))
      del mm; os.remove(path)
      log(f"  {tid}: segs {len(meta['segments'])} hack {cnt['h']} decline {cnt['d']} other {cnt['o']}  ({os.path.getsize(CACHE/'meta'/(tid+'.json'))//1000} kB meta)")
  a = np.mean(pa, 0) - np.mean(na, 0); b = np.mean(pb, 0) - np.mean(nb, 0); c = np.mean(pc, 0) - np.mean(nc, 0)
  norm = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)
  a_hat, b_hat, c_hat = norm(a), norm(b), norm(c); u = norm(a_hat - b_hat)
  cos_ab = (a_hat * b_hat).sum(-1)
  save_file({"a": torch.tensor(a, dtype=torch.float32), "b": torch.tensor(b, dtype=torch.float32),
             "u": torch.tensor(u, dtype=torch.float32), "c": torch.tensor(c, dtype=torch.float32)},
            OUT / "directions_bf16_all_layers.safetensors",
            metadata={"recipe": "per-rollout DiffMean on train (24 rollouts), a: hack vs rest, b: decline vs rest, u = norm(a_hat - b_hat), c: hack - decline; rows = layers 0..63",
                      "capture": "qwen35_27b_bf16_all103 segmeans", "n_rollouts_a": str(len(pa)), "n_rollouts_b": str(len(pb))})
  log(f"directions saved; n_rollouts a={len(pa)} b={len(pb)} c={len(pc)}")
DIRS = np.stack([a_hat, b_hat, u, c_hat], 1).astype(np.float32)   # [64, 4, 5120]

# ---------------- pass 2: val + test ----------------
def project_rollout(tid, nxt):
    meta = json.load(open(fetch_meta(tid))); path = pf.wait()
    assert path.name == f"{tid}.safetensors", (path, tid)
    if nxt is not None: pf.start(nxt, resident=tid)
    mm = open_segmeans(path); S = mm.shape[0]; P = np.zeros((S, 64, 4), dtype=np.float32)
    for s0 in range(0, S, CH):
        X = np.asarray(mm[s0:s0 + CH], dtype=np.float32)          # [n,64,5120]
        P[s0:s0 + CH] = np.einsum("nld,lkd->nlk", X, DIRS)
    del mm; os.remove(path)
    return meta, P

def evaluate(tids, tag):
    log(f"pass 2 ({tag}): {len(tids)} rollouts")
    acc = collections.defaultdict(lambda: collections.defaultdict(list))   # layer -> metric -> [scores]
    per_roll = {}
    for i, tid in enumerate(tids):
        meta, P = project_rollout(tid, tids[i + 1] if i + 1 < len(tids) else None); h, dc, o = classes(tid, meta["segments"])
        assert P.shape[0] == len(meta["segments"]), (P.shape, len(meta["segments"]))
        per_roll[tid] = {"n_hack": len(h), "n_decline": len(dc), "n_other": len(o)}
        for L in range(64):
            pa_, pb_, pu_, pc_ = P[:, L, 0], P[:, L, 1], P[:, L, 2], P[:, L, 3]
            if h and (o + dc): acc[L]["hack_det_pos"] += list(pa_[h]); acc[L]["hack_det_neg"] += list(pa_[o + dc])
            if dc and (o + h): acc[L]["anti_det_pos"] += list(pb_[dc]); acc[L]["anti_det_neg"] += list(pb_[o + h])
            if h and dc:
                acc[L]["u_hva_pos"] += list(pu_[h]); acc[L]["u_hva_neg"] += list(pu_[dc])
                acc[L]["c_hva_pos"] += list(pc_[h]); acc[L]["c_hva_neg"] += list(pc_[dc])
            if h and (o + dc): acc[L]["u_hack_vs_rest_pos"] += list(pu_[h]); acc[L]["u_hack_vs_rest_neg"] += list(pu_[o + dc])
            if dc and (o + h): acc[L]["u_decline_vs_rest_pos"] += list(-pu_[dc]); acc[L]["u_decline_vs_rest_neg"] += list(-pu_[o + h])
            acc[L]["u_mean_h"] += list(pu_[h]); acc[L]["u_mean_d"] += list(pu_[dc]); acc[L]["u_mean_o"] += list(pu_[o])
        log(f"  {tid}: hack {len(h)} decline {len(dc)} other {len(o)}")
    res = {}
    for L in range(64):
        m = acc[L]
        res[L] = {k: auroc(m[k + "_pos"], m[k + "_neg"]) for k in ("hack_det", "anti_det", "u_hva", "c_hva", "u_hack_vs_rest", "u_decline_vs_rest")}
        res[L].update({"u_mean_hack": float(np.mean(m["u_mean_h"])) if m["u_mean_h"] else None,
                       "u_mean_decline": float(np.mean(m["u_mean_d"])) if m["u_mean_d"] else None,
                       "u_mean_other": float(np.mean(m["u_mean_o"])) if m["u_mean_o"] else None,
                       "cos_ab": float(cos_ab[L]), "norm_a": float(np.linalg.norm(a[L])), "norm_b": float(np.linalg.norm(b[L]))})
    return res, per_roll

pf.start(VAL[0]); val_res, val_roll = evaluate(VAL, "val")
pf.start(TEST[0]); test_res, test_roll = evaluate(TEST, "test")
out = {"val": val_res, "test": test_res, "val_rollouts": val_roll, "test_rollouts": test_roll,
       "train": TRAIN, "val_ids": VAL, "test_ids": TEST}
json.dump(out, open(OUT / "sweep_bf16.json", "w"), indent=1)
print(f"\n{'L':>3} {'hack_det':>9} {'anti_det':>9} {'u_hva':>7} {'c_hva':>7} {'cos_ab':>7} || {'test u_hva':>10} {'u_h':>6} {'u_o':>6} {'u_d':>6}")
for L in range(64):
    v = val_res[L]; t = test_res[L]
    print(f"{L:3d} {v['hack_det']:9.3f} {v['anti_det']:9.3f} {v['u_hva']:7.3f} {v['c_hva']:7.3f} {v['cos_ab']:7.3f} || {t['u_hva']:10.3f} {t['u_mean_hack']:6.1f} {t['u_mean_other']:6.1f} {t['u_mean_decline']:6.1f}")
log("done")
