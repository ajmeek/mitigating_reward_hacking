"""Segment-mean coordinates in the (e, u_perp_e) plane for the approach-axis figure.

Reads the layer-35 per-token capture (data/captures/qwen35_27b_bf16_token_L35), the canonical u and e, and the centring mean
probe-refit-bf16/artifacts/mu_bf16_L35.safetensors; writes artifacts/segment_projections_eu_bf16.csv.
    uv run --with torch python experiments/hack-action-anchor/scripts/segment_projections_eu.py
"""
import csv, json, sys
from pathlib import Path
from safetensors import safe_open
REPO = Path(__file__).resolve().parents[3]; EXP = REPO/"experiments"; AH = EXP/"anti-hack-labeling/artifacts"; ART = EXP/"hack-action-anchor/artifacts"
sys.path.insert(0, str(REPO/"src"))
from steering_reasoning.activations import load_dump, join_judge_labels   # noqa: E402
L = 35
splits = json.loads((REPO/"src/steering_reasoning/data/splits.json").read_text()); split_of = {t: k for k in ("train", "val", "test") for t in splits[k]}
JUNK = {"no letters", "code / arithmetic / list item", "prompt quotation", "empty"}
labels = json.load(open(AH/"labels_full.json")); kinds = {(o["task_id"], o["message_index"], o["segment_idx"]): o.get("kind") for o in json.load(open(AH/"decline_unique.json"))}
DECLINE, EXCLUDED = set(), set()
for o in labels:
    k = (o["task_id"], o["message_index"], o["segment_idx"])
    if o["label"] == "EXCLUDED": EXCLUDED.add(k)
    elif o["label"] == "DECLINE" and kinds.get(k) not in JUNK: DECLINE.add(k)
SUF = "_bf16"
from steering_reasoning.canonical import load, u_perp_e
u, e, up = load("u"), load("e"), u_perp_e()
with safe_open(str(EXP/"probe-refit-bf16/artifacts/mu_bf16_L35.safetensors"), "pt") as f: mu = f.get_tensor("mu").float()
DUMP = REPO/"data/captures/qwen35_27b_bf16_token_L35"
man, rows = load_dump(DUMP, layers=[L]); join_judge_labels(rows)
rows = [r for r in rows if r.get("segment_means") is not None]; li = man["layers"].index(L)
out = []
for r in rows:
    Z = r["segment_means"][:, li].float() - mu; pe, pu = Z @ e, Z @ up; seen = {"hack": set(), "decline": set(), "other": set()}
    for i, sg in enumerate(r["segment_meta"]):
        key = (r["task_id"], sg["message_index"], sg["segment_idx"])
        if key in EXCLUDED: continue
        txt = (sg.get("text") or "").strip().lower(); c = "hack" if sg.get("judge_label") else ("decline" if key in DECLINE else "other")
        if txt in seen[c]: continue
        seen[c].add(txt); out.append({"task_id": r["task_id"], "split": split_of[r["task_id"]], "segment": i, "cls": c, "e": round(float(pe[i]), 4), "u_perp": round(float(pu[i]), 4)})
OUT = ART/f"segment_projections_eu{SUF}.csv"
with open(OUT, "w") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
print(len(out), "segments ->", OUT)
