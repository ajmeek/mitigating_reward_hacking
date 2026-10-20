"""Extract the per-token sample the approach-to-action axis e is fit on.

Streams the per-token capture of the 42 hacked rollouts at data/captures/qwen35_27b_bf16_token_L31_35 (scripts/dump_activations.py;
never loading a full token_acts tensor) and, for every reasoning turn with >= MIN_TURN_LEN tokens, selects

  - positives: all tokens with dist_to_end in [20, 60]   (41 per qualifying turn)
  - negatives: up to NEG_CAP tokens with dist_to_end > 200 AND pos_from_start >= 200
               (uniform random, seeded; positives always sit at pos_from_start >= MIN_TURN_LEN - 60 = 200)

Only train+val rollouts (splits.json) are read; test rollouts are never touched.

Outputs (in data/derived/e_token_sample/): tokens_train.safetensors, tokens_val.safetensors with acts_l31, acts_l35 (n,5120) f16,
label (n,) i8, rollout_id / turn_index / pos_from_start / dist_to_end / token_pos (n,) i32; dataset_meta.json.

  uv run --with torch python scripts/build_e_token_sample.py
"""

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "data/captures/qwen35_27b_bf16_token_L31_35"
OUT = REPO / "data/derived/e_token_sample"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260830
MIN_TURN_LEN = 260
POS_LO, POS_HI = 20, 60          # dist_to_end band for positives (inclusive)
NEG_MIN_DIST = 200               # negatives: dist_to_end > this
NEG_MIN_POS = 200                # negatives: pos_from_start >= this (matchability)
NEG_CAP = 120                    # max negatives sampled per turn
LAYERS = [31, 35]                # manifest layer order (dim-1 of token_acts)


def main():
    manifest = json.loads((SRC / "manifest.json").read_text())
    assert manifest["layers"] == LAYERS, manifest["layers"]

    splits = json.loads((REPO / "src/steering_reasoning/data/splits.json").read_text())
    rng = np.random.default_rng(SEED)

    meta = {
        "seed": SEED,
        "min_turn_len": MIN_TURN_LEN,
        "pos_band": [POS_LO, POS_HI],
        "neg_min_dist": NEG_MIN_DIST,
        "neg_min_pos": NEG_MIN_POS,
        "neg_cap_per_turn": NEG_CAP,
        "layers": LAYERS,
        "source": str(SRC.relative_to(REPO)),
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "splits": {},
        "rollouts": {},
    }

    for split in ("train", "val"):
        task_ids = [t for t in splits[split] if (SRC / f"{t}.safetensors").exists()]
        missing = [t for t in splits[split] if t not in task_ids]
        if missing:
            print(f"[{split}] no capture for {missing} (clean rollouts, expected)")
        cols = {k: [] for k in ("rollout_id", "turn_index", "pos_from_start",
                                "dist_to_end", "token_pos", "label")}
        acts = {L: [] for L in LAYERS}
        rid_map = []

        for tid in task_ids:
            t0 = time.time()
            side = json.loads((SRC / f"{tid}.json").read_text())
            rid = len(rid_map)
            rid_map.append(tid)
            with safe_open(SRC / f"{tid}.safetensors", framework="pt") as f:
                turn_tokens = f.get_tensor("turn_tokens").numpy()
                token_pos = f.get_tensor("token_pos").numpy()
                token_turn = f.get_tensor("token_turn").numpy()
                n_tok = int(turn_tokens.sum())
                assert len(token_pos) == n_tok == len(token_turn), (tid, n_tok)
                assert len(side["turns"]) == len(turn_tokens), tid
                sl = f.get_slice("token_acts")
                shape = sl.get_shape()
                assert shape == [n_tok, 2, 5120], (tid, shape)
                starts = np.concatenate([[0], np.cumsum(turn_tokens)])

                n_pos = n_neg = 0
                for ti, L_turn in enumerate(turn_tokens):
                    L_turn = int(L_turn)
                    if L_turn < MIN_TURN_LEN:
                        continue
                    a = int(starts[ti])
                    # row ordering sanity: this turn's rows carry its turn index
                    assert token_turn[a] == ti and token_turn[a + L_turn - 1] == ti

                    def add_rows(idx_local, tensor_rows, label):
                        # idx_local: positions within turn; tensor_rows: (k,2,5120) f16
                        k = len(idx_local)
                        cols["rollout_id"].extend([rid] * k)
                        cols["turn_index"].extend([ti] * k)
                        cols["pos_from_start"].extend(idx_local.tolist())
                        cols["dist_to_end"].extend((L_turn - idx_local).tolist())
                        cols["token_pos"].extend(token_pos[a + idx_local].tolist())
                        cols["label"].extend([label] * k)
                        for li, L in enumerate(LAYERS):
                            acts[L].append(tensor_rows[:, li, :].clone())

                    # positives: dist in [20,60] -> p in [L-60, L-20]
                    p_lo, p_hi = L_turn - POS_HI, L_turn - POS_LO  # inclusive
                    block = sl[a + p_lo: a + p_hi + 1]  # contiguous read
                    idx = np.arange(p_lo, p_hi + 1)
                    add_rows(idx, block, 1)
                    n_pos += len(idx)

                    # negatives: p in [NEG_MIN_POS, L-NEG_MIN_DIST-1]
                    hi = L_turn - NEG_MIN_DIST - 1
                    if hi >= NEG_MIN_POS:
                        cand = np.arange(NEG_MIN_POS, hi + 1)
                        take = rng.choice(cand, size=min(NEG_CAP, len(cand)),
                                          replace=False)
                        take.sort()
                        rows = torch.cat(
                            [sl[a + int(j): a + int(j) + 1] for j in take])
                        assert rows.shape == (len(take), 2, 5120), rows.shape
                        add_rows(take, rows, 0)
                        n_neg += len(take)
            meta["rollouts"][tid] = {"split": split, "n_pos": n_pos, "n_neg": n_neg}
            print(f"[{split}] {tid}: {n_pos} pos, {n_neg} neg "
                  f"({time.time()-t0:.1f}s)", flush=True)

        out = {
            "label": torch.tensor(cols["label"], dtype=torch.int8),
            "rollout_id": torch.tensor(cols["rollout_id"], dtype=torch.int32),
            "turn_index": torch.tensor(cols["turn_index"], dtype=torch.int32),
            "pos_from_start": torch.tensor(cols["pos_from_start"], dtype=torch.int32),
            "dist_to_end": torch.tensor(cols["dist_to_end"], dtype=torch.int32),
            "token_pos": torch.tensor(cols["token_pos"], dtype=torch.int32),
        }
        for L in LAYERS:
            out[f"acts_l{L}"] = torch.cat(acts[L])
            assert out[f"acts_l{L}"].shape == (len(out["label"]), 5120)
            assert torch.isfinite(out[f"acts_l{L}"].float()).all()
        n = len(out["label"])
        npos = int(out["label"].sum())
        print(f"[{split}] TOTAL {n} tokens ({npos} pos / {n-npos} neg) "
              f"from {len(rid_map)} rollouts")
        meta["splits"][split] = {"rollouts": rid_map, "n_tokens": n, "n_pos": npos}
        save_file(out, OUT / f"tokens_{split}.safetensors")

    (OUT / "dataset_meta.json").write_text(json.dumps(meta, indent=1))
    print("done ->", OUT)


if __name__ == "__main__":
    main()
