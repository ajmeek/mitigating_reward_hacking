#!/usr/bin/env python
"""Make the layer-35 slice of the two-layer per-token capture of the 42 hacked rollouts.

Source: a local dump written by scripts/dump_activations.py with --per-token-layers 31,35
(`--src data/captures/qwen35_27b_bf16_token_L31_35`), or, without --src, the authors' S3 copy
(`aws s3 cp`, credentials from the environment). For each rollout every layered tensor is sliced to
layer 35 and written to data/captures/qwen35_27b_bf16_token_L35/<task>.safetensors with the .json
sidecar copied through; the output manifest says layers [35]. Load it with
steering_reasoning.activations.load_dump like any dump. Resumable: rollouts already present are skipped.

    uv run --with torch python scripts/pull_bf16_token_L35.py --src data/captures/qwen35_27b_bf16_token_L31_35
"""
import argparse, json, shutil, subprocess, time
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file

S3 = "s3://steering-reasoning-artifacts/activations/label-granularity/qwen35_27b_bf16_token_L31_35"
REPO = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--src", default=None, help="local two-layer dump dir (default: download from the authors' S3 prefix)")
ap.add_argument("--out", default=str(REPO / "data/captures/qwen35_27b_bf16_token_L35"))
A = ap.parse_args()
SRC = Path(A.src).resolve() if A.src else None
OUT = Path(A.out); TMP = REPO / "data/cache/pull_tmp"; OUT.mkdir(parents=True, exist_ok=True); TMP.mkdir(parents=True, exist_ok=True)
LAYERED = ("turn_means", "segment_means", "token_acts"); L = 35
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)


def fetch(name, dst):
    """Copy <name> from the source (local or S3) to dst."""
    if SRC is not None:
        shutil.copyfile(SRC / name, dst); return
    for attempt in range(5):
        if subprocess.run(["aws", "s3", "cp", f"{S3}/{name}", str(dst), "--only-show-errors"]).returncode == 0: return
        log(f"retry {attempt + 1} for {name}")
    raise SystemExit(f"download failed: {name}")


man_path = OUT / "manifest.json"
if not man_path.exists():
    fetch("manifest.json", TMP / "manifest.json"); man = json.load(open(TMP / "manifest.json"))
    assert L in man["layers"], man["layers"]
    man["derived_from"] = {"source": str(SRC) if SRC else S3, "layers": man["layers"], "kept_layer": L,
                           "script": "scripts/pull_bf16_token_L35.py", "date": time.strftime("%Y-%m-%d")}
    man["layers"] = [L]
    json.dump(man, open(man_path, "w"), indent=1)
man = json.load(open(man_path)); li = man["derived_from"]["layers"].index(L)
if SRC is not None:
    tasks = sorted(p.stem for p in SRC.glob("*.safetensors"))
else:
    listing = subprocess.run(["aws", "s3", "ls", f"{S3}/"], capture_output=True, text=True, check=True).stdout
    tasks = sorted({l.split()[-1][:-12] for l in listing.splitlines() if l.strip().endswith(".safetensors")})
log(f"{len(tasks)} rollouts at the source; {sum((OUT / f'{t}.safetensors').exists() for t in tasks)} already local")
for t in tasks:
    dst = OUT / f"{t}.safetensors"
    if dst.exists(): continue
    t0 = time.time()
    fetch(f"{t}.json", OUT / f"{t}.json")
    full = SRC / f"{t}.safetensors" if SRC is not None else TMP / f"{t}.safetensors"
    if SRC is None: fetch(f"{t}.safetensors", full)
    out = {}
    with safe_open(str(full), "pt") as f:
        for k in f.keys():
            if k in LAYERED:
                sl = f.get_slice(k); n = sl.get_shape()[0]; parts = []
                for a in range(0, n, 20000): parts.append(sl[a:a + 20000][:, li:li + 1, :].contiguous())
                out[k] = torch.cat(parts)
            else: out[k] = f.get_tensor(k)
    save_file(out, str(dst))
    if SRC is None: full.unlink()
    log(f"{t}: token_acts {tuple(out['token_acts'].shape)} -> {dst.stat().st_size / 1e9:.2f} GB in {time.time() - t0:.0f}s")
log("done ->", OUT)
