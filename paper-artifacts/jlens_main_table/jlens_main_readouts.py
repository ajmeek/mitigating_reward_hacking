"""J-lens readouts behind the main-body J-lens table: the top-20 promoted tokens at both poles of the four
steering directions the paper deploys, read at layer 35 through the Jacobian lens (steering_reasoning.lens,
raw mode, Neuronpedia's Qwen3.5-27B lens, unembedding + final norm of Qwen/Qwen3.5-27B).

Writes jlens_main_readouts.json beside this script (tracked); jlens_main_table.py formats it. Needs torch and
transformers, which the eval env does not carry, and downloads the lens (3.3 GB) and one unembedding shard:

    uv run --with torch --with transformers python paper-artifacts/jlens_main_table/jlens_main_readouts.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from steering_reasoning.lens import DEFAULT_MODEL, LENS_FILE, LENS_REPO, jacobian_diagnostics, top_tokens  # noqa: E402

LAYER, K = 35, 20
VECTORS = {   # column key -> artifact (all single-key `vector` files)
    "u": "src/steering_reasoning/canonical/u_L35_steering.safetensors",
    "spec": "experiments/spec-vector/artifacts/v1_cites_L35.safetensors",
    "opt_add": "experiments/opt-vector/artifacts/opt_27b_L35_v3c.safetensors",
    "opt_cap": "experiments/opt-vector/artifacts/opt_27b_L35_v3c_cl.safetensors",
}
out = {"layer": LAYER, "lens": f"{LENS_REPO}/{LENS_FILE}", "mode": "raw", "unembedding": DEFAULT_MODEL, "k": K,
       "vectors": VECTORS, "diagnostics": {}, "readouts": {}}
for name, rel in VECTORS.items():
    path = ROOT / rel
    out["diagnostics"][name] = jacobian_diagnostics(path, LAYER)
    for sign, label in ((1, "+"), (-1, "-")):
        toks = top_tokens(path, LAYER, lens="jacobian", mode="raw", k=K, sign=sign)["jacobian"]
        out["readouts"][f"{label}{name}"] = [[t, round(s, 4)] for t, s in toks]
        print(f"{label}{name}: " + ", ".join(f"{t!r} {s:.2f}" for t, s in toks[:8]), flush=True)
(HERE / "jlens_main_readouts.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
print("wrote", HERE / "jlens_main_readouts.json")
