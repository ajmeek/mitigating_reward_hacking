"""Per-turn (e, u_perp) projections of the last 512 reasoning tokens of every reasoning turn that
ends in a tool call, for all 42 labelled rollouts, with per-turn label and tool metadata. Selection
and binning happen downstream (paper-artifacts/action_plane/action_plane_ewma.py); this script
stores everything so that choice stays cheap to revisit.

Sources:
  activations  data/captures/qwen35_27b_bf16_token_L35 (bf16 per-token capture, layer 35)
  tokens       data/derived/token_series.parquet (one row per reasoning
               token; label 0 other / 1 hack (judge_label) / 2 decline; seg = segment index, -1 uncovered)
  messages     load_mistrace("qwen3_5_27b"): tool_calls of the assistant message of each turn
  basis        canonical bf16 e (load("e")) and u_perp_e() (steering_reasoning.canonical)
  centring     experiments/probe-refit-bf16/artifacts/mu_bf16_L35.safetensors:mu
This is the basis and centring of the paper's approach-axis figure.

Output: experiments/hack-action-anchor/artifacts/turn_paths_last512_bf16.npz
  proj            float32 [n_turns, 512, 2]  (e, u_perp) of the last 512 reasoning tokens; index 511 is
                  the last token before the call (d = 1); NaN-padded at the front for shorter turns
  task_id, split  per turn
  turn, message_index, n_tokens
  n_hack_tokens, n_hack_segments, n_decline_tokens, n_decline_segments   (whole turn)
  tools           "+"-joined function names of the turn's tool calls (e.g. "text_editor", "submit")
  tool_detail     same with text_editor commands (e.g. "text_editor:str_replace")
  u_xy            canonical u in this basis, (u.e, u.u_perp)

    uv run --with torch python experiments/hack-action-anchor/scripts/turn_paths_bf16.py
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from safetensors import safe_open

REPO = str(Path(__file__).resolve().parents[3]); sys.path.insert(0, f"{REPO}/src")
from steering_reasoning.data.mistrace import load_mistrace  # noqa: E402
from steering_reasoning.canonical import load as load_canonical, u_perp_e  # noqa: E402

DUMP = f"{REPO}/data/captures/qwen35_27b_bf16_token_L35"; OUT = f"{REPO}/experiments/hack-action-anchor/artifacts/turn_paths_last512_bf16.npz"
N = 512; L = 35; li = json.load(open(f"{DUMP}/manifest.json"))["layers"].index(L)
e, u, u_perp = load_canonical("e"), load_canonical("u"), u_perp_e()
DIRS = torch.stack([e, u_perp], 1).float()
with safe_open(f"{REPO}/experiments/probe-refit-bf16/artifacts/mu_bf16_L35.safetensors", "pt") as f: mu = f.get_tensor("mu").float()
u_xy = np.array([float(u @ e), float(u @ u_perp)]); print("u in (e, u_perp):", u_xy.round(4), "cos(e, u_perp) =", round(float(e @ u_perp), 5))

D = pd.read_parquet(f"{REPO}/data/derived/token_series.parquet").sort_values(["task_id", "tok"])
recs = {r["task_id"]: r for r in load_mistrace("qwen3_5_27b")[0]}
fname = lambda tc: tc.get("function") or tc.get("name") or "?"
def detail(tc):
    a = tc.get("arguments") or {}; c = a.get("command") if (fname(tc) == "text_editor" and isinstance(a, dict)) else None   # bash also has a "command" arg (the shell line); only text_editor sub-commands are kept
    return fname(tc) + (f":{c}" if c else "")

rows, projs = [], []; n_no_call = 0
for tid, gg in D.groupby("task_id", sort=False):
    lab, turn, seg = gg.label.to_numpy(), gg.turn.to_numpy(), gg.seg.to_numpy(); mi_of = dict(zip(gg.turn, gg.message_index)); msgs = recs[tid]["messages"]
    with safe_open(f"{DUMP}/{tid}.safetensors", "pt") as f:
        tt = f.get_tensor("token_turn").numpy(); assert len(tt) == len(gg), (tid, len(tt), len(gg)); assert (tt == turn).all(), tid
        for k in np.unique(turn):
            mi = int(mi_of[k]); msg = msgs[mi]; assert msg.get("role") == "assistant", (tid, k, msg.get("role"))
            tcs = msg.get("tool_calls") or []
            if not tcs: n_no_call += 1; continue
            m = turn == k; gl, gs = lab[m], seg[m]; sel = np.where(tt == k)[0]; lo, hi = int(sel[-1]) + 1 - min(N, len(sel)), int(sel[-1]) + 1
            a = f.get_slice("token_acts")[lo:hi][:, li].float(); P = ((a - mu) @ DIRS).numpy()
            out = np.full((N, 2), np.nan, np.float32); out[N - len(P):] = P; projs.append(out)
            rows.append(dict(task_id=tid, split=gg.split.iloc[0], turn=int(k), message_index=mi, n_tokens=int(m.sum()),
                             n_hack_tokens=int((gl == 1).sum()), n_hack_segments=len(np.unique(gs[(gl == 1) & (gs >= 0)])),
                             n_decline_tokens=int((gl == 2).sum()), n_decline_segments=len(np.unique(gs[(gl == 2) & (gs >= 0)])),
                             tools="+".join(fname(tc) for tc in tcs), tool_detail="+".join(detail(tc) for tc in tcs)))
    print(tid, len(rows), flush=True)
M = pd.DataFrame(rows)
np.savez_compressed(OUT, proj=np.stack(projs), u_xy=u_xy, **{c: M[c].to_numpy(dtype=(object if M[c].dtype == object or str(M[c].dtype).startswith("str") else None)) for c in M.columns})
print(f"wrote {OUT}: {len(M)} turns ending in a tool call ({n_no_call} turns without a call skipped)")
print(M.groupby("split").size().to_dict(), M.tools.value_counts().to_dict())
