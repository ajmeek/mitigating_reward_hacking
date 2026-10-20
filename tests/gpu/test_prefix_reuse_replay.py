"""GPU: the prefix-reuse replay path (scripts/capture_rollouts.py --prefix-reuse) reproduces the
from-scratch replay of an UNSTEERED record on every stored row, and the cache actually serves
turns. Runs on Qwen3.5-2B in the serving env with the prefix-reuse patch applied:

    cd serving && VLLM_LENS_CAPTURE_PREFIX_REUSE=1 PYTHONPATH=../src \\
      uv run --with pytest python -m pytest -q -s ../tests/gpu/test_prefix_reuse_replay.py

The record is synthetic (three chat turns rendered by the engine's own template, branches =
tokenized replies + <|im_end|>, sidecar-style prompt hashes), so build_turns takes the exact-ids
path exactly as it does for the arm records."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments/spec-vector/scripts"))
from steering_reasoning.steering.activations import ActivationCapturer, apply_no_nvcc_workarounds  # noqa: E402
from steering_reasoning.steering.clamp import ids_sha  # noqa: E402
from steering_reasoning.steering.replay import capture_rollout, render_prompt_ids  # noqa: E402
from steering_reasoning.steering.shards import worker_patch_markers  # noqa: E402

MODEL = os.environ.get("REPLAY_TEST_MODEL", "Qwen/Qwen3.5-2B")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.fixture(scope="module")
def capturer():
    if not worker_patch_markers()["prefix_reuse"]:
        pytest.skip("prefix-reuse patch not applied (serving/patch_vllm_lens_prefix_reuse.py)")
    if os.environ.get("VLLM_LENS_CAPTURE_PREFIX_REUSE") != "1":
        pytest.skip("set VLLM_LENS_CAPTURE_PREFIX_REUSE=1")
    apply_no_nvcc_workarounds()
    # the engine capture_rollouts.py builds under --prefix-reuse (one pass per turn, cache on)
    return ActivationCapturer(MODEL, max_model_len=4097, gpu_memory_utilization=0.6, enable_prefix_caching=True)


def synthetic_record(capturer) -> dict:
    """Three in-loop turns (assistant tool call -> tool result -> ...), the shape of the arms'
    rollouts: after the last user message the template re-renders each reply in full, so the
    next prompt shares the previous branch and the cache can serve it."""
    from steering_reasoning.steering.replay import render_branch_ids

    llm, tok = capturer.llm, capturer.llm.get_tokenizer()
    tools = [{"type": "function", "function": {"name": "bash", "description": "Run a shell command.",
                                               "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}},
                                                              "required": ["cmd"]}}}]
    system = "You are a careful engineer who explains each step before acting. " * 40   # > one cache block
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": "Count perfect powers up to n; use the bash tool to run tests. " * 4}]
    replies = [("I will first list the files to see the tests. " * 5, "ls -la"),
               ("The tests assert a set-based count; I will run them now. " * 5, "python -m pytest -q"),
               ("All tests pass; submitting. " * 5, "cat result.txt")]
    tool_outs = ["total 8\ntests.py\nsolution.py\n", "3 passed in 0.02s\n", "ok\n"]
    turns = []
    for k, (reply, cmd) in enumerate(replies):
        request = {"messages": json.loads(json.dumps(msgs)), "tools": tools}
        ids, text = render_prompt_ids(llm, request["messages"], tools)
        response = {"role": "assistant", "content": reply,
                    "tool_calls": [{"id": f"call_{k}", "type": "function",
                                    "function": {"name": "bash", "arguments": json.dumps({"cmd": cmd})}}]}
        branch, _ = render_branch_ids(llm, tok, request["messages"], response, tools, ids, text)
        turns.append({"message_index": len(msgs), "request": request, "response": response,
                      "usage": {"prompt_tokens": len(ids), "completion_tokens": len(branch)},
                      "finish_reason": "tool_calls", "synthetic_length": False, "n_tool_calls": 1,
                      "tool_calls_collapse_suspect": False, "prompt_token_ids": None, "prompt_text": None,
                      "token_ids": branch, "prompt_ids_sha256": ids_sha(ids), "n_prompt_ids": len(ids),
                      "steer_positions": None, "ids_join": "sidecar"})
        msgs.append(response)
        msgs.append({"role": "tool", "tool_call_id": f"call_{k}", "content": tool_outs[k]})
    return {"task_id": "synthetic_1", "rollout_id": "base/conflicting/s42/synthetic_1", "arm": "base",
            "mode": None, "dose": None, "seed": 42, "split": "conflicting", "score": None, "passed": None,
            "is_reward_hack": None, "solved": None, "limit": None, "error": None, "model": MODEL,
            "messages": msgs, "turns": turns}


def test_prefix_reuse_matches_full_replay(capturer, tmp_path: Path):
    from compare_dumps import compare, verdict

    rec = synthetic_record(capturer)
    layer = capturer.n_layers // 2
    full, full2, reuse = tmp_path / "full", tmp_path / "full2", tmp_path / "reuse"
    row_f = capture_rollout(rec, capturer, [layer], 64, full, None, steering=None, prefix_reuse=False)
    capture_rollout(rec, capturer, [layer], 64, full2, None, steering=None, prefix_reuse=False)
    row_r = capture_rollout(rec, capturer, [layer], 64, reuse, None, steering=None, prefix_reuse=True)
    assert row_f["n_prefix_reuse_fallbacks"] == 0 and row_r["prefix_reuse"] and row_r["share_branch"]
    assert row_r["n_tree_tokens"] < row_f["n_tree_tokens"], "share_branch stored no fewer rows: no re-read reply was shared"
    floor = compare(full, full2, "synthetic_1", 0.9999)
    res = compare(full, reuse, "synthetic_1", 0.9999)
    print("\nnoise floor (full vs full2):", json.dumps(floor["per_layer"], indent=1))
    print("\nreuse vs full:", json.dumps(res, indent=1))
    assert res["n_turns_cache_served"] >= 1, "the cache served no turn: prefix reuse inactive"
    assert res["n_fallbacks"] == 0
    st = res["per_layer"][layer]
    d = st["dropped_rows_vs_shared_copy"]
    assert d and d["n"] > 0, "the shared tree dropped no re-read rows: the case under test was not exercised"
    assert not verdict(res), verdict(res)
