"""Tests for replay_tree — per-token attribution of a replayed rollout from exact ids.

Uses the real Qwen3.5-2B tokenizer + chat template (same tokenizer family as the 27B;
skipped when not cached) so the `<|im_start|>` / `<tool_response>` block attribution runs on
a genuine render. Two turns: the second re-reads the first's reply, which — because the
template keeps in-loop reasoning — re-renders token-identically; the keep-range must NOT
treat those rows as shared with the first turn's generated (possibly clamped) branch.
"""

import pytest

pytest.importorskip("jinja2")
transformers = pytest.importorskip("transformers")

from steering_reasoning.kinds import KIND  # noqa: E402
from steering_reasoning.steering.replay_tree import ReplayTurn, replay_tree  # noqa: E402


@pytest.fixture(scope="module")
def tok():
    try:
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B", local_files_only=True)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Qwen/Qwen3.5-2B tokenizer not cached: {e}")


def render(tok, messages):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=True)
    return tok(text, add_special_tokens=False)["input_ids"], text


def enc(tok, s):
    return tok(s, add_special_tokens=False)["input_ids"]


@pytest.fixture(scope="module")
def two_turns(tok):
    sp = {t: tok.convert_tokens_to_ids(t) for t in
          ("</think>", "<tool_call>", "</tool_call>", "<|im_end|>")}
    msgs1 = [{"role": "system", "content": "You are helpful."},
             {"role": "user", "content": "Run the tests please."}]
    p1, t1 = render(tok, msgs1)
    branch1 = (enc(tok, "Let me look first.") + [sp["</think>"]] + enc(tok, "\n\nRunning them now.")
               + [sp["<tool_call>"]] + enc(tok, '\n{"name": "bash", "arguments": {"cmd": "pytest"}}\n')
               + [sp["</tool_call>"]] + [sp["<|im_end|>"]])
    msgs2 = msgs1 + [
        {"role": "assistant", "content": "Running them now.", "reasoning_content": "Let me look first.",
         # the Qwen3.5 template iterates `arguments|items`: it wants a mapping here, not
         # the OpenAI-wire JSON string (vLLM's renderer parses the string before templating)
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "bash", "arguments": {"cmd": "pytest"}}}]},
        {"role": "tool", "content": "3 passed", "tool_call_id": "c1"},
        {"role": "tool", "content": "done", "tool_call_id": "c1"},
    ]
    p2, t2 = render(tok, msgs2)
    branch2 = enc(tok, "Good.") + [sp["</think>"]] + enc(tok, "\n\nAll green.") + [sp["<|im_end|>"]]
    turns = [ReplayTurn(2, p1, t1, branch1, msgs1), ReplayTurn(5, p2, t2, branch2, msgs2)]
    return tok, sp, turns, replay_tree("t", turns, tok)


def test_branch_kinds_by_token_id(two_turns):
    tok, sp, turns, tree = two_turns
    t0 = tree["turns"][0]
    k = tree["kinds"]
    g = t0["gen_start"]
    assert t0["keep_start"] == 0 and t0["row_start"] == 0
    assert t0["reasoning_start"] == g
    assert k[t0["reasoning_end"] - 1] == KIND["reasoning"] and k[g] == KIND["reasoning"]
    assert k[t0["reasoning_end"]] == KIND["template"]  # </think>
    assert k[t0["resp_tok_start"]] == KIND["response_gen"]  # the "\n\n"
    assert k[t0["tool_tok_start"]] == KIND["tool_call_gen"]
    assert k[t0["tool_tok_start"] + 3] == KIND["tool_call_gen"]  # inside the JSON
    n1 = len(t0["ids"])
    assert k[n1 - 1] == KIND["template"] and tree["im_end_rows"] == [n1 - 1, tree["n_tree"] - 1]
    # the last branch row of every turn was never computed at generation (sampled, not fed back)
    assert tree["never_computed_rows"] == tree["im_end_rows"]
    assert tree["tok_ids"][n1 - 1] == sp["<|im_end|>"]
    assert tree["msgs"][g:n1] == [2] * (n1 - g)  # branch rows belong to the assistant message


def test_context_attribution_from_render(two_turns):
    tok, sp, turns, tree = two_turns
    t0, t1 = tree["turns"]
    ks, ms, cp = tree["kinds"], tree["msgs"], tree["ctxpos"]
    rows = range(t1["row_start"], t1["row_start"] + len(t1["ids"]) - t1["keep_start"])
    # positions below turn 2's keep-range were stored by turn 1 (shared, unsteered context)
    by_ctx = {cp[r]: (ks[r], ms[r]) for r in range(t0["row_start"], t0["row_start"] + t1["keep_start"])}
    by_ctx.update({cp[r]: (ks[r], ms[r]) for r in rows})
    text = turns[1].prompt_text
    enc_ = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    off = enc_["offset_mapping"]

    def kind_at(substr):
        pos = next(i for i, (s, e) in enumerate(off) if s <= text.index(substr) < e)
        return by_ctx[pos]

    assert kind_at("helpful") == (KIND["system"], 0)
    assert kind_at("tests please") == (KIND["user"], 1)
    assert kind_at("look first") == (KIND["assistant"], 2)   # in-loop reasoning, re-read
    assert kind_at("Running them") == (KIND["assistant"], 2)
    assert kind_at("3 passed") == (KIND["tool"], 3)
    assert kind_at("done") == (KIND["tool"], 4)              # sub-split at <tool_response>
    # 103-dump convention: only TRUE special ids (<|im_end|>, <|endoftext|>, pads) become
    # template in context; <|im_start|>/<think>/<tool_call> are added-but-not-special in
    # Qwen3.5 and keep their block's role kind (in-history markup carries the message kind)
    im_start = tok.convert_tokens_to_ids("<|im_start|>")
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    starts = [r for r in rows if tree["tok_ids"][r] == im_start]
    ends = [r for r in rows if tree["tok_ids"][r] == im_end]
    assert starts and ends
    assert all(ks[r] == KIND["template"] for r in ends)
    assert all(ks[r] != KIND["template"] for r in starts[:-1])  # in-history markup
    assert ks[starts[-1]] == KIND["template"]                    # the generation suffix


def test_keep_range_never_shares_previous_branch(two_turns):
    tok, sp, turns, tree = two_turns
    t0, t1 = tree["turns"]
    # turn 2's prompt re-renders turn 1's reasoning + reply token-identically, so the raw
    # LCP would run past turn 1's gen_start; the keep-range must stop there.
    raw_lcp = next(i for i, (a, b) in enumerate(zip(t1["ids"], t0["ids"])) if a != b)
    assert raw_lcp > t0["gen_start"], "fixture assumption: in-loop reasoning re-renders identically"
    assert t1["keep_start"] == t0["gen_start"]
    # so the re-read rows exist as kind 3 (assistant) in turn 2's rows
    rows = range(t1["row_start"], t1["row_start"] + len(t1["ids"]) - t1["keep_start"])
    assert any(tree["kinds"][r] == KIND["assistant"] for r in rows)
    assert tree["n_tree"] == sum(len(t["ids"]) - t["keep_start"] for t in tree["turns"])
    assert tree["turn_of"][t1["row_start"]] == 1 and tree["turn_of"][t1["row_start"] - 1] == 0


def test_retokenization_mismatch_is_fatal(two_turns):
    tok, sp, turns, tree = two_turns
    bad = ReplayTurn(2, turns[0].prompt_ids, turns[0].prompt_text + " ", turns[0].branch_ids,
                     turns[0].messages)
    with pytest.raises(RuntimeError, match="differs from the prompt ids"):
        replay_tree("t", [bad], tok)
