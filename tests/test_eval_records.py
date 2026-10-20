"""Tests for the Inspect-log → replay-record adapter (evals/eval_records.py).

The structural tests run against two real n5 arm logs when they are present in the
sandbox (gitignored; fetched from
s3://steering-reasoning-artifacts/derived/clamp-arms-n5/data/logs/); the path/arm parsing
tests always run.
"""

import json
from pathlib import Path

import pytest

from steering_reasoning.evals.eval_records import (
    arm_steering,
    check_steering,
    export_records,
    join_ids,
    parse_arm_path,
    service_model_name,
)

REPO = Path(__file__).resolve().parents[1]
from steering_reasoning.steering.clamp import proxy_view, request_sha

SANDBOX = Path(__file__).resolve().parents[1] / "data" / "scratch" / "n5logs"
LOGS = {   # arm -> (log, mode, dose); these archived logs predate arms.tsv, so steering is given by hand
    "base_ab": (SANDBOX / "base_ab_s42.eval", None, None),
    "cn8_ab": (SANDBOX / "cn8_ab_conf_s42.eval", "clamp", -8.0),
}


@pytest.mark.parametrize("path,expected", [
    ("s3://b/derived/clamp-arms-n5/data/logs/cn8_ab/conflicting/s42/", ("cn8_ab", "conflicting", 42)),
    ("/x/logs/base_ab/conflicting/s43/foo.eval", ("base_ab", "conflicting", 43)),
    ("logs/optb_m0/original/s46", ("optb_m0", "original", 46)),
])
def test_parse_arm_path(path, expected):
    m = parse_arm_path(path)
    assert (m["arm"], m["split"], m["seed"]) == expected


def test_parse_arm_path_rejects_unknown():
    with pytest.raises(ValueError):
        parse_arm_path("/logs/cn8_ab/nosplit/s42/")


def test_arm_steering_reads_the_final_sweep_table():
    table = REPO / "experiments" / "diff-mean-steering" / "arms.tsv"
    canonical = "src/steering_reasoning/canonical/u_L35_steering.safetensors"
    assert arm_steering(table, "base", REPO) == (None, None, None)
    assert arm_steering(table, "cn8", REPO) == ("clamp", -8.0, canonical)
    assert arm_steering(table, "c0", REPO) == ("clamp", 0.0, canonical)
    assert arm_steering(table, "addn4", REPO) == ("add", -4.0, canonical)
    with pytest.raises(ValueError, match="not in"):
        arm_steering(table, "optb_m0", REPO)


def test_check_steering():
    check_steering(None, None); check_steering("clamp", -8.0); check_steering("ablate", None)
    for bad in [("clamp", None), ("add", None), (None, -8.0), ("ablate", 1.0), ("weird", 1.0)]:
        with pytest.raises(ValueError):
            check_steering(*bad)


def test_service_model_name():
    assert service_model_name("openai-api/local/Qwen/Qwen3.5-27B") == "Qwen/Qwen3.5-27B"
    assert service_model_name("Qwen/Qwen3.5-27B") == "Qwen/Qwen3.5-27B"


@pytest.fixture(scope="module", params=list(LOGS))
def records(request):
    path, mode, dose = LOGS[request.param]
    if not path.exists():
        pytest.skip(f"{path} not present")
    recs = export_records([path], arm=request.param, mode=mode, dose=dose, seed=42, split="conflicting",
                          select=["lcbhard_13"])
    assert len(recs) == 1
    return request.param, (mode, dose), recs[0]


def test_record_shape(records):
    arm, thr, r = records
    assert r["task_id"] == "lcbhard_13" and r["rollout_id"] == f"{arm}/conflicting/s42/lcbhard_13"
    assert r["arm"] == arm and (r["mode"], r["dose"]) == thr and r["seed"] == 42
    assert r["model"] == "Qwen/Qwen3.5-27B" and r["model_base_url"].startswith("http")
    assert isinstance(r["is_reward_hack"], bool) and r["solved"] is None
    assert r["max_tokens_effective"] == 16000
    assert r["messages"][0]["role"] == "system"


def test_turns_join_and_wire_format(records):
    arm, thr, r = records
    n_assistant = sum(m["role"] == "assistant" for m in r["messages"])
    assert len(r["turns"]) == n_assistant > 0
    prev_mi = -1
    for t in r["turns"]:
        mi = t["message_index"]
        assert mi > prev_mi and r["messages"][mi]["role"] == "assistant"
        prev_mi = mi
        req = t["request"]
        assert len(req["messages"]) == mi  # the request is exactly the transcript prefix
        assert req["messages"][-1]["role"] in ("user", "tool")
        assert req["tool_choice"] == "auto" and req["tools"]
        assert all(tl["function"]["strict"] is True for tl in req["tools"])
        assert req["temperature"] == 0.7 and req["top_p"] == 0.95 and req["seed"] == 42
        assert "max_tokens" not in req  # stamped by the proxy, not Inspect
        assert t["usage"]["prompt_tokens"] > 0 and t["usage"]["completion_tokens"] > 0
        assert t["synthetic_length"] is False
        assert t["response"]["role"] == "assistant" and "reasoning" in t["response"]
        assert t["prompt_token_ids"] is None and t["token_ids"] is None  # n5 logs carry none
    # prior assistant turns go on the wire with a `reasoning` field (the proxy strips think
    # tags from it and vLLM maps it to reasoning_content)
    a = next(m for m in r["turns"][-1]["request"]["messages"] if m["role"] == "assistant")
    assert isinstance(a.get("reasoning"), str)
    tc = a["tool_calls"][0]
    assert isinstance(tc["function"]["arguments"], str)  # JSON string on the wire
    # tools are identical across turns of a rollout
    assert all(t["request"]["tools"] == r["turns"][0]["request"]["tools"] for t in r["turns"])
    assert all(t["ids_join"] == "none" for t in r["turns"])  # no sidecar given


def test_ids_sidecar_join(records, tmp_path: Path):
    """A sidecar written by the proxy (keyed by the hash of the forwarded request) joins onto
    the record rebuilt from the log; retries disambiguate by completion_tokens."""
    arm, thr, r = records
    t1, t2 = r["turns"][1], r["turns"][2]
    key1 = request_sha(proxy_view(t1["request"]["messages"]), t1["request"]["tools"])
    key2 = request_sha(proxy_view(t2["request"]["messages"]), t2["request"]["tools"])
    ct1, ct2 = t1["usage"]["completion_tokens"], t2["usage"]["completion_tokens"]
    lines = [   # the proxy stamps the request's seed on every line; the log's requests carry seed 42
        {"request_sha": key1, "seed": 42, "completion_tokens": ct1 + 7, "token_ids": [9], "prompt_ids_sha256": "old", "n_prompt_ids": 1},
        {"request_sha": key1, "seed": 42, "completion_tokens": ct1, "token_ids": [1, 2, 3], "prompt_ids_sha256": "abc", "n_prompt_ids": t1["usage"]["prompt_tokens"]},
        {"request_sha": key2, "seed": 42, "completion_tokens": ct2 + 1, "token_ids": [4], "prompt_ids_sha256": "zzz", "n_prompt_ids": 5},
    ]
    p = tmp_path / "ids.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    path, mode, dose = LOGS[arm]
    (rec,) = export_records([path], arm=arm, mode=mode, dose=dose, seed=42, split="conflicting",
                            select=["lcbhard_13"], ids_logs=[p])
    a, b, c = rec["turns"][0], rec["turns"][1], rec["turns"][2]
    assert a["ids_join"] == "missing" and a["token_ids"] is None
    assert b["ids_join"] == "sidecar" and b["token_ids"] == [1, 2, 3] and b["prompt_ids_sha256"] == "abc"
    assert c["ids_join"] == "length_mismatch" and c["token_ids"] is None
    # pure join helper: ambiguity is reported, last match wins
    idx = {key1: [{"seed": 42, "completion_tokens": ct1, "token_ids": [1]}, {"seed": 42, "completion_tokens": ct1, "token_ids": [2]}]}
    j = join_ids(idx, t1["request"], t1["usage"])
    assert j["ids_join"] == "sidecar_ambiguous" and j["token_ids"] == [2]


# --- the sidecar's spans / seed join ---


def test_join_ids_uses_seed_and_carries_steer_positions():
    req = {"messages": [{"role": "user", "content": "hi"}], "tools": None, "seed": 43}
    key = request_sha(proxy_view(req["messages"]), None)
    idx = {key: [{"seed": 42, "completion_tokens": 5, "token_ids": [1], "steer_positions": []},
                 {"seed": 43, "completion_tokens": 5, "token_ids": [2], "steer_positions": [[3, 9]],
                  "prompt_ids_sha256": "s", "n_prompt_ids": 4}]}
    j = join_ids(idx, req, {"completion_tokens": 5})
    assert j["ids_join"] == "sidecar" and j["token_ids"] == [2] and j["steer_positions"] == [[3, 9]]
    # a request without a seed cannot filter and reports the ambiguity
    j = join_ids(idx, {**req, "seed": None}, {"completion_tokens": 5})
    assert j["ids_join"] == "sidecar_ambiguous" and j["token_ids"] == [2]
    # no line for this seed; a line without a seed never matches a seeded request
    assert join_ids(idx, {**req, "seed": 44}, {"completion_tokens": 5})["ids_join"] == "missing"
    assert join_ids({key: [{"completion_tokens": 5, "token_ids": [3]}]}, req, {"completion_tokens": 5})["ids_join"] == "missing"


def test_join_ids_collapses_identical_duplicates():
    """A first turn shared by two runs of the same seed, or a retried turn, leaves several sidecar
    lines with the same ids: one answer, joined as `sidecar`. Differing ids stay ambiguous."""
    req = {"messages": [{"role": "user", "content": "hi"}], "tools": None, "seed": 42}
    key = request_sha(proxy_view(req["messages"]), None)
    line = {"seed": 42, "completion_tokens": 5, "token_ids": [1, 2], "steer_positions": [[0, 3]], "prompt_ids_sha256": "p"}
    j = join_ids({key: [dict(line, ts=1.0), dict(line, ts=2.0)]}, req, {"completion_tokens": 5})
    assert j["ids_join"] == "sidecar" and j["token_ids"] == [1, 2] and j["steer_positions"] == [[0, 3]]
    j = join_ids({key: [line, dict(line, steer_positions=None)]}, req, {"completion_tokens": 5})
    assert j["ids_join"] == "sidecar_ambiguous"


def test_join_ids_window_separates_first_turns_of_different_tasks():
    """Every task's first request hashes the same (the task is in sandbox files), so a seed's
    sidecar holds one line per task under one key; the event's time window picks the rollout's own."""
    req = {"messages": [{"role": "user", "content": "go"}], "tools": None, "seed": 43}
    key = request_sha(proxy_view(req["messages"]), None)
    lines = [{"seed": 43, "completion_tokens": 68, "token_ids": [i], "ts": 1000.0 + 100 * i} for i in range(5)]
    idx = {key: lines}
    j = join_ids(idx, req, {"completion_tokens": 68}, window=(1195.0, 1203.0))   # line 2 finished at 1200
    assert j["ids_join"] == "sidecar" and j["token_ids"] == [2]
    j = join_ids(idx, req, {"completion_tokens": 68}, window=(1290.0, None))     # open end: lines 3 and 4
    assert j["ids_join"] == "sidecar_ambiguous" and j["token_ids"] == [4]
    j = join_ids(idx, req, {"completion_tokens": 68}, window=(5000.0, 5100.0))   # nothing in the window: fall back
    assert j["ids_join"] == "sidecar_ambiguous" and j["token_ids"] == [4]
    assert join_ids(idx, req, {"completion_tokens": 68})["ids_join"] == "sidecar_ambiguous"
    # a candidate without a ts disables the narrowing rather than being dropped
    idx2 = {key: [dict(lines[2]), {"seed": 43, "completion_tokens": 68, "token_ids": [9]}]}
    j = join_ids(idx2, req, {"completion_tokens": 68}, window=(1195.0, 1203.0))
    assert j["ids_join"] == "sidecar_ambiguous"


# --- one record per rollout: restarted duplicates and message-limit extensions ---


def _rec(rid, *, messages, turns, score=None, error=None, eval_file="2026-01-01T00_x.eval", resumed=None, limit=None):
    return {"rollout_id": rid, "messages": list(messages), "turns": [{"message_index": i} for i in turns],
            "score": score or {}, "error": error, "eval_file": eval_file, "limit": limit,
            "resumed_from_messages": resumed}


def test_dedup_records_prefers_scored_then_latest():
    from steering_reasoning.evals.eval_records import dedup_records

    a = _rec("b/conflicting/s42/t1", messages="ab", turns=[1], error="hung", eval_file="2026-01-02_late.eval")
    b = _rec("b/conflicting/s42/t1", messages="ab", turns=[1], score={"s": "I"}, eval_file="2026-01-01_early.eval")
    c = _rec("b/conflicting/s42/t1", messages="ab", turns=[1], score={"s": "C"}, eval_file="2026-01-03_latest.eval")
    d = _rec("b/conflicting/s42/t2", messages="ab", turns=[1], score={"s": "C"})
    out = dedup_records([a, b, c, d])
    assert [r["rollout_id"] for r in out] == ["b/conflicting/s42/t1", "b/conflicting/s42/t2"]
    assert out[0] is c
    assert dedup_records([a, b])[0] is b   # scored beats a later errored record


def test_merge_extended_joins_turns_and_checks_the_prefix():
    from steering_reasoning.evals.eval_records import merge_extended

    msgs60 = [{"role": "user", "content": str(i)} for i in range(6)]
    base = _rec("b/conflicting/s42/t1", messages=msgs60, turns=[1, 3, 5], score={"s": "I"},
                limit={"type": "message", "limit": 60}, eval_file="base.eval")
    ext = _rec("b/conflicting/s42/t1", messages=msgs60 + [{"role": "user", "content": "x"}] * 4, turns=[7, 9],
               score={"s": "C"}, resumed=6, eval_file="ext.eval")
    lone = _rec("b/conflicting/s42/t2", messages=msgs60, turns=[1], score={"s": "C"})
    (m, p) = merge_extended([base, lone], [ext], ext_name="m120")
    assert [t["message_index"] for t in m["turns"]] == [1, 3, 5, 7, 9]
    assert m["score"] == {"s": "C"} and m["limit"] is None and len(m["messages"]) == 10
    assert m["extended"] == "m120" and m["base_eval_file"] == "base.eval" and m["base_limit"]["limit"] == 60
    assert m["resumed_from_messages"] == 6
    assert p["extended"] is None and p["turns"] == lone["turns"]
    # an errored extension is dropped, the base stands
    bad = dict(ext, error="boom")
    (m2, _) = merge_extended([base, lone], [bad], ext_name="m120")
    assert m2["extended"] is None and m2["score"] == {"s": "I"}
    # a prefix that does not match is refused, as is an extension without a base
    wrong = dict(ext, messages=[{"role": "user", "content": "other"}] + ext["messages"][1:])
    with pytest.raises(ValueError, match="not the 6 restored"):
        merge_extended([base], [wrong], ext_name="m120")
    with pytest.raises(ValueError, match="no base record"):
        merge_extended([lone], [ext], ext_name="m120")
    with pytest.raises(ValueError, match="restored messages"):
        merge_extended([base], [dict(ext, turns=[{"message_index": 5}])], ext_name="m120")


def test_load_ids_log_skips_truncated_lines(tmp_path: Path, capsys):
    from steering_reasoning.evals.eval_records import load_ids_log

    p = tmp_path / "ids.jsonl"
    p.write_text(json.dumps({"request_sha": "a", "token_ids": [1]}) + "\n" + '{"foo": 1}\n' + '{"request_sha": "b", "tok')
    idx = load_ids_log([p])
    assert list(idx) == ["a"] and "skipped 2" in capsys.readouterr().out
