"""Tests for `load_dump`, the reader for dump_activations.py output dirs.

A shard holds `token_acts` (N_tokens, L, D) alongside the turn/segment means — ~10 GB
across the 27B uncapped dump. No analysis has ever consumed it, and `load_dump` never put
it in a row, but the original implementation read every tensor in the file and dropped it.
These tests pin the two things that fix depends on: the default read still returns exactly
what it used to, and `layers=` slices consistently with manual indexing while filtering the
manifest so `manifest["layers"].index(L)` keeps working.

Built on a synthetic dump rather than the real captures, which are gitignored.
"""

import json

import pytest

# Serving-env dependency: the root eval env has no torch (see CLAUDE.md). Run these
# via `uv run --with torch --with safetensors pytest`; bare pytest skips them.
torch = pytest.importorskip("torch")

from steering_reasoning.activations import load_dump  # noqa: E402

LAYERS = [3, 11, 35]
D = 8
SHAPES = [(4, 9, 40), (3, 5, 25)]  # (turns, segments, tokens) per transcript


@pytest.fixture(scope="module")
def dump(tmp_path_factory):
    from safetensors.torch import save_file

    out = tmp_path_factory.mktemp("dump")
    torch.manual_seed(0)
    transcripts = []
    for i, (n_turn, n_seg, n_tok) in enumerate(SHAPES):
        tid = f"fake_{i}"
        save_file({
            "turn_means": torch.randn(n_turn, len(LAYERS), D),
            "turn_tokens": torch.tensor([n_tok // n_turn] * n_turn, dtype=torch.int32),
            "segment_means": torch.randn(n_seg, len(LAYERS), D).half(),
            "segment_tokens": torch.tensor([3] * n_seg, dtype=torch.int32),
            "token_acts": torch.randn(n_tok, len(LAYERS), D).half(),
            "token_pos": torch.arange(n_tok, dtype=torch.int32),
            "token_turn": torch.zeros(n_tok, dtype=torch.int32),
        }, str(out / f"{tid}.safetensors"))
        (out / f"{tid}.json").write_text(json.dumps({
            "task_id": tid, "label": True, "has_gold_spans": i == 0, "turns": [],
            "segments": [{"message_index": 2, "segment_idx": j, "position": j,
                          "n_tokens": 3, "gold_label": False, "judge_label": j == 0,
                          "text": f"s{j}"} for j in range(n_seg)],
        }))
        transcripts.append({"task_id": tid, "label": True, "has_gold_spans": i == 0,
                            "n_turns": n_turn, "n_tokens": n_tok, "n_segments": n_seg})
    (out / "manifest.json").write_text(
        json.dumps({"layers": LAYERS, "transcripts": transcripts}))
    return out


def reference_load_dump(dump_dir):
    """The pre-fix implementation, verbatim, as the equivalence target."""
    from safetensors.torch import load_file

    manifest = json.loads((dump_dir / "manifest.json").read_text())
    rows = []
    for t in manifest["transcripts"]:
        tensors = load_file(str(dump_dir / f"{t['task_id']}.safetensors"))
        meta = json.loads((dump_dir / f"{t['task_id']}.json").read_text())
        w = tensors["turn_tokens"].float()
        trace_mean = (tensors["turn_means"] * w[:, None, None]).sum(0) / w.sum()
        rows.append({
            "task_id": t["task_id"], "label": t["label"], "gold": t["has_gold_spans"],
            "trace_mean": trace_mean, "turn_means": tensors["turn_means"],
            "turn_tokens": tensors["turn_tokens"],
            "segment_means": tensors.get("segment_means"),
            "segment_meta": meta["segments"],
        })
    return manifest, rows


class TestDefaultRead:
    def test_matches_the_original_implementation_exactly(self, dump):
        m_ref, r_ref = reference_load_dump(dump)
        m_new, r_new = load_dump(dump)
        assert m_ref == m_new
        assert len(r_ref) == len(r_new)
        for a, b in zip(r_ref, r_new):
            assert a.keys() == b.keys()
            for k in a:
                if torch.is_tensor(a[k]):
                    assert a[k].dtype == b[k].dtype, k
                    assert torch.equal(a[k], b[k]), k
                else:
                    assert a[k] == b[k], k

    def test_does_not_surface_token_acts(self, dump):
        # It is read only on request; a row that carried it silently would have masked
        # the cost this fix removes.
        _, rows = load_dump(dump)
        assert "token_acts" not in rows[0]

    def test_segment_means_is_none_when_the_dump_has_none(self, dump, tmp_path):
        from safetensors.torch import load_file, save_file

        bare = tmp_path / "bare"
        bare.mkdir()
        for f in dump.iterdir():
            if f.suffix == ".json":
                (bare / f.name).write_text(f.read_text())
            else:
                t = load_file(str(f))
                save_file({k: v for k, v in t.items() if not k.startswith("segment")},
                          str(bare / f.name))
        _, rows = load_dump(bare)
        assert all(r["segment_means"] is None for r in rows)


class TestLayerSubset:
    def test_slices_match_manual_indexing(self, dump):
        _, full = load_dump(dump)
        manifest, sliced = load_dump(dump, layers=[35])
        li = LAYERS.index(35)
        for a, b in zip(full, sliced):
            assert torch.equal(a["turn_means"][:, li], b["turn_means"][:, 0])
            assert torch.equal(a["segment_means"][:, li], b["segment_means"][:, 0])
            assert torch.equal(a["trace_mean"][li], b["trace_mean"][0])
            assert torch.equal(a["turn_tokens"], b["turn_tokens"])

    def test_manifest_is_filtered_so_the_index_idiom_still_works(self, dump):
        manifest, rows = load_dump(dump, layers=[35])
        assert manifest["layers"] == [35]
        assert manifest["layers"].index(35) == 0          # the idiom every caller uses
        assert rows[0]["segment_means"].shape[1] == 1
        assert manifest["transcripts"] == json.loads(
            (dump / "manifest.json").read_text())["transcripts"]

    def test_naming_every_captured_layer_is_a_no_op(self, dump):
        _, full = load_dump(dump)
        m, same = load_dump(dump, layers=LAYERS)
        assert m["layers"] == LAYERS
        for a, b in zip(full, same):
            assert torch.equal(a["turn_means"], b["turn_means"])
            assert torch.equal(a["segment_means"], b["segment_means"])

    def test_reorders_to_the_requested_order(self, dump):
        _, full = load_dump(dump)
        _, out = load_dump(dump, layers=[35, 3])
        for a, b in zip(full, out):
            assert torch.equal(a["turn_means"][:, LAYERS.index(35)], b["turn_means"][:, 0])
            assert torch.equal(a["turn_means"][:, LAYERS.index(3)], b["turn_means"][:, 1])

    def test_unknown_layer_raises(self, dump):
        with pytest.raises(ValueError):
            load_dump(dump, layers=[999])


class TestExtraTensors:
    def test_token_acts_on_request(self, dump):
        _, rows = load_dump(dump, extra_tensors=("token_acts", "token_turn"))
        acts, turn = rows[0]["token_acts"], rows[0]["token_turn"]
        n_tok = SHAPES[0][2]
        assert acts.shape == (n_tok, len(LAYERS), D)
        assert acts.dtype == torch.float16
        assert turn.shape == (n_tok,)

    def test_extra_tensors_respect_the_layer_subset(self, dump):
        _, full = load_dump(dump, extra_tensors=("token_acts",))
        _, sliced = load_dump(dump, layers=[11], extra_tensors=("token_acts",))
        li = LAYERS.index(11)
        for a, b in zip(full, sliced):
            assert b["token_acts"].shape[1] == 1
            assert torch.equal(a["token_acts"][:, li], b["token_acts"][:, 0])

    def test_absent_key_becomes_none_rather_than_raising(self, dump):
        _, rows = load_dump(dump, extra_tensors=("no_such_tensor",))
        assert rows[0]["no_such_tensor"] is None
