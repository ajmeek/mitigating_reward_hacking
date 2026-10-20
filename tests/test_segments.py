"""Tests for span-segment regrouping.

The upstream tagger splits reasoning by line and then by sentence, which cuts units that
cannot stand alone: a fenced code block becomes one unit per line, and enumeration markers,
bare block openers and colon lead-ins get split off. `regroup_segments` repairs those
boundaries without merging units the tagger got right.
"""

import pytest

from steering_reasoning.steering.segments import (
    locate_segments,
    reasoning_segments_by_message,
    regroup_segments,
)


def segs(*texts):
    return [{"text": t} for t in texts]


class TestFencedCode:
    def test_closed_fence_becomes_one_unit(self):
        assert regroup_segments(
            segs("Let me write it:", "```python", "x = 1", "y = 2", "```", "Done.")
        ) == [[0, 1, 2, 3, 4], [5]]

    def test_unclosed_fence_stops_at_prose(self):
        # The model opens a block and drifts back into prose without closing it (3 of 1593
        # messages in the reward-hacking set). Merging to end-of-message swallowed 987
        # segments in the worst case, so the block ends where the content stops being code.
        assert regroup_segments(
            segs("```python", "def f(x):", "    return x + 1", "# a comment",
                 "Let me think about this.", "More prose.")
        ) == [[0, 1, 2, 3], [4], [5]]

    def test_unclosed_fence_followed_immediately_by_prose(self):
        assert regroup_segments(segs("```python", "Actually never mind.")) == [[0], [1]]

    def test_trailing_unmatched_fence(self):
        assert regroup_segments(segs("Some prose.", "```")) == [[0], [1]]

    def test_prose_is_not_absorbed_into_a_code_block(self):
        # "This is complex." has no comment marker, so it must terminate the block rather
        # than be swallowed by it.
        assert regroup_segments(
            segs("```python", "y = 2", "This is complex.", "z = 3")
        ) == [[0, 1], [2], [3]]


class TestIntroducers:
    @pytest.mark.parametrize("marker", ["1.", "2)", "else:", "try:", "-"])
    def test_fragment_attaches_to_what_it_introduces(self, marker):
        assert regroup_segments(segs(marker, "The actual content.")) == [[0, 1]]

    def test_colon_lead_in_attaches_forward(self):
        assert regroup_segments(
            segs("The problem says:", "Do the thing.", "1.", "First.")
        ) == [[0, 1], [2, 3]]

    def test_long_colon_ender_stands_alone(self):
        # Over the length cap these are real sentences that happen to introduce a formula;
        # merging them dilutes the unit.
        long_sentence = "For a 1D line of length L, the sum of |i-j| over all pairs i<j is:"
        assert len(long_sentence) > 60
        assert regroup_segments(segs(long_sentence, "n*(n-1)/2")) == [[0], [1]]

    def test_run_of_introducers_cannot_swallow_a_message(self):
        groups = regroup_segments(segs("1.", "2.", "3.", "4.", "5.", "6.", "content"))
        assert max(len(g) for g in groups) <= 5

    def test_trailing_fragments_are_not_dropped(self):
        groups = regroup_segments(segs("Real sentence.", "1."))
        assert sorted(i for g in groups for i in g) == [0, 1]


class TestPlainProse:
    def test_ordinary_sentences_are_left_alone(self):
        assert regroup_segments(
            segs("First thought.", "Second thought.", "Third thought.")
        ) == [[0], [1], [2]]

    def test_every_segment_appears_exactly_once(self):
        s = segs("A:", "```python", "x = 1", "```", "Prose.", "1.", "Item.", "End.")
        groups = regroup_segments(s)
        assert sorted(i for g in groups for i in g) == list(range(len(s)))


class TestSpanLocation:
    def test_segments_are_located_in_order(self):
        text = "First part. Second part. Third part."
        located = locate_segments(text, segs("First part.", "Second part.", "Third part."))
        assert [(a, b) for _, a, b in located] == [(0, 11), (12, 24), (25, 36)]

    def test_missing_segment_raises(self):
        with pytest.raises(ValueError):
            locate_segments("only this text", segs("not present"))

    def test_reasoning_sections_are_grouped_by_message(self):
        record = {"span_labels": {"segments": [
            {"section": "msg4:reasoning", "text": "a"},
            {"section": "msg4:tool0", "text": "b"},
            {"section": "msg6:reasoning", "text": "c"},
        ]}}
        by_msg = reasoning_segments_by_message(record)
        assert sorted(by_msg) == [4, 6]
        assert [s["text"] for s in by_msg[4]] == ["a"]
