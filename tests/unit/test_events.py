import pytest

from hypline.bids import BIDSPath
from hypline.events import Segment, load_segments, segment_tr_slice
from hypline.layout import BIDSLayout

from .conftest import BIDSTree

SUB = "001"
TASK = "conv"
SPACE = "MNI152NLin6Asym"

_SEGMENT_ROWS = [
    {"trial_type": "block-1", "onset": 0.0, "duration": 100.0},
    {"trial_type": "block-2", "onset": 100.0, "duration": 100.0},
]


class TestSegmentTrSlice:
    """Pin the left-inclusive `[t*TR, (t+1)*TR)` boundary: onset k*TR lands in TR k."""

    def _seg(self, onset: float, duration: float) -> Segment:
        return Segment(
            entity="block", value="1", onset=onset, duration=duration, metadata={}
        )

    def test_onset_zero_lands_in_tr_zero(self):
        assert segment_tr_slice(self._seg(0.0, 10.0), 2.0) == slice(0, 5)

    def test_onset_k_times_tr_lands_in_tr_k(self):
        # onset = 3 * TR (= 6.0s) must map to TR 3, not TR 2
        assert segment_tr_slice(self._seg(6.0, 4.0), 2.0).start == 3

    def test_duration_translates_to_tr_count(self):
        assert segment_tr_slice(self._seg(10.0, 90.0), 2.0) == slice(5, 50)


class TestLoadSegments:
    def test_missing_task_entity_raises(self, tree: BIDSTree):
        stim = tree.add_stimulus(sub=SUB, task=TASK, kind="audio", ext=".wav")
        source_no_task = BIDSPath(stim).without_entity("task")
        with pytest.raises(ValueError, match="missing required 'task' entity"):
            load_segments(BIDSLayout(tree.root), source_no_task)

    def test_no_events_returns_empty(self, tree: BIDSTree):
        stim = tree.add_stimulus(sub=SUB, task=TASK, run="1", kind="audio", ext=".wav")
        segments = load_segments(BIDSLayout(tree.root), BIDSPath(stim))
        assert segments == []

    def test_source_is_stimulus_path(self, tree: BIDSTree):
        # No BOLD file on disk — must still resolve events from stimulus path
        stim = tree.add_stimulus(sub=SUB, task=TASK, run="1", kind="audio", ext=".wav")
        tree.add_events(sub=SUB, task=TASK, run="1", rows=_SEGMENT_ROWS)
        segments = load_segments(BIDSLayout(tree.root), BIDSPath(stim))
        assert [seg.value for seg in segments] == ["1", "2"]
        assert [seg.onset for seg in segments] == [0.0, 100.0]

    def test_source_is_feature_path_matches_bold(self, tree: BIDSTree):
        bold_path = tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1")
        feat_path = tree.add_feature(sub=SUB, task=TASK, run="1", kind="mfcc")
        tree.add_events(sub=SUB, task=TASK, run="1", rows=_SEGMENT_ROWS)
        layout = BIDSLayout(tree.root)
        from_feat = load_segments(layout, BIDSPath(feat_path))
        from_bold = load_segments(layout, BIDSPath(bold_path))
        assert from_feat == from_bold

    def test_levels_without_segment_rows_raises(self, tree: BIDSTree):
        # events.tsv has only flat labels (no segment rows); events.json declares Levels
        bold_path = tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=[{"trial_type": "rest", "onset": 0.0, "duration": 10.0}],
            sidecar_json={
                "trial_type": {"Levels": {"block-1": {"metadata": {"cond": "a"}}}}
            },
        )
        with pytest.raises(ValueError, match="no segment rows"):
            load_segments(BIDSLayout(tree.root), BIDSPath(bold_path))

    def test_levels_metadata_merged(self, tree: BIDSTree):
        bold_path = tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=_SEGMENT_ROWS,
            sidecar_json={
                "trial_type": {
                    "Levels": {
                        "block-1": {"metadata": {"cond": "a"}},
                        "block-2": {"metadata": {"cond": "b"}},
                    }
                }
            },
        )
        segments = load_segments(BIDSLayout(tree.root), BIDSPath(bold_path))
        assert segments[0].metadata == {"cond": "a"}
        assert segments[1].metadata == {"cond": "b"}
