import polars as pl
import pytest

from hypline.bids import BIDSPath
from hypline.events import (
    Segment,
    Turn,
    load_segments,
    load_turns,
    merge_filename_and_sidecar,
    resolve_entities,
    segment_tr_slice,
    stamp_turns,
)
from hypline.layout import BIDSLayout

from .conftest import BIDSTree

SUB = "001"
DYAD = "101"
TASK = "conv"
SPACE = "MNI152NLin6Asym"

_SEGMENT_ROWS = [
    {"trial_type": "block-1", "onset": 0.0, "duration": 100.0},
    {"trial_type": "block-2", "onset": 100.0, "duration": 100.0},
]

_SEGMENT_LEVELS = {
    "trial_type": {
        "Levels": {
            "block-1": {"metadata": {"cond": "R", "item": "101"}},
            "block-2": {"metadata": {"cond": "L", "item": "102"}},
        }
    }
}


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
        stim = tree.add_stimulus(dyad=DYAD, task=TASK, kind="audio", ext=".wav")
        source_no_task = BIDSPath(stim).without_entity("task")
        with pytest.raises(ValueError, match="missing required 'task' entity"):
            load_segments(BIDSLayout(tree.root), source_no_task)

    def test_no_events_returns_empty(self, tree: BIDSTree):
        tree.add_participants({SUB: DYAD})
        stim = tree.add_stimulus(
            dyad=DYAD, task=TASK, run="1", kind="audio", ext=".wav"
        )
        segments = load_segments(BIDSLayout(tree.root), BIDSPath(stim))
        assert segments == []

    def test_header_only_events_returns_empty(self, tree: BIDSTree):
        # A present-but-rowless events.tsv has a str-typed empty `duration` column
        # that the `> 0.0` filter cannot compare; treat it as an unsegmented run.
        tree.add_participants({SUB: DYAD})
        stim = tree.add_stimulus(
            dyad=DYAD, task=TASK, run="1", kind="audio", ext=".wav"
        )
        tree.add_events(sub=SUB, task=TASK, run="1", rows=[])
        segments = load_segments(BIDSLayout(tree.root), BIDSPath(stim))
        assert segments == []

    def test_source_is_stimulus_path(self, tree: BIDSTree):
        # No BOLD file on disk — must still resolve events from stimulus path
        tree.add_participants({SUB: DYAD})
        stim = tree.add_stimulus(
            dyad=DYAD, task=TASK, run="1", kind="audio", ext=".wav"
        )
        tree.add_events(sub=SUB, task=TASK, run="1", rows=_SEGMENT_ROWS)
        segments = load_segments(BIDSLayout(tree.root), BIDSPath(stim))
        assert [seg.value for seg in segments] == ["1", "2"]
        assert [seg.onset for seg in segments] == [0.0, 100.0]

    def test_source_is_feature_path_matches_bold(self, tree: BIDSTree):
        tree.add_participants({SUB: DYAD})
        bold_path = tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1")
        feat_path = tree.add_feature(dyad=DYAD, task=TASK, run="1", kind="mfcc")
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

    def test_turn_speaker_rows_ignored_alongside_segments(self, tree: BIDSTree):
        # The load-bearing coexistence claim: turn_speaker is a flat label, so it
        # never enters segment parsing even when segments are present.
        tree.add_participants({SUB: DYAD})
        bold_path = tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=[
                *_SEGMENT_ROWS,
                {"trial_type": "turn_speaker", "onset": 0.0, "duration": 50.0},
            ],
        )
        segments = load_segments(BIDSLayout(tree.root), BIDSPath(bold_path))
        assert [seg.entity for seg in segments] == ["block", "block"]
        assert [seg.value for seg in segments] == ["1", "2"]


class TestMergeFilenameAndSidecar:
    STRUCTURAL = frozenset({"ses", "run", "trial"})

    def test_sidecar_only_key_adopted(self):
        merged = merge_filename_and_sidecar(
            filename_entities={"ses": "1", "run": "2", "trial": "3"},
            sidecar_metadata={"cond": "R", "item": "101"},
            structural_keys=self.STRUCTURAL,
        )
        assert merged == {
            "ses": "1",
            "run": "2",
            "trial": "3",
            "cond": "R",
            "item": "101",
        }

    def test_both_same_allowed(self):
        merged = merge_filename_and_sidecar(
            filename_entities={"ses": "1", "run": "2", "trial": "3", "cond": "R"},
            sidecar_metadata={"cond": "R"},
            structural_keys=self.STRUCTURAL,
        )
        assert merged == {"ses": "1", "run": "2", "trial": "3", "cond": "R"}

    def test_structural_only_filename_returns_as_is(self):
        merged = merge_filename_and_sidecar(
            filename_entities={"ses": "1", "run": "2", "trial": "3"},
            sidecar_metadata={},
            structural_keys=self.STRUCTURAL,
        )
        assert merged == {"ses": "1", "run": "2", "trial": "3"}

    def test_both_differ_raises(self):
        with pytest.raises(ValueError, match="disagree on .cond."):
            merge_filename_and_sidecar(
                filename_entities={"run": "2", "trial": "3", "cond": "R"},
                sidecar_metadata={"cond": "L"},
                structural_keys=self.STRUCTURAL,
            )

    def test_error_names_offending_key_and_values(self):
        with pytest.raises(
            ValueError, match=r"'cond'.*filename has 'R'.*sidecar has 'L'"
        ):
            merge_filename_and_sidecar(
                filename_entities={"run": "7", "cond": "R"},
                sidecar_metadata={"cond": "L"},
                structural_keys=self.STRUCTURAL,
            )

    def test_filename_only_descriptive_raises(self):
        with pytest.raises(ValueError, match="absent from events.json"):
            merge_filename_and_sidecar(
                filename_entities={"run": "2", "trial": "3", "item": "101"},
                sidecar_metadata={"cond": "R"},
                structural_keys=self.STRUCTURAL,
            )

    def test_empty_sidecar_with_descriptive_filename_raises(self):
        with pytest.raises(ValueError, match="absent from events.json"):
            merge_filename_and_sidecar(
                filename_entities={"run": "2", "cond": "R"},
                sidecar_metadata={},
                structural_keys=self.STRUCTURAL,
            )


class TestResolveEntities:
    def test_unsegmented_run_returns_filename_entities(self, tree: BIDSTree):
        tree.add_participants({SUB: DYAD})
        stim = tree.add_stimulus(
            dyad=DYAD, task=TASK, run="1", kind="audio", ext=".wav"
        )
        merged = resolve_entities(BIDSLayout(tree.root), BIDSPath(stim))
        assert merged == {"dyad": DYAD, "task": TASK, "run": "1"}

    def test_task_escape_hatch_merges_run_level_metadata(self, tree: BIDSTree):
        # Escape hatch: `task-<value>` row reuses filename's `task` as segment entity
        tree.add_participants({SUB: DYAD})
        stim = tree.add_stimulus(
            dyad=DYAD, task=TASK, run="1", kind="audio", ext=".wav"
        )
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=[{"trial_type": f"task-{TASK}", "onset": 5.0, "duration": 100.0}],
            sidecar_json={
                "trial_type": {"Levels": {f"task-{TASK}": {"metadata": {"cond": "R"}}}}
            },
        )
        merged = resolve_entities(BIDSLayout(tree.root), BIDSPath(stim))
        assert merged == {
            "dyad": DYAD,
            "task": TASK,
            "run": "1",
            "cond": "R",
        }

    def test_segmented_stimulus_merges_sidecar_metadata(self, tree: BIDSTree):
        # No BOLD file — proves stimulus discovery works without it
        tree.add_participants({SUB: DYAD})
        stim = tree.add_stimulus(
            dyad=DYAD,
            task=TASK,
            run="1",
            kind="audio",
            ext=".wav",
            extra_entities={"block": "1"},
        )
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=_SEGMENT_ROWS,
            sidecar_json=_SEGMENT_LEVELS,
        )
        merged = resolve_entities(BIDSLayout(tree.root), BIDSPath(stim))
        assert merged == {
            "dyad": DYAD,
            "task": TASK,
            "run": "1",
            "block": "1",
            "cond": "R",
            "item": "101",
        }

    def test_segmented_feature_path_matches_stimulus(self, tree: BIDSTree):
        tree.add_participants({SUB: DYAD})
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1")
        feat = tree.add_feature(
            dyad=DYAD,
            task=TASK,
            run="1",
            kind="mfcc",
            extra_entities={"block": "2"},
        )
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=_SEGMENT_ROWS,
            sidecar_json=_SEGMENT_LEVELS,
        )
        merged = resolve_entities(BIDSLayout(tree.root), BIDSPath(feat))
        assert merged["cond"] == "L"
        assert merged["item"] == "102"

    def test_missing_segment_entity_on_filename_raises(self, tree: BIDSTree):
        tree.add_participants({SUB: DYAD})
        stim = tree.add_stimulus(
            dyad=DYAD, task=TASK, run="1", kind="audio", ext=".wav"
        )
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=_SEGMENT_ROWS,
            sidecar_json=_SEGMENT_LEVELS,
        )
        with pytest.raises(ValueError, match="missing segment entity 'block'"):
            resolve_entities(BIDSLayout(tree.root), BIDSPath(stim))

    def test_unknown_segment_value_raises(self, tree: BIDSTree):
        tree.add_participants({SUB: DYAD})
        stim = tree.add_stimulus(
            dyad=DYAD,
            task=TASK,
            run="1",
            kind="audio",
            ext=".wav",
            extra_entities={"block": "9"},
        )
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=_SEGMENT_ROWS,
            sidecar_json=_SEGMENT_LEVELS,
        )
        with pytest.raises(ValueError, match=r"block-9.*not found"):
            resolve_entities(BIDSLayout(tree.root), BIDSPath(stim))

    def test_filename_descriptive_entity_absent_from_sidecar_raises(
        self, tree: BIDSTree
    ):
        tree.add_participants({SUB: DYAD})
        stim = tree.add_stimulus(
            dyad=DYAD,
            task=TASK,
            run="1",
            kind="audio",
            ext=".wav",
            extra_entities={"block": "1", "cond": "R"},
        )
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=_SEGMENT_ROWS,
            sidecar_json={
                "trial_type": {
                    "Levels": {
                        "block-1": {"metadata": {"item": "101"}},
                        "block-2": {"metadata": {"item": "102"}},
                    }
                }
            },
        )
        with pytest.raises(ValueError, match="absent from events.json"):
            resolve_entities(BIDSLayout(tree.root), BIDSPath(stim))

    def test_filename_sidecar_disagreement_raises(self, tree: BIDSTree):
        tree.add_participants({SUB: DYAD})
        stim = tree.add_stimulus(
            dyad=DYAD,
            task=TASK,
            run="1",
            kind="audio",
            ext=".wav",
            extra_entities={"block": "1", "cond": "L"},
        )
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=_SEGMENT_ROWS,
            sidecar_json=_SEGMENT_LEVELS,
        )
        with pytest.raises(ValueError, match=r"disagree on 'cond'"):
            resolve_entities(BIDSLayout(tree.root), BIDSPath(stim))


SUB_A = "001"
SUB_B = "002"


def _row(trial_type: str, onset: float, duration: float) -> dict:
    return {"trial_type": trial_type, "onset": onset, "duration": duration}


class TestLoadTurns:
    def test_unions_both_partners_sorted(self, tree: BIDSTree):
        tree.add_participants({SUB_A: DYAD, SUB_B: DYAD})
        stim = tree.add_stimulus(
            dyad=DYAD, task=TASK, run="1", kind="audio", ext=".wav"
        )
        tree.add_events(
            sub=SUB_A, task=TASK, run="1", rows=[_row("turn_speaker", 0.0, 10.0)]
        )
        tree.add_events(
            sub=SUB_B, task=TASK, run="1", rows=[_row("turn_speaker", 10.0, 10.0)]
        )
        turns = load_turns(BIDSLayout(tree.root), BIDSPath(stim))
        assert [(t.sub, t.onset, t.offset) for t in turns] == [
            (SUB_A, 0.0, 10.0),
            (SUB_B, 10.0, 20.0),
        ]

    def test_ignores_non_turn_flat_and_segment_rows(self, tree: BIDSTree):
        tree.add_participants({SUB_A: DYAD, SUB_B: DYAD})
        stim = tree.add_stimulus(
            dyad=DYAD, task=TASK, run="1", kind="audio", ext=".wav"
        )
        tree.add_events(
            sub=SUB_A,
            task=TASK,
            run="1",
            rows=[
                _row("turn_speaker", 0.0, 10.0),
                _row("rest", 10.0, 5.0),
                _row("block-1", 0.0, 20.0),
            ],
        )
        tree.add_events(sub=SUB_B, task=TASK, run="1", rows=[])
        turns = load_turns(BIDSLayout(tree.root), BIDSPath(stim))
        assert [t.sub for t in turns] == [SUB_A]

    def test_within_subject_overlap_raises(self, tree: BIDSTree):
        tree.add_participants({SUB_A: DYAD, SUB_B: DYAD})
        stim = tree.add_stimulus(
            dyad=DYAD, task=TASK, run="1", kind="audio", ext=".wav"
        )
        tree.add_events(
            sub=SUB_A,
            task=TASK,
            run="1",
            rows=[_row("turn_speaker", 0.0, 10.0), _row("turn_speaker", 5.0, 10.0)],
        )
        tree.add_events(sub=SUB_B, task=TASK, run="1", rows=[])
        with pytest.raises(ValueError, match="windows overlap"):
            load_turns(BIDSLayout(tree.root), BIDSPath(stim))

    def test_cross_partner_overlap_raises(self, tree: BIDSTree):
        tree.add_participants({SUB_A: DYAD, SUB_B: DYAD})
        stim = tree.add_stimulus(
            dyad=DYAD, task=TASK, run="1", kind="audio", ext=".wav"
        )
        tree.add_events(
            sub=SUB_A, task=TASK, run="1", rows=[_row("turn_speaker", 0.0, 12.0)]
        )
        tree.add_events(
            sub=SUB_B, task=TASK, run="1", rows=[_row("turn_speaker", 10.0, 10.0)]
        )
        with pytest.raises(ValueError, match="cross-talk"):
            load_turns(BIDSLayout(tree.root), BIDSPath(stim))

    def test_missing_task_entity_raises(self, tree: BIDSTree):
        stim = tree.add_stimulus(dyad=DYAD, task=TASK, kind="audio", ext=".wav")
        source_no_task = BIDSPath(stim).without_entity("task")
        with pytest.raises(ValueError, match="missing required 'task' entity"):
            load_turns(BIDSLayout(tree.root), source_no_task)

    def test_no_turn_rows_returns_empty(self, tree: BIDSTree):
        tree.add_participants({SUB_A: DYAD, SUB_B: DYAD})
        stim = tree.add_stimulus(
            dyad=DYAD, task=TASK, run="1", kind="audio", ext=".wav"
        )
        tree.add_events(sub=SUB_A, task=TASK, run="1", rows=[_row("rest", 0.0, 10.0)])
        tree.add_events(sub=SUB_B, task=TASK, run="1", rows=[])
        assert load_turns(BIDSLayout(tree.root), BIDSPath(stim)) == []


class TestStampTurns:
    _TURNS = [Turn("001", 0.0, 10.0), Turn("002", 10.0, 20.0)]

    def _transcript(self, start_times: list[float | None]) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "word": [f"w{i}" for i in range(len(start_times))],
                "start_time": start_times,
            }
        )

    def test_stamps_turn_sub_by_start_time(self):
        df, n_silent = stamp_turns(self._transcript([5.0, 15.0]), self._TURNS)
        assert df.get_column("turn_sub").to_list() == ["001", "002"]
        assert n_silent == 0

    def test_boundary_belongs_to_window_it_starts_in(self):
        # half-open [onset, offset): 10.0 ends 001's window and starts 002's
        df, _ = stamp_turns(self._transcript([10.0]), self._TURNS)
        assert df.get_column("turn_sub").to_list() == ["002"]

    def test_null_start_time_gives_null_turn_sub_and_is_not_counted(self):
        df, n_silent = stamp_turns(self._transcript([5.0, None]), self._TURNS)
        assert df.get_column("turn_sub").to_list() == ["001", None]
        assert n_silent == 0

    def test_timed_word_in_gap_is_null_and_counted(self):
        gapped = [Turn("001", 0.0, 5.0), Turn("002", 10.0, 15.0)]
        df, n_silent = stamp_turns(self._transcript([7.0]), gapped)
        assert df.get_column("turn_sub").to_list() == [None]
        assert n_silent == 1
