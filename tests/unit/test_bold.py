import pytest

from hypline.bids import BIDSPath
from hypline.bold import load_bold_meta
from hypline.layout import BIDSLayout

from .conftest import BIDSTree

SUB = "001"
TASK = "conv"
SPACE = "MNI152NLin6Asym"


_SEGMENT_ROWS = [
    {"trial_type": "block-1", "onset": 0.0, "duration": 100.0},
    {"trial_type": "block-2", "onset": 100.0, "duration": 100.0},
]


class TestLoadBoldMeta:
    def test_non_bold_input_raises(self, tree: BIDSTree):
        feature_path = tree.add_feature(sub=SUB, task=TASK, run="1", kind="phonemic")
        with pytest.raises(ValueError, match="Expected a BOLD file"):
            load_bold_meta(BIDSLayout(tree.root), BIDSPath(feature_path))

    def test_missing_task_entity_raises(self, tree: BIDSTree):
        bold_path = tree.add_bold(sub=SUB, space=SPACE, run="1")
        with pytest.raises(ValueError, match="missing required 'task' entity"):
            load_bold_meta(BIDSLayout(tree.root), BIDSPath(bold_path))

    def test_no_events_yields_no_segments(self, tree: BIDSTree):
        bold_path = tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1")
        meta = load_bold_meta(BIDSLayout(tree.root), BIDSPath(bold_path))
        assert meta.segments == []

    def test_events_json_absent_passes(self, tree: BIDSTree):
        bold_path = tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1")
        tree.add_events(sub=SUB, task=TASK, run="1", rows=_SEGMENT_ROWS)
        meta = load_bold_meta(BIDSLayout(tree.root), BIDSPath(bold_path))
        assert [seg.value for seg in meta.segments] == ["1", "2"]
        assert all(seg.metadata == {} for seg in meta.segments)

    def test_events_json_without_trial_type_field_passes(self, tree: BIDSTree):
        bold_path = tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=_SEGMENT_ROWS,
            sidecar_json={"SomeOtherField": "value"},
        )
        meta = load_bold_meta(BIDSLayout(tree.root), BIDSPath(bold_path))
        assert all(seg.metadata == {} for seg in meta.segments)

    def test_trial_type_field_without_levels_passes(self, tree: BIDSTree):
        bold_path = tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=_SEGMENT_ROWS,
            sidecar_json={"trial_type": {"LongName": "Type of trial"}},
        )
        meta = load_bold_meta(BIDSLayout(tree.root), BIDSPath(bold_path))
        assert all(seg.metadata == {} for seg in meta.segments)

    def test_multiple_segment_entities_raises(self, tree: BIDSTree):
        bold_path = tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=[
                {"trial_type": "block-1", "onset": 0.0, "duration": 100.0},
                {"trial_type": "run-1", "onset": 100.0, "duration": 100.0},
            ],
        )
        with pytest.raises(ValueError, match="multiple BIDS key-value entities"):
            load_bold_meta(BIDSLayout(tree.root), BIDSPath(bold_path))

    def test_identity_entity_as_segment_entity_raises(self, tree: BIDSTree):
        bold_path = tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=[{"trial_type": "run-2", "onset": 0.0, "duration": 200.0}],
        )
        with pytest.raises(ValueError, match="not allowed"):
            load_bold_meta(BIDSLayout(tree.root), BIDSPath(bold_path))

    def test_task_segment_entity_multiple_rows_raises(self, tree: BIDSTree):
        bold_path = tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=[
                {"trial_type": f"task-{TASK}", "onset": 0.0, "duration": 100.0},
                {"trial_type": "task-other", "onset": 100.0, "duration": 100.0},
            ],
        )
        with pytest.raises(ValueError, match="at most one segment"):
            load_bold_meta(BIDSLayout(tree.root), BIDSPath(bold_path))

    def test_task_segment_value_matches_filename_passes(self, tree: BIDSTree):
        bold_path = tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=[{"trial_type": f"task-{TASK}", "onset": 10.0, "duration": 180.0}],
        )
        meta = load_bold_meta(BIDSLayout(tree.root), BIDSPath(bold_path))
        assert len(meta.segments) == 1
        assert meta.segments[0].entity == "task"
        assert meta.segments[0].value == TASK

    def test_task_segment_value_mismatches_filename_raises(self, tree: BIDSTree):
        bold_path = tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=[{"trial_type": "task-other", "onset": 0.0, "duration": 200.0}],
        )
        with pytest.raises(ValueError, match="does not match"):
            load_bold_meta(BIDSLayout(tree.root), BIDSPath(bold_path))

    def test_segment_entity_collides_with_flat_label_raises(self, tree: BIDSTree):
        bold_path = tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=[
                {"trial_type": "block-1", "onset": 0.0, "duration": 100.0},
                {"trial_type": "block", "onset": 100.0, "duration": 100.0},
            ],
        )
        with pytest.raises(ValueError, match="also appears as a flat label"):
            load_bold_meta(BIDSLayout(tree.root), BIDSPath(bold_path))

    def test_duplicate_segment_value_raises(self, tree: BIDSTree):
        bold_path = tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=[
                {"trial_type": "block-1", "onset": 0.0, "duration": 100.0},
                {"trial_type": "block-1", "onset": 100.0, "duration": 100.0},
            ],
        )
        with pytest.raises(ValueError, match="appears more than once"):
            load_bold_meta(BIDSLayout(tree.root), BIDSPath(bold_path))

    def test_overlapping_segment_slices_raises(self, tree: BIDSTree):
        bold_path = tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=[
                {"trial_type": "block-1", "onset": 0.0, "duration": 100.0},
                {"trial_type": "block-2", "onset": 50.0, "duration": 100.0},
            ],
        )
        with pytest.raises(ValueError, match="overlap"):
            load_bold_meta(BIDSLayout(tree.root), BIDSPath(bold_path))

    def test_metadata_merged_into_segments(self, tree: BIDSTree):
        bold_path = tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=_SEGMENT_ROWS,
            sidecar_json={
                "trial_type": {
                    "Levels": {
                        "block-1": {"metadata": {"cond": "R"}},
                        "block-2": {"metadata": {"cond": "L"}},
                    }
                }
            },
        )
        meta = load_bold_meta(BIDSLayout(tree.root), BIDSPath(bold_path))
        assert len(meta.segments) == 2
        by_value = {seg.value: seg for seg in meta.segments}
        assert by_value["1"].metadata == {"cond": "R"}
        assert by_value["2"].metadata == {"cond": "L"}

    def test_non_entity_value_levels_entries_ignored(self, tree: BIDSTree):
        bold_path = tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=_SEGMENT_ROWS,
            sidecar_json={
                "trial_type": {
                    "Levels": {
                        "n/a": {"LongName": "Not applicable"},
                        "rest": {"LongName": "Resting state"},
                    }
                }
            },
        )
        meta = load_bold_meta(BIDSLayout(tree.root), BIDSPath(bold_path))
        assert all(seg.metadata == {} for seg in meta.segments)

    def test_mixed_entity_and_non_entity_levels_entries(self, tree: BIDSTree):
        bold_path = tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=_SEGMENT_ROWS,
            sidecar_json={
                "trial_type": {
                    "Levels": {
                        "block-1": {"metadata": {"cond": "R"}},
                        "block-2": {"metadata": {"cond": "L"}},
                        "rest": {"LongName": "Resting state"},
                        "n/a": {"LongName": "Not applicable"},
                    }
                }
            },
        )
        meta = load_bold_meta(BIDSLayout(tree.root), BIDSPath(bold_path))
        by_value = {seg.value: seg for seg in meta.segments}
        assert by_value["1"].metadata == {"cond": "R"}
        assert by_value["2"].metadata == {"cond": "L"}

    def test_events_json_with_segments_but_no_tsv_segments_raises(self, tree: BIDSTree):
        bold_path = tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=[{"trial_type": "rest", "onset": 0.0, "duration": 200.0}],
            sidecar_json={
                "trial_type": {
                    "Levels": {
                        "block-1": {"metadata": {"cond": "R"}},
                    }
                }
            },
        )
        with pytest.raises(ValueError, match="no segment rows"):
            load_bold_meta(BIDSLayout(tree.root), BIDSPath(bold_path))

    def test_levels_entry_missing_metadata_field_raises(self, tree: BIDSTree):
        bold_path = tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=_SEGMENT_ROWS,
            sidecar_json={
                "trial_type": {
                    "Levels": {
                        "block-1": {"metadata": {"cond": "R"}},
                        "block-2": {"LongName": "no metadata key"},
                    }
                }
            },
        )
        with pytest.raises(ValueError, match="no 'metadata' field"):
            load_bold_meta(BIDSLayout(tree.root), BIDSPath(bold_path))

    def test_events_json_segment_value_mismatch_raises(self, tree: BIDSTree):
        bold_path = tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=_SEGMENT_ROWS,
            sidecar_json={
                "trial_type": {
                    "Levels": {
                        "block-1": {"metadata": {"cond": "R"}},
                        "block-99": {"metadata": {"cond": "L"}},  # "99" not in tsv
                    }
                }
            },
        )
        with pytest.raises(ValueError, match="Levels keys do not match events.tsv"):
            load_bold_meta(BIDSLayout(tree.root), BIDSPath(bold_path))

    def test_events_json_schema_mismatch_across_records_raises(self, tree: BIDSTree):
        bold_path = tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=_SEGMENT_ROWS,
            sidecar_json={
                "trial_type": {
                    "Levels": {
                        "block-1": {"metadata": {"cond": "R"}},
                        "block-2": {"metadata": {}},  # missing "cond"
                    }
                }
            },
        )
        with pytest.raises(ValueError, match="schemas differ across Levels"):
            load_bold_meta(BIDSLayout(tree.root), BIDSPath(bold_path))

    def test_events_json_extra_key_raises(self, tree: BIDSTree):
        bold_path = tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=_SEGMENT_ROWS,
            sidecar_json={
                "trial_type": {
                    "Levels": {
                        "block-1": {"metadata": {"bad-key": "R"}},
                        "block-2": {"metadata": {"bad-key": "L"}},
                    }
                }
            },
        )
        with pytest.raises(ValueError, match="must match"):
            load_bold_meta(BIDSLayout(tree.root), BIDSPath(bold_path))

    def test_events_json_reserved_key_raises(self, tree: BIDSTree):
        bold_path = tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=_SEGMENT_ROWS,
            sidecar_json={
                "trial_type": {
                    "Levels": {
                        "block-1": {"metadata": {"ses": "01"}},
                        "block-2": {"metadata": {"ses": "01"}},
                    }
                }
            },
        )
        with pytest.raises(ValueError, match="collides with a raw BOLD entity"):
            load_bold_meta(BIDSLayout(tree.root), BIDSPath(bold_path))

    def test_events_json_invalid_value_raises(self, tree: BIDSTree):
        bold_path = tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=_SEGMENT_ROWS,
            sidecar_json={
                "trial_type": {
                    "Levels": {
                        "block-1": {"metadata": {"cond": "has space"}},
                        "block-2": {"metadata": {"cond": "L"}},
                    }
                }
            },
        )
        with pytest.raises(ValueError, match="must be a string matching"):
            load_bold_meta(BIDSLayout(tree.root), BIDSPath(bold_path))
