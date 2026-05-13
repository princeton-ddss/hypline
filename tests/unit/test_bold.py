import pytest

from hypline.bids import BIDSPath
from hypline.bold import load_bold_meta, load_events_tsv

from .conftest import BIDSTree

SUB = "001"
TASK = "conv"
SPACE = "MNI152NLin6Asym"


class TestLoadEvents:
    """
    Sidecar resolution is shared between `load_events_tsv` and `load_events_json`
    via `_resolve_run_sidecar`; exercising the `.tsv` variant here covers both.
    The `.json` branch is exercised end-to-end through `TestLoadBoldMeta`.
    """

    def test_spec_compliant_events_returned(self, tree: BIDSTree):
        bold_path = tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1")
        tree.add_events(sub=SUB, task=TASK, run="1")
        result = load_events_tsv(bold_path)
        assert result is not None
        assert "trial_type" in result.columns

    def test_no_events_returns_none(self, tree: BIDSTree):
        bold_path = tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1")
        assert load_events_tsv(bold_path) is None

    def test_misnamed_sibling_with_space_raises(self, tree: BIDSTree):
        bold_path = tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1")
        events_path = tree.add_events(sub=SUB, task=TASK, run="1")
        events_path.rename(BIDSPath(events_path).with_entity("space", SPACE).path)
        with pytest.raises(ValueError, match="unexpected events"):
            load_events_tsv(bold_path)

    def test_misnamed_sibling_with_desc_raises(self, tree: BIDSTree):
        bold_path = tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1")
        events_path = tree.add_events(sub=SUB, task=TASK, run="1")
        events_path.rename(BIDSPath(events_path).with_entity("desc", "preproc").path)
        with pytest.raises(ValueError, match="unexpected events"):
            load_events_tsv(bold_path)

    def test_misnamed_sibling_with_space_and_desc_raises(self, tree: BIDSTree):
        bold_path = tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1")
        events_path = tree.add_events(sub=SUB, task=TASK, run="1")
        events_path = events_path.rename(
            BIDSPath(events_path)
            .with_entity("space", SPACE)
            .with_entity("desc", "preproc")
            .path
        )
        with pytest.raises(ValueError, match=events_path.name):
            load_events_tsv(bold_path)

    def test_misnamed_sibling_with_reordered_entities_raises(self, tree: BIDSTree):
        bold_path = tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1")
        events_path = bold_path.parent / f"sub-{SUB}_run-1_task-{TASK}_events.tsv"
        events_path.write_text("trial_type\tonset\tduration\n")
        with pytest.raises(ValueError, match="unexpected events"):
            load_events_tsv(bold_path)

    def test_multiple_misnamed_siblings_all_listed(self, tree: BIDSTree):
        bold_path = tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1")
        for space in (SPACE, "T1w"):
            events_path = tree.add_events(sub=SUB, task=TASK, run="1")
            events_path.rename(BIDSPath(events_path).with_entity("space", space).path)
        with pytest.raises(ValueError) as exc_info:
            load_events_tsv(bold_path)
        msg = str(exc_info.value)
        assert SPACE in msg
        assert "T1w" in msg

    def test_unrelated_events_in_same_dir_ignored(self, tree: BIDSTree):
        bold_path = tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1")
        foreign = bold_path.parent / f"sub-002_task-{TASK}_run-1_events.tsv"
        foreign.write_text("trial_type\tonset\tduration\n")
        assert load_events_tsv(bold_path) is None

    def test_different_run_events_in_same_dir_ignored(self, tree: BIDSTree):
        bold_path = tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1")
        tree.add_events(sub=SUB, task=TASK, run="2")
        assert load_events_tsv(bold_path) is None

    def test_misnamed_sibling_for_different_run_ignored(self, tree: BIDSTree):
        bold_path = tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1")
        events_path = tree.add_events(sub=SUB, task=TASK, run="2")
        events_path.rename(BIDSPath(events_path).with_entity("space", SPACE).path)
        assert load_events_tsv(bold_path) is None


_SEGMENT_ROWS = [
    {"trial_type": "block-1", "onset": 0.0, "duration": 100.0},
    {"trial_type": "block-2", "onset": 100.0, "duration": 100.0},
]


class TestLoadBoldMeta:
    def test_events_json_absent_passes(self, tree: BIDSTree):
        bold_path = tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1")
        tree.add_events(sub=SUB, task=TASK, run="1", rows=_SEGMENT_ROWS)
        meta = load_bold_meta(bold_path)
        assert all(seg.metadata == {} for seg in meta.segments)

    def test_events_json_without_trial_type_field_passes(self, tree: BIDSTree):
        bold_path = tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=_SEGMENT_ROWS,
            events_json={"SomeOtherField": "value"},
        )
        meta = load_bold_meta(bold_path)
        assert all(seg.metadata == {} for seg in meta.segments)

    def test_trial_type_field_without_levels_passes(self, tree: BIDSTree):
        bold_path = tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=_SEGMENT_ROWS,
            events_json={"trial_type": {"LongName": "Type of trial"}},
        )
        meta = load_bold_meta(bold_path)
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
            load_bold_meta(bold_path)

    def test_identity_entity_as_segment_entity_raises(self, tree: BIDSTree):
        bold_path = tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=[{"trial_type": "run-2", "onset": 0.0, "duration": 200.0}],
        )
        with pytest.raises(ValueError, match="not allowed"):
            load_bold_meta(bold_path)

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
            load_bold_meta(bold_path)

    def test_task_segment_value_matches_filename_passes(self, tree: BIDSTree):
        bold_path = tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=[{"trial_type": f"task-{TASK}", "onset": 10.0, "duration": 180.0}],
        )
        meta = load_bold_meta(bold_path)
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
            load_bold_meta(bold_path)

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
            load_bold_meta(bold_path)

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
            load_bold_meta(bold_path)

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
            load_bold_meta(bold_path)

    def test_metadata_merged_into_segments(self, tree: BIDSTree):
        bold_path = tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=_SEGMENT_ROWS,
            events_json={
                "trial_type": {
                    "Levels": {
                        "block-1": {"metadata": {"cond": "R"}},
                        "block-2": {"metadata": {"cond": "L"}},
                    }
                }
            },
        )
        meta = load_bold_meta(bold_path)
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
            events_json={
                "trial_type": {
                    "Levels": {
                        "n/a": {"LongName": "Not applicable"},
                        "rest": {"LongName": "Resting state"},
                    }
                }
            },
        )
        meta = load_bold_meta(bold_path)
        assert all(seg.metadata == {} for seg in meta.segments)

    def test_mixed_entity_and_non_entity_levels_entries(self, tree: BIDSTree):
        bold_path = tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=_SEGMENT_ROWS,
            events_json={
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
        meta = load_bold_meta(bold_path)
        by_value = {seg.value: seg for seg in meta.segments}
        assert by_value["1"].metadata == {"cond": "R"}
        assert by_value["2"].metadata == {"cond": "L"}

    def test_events_json_misnamed_sibling_raises(self, tree: BIDSTree):
        bold_path = tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1")
        events_path = tree.add_events(
            sub=SUB, task=TASK, run="1", rows=_SEGMENT_ROWS, events_json={}
        )
        events_json = events_path.with_suffix(".json")
        events_json.rename(BIDSPath(events_json).with_entity("space", SPACE).path)
        with pytest.raises(ValueError, match="unexpected events"):
            load_bold_meta(bold_path)

    def test_events_json_with_segments_but_no_tsv_segments_raises(self, tree: BIDSTree):
        bold_path = tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=[{"trial_type": "rest", "onset": 0.0, "duration": 200.0}],
            events_json={
                "trial_type": {
                    "Levels": {
                        "block-1": {"metadata": {"cond": "R"}},
                    }
                }
            },
        )
        with pytest.raises(ValueError, match="no segment rows"):
            load_bold_meta(bold_path)

    def test_levels_entry_missing_metadata_field_raises(self, tree: BIDSTree):
        bold_path = tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=_SEGMENT_ROWS,
            events_json={
                "trial_type": {
                    "Levels": {
                        "block-1": {"metadata": {"cond": "R"}},
                        "block-2": {"LongName": "no metadata key"},
                    }
                }
            },
        )
        with pytest.raises(ValueError, match="no 'metadata' field"):
            load_bold_meta(bold_path)

    def test_events_json_segment_value_mismatch_raises(self, tree: BIDSTree):
        bold_path = tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=_SEGMENT_ROWS,
            events_json={
                "trial_type": {
                    "Levels": {
                        "block-1": {"metadata": {"cond": "R"}},
                        "block-99": {"metadata": {"cond": "L"}},  # "99" not in tsv
                    }
                }
            },
        )
        with pytest.raises(ValueError, match="Levels keys do not match events.tsv"):
            load_bold_meta(bold_path)

    def test_events_json_schema_mismatch_across_records_raises(self, tree: BIDSTree):
        bold_path = tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=_SEGMENT_ROWS,
            events_json={
                "trial_type": {
                    "Levels": {
                        "block-1": {"metadata": {"cond": "R"}},
                        "block-2": {"metadata": {}},  # missing "cond"
                    }
                }
            },
        )
        with pytest.raises(ValueError, match="schemas differ across Levels"):
            load_bold_meta(bold_path)

    def test_events_json_extra_key_raises(self, tree: BIDSTree):
        bold_path = tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=_SEGMENT_ROWS,
            events_json={
                "trial_type": {
                    "Levels": {
                        "block-1": {"metadata": {"bad-key": "R"}},
                        "block-2": {"metadata": {"bad-key": "L"}},
                    }
                }
            },
        )
        with pytest.raises(ValueError, match="must match"):
            load_bold_meta(bold_path)

    def test_events_json_reserved_key_raises(self, tree: BIDSTree):
        bold_path = tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=_SEGMENT_ROWS,
            events_json={
                "trial_type": {
                    "Levels": {
                        "block-1": {"metadata": {"ses": "01"}},
                        "block-2": {"metadata": {"ses": "01"}},
                    }
                }
            },
        )
        with pytest.raises(ValueError, match="collides with BOLD identity entity"):
            load_bold_meta(bold_path)

    def test_events_json_invalid_value_raises(self, tree: BIDSTree):
        bold_path = tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=_SEGMENT_ROWS,
            events_json={
                "trial_type": {
                    "Levels": {
                        "block-1": {"metadata": {"cond": "has space"}},
                        "block-2": {"metadata": {"cond": "L"}},
                    }
                }
            },
        )
        with pytest.raises(ValueError, match="must be a string matching"):
            load_bold_meta(bold_path)
