import pytest

from hypline.bids import BIDSPath
from hypline.bold import load_bold_meta, load_events_tsv

from .conftest import SPACE, SUB, TASK, BIDSTree


class TestLoadEvents:
    """
    Sidecar resolution is shared between `load_events_tsv` and `load_events_json`
    via `_resolve_run_sidecar`; exercising the `.tsv` variant here covers both.
    The `.json` branch is exercised end-to-end through `TestLoadBoldMeta`.
    """

    def test_spec_compliant_events_returned(self, tree: BIDSTree):
        bold_path = tree.add_bold(run="1")
        tree.add_events(run="1")
        result = load_events_tsv(bold_path)
        assert result is not None
        assert "trial_type" in result.columns

    def test_no_events_returns_none(self, tree: BIDSTree):
        bold_path = tree.add_bold(run="1")
        assert load_events_tsv(bold_path) is None

    def test_misnamed_sibling_with_space_raises(self, tree: BIDSTree):
        bold_path = tree.add_bold(run="1")
        events_path = tree.add_events(run="1")
        events_path.rename(BIDSPath(events_path).with_entity("space", SPACE).path)
        with pytest.raises(ValueError, match="unexpected events"):
            load_events_tsv(bold_path)

    def test_misnamed_sibling_with_desc_raises(self, tree: BIDSTree):
        bold_path = tree.add_bold(run="1")
        events_path = tree.add_events(run="1")
        events_path.rename(BIDSPath(events_path).with_entity("desc", "preproc").path)
        with pytest.raises(ValueError, match="unexpected events"):
            load_events_tsv(bold_path)

    def test_misnamed_sibling_with_space_and_desc_raises(self, tree: BIDSTree):
        bold_path = tree.add_bold(run="1")
        events_path = tree.add_events(run="1")
        events_path = events_path.rename(
            BIDSPath(events_path)
            .with_entity("space", SPACE)
            .with_entity("desc", "preproc")
            .path
        )
        with pytest.raises(ValueError, match=events_path.name):
            load_events_tsv(bold_path)

    def test_misnamed_sibling_with_reordered_entities_raises(self, tree: BIDSTree):
        bold_path = tree.add_bold(run="1")
        events_path = tree.bold_dir / f"sub-{SUB}_run-1_task-{TASK}_events.tsv"
        events_path.write_text("trial_type\tonset\tduration\n")
        with pytest.raises(ValueError, match="unexpected events"):
            load_events_tsv(bold_path)

    def test_multiple_misnamed_siblings_all_listed(self, tree: BIDSTree):
        bold_path = tree.add_bold(run="1")
        for space in (SPACE, "T1w"):
            events_path = tree.add_events(run="1")
            events_path.rename(BIDSPath(events_path).with_entity("space", space).path)
        with pytest.raises(ValueError) as exc_info:
            load_events_tsv(bold_path)
        msg = str(exc_info.value)
        assert SPACE in msg
        assert "T1w" in msg

    def test_unrelated_events_in_same_dir_ignored(self, tree: BIDSTree):
        bold_path = tree.add_bold(run="1")
        tree.add_events(sub="002", run="1")
        assert load_events_tsv(bold_path) is None

    def test_different_run_events_in_same_dir_ignored(self, tree: BIDSTree):
        bold_path = tree.add_bold(run="1")
        tree.add_events(run="2")
        assert load_events_tsv(bold_path) is None

    def test_misnamed_sibling_for_different_run_ignored(self, tree: BIDSTree):
        bold_path = tree.add_bold(run="1")
        events_path = tree.add_events(run="2")
        events_path.rename(BIDSPath(events_path).with_entity("space", SPACE).path)
        assert load_events_tsv(bold_path) is None


_SEGMENT_ROWS = [
    {"trial_type": "block-1", "onset": 0.0, "duration": 100.0},
    {"trial_type": "block-2", "onset": 100.0, "duration": 100.0},
]


class TestLoadBoldMeta:
    def test_events_json_absent_passes(self, tree: BIDSTree):
        bold_path = tree.add_bold(run="1")
        tree.add_events(run="1", rows=_SEGMENT_ROWS)
        meta = load_bold_meta(bold_path)
        assert all(seg.metadata == {} for seg in meta.segments)

    def test_events_json_without_segments_field_passes(self, tree: BIDSTree):
        bold_path = tree.add_bold(run="1")
        tree.add_events(
            run="1", rows=_SEGMENT_ROWS, events_json={"SomeOtherField": "value"}
        )
        meta = load_bold_meta(bold_path)
        assert all(seg.metadata == {} for seg in meta.segments)

    def test_metadata_merged_into_segments(self, tree: BIDSTree):
        bold_path = tree.add_bold(run="1")
        tree.add_events(
            run="1",
            rows=_SEGMENT_ROWS,
            events_json={
                "Segments": [
                    {"block": "1", "cond": "R"},
                    {"block": "2", "cond": "L"},
                ]
            },
        )
        meta = load_bold_meta(bold_path)
        assert len(meta.segments) == 2
        by_value = {seg.value: seg for seg in meta.segments}
        assert by_value["1"].metadata == {"cond": "R"}
        assert by_value["2"].metadata == {"cond": "L"}

    def test_reversed_record_order_merges_by_value(self, tree: BIDSTree):
        bold_path = tree.add_bold(run="1")
        tree.add_events(
            run="1",
            rows=_SEGMENT_ROWS,
            events_json={
                "Segments": [
                    {"block": "2", "cond": "L"},
                    {"block": "1", "cond": "R"},
                ]
            },
        )
        meta = load_bold_meta(bold_path)
        by_value = {seg.value: seg for seg in meta.segments}
        assert by_value["1"].metadata == {"cond": "R"}
        assert by_value["2"].metadata == {"cond": "L"}

    def test_records_with_only_segment_entity_key(self, tree: BIDSTree):
        bold_path = tree.add_bold(run="1")
        tree.add_events(
            run="1",
            rows=_SEGMENT_ROWS,
            events_json={"Segments": [{"block": "1"}, {"block": "2"}]},
        )
        meta = load_bold_meta(bold_path)
        assert all(seg.metadata == {} for seg in meta.segments)

    def test_events_json_record_missing_segment_entity_key_raises(self, tree: BIDSTree):
        bold_path = tree.add_bold(run="1")
        tree.add_events(
            run="1",
            rows=_SEGMENT_ROWS,
            events_json={"Segments": [{"cond": "R"}, {"block": "2", "cond": "L"}]},
        )
        with pytest.raises(ValueError, match="missing segment entity key"):
            load_bold_meta(bold_path)

    def test_events_json_segment_value_mismatch_raises(self, tree: BIDSTree):
        bold_path = tree.add_bold(run="1")
        tree.add_events(
            run="1",
            rows=_SEGMENT_ROWS,
            events_json={
                "Segments": [
                    {"block": "1", "cond": "R"},
                    {"block": "99", "cond": "L"},  # "99" not in tsv
                ]
            },
        )
        with pytest.raises(ValueError, match="do not match events.tsv"):
            load_bold_meta(bold_path)

    def test_events_json_schema_mismatch_across_records_raises(self, tree: BIDSTree):
        bold_path = tree.add_bold(run="1")
        tree.add_events(
            run="1",
            rows=_SEGMENT_ROWS,
            events_json={
                "Segments": [
                    {"block": "1", "cond": "R"},
                    {"block": "2"},  # missing "cond"
                ]
            },
        )
        with pytest.raises(ValueError, match="inconsistent metadata keys"):
            load_bold_meta(bold_path)

    def test_events_json_extra_key_raises(self, tree: BIDSTree):
        bold_path = tree.add_bold(run="1")
        tree.add_events(
            run="1",
            rows=_SEGMENT_ROWS,
            events_json={
                "Segments": [
                    {"block": "1", "bad-key": "R"},
                    {"block": "2", "bad-key": "L"},
                ]
            },
        )
        with pytest.raises(ValueError, match="invalid"):
            load_bold_meta(bold_path)

    def test_events_json_reserved_key_raises(self, tree: BIDSTree):
        bold_path = tree.add_bold(run="1")
        tree.add_events(
            run="1",
            rows=_SEGMENT_ROWS,
            events_json={
                "Segments": [
                    {"block": "1", "ses": "01"},
                    {"block": "2", "ses": "01"},
                ]
            },
        )
        with pytest.raises(ValueError, match="collides with BOLD identity entity"):
            load_bold_meta(bold_path)

    def test_events_json_invalid_value_raises(self, tree: BIDSTree):
        bold_path = tree.add_bold(run="1")
        tree.add_events(
            run="1",
            rows=_SEGMENT_ROWS,
            events_json={
                "Segments": [
                    {"block": "1", "cond": "has space"},
                    {"block": "2", "cond": "L"},
                ]
            },
        )
        with pytest.raises(ValueError, match="invalid"):
            load_bold_meta(bold_path)

    def test_events_json_misnamed_sibling_raises(self, tree: BIDSTree):
        bold_path = tree.add_bold(run="1")
        events_path = tree.add_events(run="1", rows=_SEGMENT_ROWS, events_json={})
        events_json = events_path.with_suffix(".json")
        events_json.rename(BIDSPath(events_json).with_entity("space", SPACE).path)
        with pytest.raises(ValueError, match="unexpected events"):
            load_bold_meta(bold_path)

    def test_events_json_with_segments_but_no_tsv_segments_raises(self, tree: BIDSTree):
        bold_path = tree.add_bold(run="1")
        tree.add_events(
            run="1",
            rows=[{"trial_type": "rest", "onset": 0.0, "duration": 200.0}],
            events_json={"Segments": [{"block": "1", "cond": "R"}]},
        )
        with pytest.raises(ValueError, match="no BIDS key-value rows"):
            load_bold_meta(bold_path)
