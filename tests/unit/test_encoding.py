import polars as pl
import pyarrow.parquet as pq
import pytest

from hypline.encoding import BoldKey, CellKey, Encoding, EncodingConfig, FeatureKey
from hypline.layout import BIDSLayout

from .conftest import BIDSTree

SUB = "001"
TASK = "conv"
SPACE = "MNI152NLin6Asym"


def _make_encoding(
    tree: BIDSTree,
    features: list[str],
    *,
    bold_space: str = SPACE,
    bids_filters: list[str] | None = None,
) -> Encoding:
    return Encoding(
        EncodingConfig(),
        layout=BIDSLayout(tree.root),
        features=features,
        bold_space=bold_space,
        bids_filters=bids_filters,
    )


class TestEncodingInit:
    def test_valid_config_succeeds(self, tree: BIDSTree):
        enc = _make_encoding(tree, ["mfcc"])
        assert enc.features == ["mfcc"]

    def test_empty_features_raises(self, tree: BIDSTree):
        with pytest.raises(ValueError, match="non-empty"):
            _make_encoding(tree, [])

    def test_duplicate_features_raises(self, tree: BIDSTree):
        with pytest.raises(ValueError, match="Duplicate"):
            _make_encoding(tree, ["mfcc", "mfcc"])

    def test_reserved_entity_in_filters_raises(self, tree: BIDSTree):
        with pytest.raises(ValueError, match="sub"):
            _make_encoding(tree, ["mfcc"], bids_filters=["sub-001"])

    def test_unknown_entity_accepted_at_init(self, tree: BIDSTree):
        enc = _make_encoding(tree, ["mfcc"], bids_filters=["xyz-foo"])
        assert enc.bids_filters == ["xyz-foo"]

    def test_invalid_bold_space_raises(self, tree: BIDSTree):
        with pytest.raises(ValueError, match="Unsupported BOLD data space"):
            _make_encoding(tree, ["mfcc"], bold_space="notaspace")


class TestCellKey:
    def test_excluded_entity_raises(self):
        for entity in CellKey.EXCLUDE:
            with pytest.raises(ValueError, match="CellKey does not accept"):
                CellKey(**{entity: "x"})

    def test_equality_is_order_independent(self):
        assert CellKey(ses="1", run="2") == CellKey(run="2", ses="1")

    def test_keys_returns_present_entities(self):
        assert CellKey(ses="1", run="2").keys() == {"ses", "run"}

    def test_getitem_missing_raises(self):
        with pytest.raises(KeyError):
            CellKey(run="1")["ses"]

    def test_get_missing_returns_default(self):
        assert CellKey(run="1").get("ses") is None
        assert CellKey(run="1").get("ses", "fallback") == "fallback"


class TestDiscoverFeatures:
    def test_returns_expected_keys(self, tree: BIDSTree):
        tree.add_feature(sub=SUB, task=TASK, kind="mfcc", run="1")
        enc = _make_encoding(tree, ["mfcc"])
        feature_paths = enc._discover_features(SUB)
        expected = FeatureKey(cell=CellKey(run="1"), feature="mfcc")
        assert expected in feature_paths

    def test_no_files_raises(self, tree: BIDSTree):
        enc = _make_encoding(tree, ["mfcc"])
        with pytest.raises(FileNotFoundError, match="features"):
            enc._discover_features(SUB)

    def test_duplicate_feature_file_raises(self, tree: BIDSTree):
        # Two filenames with identical BIDS entities (reordered) collide on FeatureKey
        original = tree.add_feature(sub=SUB, task=TASK, kind="mfcc", run="1")
        dup = original.parent / f"sub-{SUB}_run-1_task-{TASK}_feat-mfcc.parquet"
        dup.write_bytes(original.read_bytes())
        enc = _make_encoding(tree, ["mfcc"])
        with pytest.raises(ValueError, match="Multiple feature files"):
            enc._discover_features(SUB)

    def test_missing_feature_at_one_cell_raises(self, tree: BIDSTree):
        tree.add_feature(sub=SUB, task=TASK, kind="mfcc", run="1")
        tree.add_feature(sub=SUB, task=TASK, kind="mfcc", run="2")
        tree.add_feature(sub=SUB, task=TASK, kind="clip", run="1")
        enc = _make_encoding(tree, ["mfcc", "clip"])
        with pytest.raises(FileNotFoundError, match="Missing feat=clip"):
            enc._discover_features(SUB)

    def test_task_invariance_violation_raises(self, tree: BIDSTree):
        tree.add_feature(sub=SUB, task="rest", kind="mfcc", run="1")
        tree.add_feature(sub=SUB, task="conv", kind="mfcc", run="2")
        enc = _make_encoding(tree, ["mfcc"])
        with pytest.raises(ValueError, match="task"):
            enc._discover_features(SUB)

    def test_mixed_segmented_unsegmented_runs_raises(self, tree: BIDSTree):
        tree.add_feature(
            sub=SUB, task=TASK, kind="mfcc", run="1", extra_entities={"block": "1"}
        )
        tree.add_feature(sub=SUB, task=TASK, kind="mfcc", run="2")
        enc = _make_encoding(tree, ["mfcc"])
        with pytest.raises(ValueError, match="Inconsistent feature file schemas"):
            enc._discover_features(SUB)

    def test_schema_error_fires_before_coverage_error(self, tree: BIDSTree):
        # clip missing at run-2, but schema mismatch should raise first
        tree.add_feature(
            sub=SUB, task=TASK, kind="mfcc", run="1", extra_entities={"block": "1"}
        )
        tree.add_feature(sub=SUB, task=TASK, kind="mfcc", run="2")
        tree.add_feature(
            sub=SUB, task=TASK, kind="clip", run="1", extra_entities={"block": "1"}
        )
        enc = _make_encoding(tree, ["mfcc", "clip"])
        with pytest.raises(ValueError, match="Inconsistent feature file schemas"):
            enc._discover_features(SUB)

    def test_file_without_hypline_metadata_raises(self, tree: BIDSTree, tmp_path):
        kind_dir = tree.features_dir / f"sub-{SUB}" / "mfcc"
        kind_dir.mkdir(parents=True)
        path = kind_dir / f"sub-{SUB}_task-{TASK}_run-1_feat-mfcc.parquet"
        df = pl.DataFrame({"start_time": [0.0], "feature": [[0.0]]})
        pq.write_table(df.to_arrow(), path)
        enc = _make_encoding(tree, ["mfcc"])
        with pytest.raises(ValueError, match="no hypline metadata"):
            enc._discover_features(SUB)

    def test_bids_filters_not_applied_at_discovery(self, tree: BIDSTree):
        tree.add_feature(sub=SUB, task=TASK, kind="mfcc", run="1")
        tree.add_feature(sub=SUB, task=TASK, kind="mfcc", run="2")
        enc = _make_encoding(tree, ["mfcc"], bids_filters=["run-1"])
        feature_paths = enc._discover_features(SUB)
        assert len(feature_paths) == 2

    def test_inconsistent_metadata_across_files_raises(self, tree: BIDSTree):
        tree.add_feature(
            sub=SUB, task=TASK, kind="mfcc", run="1", metadata={"model": "v1"}
        )
        tree.add_feature(
            sub=SUB, task=TASK, kind="mfcc", run="2", metadata={"model": "v2"}
        )
        enc = _make_encoding(tree, ["mfcc"])
        with pytest.raises(ValueError, match="Inconsistent metadata for feat=mfcc"):
            enc._discover_features(SUB)

    def test_inconsistent_metadata_diff_in_error_message(self, tree: BIDSTree):
        tree.add_feature(
            sub=SUB, task=TASK, kind="mfcc", run="1", metadata={"model": "v1"}
        )
        tree.add_feature(
            sub=SUB, task=TASK, kind="mfcc", run="2", metadata={"model": "v2"}
        )
        enc = _make_encoding(tree, ["mfcc"])
        with pytest.raises(ValueError, match="model:"):
            enc._discover_features(SUB)

    def test_missing_key_shows_as_missing_in_diff(self, tree: BIDSTree):
        tree.add_feature(
            sub=SUB, task=TASK, kind="mfcc", run="1", metadata={"model": "v1"}
        )
        tree.add_feature(
            sub=SUB,
            task=TASK,
            kind="mfcc",
            run="2",
            metadata={"model": "v1", "extra": "x"},
        )
        enc = _make_encoding(tree, ["mfcc"])
        with pytest.raises(ValueError, match="<missing>"):
            enc._discover_features(SUB)

    def test_third_file_diverges_from_first(self, tree: BIDSTree):
        tree.add_feature(
            sub=SUB, task=TASK, kind="mfcc", run="1", metadata={"model": "v1"}
        )
        tree.add_feature(
            sub=SUB, task=TASK, kind="mfcc", run="2", metadata={"model": "v1"}
        )
        tree.add_feature(
            sub=SUB, task=TASK, kind="mfcc", run="3", metadata={"model": "v2"}
        )
        enc = _make_encoding(tree, ["mfcc"])
        with pytest.raises(ValueError, match="Inconsistent metadata for feat=mfcc"):
            enc._discover_features(SUB)

    def test_underscore_prefixed_keys_exempt_from_metadata_check(self, tree: BIDSTree):
        tree.add_feature(
            sub=SUB,
            task=TASK,
            kind="mfcc",
            run="1",
            metadata={"model": "v1", "_run_id": "abc"},
        )
        tree.add_feature(
            sub=SUB,
            task=TASK,
            kind="mfcc",
            run="2",
            metadata={"model": "v1", "_run_id": "xyz"},
        )
        enc = _make_encoding(tree, ["mfcc"])
        enc._discover_features(SUB)

    def test_metadata_check_independent_across_features(self, tree: BIDSTree):
        tree.add_feature(
            sub=SUB, task=TASK, kind="mfcc", run="1", metadata={"model": "v1"}
        )
        tree.add_feature(
            sub=SUB, task=TASK, kind="mfcc", run="2", metadata={"model": "v1"}
        )
        tree.add_feature(
            sub=SUB, task=TASK, kind="clip", run="1", metadata={"model": "v2"}
        )
        tree.add_feature(
            sub=SUB, task=TASK, kind="clip", run="2", metadata={"model": "v2"}
        )
        enc = _make_encoding(tree, ["mfcc", "clip"])
        enc._discover_features(SUB)


class TestDiscoverBold:
    def test_returns_expected_keys(self, tree: BIDSTree):
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", desc="clean")
        enc = _make_encoding(tree, ["mfcc"])
        bold_metas = enc._discover_bold(SUB)
        assert BoldKey(ses=None, run="1") in bold_metas

    def test_no_files_raises(self, tree: BIDSTree):
        enc = _make_encoding(tree, ["mfcc"])
        with pytest.raises(FileNotFoundError, match="fmriprep"):
            enc._discover_bold(SUB)

    def test_duplicate_bold_raises(self, tree: BIDSTree):
        # Two BOLDs sharing identity entities but distinguished by a variant
        # entity (`res`) collide on BoldKey.
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", desc="clean")
        tree.add_bold(
            sub=SUB,
            task=TASK,
            space=SPACE,
            run="1",
            desc="clean",
            extra_entities={"res": "2"},
        )
        enc = _make_encoding(tree, ["mfcc"])
        with pytest.raises(ValueError, match="Duplicate BOLD"):
            enc._discover_bold(SUB)

    def test_cross_session_runs_distinguished(self, tree: BIDSTree):
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, ses="1", run="1", desc="clean")
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, ses="2", run="1", desc="clean")
        enc = _make_encoding(tree, ["mfcc"])
        bold_metas = enc._discover_bold(SUB)
        assert BoldKey(ses="1", run="1") in bold_metas
        assert BoldKey(ses="2", run="1") in bold_metas

    def test_bold_siblings_excluded(self, tree: BIDSTree):
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", desc="clean")
        tree.add_bold_siblings(sub=SUB, task=TASK, space=SPACE, run="1")
        enc = _make_encoding(tree, ["mfcc"])
        bold_metas = enc._discover_bold(SUB)
        assert list(bold_metas.keys()) == [BoldKey(ses=None, run="1")]

    def test_wrong_space_bold_excluded(self, tree: BIDSTree):
        tree.add_bold(
            sub=SUB, task=TASK, run="1", space="MNI152NLin6Asym", desc="clean"
        )
        tree.add_bold(sub=SUB, task=TASK, run="1", space="T1w", desc="clean")
        enc = _make_encoding(tree, ["mfcc"], bold_space="MNI152NLin6Asym")
        bold_metas = enc._discover_bold(SUB)
        assert list(bold_metas.keys()) == [BoldKey(ses=None, run="1")]

    def test_inconsistent_tr_raises(self, tree: BIDSTree):
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="clean")
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="2", tr=1.5, desc="clean")
        enc = _make_encoding(tree, ["mfcc"])
        with pytest.raises(ValueError, match="Inconsistent repetition times"):
            enc._discover_bold(SUB)

    def test_task_invariance_violation_raises(self, tree: BIDSTree):
        tree.add_bold(sub=SUB, space=SPACE, task="rest", run="1", desc="clean")
        tree.add_bold(sub=SUB, space=SPACE, task="conv", run="2", desc="clean")
        enc = _make_encoding(tree, ["mfcc"])
        with pytest.raises(ValueError, match="task"):
            enc._discover_bold(SUB)

    def test_no_events_gives_no_segments(self, tree: BIDSTree):
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="clean")
        enc = _make_encoding(tree, ["mfcc"])
        bold_metas = enc._discover_bold(SUB)
        bold_meta = bold_metas[BoldKey(ses=None, run="1")]
        assert bold_meta.segments == []

    def test_structural_entity_slices_parsed(self, tree: BIDSTree):
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="clean")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=[
                {"trial_type": "block-1", "onset": 0.0, "duration": 100.0},
                {"trial_type": "block-2", "onset": 100.0, "duration": 100.0},
            ],
        )
        enc = _make_encoding(tree, ["mfcc"])
        bold_metas = enc._discover_bold(SUB)
        bold_meta = bold_metas[BoldKey(ses=None, run="1")]
        assert len(bold_meta.segments) == 2
        assert bold_meta.segments[0].entity == "block"
        assert bold_meta.segments[0].value == "1"
        assert bold_meta.segments[0].tr_slice == slice(0, 50)
        assert bold_meta.segments[1].value == "2"
        assert bold_meta.segments[1].tr_slice == slice(50, 100)

    def test_multiple_kv_entities_raises(self, tree: BIDSTree):
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="clean")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=[
                {"trial_type": "block-1", "onset": 0.0, "duration": 100.0},
                {"trial_type": "trial-1", "onset": 100.0, "duration": 100.0},
            ],
        )
        enc = _make_encoding(tree, ["mfcc"])
        with pytest.raises(ValueError, match="multiple BIDS key-value entities"):
            enc._discover_bold(SUB)

    def test_segment_entity_as_flat_label_raises(self, tree: BIDSTree):
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="clean")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=[
                {"trial_type": "block-1", "onset": 0.0, "duration": 100.0},
                {"trial_type": "block", "onset": 100.0, "duration": 100.0},
            ],
        )
        enc = _make_encoding(tree, ["mfcc"])
        with pytest.raises(ValueError, match="also appears as a flat label"):
            enc._discover_bold(SUB)

    def test_duplicate_segment_values_raises(self, tree: BIDSTree):
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="clean")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=[
                {"trial_type": "block-1", "onset": 0.0, "duration": 100.0},
                {"trial_type": "block-1", "onset": 100.0, "duration": 100.0},
            ],
        )
        enc = _make_encoding(tree, ["mfcc"])
        with pytest.raises(ValueError, match="appears more than once"):
            enc._discover_bold(SUB)

    def test_overlapping_slices_raises(self, tree: BIDSTree):
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="clean")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=[
                {"trial_type": "block-1", "onset": 0.0, "duration": 120.0},
                {"trial_type": "block-2", "onset": 100.0, "duration": 100.0},
            ],
        )
        enc = _make_encoding(tree, ["mfcc"])
        with pytest.raises(ValueError, match="overlap"):
            enc._discover_bold(SUB)

    def test_leading_break_allowed(self, tree: BIDSTree):
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="clean")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=[
                {"trial_type": "block-1", "onset": 10.0, "duration": 90.0},
                {"trial_type": "block-2", "onset": 100.0, "duration": 100.0},
            ],
        )
        enc = _make_encoding(tree, ["mfcc"])
        bold_metas = enc._discover_bold(SUB)
        bold_meta = bold_metas[BoldKey(ses=None, run="1")]
        assert len(bold_meta.segments) == 2
        assert bold_meta.segments[0].tr_slice == slice(5, 50)

    def test_hyphen_free_trial_type_ignored(self, tree: BIDSTree):
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="clean")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=[{"trial_type": "rest", "onset": 0.0, "duration": 100.0}],
        )
        enc = _make_encoding(tree, ["mfcc"])
        bold_metas = enc._discover_bold(SUB)
        bold_meta = bold_metas[BoldKey(ses=None, run="1")]
        assert bold_meta.segments == []

    def test_kv_entity_with_gap_passes(self, tree: BIDSTree):
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="clean")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=[
                {"trial_type": "block-1", "onset": 0.0, "duration": 100.0},
                {"trial_type": "block-2", "onset": 120.0, "duration": 80.0},
            ],
        )
        enc = _make_encoding(tree, ["mfcc"])
        bold_metas = enc._discover_bold(SUB)
        bold_meta = bold_metas[BoldKey(ses=None, run="1")]
        assert len(bold_meta.segments) == 2

    def test_flat_labels_alongside_segments_ignored(self, tree: BIDSTree):
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="clean")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=[
                {"trial_type": "block-1", "onset": 0.0, "duration": 100.0},
                {"trial_type": "block-2", "onset": 100.0, "duration": 100.0},
                {"trial_type": "fixation", "onset": 0.0, "duration": 10.0},
            ],
        )
        enc = _make_encoding(tree, ["mfcc"])
        bold_metas = enc._discover_bold(SUB)
        bold_meta = bold_metas[BoldKey(ses=None, run="1")]
        assert len(bold_meta.segments) == 2
        assert bold_meta.segments[0].entity == "block"

    def test_runs_disagree_on_segment_entity_raises(self, tree: BIDSTree):
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="clean")
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="2", tr=2.0, desc="clean")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=[
                {"trial_type": "block-1", "onset": 0.0, "duration": 100.0},
                {"trial_type": "block-2", "onset": 100.0, "duration": 100.0},
            ],
        )
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="2",
            rows=[
                {"trial_type": "trial-1", "onset": 0.0, "duration": 100.0},
                {"trial_type": "trial-2", "onset": 100.0, "duration": 100.0},
            ],
        )
        enc = _make_encoding(tree, ["mfcc"])
        with pytest.raises(ValueError, match="disagree on segment entity"):
            enc._discover_bold(SUB)

    def test_mixed_segmented_and_unsegmented_raises(self, tree: BIDSTree):
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="clean")
        tree.add_bold(
            sub=SUB, task=TASK, space=SPACE, run="2", tr=2.0, desc="clean"
        )  # no events → unsegmented
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=[
                {"trial_type": "block-1", "onset": 0.0, "duration": 100.0},
                {"trial_type": "block-2", "onset": 100.0, "duration": 100.0},
            ],
        )
        enc = _make_encoding(tree, ["mfcc"])
        with pytest.raises(ValueError, match="disagree on segment entity"):
            enc._discover_bold(SUB)

    def test_events_json_cross_run_schema_mismatch_raises(self, tree: BIDSTree):
        rows = [
            {"trial_type": "block-1", "onset": 0.0, "duration": 100.0},
            {"trial_type": "block-2", "onset": 100.0, "duration": 100.0},
        ]
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="clean")
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="2", tr=2.0, desc="clean")

        # run-1 has metadata key "cond"; run-2 has metadata key "item"
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=rows,
            sidecar_json={
                "trial_type": {
                    "Levels": {
                        "block-1": {"metadata": {"cond": "R"}},
                        "block-2": {"metadata": {"cond": "L"}},
                    }
                }
            },
        )
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="2",
            rows=rows,
            sidecar_json={
                "trial_type": {
                    "Levels": {
                        "block-1": {"metadata": {"item": "A"}},
                        "block-2": {"metadata": {"item": "B"}},
                    }
                }
            },
        )

        enc = _make_encoding(tree, ["mfcc"])
        with pytest.raises(ValueError, match="disagree on segment metadata schema"):
            enc._discover_bold(SUB)

    def test_all_unsegmented_passes(self, tree: BIDSTree):
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="clean")
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="2", tr=2.0, desc="clean")
        enc = _make_encoding(tree, ["mfcc"])
        bold_metas = enc._discover_bold(SUB)
        assert all(bold_meta.segments == [] for bold_meta in bold_metas.values())

    def test_bids_filters_not_applied_at_discovery(self, tree: BIDSTree):
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="clean")
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="2", tr=2.0, desc="clean")
        enc = _make_encoding(tree, ["mfcc"], bids_filters=["run-1"])
        bold_metas = enc._discover_bold(SUB)
        assert len(bold_metas) == 2


class TestResolveCellKeys:
    # Orphan check (feature cell has no matching BOLD run)

    def test_features_without_bold_raises(self, tree: BIDSTree):
        tree.add_feature(sub=SUB, task=TASK, kind="mfcc", run="1")
        tree.add_feature(sub=SUB, task=TASK, kind="mfcc", run="2")
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", desc="clean")
        enc = _make_encoding(tree, ["mfcc"])
        feature_paths = enc._discover_features(SUB)
        bold_metas = enc._discover_bold(SUB)
        with pytest.raises(FileNotFoundError, match="No BOLD file found for features"):
            enc._resolve_cell_keys(SUB, feature_paths, bold_metas)

    def test_multiple_features_without_bold_reports_count(self, tree: BIDSTree):
        for run in ("1", "2", "3"):
            tree.add_feature(sub=SUB, task=TASK, kind="mfcc", run=run)
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", desc="clean")
        enc = _make_encoding(tree, ["mfcc"])
        feature_paths = enc._discover_features(SUB)
        bold_metas = enc._discover_bold(SUB)
        with pytest.raises(FileNotFoundError, match="other coverage gaps"):
            enc._resolve_cell_keys(SUB, feature_paths, bold_metas)

    # Unsegmented run cases

    def test_unsegmented_run_single_cell_passes(self, tree: BIDSTree):
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="clean")
        tree.add_feature(sub=SUB, task=TASK, kind="mfcc", run="1")
        enc = _make_encoding(tree, ["mfcc"])
        feature_paths = enc._discover_features(SUB)
        bold_metas = enc._discover_bold(SUB)
        feature_paths = enc._resolve_cell_keys(SUB, feature_paths, bold_metas)
        assert len(feature_paths) == 1

    def test_unsegmented_run_multiple_cells_raises(self, tree: BIDSTree):
        tree.add_bold(
            sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="clean"
        )  # no events → unsegmented
        tree.add_feature(
            sub=SUB, task=TASK, kind="mfcc", run="1", extra_entities={"trial": "1"}
        )
        tree.add_feature(
            sub=SUB, task=TASK, kind="mfcc", run="1", extra_entities={"trial": "2"}
        )
        enc = _make_encoding(tree, ["mfcc"])
        feature_paths = enc._discover_features(SUB)
        bold_metas = enc._discover_bold(SUB)
        with pytest.raises(ValueError, match="unsegmented but has 2 feature files"):
            enc._resolve_cell_keys(SUB, feature_paths, bold_metas)

    def test_extra_entity_on_unsegmented_run_raises(self, tree: BIDSTree):
        tree.add_bold(
            sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="clean"
        )  # no events → unsegmented
        tree.add_feature(
            sub=SUB, task=TASK, kind="mfcc", run="1", extra_entities={"trial": "1"}
        )
        enc = _make_encoding(tree, ["mfcc"])
        feature_paths = enc._discover_features(SUB)
        bold_metas = enc._discover_bold(SUB)
        with pytest.raises(ValueError, match="only ses and run are valid"):
            enc._resolve_cell_keys(SUB, feature_paths, bold_metas)

    # Segmented run cases

    def test_valid_segmented_cells_pass(self, tree: BIDSTree):
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="clean")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=[
                {"trial_type": "block-1", "onset": 0.0, "duration": 100.0},
                {"trial_type": "block-2", "onset": 100.0, "duration": 100.0},
            ],
        )
        tree.add_feature(
            sub=SUB, task=TASK, kind="mfcc", run="1", extra_entities={"block": "1"}
        )
        tree.add_feature(
            sub=SUB, task=TASK, kind="mfcc", run="1", extra_entities={"block": "2"}
        )
        enc = _make_encoding(tree, ["mfcc"])
        feature_paths = enc._discover_features(SUB)
        bold_metas = enc._discover_bold(SUB)
        feature_paths = enc._resolve_cell_keys(SUB, feature_paths, bold_metas)
        assert len(feature_paths) == 2

    def test_cell_missing_segment_entity_raises(self, tree: BIDSTree):
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="clean")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=[
                {"trial_type": "block-1", "onset": 0.0, "duration": 100.0},
                {"trial_type": "block-2", "onset": 100.0, "duration": 100.0},
            ],
        )
        tree.add_feature(
            sub=SUB, task=TASK, kind="mfcc", run="1"
        )  # no block entity on the filename
        enc = _make_encoding(tree, ["mfcc"])
        feature_paths = enc._discover_features(SUB)
        bold_metas = enc._discover_bold(SUB)
        with pytest.raises(ValueError, match="missing segment entity"):
            enc._resolve_cell_keys(SUB, feature_paths, bold_metas)

    def test_cell_value_not_in_events_raises(self, tree: BIDSTree):
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="clean")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=[
                {"trial_type": "block-1", "onset": 0.0, "duration": 100.0},
                {"trial_type": "block-2", "onset": 100.0, "duration": 100.0},
            ],
        )
        tree.add_feature(
            sub=SUB, task=TASK, kind="mfcc", run="1", extra_entities={"block": "3"}
        )  # block-3 not in events
        enc = _make_encoding(tree, ["mfcc"])
        feature_paths = enc._discover_features(SUB)
        bold_metas = enc._discover_bold(SUB)
        with pytest.raises(ValueError, match="not found in events"):
            enc._resolve_cell_keys(SUB, feature_paths, bold_metas)

    def test_extra_entity_on_segmented_run_raises(self, tree: BIDSTree):
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="clean")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=[
                {"trial_type": "block-1", "onset": 0.0, "duration": 100.0},
            ],
        )
        tree.add_feature(
            sub=SUB,
            task=TASK,
            kind="mfcc",
            run="1",
            extra_entities={"block": "1", "extra": "foo"},
        )
        enc = _make_encoding(tree, ["mfcc"])
        feature_paths = enc._discover_features(SUB)
        bold_metas = enc._discover_bold(SUB)
        with pytest.raises(ValueError, match="absent from events.json"):
            enc._resolve_cell_keys(SUB, feature_paths, bold_metas)

    # Metadata merge cases (filename × sidecar)

    def test_sidecar_only_metadata_merged_onto_cell_key(self, tree: BIDSTree):
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="clean")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=[
                {"trial_type": "block-1", "onset": 0.0, "duration": 100.0},
                {"trial_type": "block-2", "onset": 100.0, "duration": 100.0},
            ],
            sidecar_json={
                "trial_type": {
                    "Levels": {
                        "block-1": {"metadata": {"cond": "R"}},
                        "block-2": {"metadata": {"cond": "L"}},
                    }
                }
            },
        )
        tree.add_feature(
            sub=SUB, task=TASK, kind="mfcc", run="1", extra_entities={"block": "1"}
        )
        tree.add_feature(
            sub=SUB, task=TASK, kind="mfcc", run="1", extra_entities={"block": "2"}
        )
        enc = _make_encoding(tree, ["mfcc"])
        feature_paths = enc._discover_features(SUB)
        bold_metas = enc._discover_bold(SUB)
        feature_paths = enc._resolve_cell_keys(SUB, feature_paths, bold_metas)
        cell_cond_values = {
            feature_key.cell.get("cond") for feature_key in feature_paths
        }
        assert cell_cond_values == {"R", "L"}

    def test_filename_value_agrees_with_sidecar_passes(self, tree: BIDSTree):
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="clean")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=[
                {"trial_type": "block-1", "onset": 0.0, "duration": 100.0},
            ],
            sidecar_json={
                "trial_type": {
                    "Levels": {
                        "block-1": {"metadata": {"cond": "R"}},
                    }
                }
            },
        )
        tree.add_feature(
            sub=SUB,
            task=TASK,
            kind="mfcc",
            run="1",
            extra_entities={"block": "1", "cond": "R"},
        )
        enc = _make_encoding(tree, ["mfcc"])
        feature_paths = enc._discover_features(SUB)
        bold_metas = enc._discover_bold(SUB)
        feature_paths = enc._resolve_cell_keys(SUB, feature_paths, bold_metas)
        assert next(iter(feature_paths)).cell.get("cond") == "R"

    def test_filename_value_disagrees_with_sidecar_raises(self, tree: BIDSTree):
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="clean")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=[
                {"trial_type": "block-1", "onset": 0.0, "duration": 100.0},
            ],
            sidecar_json={
                "trial_type": {
                    "Levels": {
                        "block-1": {"metadata": {"cond": "R"}},
                    }
                }
            },
        )
        tree.add_feature(
            sub=SUB,
            task=TASK,
            kind="mfcc",
            run="1",
            extra_entities={"block": "1", "cond": "L"},
        )
        enc = _make_encoding(tree, ["mfcc"])
        feature_paths = enc._discover_features(SUB)
        bold_metas = enc._discover_bold(SUB)
        with pytest.raises(ValueError, match="disagree on"):
            enc._resolve_cell_keys(SUB, feature_paths, bold_metas)


class TestApplyFilters:
    def test_no_filters_returns_unchanged(self, tree: BIDSTree):
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="clean")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=[
                {"trial_type": "block-1", "onset": 0.0, "duration": 100.0},
                {"trial_type": "block-2", "onset": 100.0, "duration": 100.0},
            ],
        )
        tree.add_feature(
            sub=SUB, task=TASK, kind="mfcc", run="1", extra_entities={"block": "1"}
        )
        tree.add_feature(
            sub=SUB, task=TASK, kind="mfcc", run="1", extra_entities={"block": "2"}
        )
        enc = _make_encoding(tree, ["mfcc"])
        feature_paths = enc._discover_features(SUB)
        bold_metas = enc._discover_bold(SUB)
        feature_paths = enc._resolve_cell_keys(SUB, feature_paths, bold_metas)
        filtered_features, filtered_bold = enc._apply_filters(
            SUB, feature_paths, bold_metas
        )
        assert filtered_features == feature_paths
        assert filtered_bold == bold_metas

    def test_filter_narrows_features(self, tree: BIDSTree):
        for run in ("1", "2"):
            tree.add_bold(
                sub=SUB, task=TASK, space=SPACE, run=run, tr=2.0, desc="clean"
            )
            tree.add_events(
                sub=SUB,
                task=TASK,
                run=run,
                rows=[
                    {"trial_type": "block-1", "onset": 0.0, "duration": 100.0},
                    {"trial_type": "block-2", "onset": 100.0, "duration": 100.0},
                ],
            )
        tree.add_feature(
            sub=SUB, task=TASK, kind="mfcc", run="1", extra_entities={"block": "1"}
        )
        tree.add_feature(
            sub=SUB, task=TASK, kind="mfcc", run="1", extra_entities={"block": "2"}
        )
        tree.add_feature(
            sub=SUB, task=TASK, kind="mfcc", run="2", extra_entities={"block": "1"}
        )
        tree.add_feature(
            sub=SUB, task=TASK, kind="mfcc", run="2", extra_entities={"block": "2"}
        )
        enc = _make_encoding(tree, ["mfcc"], bids_filters=["run-1"])
        feature_paths = enc._discover_features(SUB)
        bold_metas = enc._discover_bold(SUB)
        feature_paths = enc._resolve_cell_keys(SUB, feature_paths, bold_metas)
        feature_paths, bold_metas = enc._apply_filters(SUB, feature_paths, bold_metas)
        assert all(feature_key.cell.get("run") == "1" for feature_key in feature_paths)
        assert all(bold_key.run == "1" for bold_key in bold_metas)

    def test_or_within_entity_and_across_entities(self, tree: BIDSTree):
        for ses, run in (("1", "1"), ("1", "2"), ("2", "1")):
            tree.add_bold(
                sub=SUB, task=TASK, space=SPACE, ses=ses, run=run, tr=2.0, desc="clean"
            )
            tree.add_events(
                sub=SUB,
                task=TASK,
                ses=ses,
                run=run,
                rows=[
                    {"trial_type": "block-1", "onset": 0.0, "duration": 100.0},
                ],
            )
            tree.add_feature(
                sub=SUB,
                task=TASK,
                kind="mfcc",
                ses=ses,
                run=run,
                extra_entities={"block": "1"},
            )
        enc = _make_encoding(
            tree,
            ["mfcc"],
            bids_filters=["ses-1", "run-1", "run-2"],  # ses-1 AND (run-1 OR run-2)
        )
        feature_paths = enc._discover_features(SUB)
        bold_metas = enc._discover_bold(SUB)
        feature_paths = enc._resolve_cell_keys(SUB, feature_paths, bold_metas)
        feature_paths, bold_metas = enc._apply_filters(SUB, feature_paths, bold_metas)
        assert BoldKey(ses="1", run="1") in bold_metas
        assert BoldKey(ses="1", run="2") in bold_metas
        assert BoldKey(ses="2", run="1") not in bold_metas

    def test_filter_on_bold_only_entity_skipped_on_features(self, tree: BIDSTree):
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="clean")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=[
                {"trial_type": "block-1", "onset": 0.0, "duration": 100.0},
                {"trial_type": "block-2", "onset": 100.0, "duration": 100.0},
            ],
        )
        tree.add_feature(
            sub=SUB, task=TASK, kind="mfcc", run="1", extra_entities={"block": "1"}
        )
        tree.add_feature(
            sub=SUB, task=TASK, kind="mfcc", run="1", extra_entities={"block": "2"}
        )
        enc = _make_encoding(
            tree,
            ["mfcc"],
            bids_filters=["desc-clean"],  # on BOLD filenames but not on CellKey
        )
        feature_paths = enc._discover_features(SUB)
        bold_metas = enc._discover_bold(SUB)
        feature_paths = enc._resolve_cell_keys(SUB, feature_paths, bold_metas)
        feature_paths, bold_metas = enc._apply_filters(SUB, feature_paths, bold_metas)
        assert len(feature_paths) == 2
        assert len(bold_metas) == 1

    def test_filter_on_cell_only_entity_skipped_on_bold(self, tree: BIDSTree):
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="clean")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=[
                {"trial_type": "block-1", "onset": 0.0, "duration": 100.0},
                {"trial_type": "block-2", "onset": 100.0, "duration": 100.0},
            ],
            sidecar_json={
                "trial_type": {
                    "Levels": {
                        "block-1": {"metadata": {"cond": "R"}},
                        "block-2": {"metadata": {"cond": "L"}},
                    }
                }
            },
        )
        tree.add_feature(
            sub=SUB, task=TASK, kind="mfcc", run="1", extra_entities={"block": "1"}
        )
        tree.add_feature(
            sub=SUB, task=TASK, kind="mfcc", run="1", extra_entities={"block": "2"}
        )
        enc = _make_encoding(
            tree,
            ["mfcc"],
            bids_filters=["cond-R"],  # on CellKey but not on BOLD filenames
        )
        feature_paths = enc._discover_features(SUB)
        bold_metas = enc._discover_bold(SUB)
        feature_paths = enc._resolve_cell_keys(SUB, feature_paths, bold_metas)
        feature_paths, bold_metas = enc._apply_filters(SUB, feature_paths, bold_metas)
        assert len(feature_paths) == 1
        assert len(bold_metas) == 1

    def test_typo_filter_entity_raises(self, tree: BIDSTree):
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="clean")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=[
                {"trial_type": "block-1", "onset": 0.0, "duration": 100.0},
                {"trial_type": "block-2", "onset": 100.0, "duration": 100.0},
            ],
        )
        tree.add_feature(
            sub=SUB, task=TASK, kind="mfcc", run="1", extra_entities={"block": "1"}
        )
        tree.add_feature(
            sub=SUB, task=TASK, kind="mfcc", run="1", extra_entities={"block": "2"}
        )
        enc = _make_encoding(tree, ["mfcc"], bids_filters=["typo-foo"])
        feature_paths = enc._discover_features(SUB)
        bold_metas = enc._discover_bold(SUB)
        feature_paths = enc._resolve_cell_keys(SUB, feature_paths, bold_metas)
        with pytest.raises(ValueError, match="typo"):
            enc._apply_filters(SUB, feature_paths, bold_metas)

    def test_valid_entity_wrong_value_passes_filter_raises_at_coverage(
        self, tree: BIDSTree
    ):
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="clean")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=[
                {"trial_type": "block-1", "onset": 0.0, "duration": 100.0},
                {"trial_type": "block-2", "onset": 100.0, "duration": 100.0},
            ],
        )
        tree.add_feature(
            sub=SUB, task=TASK, kind="mfcc", run="1", extra_entities={"block": "1"}
        )
        tree.add_feature(
            sub=SUB, task=TASK, kind="mfcc", run="1", extra_entities={"block": "2"}
        )
        enc = _make_encoding(tree, ["mfcc"], bids_filters=["block-99"])
        feature_paths = enc._discover_features(SUB)
        bold_metas = enc._discover_bold(SUB)
        feature_paths = enc._resolve_cell_keys(SUB, feature_paths, bold_metas)
        feature_paths, bold_metas = enc._apply_filters(SUB, feature_paths, bold_metas)
        with pytest.raises(FileNotFoundError, match="No feature files match"):
            enc._validate_coverage(SUB, feature_paths, bold_metas)


class TestValidateCoverage:
    def test_valid_alignment_passes(self, tree: BIDSTree):
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", desc="clean")
        tree.add_feature(sub=SUB, task=TASK, kind="mfcc", run="1")
        enc = _make_encoding(tree, ["mfcc"])
        feature_paths = enc._discover_features(SUB)
        bold_metas = enc._discover_bold(SUB)
        feature_paths = enc._resolve_cell_keys(SUB, feature_paths, bold_metas)
        feature_paths, bold_metas = enc._apply_filters(SUB, feature_paths, bold_metas)
        enc._validate_coverage(SUB, feature_paths, bold_metas)

    def test_cross_file_task_invariance_raises(self, tree: BIDSTree):
        tree.add_bold(sub=SUB, space=SPACE, task="conv", run="1", desc="clean")
        tree.add_feature(sub=SUB, task="rest", kind="mfcc", run="1")
        enc = _make_encoding(tree, ["mfcc"])
        feature_paths = enc._discover_features(SUB)
        bold_metas = enc._discover_bold(SUB)
        feature_paths = enc._resolve_cell_keys(SUB, feature_paths, bold_metas)
        feature_paths, bold_metas = enc._apply_filters(SUB, feature_paths, bold_metas)
        with pytest.raises(ValueError, match="task"):
            enc._validate_coverage(SUB, feature_paths, bold_metas)

    def test_empty_features_after_filter_raises(self, tree: BIDSTree):
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", desc="clean")
        tree.add_feature(sub=SUB, task=TASK, kind="mfcc", run="1")
        enc = _make_encoding(tree, ["mfcc"])
        feature_paths = enc._discover_features(SUB)
        bold_metas = enc._discover_bold(SUB)
        feature_paths = enc._resolve_cell_keys(SUB, feature_paths, bold_metas)
        feature_paths, bold_metas = enc._apply_filters(SUB, feature_paths, bold_metas)
        with pytest.raises(FileNotFoundError, match="No feature files match"):
            enc._validate_coverage(SUB, {}, bold_metas)

    def test_empty_bold_after_filter_raises(self, tree: BIDSTree):
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", desc="clean")
        tree.add_feature(sub=SUB, task=TASK, kind="mfcc", run="1")
        enc = _make_encoding(tree, ["mfcc"])
        feature_paths = enc._discover_features(SUB)
        bold_metas = enc._discover_bold(SUB)
        feature_paths = enc._resolve_cell_keys(SUB, feature_paths, bold_metas)
        feature_paths, bold_metas = enc._apply_filters(SUB, feature_paths, bold_metas)
        with pytest.raises(FileNotFoundError, match="No BOLD files match"):
            enc._validate_coverage(SUB, feature_paths, {})

    def test_bold_without_features_raises(self, tree: BIDSTree):
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", desc="clean")
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="2", desc="clean")
        tree.add_feature(sub=SUB, task=TASK, kind="mfcc", run="1")
        enc = _make_encoding(tree, ["mfcc"])
        feature_paths = enc._discover_features(SUB)
        bold_metas = enc._discover_bold(SUB)
        feature_paths = enc._resolve_cell_keys(SUB, feature_paths, bold_metas)
        feature_paths, bold_metas = enc._apply_filters(SUB, feature_paths, bold_metas)
        with pytest.raises(FileNotFoundError, match="No feature files found for BOLD"):
            enc._validate_coverage(SUB, feature_paths, bold_metas)

    def test_features_without_bold_after_filter_raises(self, tree: BIDSTree):
        # `res` is a BOLD-only entity (features carry no res), so filtering on it
        # narrows BOLDs but not features, leaving run-2 features without a match
        tree.add_bold(
            sub=SUB,
            task=TASK,
            space=SPACE,
            run="1",
            desc="clean",
            extra_entities={"res": "2"},
        )
        tree.add_bold(
            sub=SUB,
            task=TASK,
            space=SPACE,
            run="2",
            desc="clean",
            extra_entities={"res": "3"},
        )
        tree.add_feature(sub=SUB, task=TASK, kind="mfcc", run="1")
        tree.add_feature(sub=SUB, task=TASK, kind="mfcc", run="2")
        enc = _make_encoding(tree, ["mfcc"], bids_filters=["res-2"])
        feature_paths = enc._discover_features(SUB)
        bold_metas = enc._discover_bold(SUB)
        feature_paths = enc._resolve_cell_keys(SUB, feature_paths, bold_metas)
        feature_paths, bold_metas = enc._apply_filters(SUB, feature_paths, bold_metas)
        with pytest.raises(FileNotFoundError, match="No BOLD file found for features"):
            enc._validate_coverage(SUB, feature_paths, bold_metas)

    def test_multiple_bold_gaps_reports_count(self, tree: BIDSTree):
        for run in ("1", "2", "3"):
            tree.add_bold(sub=SUB, task=TASK, space=SPACE, run=run, desc="clean")
            tree.add_feature(sub=SUB, task=TASK, kind="mfcc", run=run)
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="4", desc="clean")
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="5", desc="clean")
        enc = _make_encoding(tree, ["mfcc"])
        feature_paths = enc._discover_features(SUB)
        bold_metas = enc._discover_bold(SUB)
        feature_paths = enc._resolve_cell_keys(SUB, feature_paths, bold_metas)
        feature_paths, bold_metas = enc._apply_filters(SUB, feature_paths, bold_metas)
        with pytest.raises(FileNotFoundError, match="other coverage gaps"):
            enc._validate_coverage(SUB, feature_paths, bold_metas)
