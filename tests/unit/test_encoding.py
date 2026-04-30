import pytest

from hypline.encoding import BoldKey, CellKey, Encoding, EncodingConfig, FeatureKey

from .conftest import SPACE, SUB, TASK, BIDSTree


def make_encoding(
    tree: BIDSTree,
    features: list[str],
    *,
    bold_space: str = SPACE,
    bids_filters: list[str] | None = None,
) -> Encoding:
    return Encoding(
        EncodingConfig(),
        features=features,
        features_dir=tree.features_dir,
        bold_dir=tree.bold_dir,
        output_dir=tree.output_dir,
        bold_space=bold_space,
        bids_filters=bids_filters,
    )


class TestEncodingInit:
    def test_valid_config_succeeds(self, tree: BIDSTree):
        enc = make_encoding(tree, ["mfcc"])
        assert enc.features == ["mfcc"]

    def test_empty_features_raises(self, tree: BIDSTree):
        with pytest.raises(ValueError, match="non-empty"):
            make_encoding(tree, [])

    def test_duplicate_features_raises(self, tree: BIDSTree):
        with pytest.raises(ValueError, match="Duplicate"):
            make_encoding(tree, ["mfcc", "mfcc"])

    def test_reserved_entity_in_filters_raises(self, tree: BIDSTree):
        with pytest.raises(ValueError, match="sub"):
            make_encoding(tree, ["mfcc"], bids_filters=["sub-001"])

    def test_unknown_entity_accepted_at_init(self, tree: BIDSTree):
        enc = make_encoding(tree, ["mfcc"], bids_filters=["xyz-foo"])
        assert enc.bids_filters == ["xyz-foo"]

    def test_invalid_bold_space_raises(self, tree: BIDSTree):
        with pytest.raises(ValueError, match="Unsupported BOLD data space"):
            make_encoding(tree, ["mfcc"], bold_space="notaspace")


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
        tree.add_feature("mfcc", run="1")
        enc = make_encoding(tree, ["mfcc"])
        feature_paths = enc._discover_features(SUB)
        expected = FeatureKey(cell=CellKey(run="1"), feature="mfcc")
        assert expected in feature_paths

    def test_no_files_raises(self, tree: BIDSTree):
        enc = make_encoding(tree, ["mfcc"])
        with pytest.raises(FileNotFoundError, match="No matching feature files"):
            enc._discover_features(SUB)

    def test_duplicate_feature_file_raises(self, tree: BIDSTree):
        tree.add_feature("mfcc", run="1")
        tree.add_feature("mfcc", run="1", subdir="sub")
        enc = make_encoding(tree, ["mfcc"])
        with pytest.raises(ValueError, match="Multiple feature files"):
            enc._discover_features(SUB)

    def test_missing_feature_at_one_cell_raises(self, tree: BIDSTree):
        tree.add_feature("mfcc", run="1")
        tree.add_feature("mfcc", run="2")
        tree.add_feature("clip", run="1")
        enc = make_encoding(tree, ["mfcc", "clip"])
        with pytest.raises(FileNotFoundError, match="Missing feature=clip"):
            enc._discover_features(SUB)

    def test_semantic_entity_filter_narrows_results(self, tree: BIDSTree):
        tree.add_feature("mfcc", block="a")
        tree.add_feature("mfcc", block="b")
        enc = make_encoding(tree, ["mfcc"], bids_filters=["block-a"])
        feature_paths = enc._discover_features(SUB)
        assert {k.cell.get("block") for k in feature_paths} == {"a"}

    def test_task_invariance_violation_raises(self, tree: BIDSTree):
        tree.add_feature("mfcc", task="rest", run="1")
        tree.add_feature("mfcc", task="conv", run="2")
        enc = make_encoding(tree, ["mfcc"])
        with pytest.raises(ValueError, match="task"):
            enc._discover_features(SUB)

    def test_acquisition_entities_not_required_on_features(self, tree: BIDSTree):
        tree.add_feature("mfcc", run="1")
        enc = make_encoding(tree, ["mfcc"], bids_filters=["desc-preproc"])
        feature_paths = enc._discover_features(SUB)
        expected = FeatureKey(cell=CellKey(run="1"), feature="mfcc")
        assert expected in feature_paths

    def test_mixed_segmented_unsegmented_runs_raise(self, tree: BIDSTree):
        tree.add_feature("mfcc", run="1", block="1")
        tree.add_feature("mfcc", run="2")
        enc = make_encoding(tree, ["mfcc"])
        with pytest.raises(ValueError, match="Inconsistent feature file schemas"):
            enc._discover_features(SUB)

    def test_typo_filter_entity_raises(self, tree: BIDSTree):
        tree.add_feature("mfcc", run="1")
        enc = make_encoding(tree, ["mfcc"], bids_filters=["typo-foo"])
        with pytest.raises(ValueError, match="typo"):
            enc._discover_features(SUB)

    def test_absent_common_entity_filter_raises(self, tree: BIDSTree):
        # ses not on any file — filter can't match, entity check fires
        tree.add_feature("mfcc", run="1")
        enc = make_encoding(tree, ["mfcc"], bids_filters=["ses-99"])
        with pytest.raises(ValueError, match="ses"):
            enc._discover_features(SUB)

    def test_valid_entity_wrong_value_gives_file_not_found(self, tree: BIDSTree):
        # block is on files — filter entity is valid, just no match for block-99
        tree.add_feature("mfcc", block="1")
        tree.add_feature("mfcc", block="2")
        enc = make_encoding(tree, ["mfcc"], bids_filters=["block-99"])
        with pytest.raises(FileNotFoundError):
            enc._discover_features(SUB)

    def test_schema_error_fires_before_coverage_error(self, tree: BIDSTree):
        # clip missing at run-2, but schema mismatch should raise first
        tree.add_feature("mfcc", run="1", block="1")
        tree.add_feature("mfcc", run="2")
        tree.add_feature("clip", run="1", block="1")
        enc = make_encoding(tree, ["mfcc", "clip"])
        with pytest.raises(ValueError, match="Inconsistent feature file schemas"):
            enc._discover_features(SUB)


class TestDiscoverBold:
    def test_returns_expected_keys(self, tree: BIDSTree):
        tree.add_bold(run="1")
        enc = make_encoding(tree, ["mfcc"])
        bold_metas = enc._discover_bold(SUB)
        assert BoldKey(ses=None, run="1") in bold_metas

    def test_no_files_raises(self, tree: BIDSTree):
        enc = make_encoding(tree, ["mfcc"])
        with pytest.raises(FileNotFoundError, match="No BOLD files"):
            enc._discover_bold(SUB)

    def test_duplicate_bold_raises(self, tree: BIDSTree):
        tree.add_bold(run="1")
        tree.add_bold(run="1", subdir="sub")
        enc = make_encoding(tree, ["mfcc"])
        with pytest.raises(ValueError, match="Duplicate BOLD"):
            enc._discover_bold(SUB)

    def test_cross_session_runs_distinguished(self, tree: BIDSTree):
        tree.add_bold(ses="1", run="1")
        tree.add_bold(ses="2", run="1")
        enc = make_encoding(tree, ["mfcc"])
        bold_metas = enc._discover_bold(SUB)
        assert BoldKey(ses="1", run="1") in bold_metas
        assert BoldKey(ses="2", run="1") in bold_metas

    def test_bold_siblings_excluded(self, tree: BIDSTree):
        tree.add_bold(run="1")
        tree.add_bold_siblings(run="1")
        enc = make_encoding(tree, ["mfcc"])
        bold_metas = enc._discover_bold(SUB)
        assert list(bold_metas.keys()) == [BoldKey(ses=None, run="1")]

    def test_wrong_space_bold_excluded(self, tree: BIDSTree):
        tree.add_bold(run="1", space="MNI152NLin6Asym")
        tree.add_bold(run="1", space="T1w")
        enc = make_encoding(tree, ["mfcc"], bold_space="MNI152NLin6Asym")
        bold_metas = enc._discover_bold(SUB)
        assert list(bold_metas.keys()) == [BoldKey(ses=None, run="1")]

    def test_inconsistent_tr_raises(self, tree: BIDSTree):
        tree.add_bold(run="1", tr=2.0)
        tree.add_bold(run="2", tr=1.5)
        enc = make_encoding(tree, ["mfcc"])
        with pytest.raises(ValueError, match="Inconsistent repetition times"):
            enc._discover_bold(SUB)

    def test_task_invariance_violation_raises(self, tree: BIDSTree):
        tree.add_bold(task="rest", run="1")
        tree.add_bold(task="conv", run="2")
        enc = make_encoding(tree, ["mfcc"])
        with pytest.raises(ValueError, match="task"):
            enc._discover_bold(SUB)

    def test_acq_invariance_violation_raises(self, tree: BIDSTree):
        for acq, run in (("hi", "1"), ("lo", "2")):
            stem = (
                f"sub-{SUB}_task-{TASK}_acq-{acq}_run-{run}_"
                f"space-{SPACE}_desc-preproc_bold"
            )
            (tree.bold_dir / f"{stem}.nii.gz").touch()
            (tree.bold_dir / f"{stem}.json").write_text('{"RepetitionTime": 2.0}')
        enc = make_encoding(tree, ["mfcc"])
        with pytest.raises(ValueError, match="acq"):
            enc._discover_bold(SUB)

    def test_no_events_gives_no_segments(self, tree: BIDSTree):
        tree.add_bold(run="1", tr=2.0)
        enc = make_encoding(tree, ["mfcc"])
        bold_metas = enc._discover_bold(SUB)
        meta = bold_metas[BoldKey(ses=None, run="1")]
        assert meta.segments == []

    def test_structural_entity_slices_parsed(self, tree: BIDSTree):
        tree.add_bold(run="1", tr=2.0)
        tree.add_events(
            run="1",
            rows=[
                {"trial_type": "block-1", "onset": 0.0, "duration": 100.0},
                {"trial_type": "block-2", "onset": 100.0, "duration": 100.0},
            ],
        )
        enc = make_encoding(tree, ["mfcc"])
        bold_metas = enc._discover_bold(SUB)
        meta = bold_metas[BoldKey(ses=None, run="1")]
        assert len(meta.segments) == 2
        assert meta.segments[0].entity == "block"
        assert meta.segments[0].value == "1"
        assert meta.segments[0].slice == slice(0, 50)
        assert meta.segments[1].value == "2"
        assert meta.segments[1].slice == slice(50, 100)

    def test_multiple_kv_entities_raises(self, tree: BIDSTree):
        tree.add_bold(run="1", tr=2.0)
        tree.add_events(
            run="1",
            rows=[
                {"trial_type": "block-1", "onset": 0.0, "duration": 100.0},
                {"trial_type": "trial-1", "onset": 100.0, "duration": 100.0},
            ],
        )
        enc = make_encoding(tree, ["mfcc"])
        with pytest.raises(ValueError, match="multiple BIDS key-value entities"):
            enc._discover_bold(SUB)

    def test_segment_entity_as_flat_label_raises(self, tree: BIDSTree):
        tree.add_bold(run="1", tr=2.0)
        tree.add_events(
            run="1",
            rows=[
                {"trial_type": "block-1", "onset": 0.0, "duration": 100.0},
                {"trial_type": "block", "onset": 100.0, "duration": 100.0},
            ],
        )
        enc = make_encoding(tree, ["mfcc"])
        with pytest.raises(ValueError, match="also appears as a flat label"):
            enc._discover_bold(SUB)

    def test_duplicate_segment_values_raise(self, tree: BIDSTree):
        tree.add_bold(run="1", tr=2.0)
        tree.add_events(
            run="1",
            rows=[
                {"trial_type": "block-1", "onset": 0.0, "duration": 100.0},
                {"trial_type": "block-1", "onset": 100.0, "duration": 100.0},
            ],
        )
        enc = make_encoding(tree, ["mfcc"])
        with pytest.raises(ValueError, match="appears more than once"):
            enc._discover_bold(SUB)

    def test_overlapping_slices_raise(self, tree: BIDSTree):
        tree.add_bold(run="1", tr=2.0)
        tree.add_events(
            run="1",
            rows=[
                {"trial_type": "block-1", "onset": 0.0, "duration": 120.0},
                {"trial_type": "block-2", "onset": 100.0, "duration": 100.0},
            ],
        )
        enc = make_encoding(tree, ["mfcc"])
        with pytest.raises(ValueError, match="overlap"):
            enc._discover_bold(SUB)

    def test_leading_break_allowed(self, tree: BIDSTree):
        tree.add_bold(run="1", tr=2.0)
        tree.add_events(
            run="1",
            rows=[
                {"trial_type": "block-1", "onset": 10.0, "duration": 90.0},
                {"trial_type": "block-2", "onset": 100.0, "duration": 100.0},
            ],
        )
        enc = make_encoding(tree, ["mfcc"])
        bold_metas = enc._discover_bold(SUB)
        meta = bold_metas[BoldKey(ses=None, run="1")]
        assert len(meta.segments) == 2
        assert meta.segments[0].slice == slice(5, 50)

    def test_hyphen_free_trial_type_ignored(self, tree: BIDSTree):
        tree.add_bold(run="1", tr=2.0)
        events_path = (
            tree.bold_dir / f"sub-{SUB}_task-{TASK}_run-1_events.tsv"
        )
        events_path.write_text("trial_type\tonset\tduration\nrest\t0.0\t100.0\n")
        enc = make_encoding(tree, ["mfcc"])
        bold_metas = enc._discover_bold(SUB)
        meta = bold_metas[BoldKey(ses=None, run="1")]
        assert meta.segments == []

    def test_kv_entity_with_gap_passes(self, tree: BIDSTree):
        tree.add_bold(run="1", tr=2.0)
        tree.add_events(
            run="1",
            rows=[
                {"trial_type": "block-1", "onset": 0.0, "duration": 100.0},
                {"trial_type": "block-2", "onset": 120.0, "duration": 80.0},
            ],
        )
        enc = make_encoding(tree, ["mfcc"])
        bold_metas = enc._discover_bold(SUB)
        meta = bold_metas[BoldKey(ses=None, run="1")]
        assert len(meta.segments) == 2

    def test_flat_labels_alongside_segments_ignored(self, tree: BIDSTree):
        tree.add_bold(run="1", tr=2.0)
        tree.add_events(
            run="1",
            rows=[
                {"trial_type": "block-1", "onset": 0.0, "duration": 100.0},
                {"trial_type": "block-2", "onset": 100.0, "duration": 100.0},
                {"trial_type": "fixation", "onset": 0.0, "duration": 10.0},
            ],
        )
        enc = make_encoding(tree, ["mfcc"])
        bold_metas = enc._discover_bold(SUB)
        meta = bold_metas[BoldKey(ses=None, run="1")]
        assert len(meta.segments) == 2
        assert meta.segments[0].entity == "block"

    def test_runs_disagree_on_segment_entity_raises(self, tree: BIDSTree):
        tree.add_bold(run="1", tr=2.0)
        tree.add_bold(run="2", tr=2.0)
        tree.add_events(
            run="1",
            rows=[
                {"trial_type": "block-1", "onset": 0.0, "duration": 100.0},
                {"trial_type": "block-2", "onset": 100.0, "duration": 100.0},
            ],
        )
        tree.add_events(
            run="2",
            rows=[
                {"trial_type": "trial-1", "onset": 0.0, "duration": 100.0},
                {"trial_type": "trial-2", "onset": 100.0, "duration": 100.0},
            ],
        )
        enc = make_encoding(tree, ["mfcc"])
        with pytest.raises(ValueError, match="disagree on segment entity"):
            enc._discover_bold(SUB)

    def test_mixed_segmented_and_unsegmented_raises(self, tree: BIDSTree):
        tree.add_bold(run="1", tr=2.0)
        tree.add_bold(run="2", tr=2.0)  # no events → unsegmented
        tree.add_events(
            run="1",
            rows=[
                {"trial_type": "block-1", "onset": 0.0, "duration": 100.0},
                {"trial_type": "block-2", "onset": 100.0, "duration": 100.0},
            ],
        )
        enc = make_encoding(tree, ["mfcc"])
        with pytest.raises(ValueError, match="disagree on segment entity"):
            enc._discover_bold(SUB)

    def test_events_json_cross_run_schema_mismatch_raises(self, tree: BIDSTree):
        rows = [
            {"trial_type": "block-1", "onset": 0.0, "duration": 100.0},
            {"trial_type": "block-2", "onset": 100.0, "duration": 100.0},
        ]
        tree.add_bold(run="1", tr=2.0)
        tree.add_bold(run="2", tr=2.0)

        # run-1 has metadata key "cond"; run-2 has metadata key "item"
        tree.add_events(
            run="1",
            rows=rows,
            events_json={
                "Segments": [
                    {"block": "1", "cond": "R"},
                    {"block": "2", "cond": "L"},
                ]
            },
        )
        tree.add_events(
            run="2",
            rows=rows,
            events_json={
                "Segments": [
                    {"block": "1", "item": "A"},
                    {"block": "2", "item": "B"},
                ]
            },
        )

        enc = make_encoding(tree, ["mfcc"])
        with pytest.raises(ValueError, match="disagree on segment metadata schema"):
            enc._discover_bold(SUB)

    def test_all_unsegmented_passes(self, tree: BIDSTree):
        tree.add_bold(run="1", tr=2.0)
        tree.add_bold(run="2", tr=2.0)
        enc = make_encoding(tree, ["mfcc"])
        bold_metas = enc._discover_bold(SUB)
        assert all(meta.segments == [] for meta in bold_metas.values())


class TestValidateAlignment:
    def test_valid_alignment_passes(self, tree: BIDSTree):
        tree.add_feature("mfcc", run="1")
        tree.add_bold(run="1")
        enc = make_encoding(tree, ["mfcc"])
        feature_paths = enc._discover_features(SUB)
        bold_metas = enc._discover_bold(SUB)
        enc._validate_alignment(feature_paths, bold_metas)

    def test_cross_file_task_invariance_raises(self, tree: BIDSTree):
        tree.add_feature("mfcc", task="rest", run="1")
        tree.add_bold(task="conv", run="1")
        enc = make_encoding(tree, ["mfcc"])
        feature_paths = enc._discover_features(SUB)
        bold_metas = enc._discover_bold(SUB)
        with pytest.raises(ValueError, match="task"):
            enc._validate_alignment(feature_paths, bold_metas)

    def test_bold_without_features_raises(self, tree: BIDSTree):
        tree.add_feature("mfcc", run="1")
        tree.add_bold(run="1")
        tree.add_bold(run="2")
        enc = make_encoding(tree, ["mfcc"])
        feature_paths = enc._discover_features(SUB)
        bold_metas = enc._discover_bold(SUB)
        with pytest.raises(FileNotFoundError, match="No feature files found for BOLD"):
            enc._validate_alignment(feature_paths, bold_metas)

    def test_features_without_bold_raises(self, tree: BIDSTree):
        tree.add_feature("mfcc", run="1")
        tree.add_feature("mfcc", run="2")
        tree.add_bold(run="1")
        enc = make_encoding(tree, ["mfcc"])
        feature_paths = enc._discover_features(SUB)
        bold_metas = enc._discover_bold(SUB)
        with pytest.raises(FileNotFoundError, match="No BOLD file found for features"):
            enc._validate_alignment(feature_paths, bold_metas)

    def test_multiple_bold_gaps_reports_count(self, tree: BIDSTree):
        for run in ("1", "2", "3"):
            tree.add_feature("mfcc", run=run)
            tree.add_bold(run=run)
        tree.add_bold(run="4")
        tree.add_bold(run="5")
        enc = make_encoding(tree, ["mfcc"])
        feature_paths = enc._discover_features(SUB)
        bold_metas = enc._discover_bold(SUB)
        with pytest.raises(FileNotFoundError, match="other coverage gaps"):
            enc._validate_alignment(feature_paths, bold_metas)

    def test_multiple_feature_gaps_reports_count(self, tree: BIDSTree):
        for run in ("1", "2", "3"):
            tree.add_feature("mfcc", run=run)
            tree.add_bold(run=run)
        tree.add_feature("mfcc", run="4")
        tree.add_feature("mfcc", run="5")
        enc = make_encoding(tree, ["mfcc"])
        feature_paths = enc._discover_features(SUB)
        bold_metas = enc._discover_bold(SUB)
        with pytest.raises(FileNotFoundError, match="other coverage gaps"):
            enc._validate_alignment(feature_paths, bold_metas)

    def test_segmented_valid_cells_pass(self, tree: BIDSTree):
        tree.add_bold(run="1", tr=2.0)
        tree.add_events(
            run="1",
            rows=[
                {"trial_type": "block-1", "onset": 0.0, "duration": 100.0},
                {"trial_type": "block-2", "onset": 100.0, "duration": 100.0},
            ],
        )
        tree.add_feature("mfcc", run="1", block="1")
        tree.add_feature("mfcc", run="1", block="2")
        enc = make_encoding(tree, ["mfcc"])
        feature_paths = enc._discover_features(SUB)
        bold_metas = enc._discover_bold(SUB)
        enc._validate_alignment(feature_paths, bold_metas)

    def test_cell_missing_segment_entity_raises(self, tree: BIDSTree):
        tree.add_bold(run="1", tr=2.0)
        tree.add_events(
            run="1",
            rows=[
                {"trial_type": "block-1", "onset": 0.0, "duration": 100.0},
                {"trial_type": "block-2", "onset": 100.0, "duration": 100.0},
            ],
        )
        tree.add_feature("mfcc", run="1")  # no block entity on the filename
        enc = make_encoding(tree, ["mfcc"])
        feature_paths = enc._discover_features(SUB)
        bold_metas = enc._discover_bold(SUB)
        with pytest.raises(ValueError, match="missing segment entity"):
            enc._validate_alignment(feature_paths, bold_metas)

    def test_cell_value_not_in_events_raises(self, tree: BIDSTree):
        tree.add_bold(run="1", tr=2.0)
        tree.add_events(
            run="1",
            rows=[
                {"trial_type": "block-1", "onset": 0.0, "duration": 100.0},
                {"trial_type": "block-2", "onset": 100.0, "duration": 100.0},
            ],
        )
        tree.add_feature("mfcc", run="1", block="3")  # block-3 not in events
        enc = make_encoding(tree, ["mfcc"])
        feature_paths = enc._discover_features(SUB)
        bold_metas = enc._discover_bold(SUB)
        with pytest.raises(ValueError, match="not found in events"):
            enc._validate_alignment(feature_paths, bold_metas)

    def test_unsegmented_run_multiple_cells_raises(self, tree: BIDSTree):
        tree.add_bold(run="1", tr=2.0)  # no events → unsegmented
        tree.add_feature("mfcc", run="1", trial="1")
        tree.add_feature("mfcc", run="1", trial="2")
        enc = make_encoding(tree, ["mfcc"])
        feature_paths = enc._discover_features(SUB)
        bold_metas = enc._discover_bold(SUB)
        with pytest.raises(ValueError, match="unsegmented but has 2 feature cells"):
            enc._validate_alignment(feature_paths, bold_metas)

    def test_apparent_segment_entity_without_events_passes_silently(
        self, tree: BIDSTree
    ):
        # Blind spot: trial-1 on filename without events.tsv is indistinguishable
        # from a descriptive tag — mismatch surfaces only as row-count errors later
        tree.add_bold(run="1", tr=2.0)
        tree.add_feature("mfcc", run="1", trial="1")
        enc = make_encoding(tree, ["mfcc"])
        feature_paths = enc._discover_features(SUB)
        bold_metas = enc._discover_bold(SUB)
        enc._validate_alignment(feature_paths, bold_metas)  # does not raise
