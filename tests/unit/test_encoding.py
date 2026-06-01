import numpy as np
import polars as pl
import pyarrow.parquet as pq
import pytest

from hypline.encoding import (
    BoldKey,
    CellDelayer,
    CellKey,
    Encoding,
    EncodingArtifact,
    EncodingConfig,
    FeatureKey,
    TrainingData,
    _build_pipeline,
)

from .conftest import BIDSTree

SUB = "001"
TASK = "conv"
SPACE = "MNI152NLin6Asym"


def _make_encoding(
    tree: BIDSTree,
    features: list[str],
    *,
    tasks: list[str] | None = None,
    bold_space: str = SPACE,
    bold_desc: str = "clean",
    bids_filters: list[str] | None = None,
    desc: str = "v1",
    force: bool = False,
) -> Encoding:
    return Encoding(
        EncodingConfig(),
        bids_root=tree.root,
        features=features,
        tasks=tasks if tasks is not None else [TASK],
        bold_space=bold_space,
        bold_desc=bold_desc,
        bids_filters=bids_filters,
        desc=desc,
        force=force,
    )


class TestCellDelayer:
    def test_cell_delayer_resets_at_boundaries(self):
        # Two stacked cells of 4 rows; values encode (cell, row) so bleed is visible
        cell_lengths = [4, 4]
        X = np.arange(1, 9, dtype=float).reshape(-1, 1)
        delays = [0, 1, 2]
        out = CellDelayer(delays=delays, cell_lengths=cell_lengths).transform(X)

        # Output columns are [delay0, delay1, delay2]; cell 2 spans rows 4..7
        # The first max(delays)=2 rows of cell 2 must not pull from cell 1
        col_d1, col_d2 = out[:, 1], out[:, 2]
        assert col_d1[4] == 0  # row 4, delay 1 would source row 3 (cell 1)
        assert col_d2[4] == 0  # row 4, delay 2 would source row 2 (cell 1)
        assert col_d2[5] == 0  # row 5, delay 2 would source row 3 (cell 1)
        # Within-cell delays still work
        assert col_d1[5] == X[4, 0]  # row 5, delay 1 sources row 4 (same cell)
        assert col_d2[6] == X[4, 0]  # row 6, delay 2 sources row 4 (same cell)

    def test_cell_delayer_single_cell_matches_plain_delay(self):
        cell_lengths = [6]
        X = np.arange(1, 7, dtype=float).reshape(-1, 1)
        delays = [0, 1, 2]
        out = CellDelayer(delays=delays, cell_lengths=cell_lengths).transform(X)

        expected_blocks = []
        for d in delays:
            block = np.zeros_like(X)
            if d == 0:
                block[:] = X
            else:
                block[d:] = X[:-d]
            expected_blocks.append(block)
        np.testing.assert_array_equal(out, np.hstack(expected_blocks))

    def test_cell_delayer_short_cell_all_zero_deep_delay(self):
        # Cell of 2 rows with a delay of 3 → that delay block is all-zero, no error
        cell_lengths = [2]
        X = np.arange(1, 3, dtype=float).reshape(-1, 1)
        out = CellDelayer(delays=[0, 3], cell_lengths=cell_lengths).transform(X)
        assert np.all(out[:, 1] == 0)

    def test_cell_delayer_negative_delay_raises(self):
        with pytest.raises(ValueError, match="delays >= 0"):
            CellDelayer(delays=[-1], cell_lengths=[3]).transform(np.zeros((3, 1)))

    def test_cell_delayer_row_count_mismatch_raises(self):
        with pytest.raises(ValueError, match="cell_lengths sum"):
            CellDelayer(delays=[0], cell_lengths=[3]).transform(np.zeros((4, 1)))


class TestBuildPipeline:
    def test_build_pipeline_fits_on_synthetic_training_data(self):
        from himalaya.backend import set_backend

        set_backend("torch")

        rng = np.random.RandomState(0)
        n_rows, n_voxels = 20, 5
        X = rng.randn(n_rows, 7).astype(np.float32)
        Y = rng.randn(n_rows, n_voxels).astype(np.float32)
        data = TrainingData(
            X=X,
            Y=Y,
            row_slices={
                CellKey(task="a", run="1"): slice(0, 10),
                CellKey(task="a", run="2"): slice(10, 20),
            },
            col_slices={"f1": slice(0, 3), "f2": slice(3, 7)},
        )
        cell_lengths = [s.stop - s.start for s in data.row_slices.values()]
        pipeline = _build_pipeline(
            col_slices=data.col_slices,
            cell_lengths=cell_lengths,
            delays=[0, 1, 2],
            alphas=[1.0, 10.0, 100.0],
        )
        pipeline.fit(data.X, data.Y)
        pred = np.asarray(pipeline.predict(data.X))
        assert pred.shape == (n_rows, n_voxels)


class TestEncodingInit:
    def test_valid_config_succeeds(self, tree: BIDSTree):
        enc = _make_encoding(tree, ["mfcc"])
        assert list(enc._features) == ["mfcc"]

    def test_empty_features_raises(self, tree: BIDSTree):
        with pytest.raises(ValueError, match="non-empty"):
            _make_encoding(tree, [])

    def test_duplicate_features_raises(self, tree: BIDSTree):
        with pytest.raises(ValueError, match="Duplicate"):
            _make_encoding(tree, ["mfcc", "mfcc"])

    def test_duplicate_kind_across_variants_raises(self, tree: BIDSTree):
        with pytest.raises(ValueError, match="Duplicate feature kind"):
            _make_encoding(tree, ["semantic", "semantic-gpt2"])

    @pytest.mark.parametrize("entry", ["a_b", "a-", "-b", "a-b-c", ""])
    def test_malformed_feature_entry_raises(self, tree: BIDSTree, entry: str):
        with pytest.raises(ValueError, match="Invalid (kind|desc)"):
            _make_encoding(tree, [entry])

    def test_variant_entry_parsed(self, tree: BIDSTree):
        enc = _make_encoding(tree, ["semantic-gpt3"])
        assert enc._features == {"semantic-gpt3": ("semantic", "gpt3")}

    def test_desc_reserved_in_filters_raises(self, tree: BIDSTree):
        with pytest.raises(ValueError, match="desc"):
            _make_encoding(tree, ["mfcc"], bids_filters=["desc-gpt3"])

    def test_reserved_entity_in_filters_raises(self, tree: BIDSTree):
        with pytest.raises(ValueError, match="sub"):
            _make_encoding(tree, ["mfcc"], bids_filters=["sub-001"])

    def test_unknown_entity_accepted_at_init(self, tree: BIDSTree):
        enc = _make_encoding(tree, ["mfcc"], bids_filters=["xyz-foo"])
        assert enc.bids_filters == ["xyz-foo"]

    def test_invalid_bold_space_raises(self, tree: BIDSTree):
        with pytest.raises(ValueError, match="Unsupported BOLD data space"):
            _make_encoding(tree, ["mfcc"], bold_space="notaspace")

    def test_invalid_bold_desc_raises(self, tree: BIDSTree):
        with pytest.raises(ValueError, match="Invalid bold_desc"):
            _make_encoding(tree, ["mfcc"], bold_desc="not-valid")


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
        expected = FeatureKey(cell=CellKey(task=TASK, run="1"), feature="mfcc")
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

    def test_canonical_reads_bare_folder_not_variants(self, tree: BIDSTree):
        # The user-visible bug fix: variants on disk no longer collide
        bare = tree.add_feature(sub=SUB, task=TASK, kind="phonemic", run="1")
        tree.add_feature(sub=SUB, task=TASK, kind="phonemic", run="1", desc="gpt3")
        enc = _make_encoding(tree, ["phonemic"])
        feature_paths = enc._discover_features(SUB)
        key = FeatureKey(cell=CellKey(task=TASK, run="1"), feature="phonemic")
        assert feature_paths[key].path == bare

    def test_variant_reads_variant_folder_only(self, tree: BIDSTree):
        tree.add_feature(sub=SUB, task=TASK, kind="phonemic", run="1")
        variant = tree.add_feature(
            sub=SUB, task=TASK, kind="phonemic", run="1", desc="gpt3"
        )
        enc = _make_encoding(tree, ["phonemic-gpt3"])
        feature_paths = enc._discover_features(SUB)
        key = FeatureKey(cell=CellKey(task=TASK, run="1"), feature="phonemic-gpt3")
        assert feature_paths[key].path == variant

    def test_missing_variant_raises(self, tree: BIDSTree):
        tree.add_feature(sub=SUB, task=TASK, kind="phonemic", run="1")
        enc = _make_encoding(tree, ["phonemic-gpt3"])
        with pytest.raises(FileNotFoundError):
            enc._discover_features(SUB)

    def test_distinct_variants_form_separate_feature_groups(self, tree: BIDSTree):
        # Distinct kinds, each a variant — verbatim strings key the groups
        tree.add_feature(sub=SUB, task=TASK, kind="phonemic", run="1", desc="gpt3")
        tree.add_feature(sub=SUB, task=TASK, kind="semantic", run="1", desc="bert")
        enc = _make_encoding(tree, ["phonemic-gpt3", "semantic-bert"])
        feature_paths = enc._discover_features(SUB)
        features = {fk.feature for fk in feature_paths}
        assert features == {"phonemic-gpt3", "semantic-bert"}

    def test_unrequested_task_files_filtered_out(self, tree: BIDSTree):
        tree.add_feature(sub=SUB, task="rest", kind="mfcc", run="1")
        tree.add_feature(sub=SUB, task="conv", kind="mfcc", run="2")
        enc = _make_encoding(tree, ["mfcc"], tasks=["conv"])
        feature_paths = enc._discover_features(SUB)
        cell_keys = {fk.cell for fk in feature_paths}
        assert cell_keys == {CellKey(task="conv", run="2")}

    def test_multi_task_cells_distinct(self, tree: BIDSTree):
        tree.add_feature(sub=SUB, task="rest", kind="mfcc", run="1")
        tree.add_feature(sub=SUB, task="conv", kind="mfcc", run="1")
        enc = _make_encoding(tree, ["mfcc"], tasks=["rest", "conv"])
        feature_paths = enc._discover_features(SUB)
        cell_keys = {fk.cell for fk in feature_paths}
        assert cell_keys == {
            CellKey(task="rest", run="1"),
            CellKey(task="conv", run="1"),
        }

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
        assert BoldKey(ses=None, task=TASK, run="1") in bold_metas

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
        assert BoldKey(ses="1", task=TASK, run="1") in bold_metas
        assert BoldKey(ses="2", task=TASK, run="1") in bold_metas

    def test_bold_siblings_excluded(self, tree: BIDSTree):
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", desc="clean")
        tree.add_bold_siblings(sub=SUB, task=TASK, space=SPACE, run="1")
        enc = _make_encoding(tree, ["mfcc"])
        bold_metas = enc._discover_bold(SUB)
        assert list(bold_metas.keys()) == [BoldKey(ses=None, task=TASK, run="1")]

    def test_wrong_space_bold_excluded(self, tree: BIDSTree):
        tree.add_bold(
            sub=SUB, task=TASK, run="1", space="MNI152NLin6Asym", desc="clean"
        )
        tree.add_bold(sub=SUB, task=TASK, run="1", space="T1w", desc="clean")
        enc = _make_encoding(tree, ["mfcc"], bold_space="MNI152NLin6Asym")
        bold_metas = enc._discover_bold(SUB)
        assert list(bold_metas.keys()) == [BoldKey(ses=None, task=TASK, run="1")]

    def test_bold_desc_selects_matching_derivative(self, tree: BIDSTree):
        # bold_desc picks one derivative flavor; sibling flavors are not discovered
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="clean")
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="2", tr=2.0, desc="smooth")
        enc = _make_encoding(tree, ["mfcc"], bold_desc="smooth")
        bold_metas = enc._discover_bold(SUB)
        assert set(bold_metas) == {BoldKey(ses=None, task=TASK, run="2")}

    def test_inconsistent_tr_raises(self, tree: BIDSTree):
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="clean")
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="2", tr=1.5, desc="clean")
        enc = _make_encoding(tree, ["mfcc"])
        with pytest.raises(ValueError, match="Inconsistent repetition times"):
            enc._discover_bold(SUB)

    def test_unrequested_task_bold_filtered_out(self, tree: BIDSTree):
        tree.add_bold(sub=SUB, space=SPACE, task="rest", run="1", desc="clean")
        tree.add_bold(sub=SUB, space=SPACE, task="conv", run="2", desc="clean")
        enc = _make_encoding(tree, ["mfcc"], tasks=["conv"])
        bold_metas = enc._discover_bold(SUB)
        assert list(bold_metas.keys()) == [BoldKey(ses=None, task="conv", run="2")]

    def test_multi_task_bold_distinct(self, tree: BIDSTree):
        tree.add_bold(sub=SUB, space=SPACE, task="rest", run="1", desc="clean")
        tree.add_bold(sub=SUB, space=SPACE, task="conv", run="1", desc="clean")
        enc = _make_encoding(tree, ["mfcc"], tasks=["rest", "conv"])
        bold_metas = enc._discover_bold(SUB)
        assert set(bold_metas.keys()) == {
            BoldKey(ses=None, task="rest", run="1"),
            BoldKey(ses=None, task="conv", run="1"),
        }

    def test_no_events_gives_no_segments(self, tree: BIDSTree):
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="clean")
        enc = _make_encoding(tree, ["mfcc"])
        bold_metas = enc._discover_bold(SUB)
        bold_meta = bold_metas[BoldKey(ses=None, task=TASK, run="1")]
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
        bold_meta = bold_metas[BoldKey(ses=None, task=TASK, run="1")]
        assert len(bold_meta.segments) == 2
        assert bold_meta.segments[0].entity == "block"
        assert bold_meta.segments[0].value == "1"
        assert bold_meta.segments[0].onset == 0.0
        assert bold_meta.segments[0].duration == 100.0
        assert bold_meta.segments[1].value == "2"
        assert bold_meta.segments[1].onset == 100.0
        assert bold_meta.segments[1].duration == 100.0

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
        bold_meta = bold_metas[BoldKey(ses=None, task=TASK, run="1")]
        assert len(bold_meta.segments) == 2
        assert bold_meta.segments[0].onset == 10.0
        assert bold_meta.segments[0].duration == 90.0

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
        bold_meta = bold_metas[BoldKey(ses=None, task=TASK, run="1")]
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
        bold_meta = bold_metas[BoldKey(ses=None, task=TASK, run="1")]
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
        bold_meta = bold_metas[BoldKey(ses=None, task=TASK, run="1")]
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
        with pytest.raises(ValueError, match="only ses, task, and run are valid"):
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
        assert BoldKey(ses="1", task=TASK, run="1") in bold_metas
        assert BoldKey(ses="1", task=TASK, run="2") in bold_metas
        assert BoldKey(ses="2", task=TASK, run="1") not in bold_metas

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


class TestTrainWiring:
    def test_train_fits_and_returns_artifact(
        self, tree: BIDSTree, monkeypatch: pytest.MonkeyPatch
    ):
        # Exercise the train() wiring (backend, float32 cast, cell_lengths,
        # fit, return) without _discover_*/BIDS plumbing. A hyphenated feature
        # name checks it survives ColumnKernelizer's transformer-name handling.
        n_rows, n_voxels = 20, 5
        rng = np.random.RandomState(0)
        data = TrainingData(
            X=rng.randn(n_rows, 7).astype(np.float64),
            Y=rng.randn(n_rows, n_voxels).astype(np.float64),
            row_slices={
                CellKey(task="a", run="1"): slice(0, 10),
                CellKey(task="a", run="2"): slice(10, 20),
            },
            col_slices={"phonemic-gpt3": slice(0, 3), "mfcc": slice(3, 7)},
        )
        enc = _make_encoding(tree, ["phonemic-gpt3", "mfcc"])
        for step in (
            "_discover_features",
            "_discover_bold",
            "_resolve_cell_keys",
            "_validate_coverage",
        ):
            monkeypatch.setattr(enc, step, lambda *a, **k: None)
        monkeypatch.setattr(enc, "_apply_filters", lambda *a, **k: (None, None))
        monkeypatch.setattr(enc, "_build_xy", lambda *a, **k: data)

        from sklearn.pipeline import Pipeline

        artifact = enc.train(SUB)
        assert isinstance(artifact, EncodingArtifact)
        assert len(artifact.models) == 1
        pipeline = artifact.models[0].pipeline
        assert isinstance(pipeline, Pipeline)
        pred = np.asarray(pipeline.predict(data.X.astype(np.float32)))
        assert pred.shape == (n_rows, n_voxels)
        # train records the cells it fit on and leaves universe unbound
        assert artifact.models[0].train_cells == set(data.row_slices)
        assert artifact.universe is None
        assert artifact.recipe.col_slices == data.col_slices


class TestArtifactRoundTrip:
    """Write → load reproduces the recipe, cell set, and predictions exactly."""

    def _trained(
        self, tree: BIDSTree, monkeypatch: pytest.MonkeyPatch, *, force: bool = False
    ) -> tuple[Encoding, TrainingData]:
        n_rows, n_voxels = 20, 5
        rng = np.random.RandomState(0)
        data = TrainingData(
            X=rng.randn(n_rows, 7).astype(np.float64),
            Y=rng.randn(n_rows, n_voxels).astype(np.float64),
            row_slices={
                CellKey(task="a", run="1"): slice(0, 10),
                CellKey(task="a", run="2"): slice(10, 20),
            },
            col_slices={"phonemic-gpt3": slice(0, 3), "mfcc": slice(3, 7)},
        )
        enc = _make_encoding(tree, ["phonemic-gpt3", "mfcc"], force=force)
        for step in (
            "_discover_features",
            "_discover_bold",
            "_resolve_cell_keys",
            "_validate_coverage",
        ):
            monkeypatch.setattr(enc, step, lambda *a, **k: None)
        monkeypatch.setattr(enc, "_apply_filters", lambda *a, **k: (None, None))
        monkeypatch.setattr(enc, "_build_xy", lambda *a, **k: data)
        return enc, data

    def test_round_trip(self, tree: BIDSTree, monkeypatch: pytest.MonkeyPatch):
        from himalaya.backend import set_backend

        enc, data = self._trained(tree, monkeypatch)
        X = data.X.astype(np.float32)

        artifact = enc.train(SUB)

        # Predictions compare numpy-vs-numpy: train already forced the in-memory
        # pipeline to the numpy backend during the write, so a plain predict here
        # is the numpy reference for the reloaded pipeline.
        set_backend("numpy")
        ref = np.asarray(artifact.models[0].pipeline.predict(X))

        loaded = Encoding.load(tree.root, sub=SUB, desc="v1")
        got = np.asarray(loaded.models[0].pipeline.predict(X))
        np.testing.assert_array_equal(got, ref)

        # Persisted weights are numpy, not torch — guards a regressed converter
        # (a no-op converter would still round-trip while torch is installed)
        model = loaded.models[0].pipeline.named_steps["model"]
        assert isinstance(model.dual_coef_, np.ndarray)

        assert loaded.recipe == artifact.recipe
        assert loaded.models[0].train_cells == artifact.models[0].train_cells
        assert loaded.universe is None

    def test_sidecar_mirrors_non_pipeline_fields(
        self, tree: BIDSTree, monkeypatch: pytest.MonkeyPatch
    ):
        import json

        enc, _ = self._trained(tree, monkeypatch)
        enc.train(SUB)

        out = enc._layout.path.result(sub=SUB, kind="encoding", desc="v1")
        sidecar = json.loads(out.path.with_suffix(".json").read_text())
        assert sidecar["recipe"]["tasks"] == [TASK]
        assert sidecar["recipe"]["col_slices"] == {
            "phonemic-gpt3": [0, 3],
            "mfcc": [3, 7],
        }
        assert sidecar["universe"] is None
        assert {frozenset(c.items()) for c in sidecar["models"][0]["train_cells"]} == {
            frozenset({("task", "a"), ("run", "1")}),
            frozenset({("task", "a"), ("run", "2")}),
        }

    def test_force_governs_overwrite(
        self, tree: BIDSTree, monkeypatch: pytest.MonkeyPatch
    ):
        import os

        enc, _ = self._trained(tree, monkeypatch)
        enc.train(SUB)
        out = enc._layout.path.result(sub=SUB, kind="encoding", desc="v1")

        # Backdate the blob so a rewrite is unambiguous (avoids ns-resolution ties)
        old = 1_000_000_000
        os.utime(out.path, ns=(old, old))

        # force=False skips: existing file is loaded, not rewritten
        enc_skip, _ = self._trained(tree, monkeypatch)
        enc_skip.train(SUB)
        assert out.path.stat().st_mtime_ns == old

        enc_force, _ = self._trained(tree, monkeypatch, force=True)
        enc_force.train(SUB)
        assert out.path.stat().st_mtime_ns != old
