import numpy as np
import polars as pl
import pyarrow.parquet as pq
import pytest

from hypline.encoding._context import _SCREEN_BAND, _build_pipeline
from hypline.encoding._schema import BoldKey, CellKey, RegressorKey, TrainingData

from ..conftest import BIDSTree
from .conftest import DYAD, SPACE, SUB, TASK, _make_encoding


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
        from sklearn.model_selection import KFold

        cell_lengths = [s.stop - s.start for s in data.row_slices.values()]
        pipeline = _build_pipeline(
            col_slices=data.col_slices,
            cell_lengths=cell_lengths,
            delays=[0, 1, 2],
            alphas=[1.0, 10.0, 100.0],
            cv=KFold(n_splits=2, shuffle=False),
        )
        pipeline.fit(data.X, data.Y)
        pred = np.asarray(pipeline.predict(data.X))
        assert pred.shape == (n_rows, n_voxels)

    def test_screen_band_skips_standard_scaler(self):
        # The screen boxcars must stay unscaled: standardizing maps their off-state
        # 0 to `-mean` and destroys the baseline the screens exist to carry. Assert on
        # structure (no fit) so deleting the bypass fails here, not just numerically.
        from sklearn.model_selection import KFold
        from sklearn.preprocessing import StandardScaler

        pipeline = _build_pipeline(
            col_slices={_SCREEN_BAND: slice(0, 2), "f1": slice(2, 5)},
            cell_lengths=[10, 10],
            delays=[0, 1],
            alphas=[1.0],
            cv=KFold(n_splits=2, shuffle=False),
        )
        bands = {
            name: sub
            for name, sub, _ in pipeline.named_steps["kernelizer"].transformers
        }
        screen_steps = [type(s) for _, s in bands[_SCREEN_BAND].steps]
        feature_steps = [type(s) for _, s in bands["f1"].steps]
        assert StandardScaler not in screen_steps
        assert StandardScaler in feature_steps


class TestDiscoverFeatures:
    def test_returns_expected_keys(self, tree: BIDSTree):
        tree.add_participants({SUB: DYAD})
        tree.add_feature(dyad=DYAD, task=TASK, kind="mfcc", run="1")
        enc = _make_encoding(tree, ["mfcc"])
        feature_paths = enc._discover_features(SUB)
        expected = RegressorKey(cell=CellKey(task=TASK, run="1"), name="mfcc")
        assert expected in feature_paths

    def test_no_files_raises(self, tree: BIDSTree):
        tree.add_participants({SUB: DYAD})
        enc = _make_encoding(tree, ["mfcc"])
        with pytest.raises(FileNotFoundError, match="features"):
            enc._discover_features(SUB)

    def test_duplicate_feature_file_raises(self, tree: BIDSTree):
        # Two filenames with identical BIDS entities (reordered) collide on RegressorKey
        tree.add_participants({SUB: DYAD})
        original = tree.add_feature(dyad=DYAD, task=TASK, kind="mfcc", run="1")
        dup = original.parent / f"dyad-{DYAD}_run-1_task-{TASK}_feat-mfcc.parquet"
        dup.write_bytes(original.read_bytes())
        enc = _make_encoding(tree, ["mfcc"])
        with pytest.raises(ValueError, match="Multiple feature files"):
            enc._discover_features(SUB)

    def test_missing_feature_at_one_cell_raises(self, tree: BIDSTree):
        tree.add_participants({SUB: DYAD})
        tree.add_feature(dyad=DYAD, task=TASK, kind="mfcc", run="1")
        tree.add_feature(dyad=DYAD, task=TASK, kind="mfcc", run="2")
        tree.add_feature(dyad=DYAD, task=TASK, kind="clip", run="1")
        enc = _make_encoding(tree, ["mfcc", "clip"])
        with pytest.raises(FileNotFoundError, match="Missing feat=clip"):
            enc._discover_features(SUB)

    def test_canonical_reads_bare_folder_not_variants(self, tree: BIDSTree):
        # The user-visible bug fix: variants on disk no longer collide
        tree.add_participants({SUB: DYAD})
        bare = tree.add_feature(dyad=DYAD, task=TASK, kind="phonemic", run="1")
        tree.add_feature(dyad=DYAD, task=TASK, kind="phonemic", run="1", desc="gpt3")
        enc = _make_encoding(tree, ["phonemic"])
        feature_paths = enc._discover_features(SUB)
        key = RegressorKey(cell=CellKey(task=TASK, run="1"), name="phonemic")
        assert feature_paths[key].path == bare

    def test_variant_reads_variant_folder_only(self, tree: BIDSTree):
        tree.add_participants({SUB: DYAD})
        tree.add_feature(dyad=DYAD, task=TASK, kind="phonemic", run="1")
        variant = tree.add_feature(
            dyad=DYAD, task=TASK, kind="phonemic", run="1", desc="gpt3"
        )
        enc = _make_encoding(tree, ["phonemic-gpt3"])
        feature_paths = enc._discover_features(SUB)
        key = RegressorKey(cell=CellKey(task=TASK, run="1"), name="phonemic-gpt3")
        assert feature_paths[key].path == variant

    def test_missing_variant_raises(self, tree: BIDSTree):
        tree.add_feature(dyad=DYAD, task=TASK, kind="phonemic", run="1")
        enc = _make_encoding(tree, ["phonemic-gpt3"])
        with pytest.raises(FileNotFoundError):
            enc._discover_features(SUB)

    def test_distinct_variants_form_separate_feature_groups(self, tree: BIDSTree):
        # Distinct kinds, each a variant — verbatim strings key the groups
        tree.add_participants({SUB: DYAD})
        tree.add_feature(dyad=DYAD, task=TASK, kind="phonemic", run="1", desc="gpt3")
        tree.add_feature(dyad=DYAD, task=TASK, kind="semantic", run="1", desc="bert")
        enc = _make_encoding(tree, ["phonemic-gpt3", "semantic-bert"])
        feature_paths = enc._discover_features(SUB)
        features = {fk.name for fk in feature_paths}
        assert features == {"phonemic-gpt3", "semantic-bert"}

    def test_unrequested_task_files_filtered_out(self, tree: BIDSTree):
        tree.add_participants({SUB: DYAD})
        tree.add_feature(dyad=DYAD, task="rest", kind="mfcc", run="1")
        tree.add_feature(dyad=DYAD, task="conv", kind="mfcc", run="2")
        enc = _make_encoding(tree, ["mfcc"], tasks=["conv"])
        feature_paths = enc._discover_features(SUB)
        cell_keys = {fk.cell for fk in feature_paths}
        assert cell_keys == {CellKey(task="conv", run="2")}

    def test_multi_task_cells_distinct(self, tree: BIDSTree):
        tree.add_participants({SUB: DYAD})
        tree.add_feature(dyad=DYAD, task="rest", kind="mfcc", run="1")
        tree.add_feature(dyad=DYAD, task="conv", kind="mfcc", run="1")
        enc = _make_encoding(tree, ["mfcc"], tasks=["rest", "conv"])
        feature_paths = enc._discover_features(SUB)
        cell_keys = {fk.cell for fk in feature_paths}
        assert cell_keys == {
            CellKey(task="rest", run="1"),
            CellKey(task="conv", run="1"),
        }

    def test_mixed_segmented_unsegmented_runs_raises(self, tree: BIDSTree):
        tree.add_participants({SUB: DYAD})
        tree.add_feature(
            dyad=DYAD, task=TASK, kind="mfcc", run="1", extra_entities={"block": "1"}
        )
        tree.add_feature(dyad=DYAD, task=TASK, kind="mfcc", run="2")
        enc = _make_encoding(tree, ["mfcc"])
        with pytest.raises(
            ValueError, match="Inconsistent schemas across regressor files"
        ):
            enc._discover_features(SUB)

    def test_schema_error_fires_before_coverage_error(self, tree: BIDSTree):
        # clip missing at run-2, but schema mismatch should raise first
        tree.add_participants({SUB: DYAD})
        tree.add_feature(
            dyad=DYAD, task=TASK, kind="mfcc", run="1", extra_entities={"block": "1"}
        )
        tree.add_feature(dyad=DYAD, task=TASK, kind="mfcc", run="2")
        tree.add_feature(
            dyad=DYAD, task=TASK, kind="clip", run="1", extra_entities={"block": "1"}
        )
        enc = _make_encoding(tree, ["mfcc", "clip"])
        with pytest.raises(
            ValueError, match="Inconsistent schemas across regressor files"
        ):
            enc._discover_features(SUB)

    def test_file_without_hypline_metadata_raises(self, tree: BIDSTree, tmp_path):
        tree.add_participants({SUB: DYAD})
        kind_dir = tree.features_dir / f"dyad-{DYAD}" / "mfcc"
        kind_dir.mkdir(parents=True)
        path = kind_dir / f"dyad-{DYAD}_task-{TASK}_run-1_feat-mfcc.parquet"
        df = pl.DataFrame({"start_time": [0.0], "feature": [[0.0]]})
        pq.write_table(df.to_arrow(), path)
        enc = _make_encoding(tree, ["mfcc"])
        with pytest.raises(ValueError, match="no hypline metadata"):
            enc._discover_features(SUB)

    def test_bids_filters_not_applied_at_discovery(self, tree: BIDSTree):
        tree.add_participants({SUB: DYAD})
        tree.add_feature(dyad=DYAD, task=TASK, kind="mfcc", run="1")
        tree.add_feature(dyad=DYAD, task=TASK, kind="mfcc", run="2")
        enc = _make_encoding(tree, ["mfcc"], bids_filters=["run-1"])
        feature_paths = enc._discover_features(SUB)
        assert len(feature_paths) == 2

    def test_inconsistent_metadata_across_files_raises(self, tree: BIDSTree):
        tree.add_participants({SUB: DYAD})
        tree.add_feature(
            dyad=DYAD, task=TASK, kind="mfcc", run="1", metadata={"model": "v1"}
        )
        tree.add_feature(
            dyad=DYAD, task=TASK, kind="mfcc", run="2", metadata={"model": "v2"}
        )
        enc = _make_encoding(tree, ["mfcc"])
        with pytest.raises(ValueError, match="Inconsistent metadata for feat=mfcc"):
            enc._discover_features(SUB)

    def test_inconsistent_metadata_diff_in_error_message(self, tree: BIDSTree):
        tree.add_participants({SUB: DYAD})
        tree.add_feature(
            dyad=DYAD, task=TASK, kind="mfcc", run="1", metadata={"model": "v1"}
        )
        tree.add_feature(
            dyad=DYAD, task=TASK, kind="mfcc", run="2", metadata={"model": "v2"}
        )
        enc = _make_encoding(tree, ["mfcc"])
        with pytest.raises(ValueError, match="model:"):
            enc._discover_features(SUB)

    def test_missing_key_shows_as_missing_in_diff(self, tree: BIDSTree):
        tree.add_participants({SUB: DYAD})
        tree.add_feature(
            dyad=DYAD, task=TASK, kind="mfcc", run="1", metadata={"model": "v1"}
        )
        tree.add_feature(
            dyad=DYAD,
            task=TASK,
            kind="mfcc",
            run="2",
            metadata={"model": "v1", "extra": "x"},
        )
        enc = _make_encoding(tree, ["mfcc"])
        with pytest.raises(ValueError, match="<missing>"):
            enc._discover_features(SUB)

    def test_third_file_diverges_from_first(self, tree: BIDSTree):
        tree.add_participants({SUB: DYAD})
        tree.add_feature(
            dyad=DYAD, task=TASK, kind="mfcc", run="1", metadata={"model": "v1"}
        )
        tree.add_feature(
            dyad=DYAD, task=TASK, kind="mfcc", run="2", metadata={"model": "v1"}
        )
        tree.add_feature(
            dyad=DYAD, task=TASK, kind="mfcc", run="3", metadata={"model": "v2"}
        )
        enc = _make_encoding(tree, ["mfcc"])
        with pytest.raises(ValueError, match="Inconsistent metadata for feat=mfcc"):
            enc._discover_features(SUB)

    def test_underscore_prefixed_keys_exempt_from_metadata_check(self, tree: BIDSTree):
        tree.add_participants({SUB: DYAD})
        tree.add_feature(
            dyad=DYAD,
            task=TASK,
            kind="mfcc",
            run="1",
            metadata={"model": "v1", "_run_id": "abc"},
        )
        tree.add_feature(
            dyad=DYAD,
            task=TASK,
            kind="mfcc",
            run="2",
            metadata={"model": "v1", "_run_id": "xyz"},
        )
        enc = _make_encoding(tree, ["mfcc"])
        enc._discover_features(SUB)

    def test_metadata_check_independent_across_features(self, tree: BIDSTree):
        tree.add_participants({SUB: DYAD})
        tree.add_feature(
            dyad=DYAD, task=TASK, kind="mfcc", run="1", metadata={"model": "v1"}
        )
        tree.add_feature(
            dyad=DYAD, task=TASK, kind="mfcc", run="2", metadata={"model": "v1"}
        )
        tree.add_feature(
            dyad=DYAD, task=TASK, kind="clip", run="1", metadata={"model": "v2"}
        )
        tree.add_feature(
            dyad=DYAD, task=TASK, kind="clip", run="2", metadata={"model": "v2"}
        )
        enc = _make_encoding(tree, ["mfcc", "clip"])
        enc._discover_features(SUB)


class TestDiscoverConfounds:
    def test_empty_when_unconfigured(self, tree: BIDSTree):
        tree.add_participants({SUB: DYAD})
        enc = _make_encoding(tree, ["mfcc"])
        assert enc._discover_confounds(SUB) == {}

    def test_returns_expected_keys(self, tree: BIDSTree):
        tree.add_participants({SUB: DYAD})
        tree.add_confound(dyad=DYAD, task=TASK, kind="phonemic", run="1")
        enc = _make_encoding(tree, ["mfcc"], confounds=["phonemic"])
        confound_paths = enc._discover_confounds(SUB)
        expected = RegressorKey(cell=CellKey(task=TASK, run="1"), name="phonemic")
        assert expected in confound_paths

    def test_variant_keyed_by_name(self, tree: BIDSTree):
        tree.add_participants({SUB: DYAD})
        variant = tree.add_confound(
            dyad=DYAD, task=TASK, kind="phonemic", run="1", desc="onset"
        )
        enc = _make_encoding(tree, ["mfcc"], confounds=["phonemic-onset"])
        confound_paths = enc._discover_confounds(SUB)
        key = RegressorKey(cell=CellKey(task=TASK, run="1"), name="phonemic-onset")
        assert confound_paths[key].path == variant

    def test_duplicate_confound_file_raises(self, tree: BIDSTree):
        tree.add_participants({SUB: DYAD})
        original = tree.add_confound(dyad=DYAD, task=TASK, kind="phonemic", run="1")
        dup = original.parent / f"dyad-{DYAD}_run-1_task-{TASK}_conf-phonemic.parquet"
        dup.write_bytes(original.read_bytes())
        enc = _make_encoding(tree, ["mfcc"], confounds=["phonemic"])
        with pytest.raises(ValueError, match="Multiple confound files"):
            enc._discover_confounds(SUB)

    def test_missing_confound_at_one_cell_raises(self, tree: BIDSTree):
        tree.add_participants({SUB: DYAD})
        tree.add_confound(dyad=DYAD, task=TASK, kind="phonemic", run="1")
        tree.add_confound(dyad=DYAD, task=TASK, kind="phonemic", run="2")
        tree.add_confound(dyad=DYAD, task=TASK, kind="motion", run="1")
        enc = _make_encoding(tree, ["mfcc"], confounds=["phonemic", "motion"])
        with pytest.raises(FileNotFoundError, match="Missing conf=motion"):
            enc._discover_confounds(SUB)

    def test_inconsistent_metadata_across_files_raises(self, tree: BIDSTree):
        tree.add_participants({SUB: DYAD})
        tree.add_confound(
            dyad=DYAD, task=TASK, kind="phonemic", run="1", metadata={"model": "v1"}
        )
        tree.add_confound(
            dyad=DYAD, task=TASK, kind="phonemic", run="2", metadata={"model": "v2"}
        )
        enc = _make_encoding(tree, ["mfcc"], confounds=["phonemic"])
        with pytest.raises(ValueError, match="Inconsistent metadata for conf=phonemic"):
            enc._discover_confounds(SUB)


class TestDiscoverBold:
    def test_returns_expected_keys(self, tree: BIDSTree):
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", desc="denoised")
        enc = _make_encoding(tree, ["mfcc"])
        bold_metas = enc._discover_bold(SUB)
        assert BoldKey(ses=None, task=TASK, run="1") in bold_metas

    def test_no_files_raises(self, tree: BIDSTree):
        enc = _make_encoding(tree, ["mfcc"])
        with pytest.raises(FileNotFoundError, match="hypline"):
            enc._discover_bold(SUB)

    def test_duplicate_bold_raises(self, tree: BIDSTree):
        # Two BOLDs sharing identity entities but distinguished by a variant
        # entity (`res`) collide on BoldKey.
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", desc="denoised")
        tree.add_bold(
            sub=SUB,
            task=TASK,
            space=SPACE,
            run="1",
            desc="denoised",
            extra_entities={"res": "2"},
        )
        enc = _make_encoding(tree, ["mfcc"])
        with pytest.raises(ValueError, match="Duplicate BOLD"):
            enc._discover_bold(SUB)

    def test_cross_session_runs_distinguished(self, tree: BIDSTree):
        tree.add_bold(
            sub=SUB, task=TASK, space=SPACE, ses="1", run="1", desc="denoised"
        )
        tree.add_bold(
            sub=SUB, task=TASK, space=SPACE, ses="2", run="1", desc="denoised"
        )
        enc = _make_encoding(tree, ["mfcc"])
        bold_metas = enc._discover_bold(SUB)
        assert BoldKey(ses="1", task=TASK, run="1") in bold_metas
        assert BoldKey(ses="2", task=TASK, run="1") in bold_metas

    def test_bold_siblings_excluded(self, tree: BIDSTree):
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", desc="denoised")
        tree.add_bold_siblings(sub=SUB, task=TASK, space=SPACE, run="1")
        enc = _make_encoding(tree, ["mfcc"])
        bold_metas = enc._discover_bold(SUB)
        assert list(bold_metas.keys()) == [BoldKey(ses=None, task=TASK, run="1")]

    def test_wrong_space_bold_excluded(self, tree: BIDSTree):
        tree.add_bold(
            sub=SUB, task=TASK, run="1", space="MNI152NLin6Asym", desc="denoised"
        )
        tree.add_bold(sub=SUB, task=TASK, run="1", space="T1w", desc="denoised")
        enc = _make_encoding(tree, ["mfcc"], bold_space="MNI152NLin6Asym")
        bold_metas = enc._discover_bold(SUB)
        assert list(bold_metas.keys()) == [BoldKey(ses=None, task=TASK, run="1")]

    def test_bold_desc_selects_matching_derivative(self, tree: BIDSTree):
        # bold_desc picks one derivative flavor; sibling flavors are not discovered
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="denoised")
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="2", tr=2.0, desc="smooth")
        enc = _make_encoding(tree, ["mfcc"], bold_desc="smooth")
        bold_metas = enc._discover_bold(SUB)
        assert set(bold_metas) == {BoldKey(ses=None, task=TASK, run="2")}

    def test_inconsistent_tr_raises(self, tree: BIDSTree):
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="denoised")
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="2", tr=1.5, desc="denoised")
        enc = _make_encoding(tree, ["mfcc"])
        with pytest.raises(ValueError, match="Inconsistent repetition times"):
            enc._discover_bold(SUB)

    def test_unrequested_task_bold_filtered_out(self, tree: BIDSTree):
        tree.add_bold(sub=SUB, space=SPACE, task="rest", run="1", desc="denoised")
        tree.add_bold(sub=SUB, space=SPACE, task="conv", run="2", desc="denoised")
        enc = _make_encoding(tree, ["mfcc"], tasks=["conv"])
        bold_metas = enc._discover_bold(SUB)
        assert list(bold_metas.keys()) == [BoldKey(ses=None, task="conv", run="2")]

    def test_multi_task_bold_distinct(self, tree: BIDSTree):
        tree.add_bold(sub=SUB, space=SPACE, task="rest", run="1", desc="denoised")
        tree.add_bold(sub=SUB, space=SPACE, task="conv", run="1", desc="denoised")
        enc = _make_encoding(tree, ["mfcc"], tasks=["rest", "conv"])
        bold_metas = enc._discover_bold(SUB)
        assert set(bold_metas.keys()) == {
            BoldKey(ses=None, task="rest", run="1"),
            BoldKey(ses=None, task="conv", run="1"),
        }

    def test_no_events_gives_no_segments(self, tree: BIDSTree):
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="denoised")
        enc = _make_encoding(tree, ["mfcc"])
        bold_metas = enc._discover_bold(SUB)
        bold_meta = bold_metas[BoldKey(ses=None, task=TASK, run="1")]
        assert bold_meta.segments == []

    def test_structural_entity_slices_parsed(self, tree: BIDSTree):
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="denoised")
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
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="denoised")
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
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="denoised")
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
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="denoised")
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
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="denoised")
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
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="denoised")
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
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="denoised")
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
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="denoised")
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
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="denoised")
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
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="denoised")
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="2", tr=2.0, desc="denoised")
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
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="denoised")
        tree.add_bold(
            sub=SUB, task=TASK, space=SPACE, run="2", tr=2.0, desc="denoised"
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
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="denoised")
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="2", tr=2.0, desc="denoised")

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
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="denoised")
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="2", tr=2.0, desc="denoised")
        enc = _make_encoding(tree, ["mfcc"])
        bold_metas = enc._discover_bold(SUB)
        assert all(bold_meta.segments == [] for bold_meta in bold_metas.values())

    def test_bids_filters_not_applied_at_discovery(self, tree: BIDSTree):
        tree.add_participants({SUB: DYAD})
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="denoised")
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="2", tr=2.0, desc="denoised")
        enc = _make_encoding(tree, ["mfcc"], bids_filters=["run-1"])
        bold_metas = enc._discover_bold(SUB)
        assert len(bold_metas) == 2


class TestResolveCellKeys:
    # Orphan check (regressor cell has no matching BOLD run)

    def test_features_without_bold_raises(self, tree: BIDSTree):
        tree.add_participants({SUB: DYAD})
        tree.add_feature(dyad=DYAD, task=TASK, kind="mfcc", run="1")
        tree.add_feature(dyad=DYAD, task=TASK, kind="mfcc", run="2")
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", desc="denoised")
        enc = _make_encoding(tree, ["mfcc"])
        feature_paths = enc._discover_features(SUB)
        bold_metas = enc._discover_bold(SUB)
        with pytest.raises(FileNotFoundError, match="No BOLD file found for regressor"):
            enc._resolve_cell_keys(SUB, feature_paths, bold_metas)

    def test_multiple_features_without_bold_reports_count(self, tree: BIDSTree):
        tree.add_participants({SUB: DYAD})
        for run in ("1", "2", "3"):
            tree.add_feature(dyad=DYAD, task=TASK, kind="mfcc", run=run)
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", desc="denoised")
        enc = _make_encoding(tree, ["mfcc"])
        feature_paths = enc._discover_features(SUB)
        bold_metas = enc._discover_bold(SUB)
        with pytest.raises(FileNotFoundError, match="other coverage gaps"):
            enc._resolve_cell_keys(SUB, feature_paths, bold_metas)

    # Unsegmented run cases

    def test_unsegmented_run_single_cell_passes(self, tree: BIDSTree):
        tree.add_participants({SUB: DYAD})
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="denoised")
        tree.add_feature(dyad=DYAD, task=TASK, kind="mfcc", run="1")
        enc = _make_encoding(tree, ["mfcc"])
        feature_paths = enc._discover_features(SUB)
        bold_metas = enc._discover_bold(SUB)
        feature_paths = enc._resolve_cell_keys(SUB, feature_paths, bold_metas)
        assert len(feature_paths) == 1

    def test_unsegmented_run_multiple_cells_raises(self, tree: BIDSTree):
        tree.add_participants({SUB: DYAD})
        tree.add_bold(
            sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="denoised"
        )  # no events → unsegmented
        tree.add_feature(
            dyad=DYAD, task=TASK, kind="mfcc", run="1", extra_entities={"trial": "1"}
        )
        tree.add_feature(
            dyad=DYAD, task=TASK, kind="mfcc", run="1", extra_entities={"trial": "2"}
        )
        enc = _make_encoding(tree, ["mfcc"])
        feature_paths = enc._discover_features(SUB)
        bold_metas = enc._discover_bold(SUB)
        with pytest.raises(ValueError, match="unsegmented but has 2 regressor files"):
            enc._resolve_cell_keys(SUB, feature_paths, bold_metas)

    def test_extra_entity_on_unsegmented_run_raises(self, tree: BIDSTree):
        tree.add_participants({SUB: DYAD})
        tree.add_bold(
            sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="denoised"
        )  # no events → unsegmented
        tree.add_feature(
            dyad=DYAD, task=TASK, kind="mfcc", run="1", extra_entities={"trial": "1"}
        )
        enc = _make_encoding(tree, ["mfcc"])
        feature_paths = enc._discover_features(SUB)
        bold_metas = enc._discover_bold(SUB)
        with pytest.raises(ValueError, match="only ses, task, and run are valid"):
            enc._resolve_cell_keys(SUB, feature_paths, bold_metas)

    # Segmented run cases

    def test_valid_segmented_cells_pass(self, tree: BIDSTree):
        tree.add_participants({SUB: DYAD})
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="denoised")
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
            dyad=DYAD, task=TASK, kind="mfcc", run="1", extra_entities={"block": "1"}
        )
        tree.add_feature(
            dyad=DYAD, task=TASK, kind="mfcc", run="1", extra_entities={"block": "2"}
        )
        enc = _make_encoding(tree, ["mfcc"])
        feature_paths = enc._discover_features(SUB)
        bold_metas = enc._discover_bold(SUB)
        feature_paths = enc._resolve_cell_keys(SUB, feature_paths, bold_metas)
        assert len(feature_paths) == 2

    def test_cell_missing_segment_entity_raises(self, tree: BIDSTree):
        tree.add_participants({SUB: DYAD})
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="denoised")
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
            dyad=DYAD, task=TASK, kind="mfcc", run="1"
        )  # no block entity on the filename
        enc = _make_encoding(tree, ["mfcc"])
        feature_paths = enc._discover_features(SUB)
        bold_metas = enc._discover_bold(SUB)
        with pytest.raises(ValueError, match="missing segment entity"):
            enc._resolve_cell_keys(SUB, feature_paths, bold_metas)

    def test_cell_value_not_in_events_raises(self, tree: BIDSTree):
        tree.add_participants({SUB: DYAD})
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="denoised")
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
            dyad=DYAD, task=TASK, kind="mfcc", run="1", extra_entities={"block": "3"}
        )  # block-3 not in events
        enc = _make_encoding(tree, ["mfcc"])
        feature_paths = enc._discover_features(SUB)
        bold_metas = enc._discover_bold(SUB)
        with pytest.raises(ValueError, match="not found in events"):
            enc._resolve_cell_keys(SUB, feature_paths, bold_metas)

    def test_extra_entity_on_segmented_run_raises(self, tree: BIDSTree):
        tree.add_participants({SUB: DYAD})
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="denoised")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=[
                {"trial_type": "block-1", "onset": 0.0, "duration": 100.0},
            ],
        )
        tree.add_feature(
            dyad=DYAD,
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
        tree.add_participants({SUB: DYAD})
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="denoised")
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
            dyad=DYAD, task=TASK, kind="mfcc", run="1", extra_entities={"block": "1"}
        )
        tree.add_feature(
            dyad=DYAD, task=TASK, kind="mfcc", run="1", extra_entities={"block": "2"}
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
        tree.add_participants({SUB: DYAD})
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="denoised")
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
            dyad=DYAD,
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
        tree.add_participants({SUB: DYAD})
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="denoised")
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
            dyad=DYAD,
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
