from pathlib import Path

import pytest

from hypline.encoding import BoldKey, CellKey, Encoding, EncodingConfig, FeatureKey

_DEFAULT_SUB = "001"
_DEFAULT_TASK = "conv"
_DEFAULT_SPACE = "MNI152NLin6Asym"


class BIDSTree:
    """Minimal on-disk fixture for an Encoding test session.

    Files are empty (touch-only); naming follows fMRIPrep BIDS-derivatives
    convention. All three dirs are pre-created under a single tmp_path root.
    """

    def __init__(self, root: Path):
        self.features_dir = root / "features"
        self.bold_dir = root / "bold"
        self.output_dir = root / "output"
        for dir in (self.features_dir, self.bold_dir, self.output_dir):
            dir.mkdir(parents=True)

    def _stem(self, entities: dict[str, str]) -> str:
        return "_".join(f"{k}-{v}" for k, v in entities.items())

    def _write(
        self,
        directory: Path,
        entities: dict[str, str],
        *,
        suffix: str | None = None,
        ext: str,
        sidecar_json: str | None = None,
    ) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        stem = self._stem(entities)
        name = stem if suffix is None else f"{stem}_{suffix}"
        path = directory / f"{name}{ext}"
        path.touch()
        if sidecar_json is not None:
            (directory / f"{name}.json").write_text(sidecar_json)
        return path

    def _identity_entities(
        self,
        sub: str,
        ses: str | None,
        task: str | None,
        run: str | None,
    ) -> dict[str, str]:
        entities: dict[str, str] = {"sub": sub}
        if ses is not None:
            entities["ses"] = ses
        if task is not None:
            entities["task"] = task
        if run is not None:
            entities["run"] = run
        return entities

    def _bold_entities(
        self,
        sub: str,
        ses: str | None,
        task: str | None,
        run: str | None,
        space: str,
    ) -> dict[str, str]:
        return {
            **self._identity_entities(sub, ses, task, run),
            "space": space,
            "desc": "preproc",
        }

    def add_feature(
        self,
        feature: str,
        *,
        sub: str = _DEFAULT_SUB,
        ses: str | None = None,
        task: str | None = _DEFAULT_TASK,
        run: str | None = None,
        subdir: str | None = None,
        **extra_entities: str,
    ) -> Path:
        entities = self._identity_entities(sub, ses, task, run)
        entities.update(extra_entities)
        entities["feature"] = feature
        directory = self.features_dir if subdir is None else self.features_dir / subdir
        return self._write(directory, entities, ext=".parquet")

    def add_bold(
        self,
        *,
        sub: str = _DEFAULT_SUB,
        ses: str | None = None,
        task: str | None = _DEFAULT_TASK,
        run: str | None = None,
        tr: float = 2.0,
        space: str = _DEFAULT_SPACE,
        subdir: str | None = None,
    ) -> Path:
        entities = self._bold_entities(sub, ses, task, run, space)
        directory = self.bold_dir if subdir is None else self.bold_dir / subdir
        return self._write(
            directory,
            entities,
            suffix="bold",
            ext=".nii.gz",
            sidecar_json=f'{{"RepetitionTime": {tr}}}',
        )

    def add_events(
        self,
        *,
        sub: str = _DEFAULT_SUB,
        ses: str | None = None,
        task: str | None = _DEFAULT_TASK,
        run: str | None = None,
        rows: list[dict[str, str | float]] | None = None,
    ) -> Path:
        """Write an events.tsv file.

        rows is a list of dicts with trial_type, onset, duration, e.g.:
            [{"trial_type": "block-1", "onset": 0.0, "duration": 100.0}]
        """
        identity = self._identity_entities(sub, ses, task, run)
        tsv_rows = []
        if rows:
            for row in rows:
                tsv_rows.append(
                    f"{row['trial_type']}\t{row['onset']}\t{row['duration']}"
                )
        events_path = self.bold_dir / f"{self._stem(identity)}_events.tsv"
        events_path.write_text("trial_type\tonset\tduration\n" + "\n".join(tsv_rows))
        return events_path

    def add_boldref(
        self,
        *,
        sub: str = _DEFAULT_SUB,
        ses: str | None = None,
        task: str | None = _DEFAULT_TASK,
        run: str | None = None,
        space: str = _DEFAULT_SPACE,
    ) -> Path:
        return self._write(
            self.bold_dir,
            self._bold_entities(sub, ses, task, run, space),
            suffix="boldref",
            ext=".nii.gz",
        )

    def add_brain_mask(
        self,
        *,
        sub: str = _DEFAULT_SUB,
        ses: str | None = None,
        task: str | None = _DEFAULT_TASK,
        run: str | None = None,
        space: str = _DEFAULT_SPACE,
    ) -> Path:
        entities = {**self._bold_entities(sub, ses, task, run, space), "desc": "brain"}
        return self._write(
            self.bold_dir, entities, suffix="mask", ext=".nii.gz", sidecar_json="{}"
        )

    def add_dseg(
        self,
        *,
        sub: str = _DEFAULT_SUB,
        ses: str | None = None,
        task: str | None = _DEFAULT_TASK,
        run: str | None = None,
        space: str = _DEFAULT_SPACE,
        label: str = "aparcaseg",
    ) -> Path:
        entities = {**self._bold_entities(sub, ses, task, run, space), "desc": label}
        return self._write(self.bold_dir, entities, suffix="dseg", ext=".nii.gz")

    def add_confounds(
        self,
        *,
        sub: str = _DEFAULT_SUB,
        ses: str | None = None,
        task: str | None = _DEFAULT_TASK,
        run: str | None = None,
    ) -> Path:
        entities = {**self._identity_entities(sub, ses, task, run), "desc": "confounds"}
        return self._write(
            self.bold_dir, entities, suffix="timeseries", ext=".tsv", sidecar_json="{}"
        )

    def add_xfm(
        self,
        *,
        sub: str = _DEFAULT_SUB,
        ses: str | None = None,
        task: str | None = _DEFAULT_TASK,
        run: str | None = None,
    ) -> Path:
        entities = {
            **self._identity_entities(sub, ses, task, run),
            "from": "T1w",
            "to": "scanner",
            "mode": "image",
        }
        return self._write(self.bold_dir, entities, suffix="xfm", ext=".txt")

    def add_bold_siblings(
        self,
        *,
        sub: str = _DEFAULT_SUB,
        ses: str | None = None,
        task: str | None = _DEFAULT_TASK,
        run: str | None = None,
        space: str = _DEFAULT_SPACE,
    ) -> None:
        """Drop the full set of realistic fMRIPrep sibling files for a BOLD run."""
        self.add_boldref(sub=sub, ses=ses, task=task, run=run, space=space)
        self.add_brain_mask(sub=sub, ses=ses, task=task, run=run, space=space)
        self.add_dseg(
            sub=sub, ses=ses, task=task, run=run, space=space, label="aparcaseg"
        )
        self.add_dseg(sub=sub, ses=ses, task=task, run=run, space=space, label="aseg")
        self.add_confounds(sub=sub, ses=ses, task=task, run=run)
        self.add_xfm(sub=sub, ses=ses, task=task, run=run)


@pytest.fixture()
def tree(tmp_path: Path) -> BIDSTree:
    return BIDSTree(tmp_path)


def make_encoding(
    tree: BIDSTree,
    features: list[str],
    *,
    bold_space: str = _DEFAULT_SPACE,
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
        feature_paths = enc._discover_features(_DEFAULT_SUB)
        expected = FeatureKey(cell=CellKey(run="1"), feature="mfcc")
        assert expected in feature_paths

    def test_no_files_raises(self, tree: BIDSTree):
        enc = make_encoding(tree, ["mfcc"])
        with pytest.raises(FileNotFoundError, match="No matching feature files"):
            enc._discover_features(_DEFAULT_SUB)

    def test_duplicate_feature_file_raises(self, tree: BIDSTree):
        tree.add_feature("mfcc", run="1")
        tree.add_feature("mfcc", run="1", subdir="sub")
        enc = make_encoding(tree, ["mfcc"])
        with pytest.raises(ValueError, match="Multiple feature files"):
            enc._discover_features(_DEFAULT_SUB)

    def test_missing_feature_at_one_cell_raises(self, tree: BIDSTree):
        tree.add_feature("mfcc", run="1")
        tree.add_feature("mfcc", run="2")
        tree.add_feature("clip", run="1")
        enc = make_encoding(tree, ["mfcc", "clip"])
        with pytest.raises(FileNotFoundError, match="Missing feature=clip"):
            enc._discover_features(_DEFAULT_SUB)

    def test_semantic_entity_filter_narrows_results(self, tree: BIDSTree):
        tree.add_feature("mfcc", block="a")
        tree.add_feature("mfcc", block="b")
        enc = make_encoding(tree, ["mfcc"], bids_filters=["block-a"])
        feature_paths = enc._discover_features(_DEFAULT_SUB)
        assert {k.cell.get("block") for k in feature_paths} == {"a"}

    def test_task_invariance_violation_raises(self, tree: BIDSTree):
        tree.add_feature("mfcc", task="rest", run="1")
        tree.add_feature("mfcc", task="conv", run="2")
        enc = make_encoding(tree, ["mfcc"])
        with pytest.raises(ValueError, match="task"):
            enc._discover_features(_DEFAULT_SUB)

    def test_acquisition_entities_not_required_on_features(self, tree: BIDSTree):
        tree.add_feature("mfcc", run="1")
        enc = make_encoding(tree, ["mfcc"], bids_filters=["desc-preproc"])
        feature_paths = enc._discover_features(_DEFAULT_SUB)
        expected = FeatureKey(cell=CellKey(run="1"), feature="mfcc")
        assert expected in feature_paths

    def test_mixed_partitioned_unpartitioned_runs_raise(self, tree: BIDSTree):
        tree.add_feature("mfcc", run="1", block="1")
        tree.add_feature("mfcc", run="2")
        enc = make_encoding(tree, ["mfcc"])
        with pytest.raises(ValueError, match="Inconsistent feature file schemas"):
            enc._discover_features(_DEFAULT_SUB)

    def test_typo_filter_entity_raises(self, tree: BIDSTree):
        tree.add_feature("mfcc", run="1")
        enc = make_encoding(tree, ["mfcc"], bids_filters=["typo-foo"])
        with pytest.raises(ValueError, match="typo"):
            enc._discover_features(_DEFAULT_SUB)

    def test_absent_common_entity_filter_raises(self, tree: BIDSTree):
        # ses not on any file — filter can't match, entity check fires
        tree.add_feature("mfcc", run="1")
        enc = make_encoding(tree, ["mfcc"], bids_filters=["ses-99"])
        with pytest.raises(ValueError, match="ses"):
            enc._discover_features(_DEFAULT_SUB)

    def test_valid_entity_wrong_value_gives_file_not_found(self, tree: BIDSTree):
        # block is on files — filter entity is valid, just no match for block-99
        tree.add_feature("mfcc", block="1")
        tree.add_feature("mfcc", block="2")
        enc = make_encoding(tree, ["mfcc"], bids_filters=["block-99"])
        with pytest.raises(FileNotFoundError):
            enc._discover_features(_DEFAULT_SUB)

    def test_schema_error_fires_before_coverage_error(self, tree: BIDSTree):
        # clip missing at run-2, but schema mismatch should raise first
        tree.add_feature("mfcc", run="1", block="1")
        tree.add_feature("mfcc", run="2")
        tree.add_feature("clip", run="1", block="1")
        enc = make_encoding(tree, ["mfcc", "clip"])
        with pytest.raises(ValueError, match="Inconsistent feature file schemas"):
            enc._discover_features(_DEFAULT_SUB)


class TestDiscoverBold:
    def test_returns_expected_keys(self, tree: BIDSTree):
        tree.add_bold(run="1")
        enc = make_encoding(tree, ["mfcc"])
        bold_metas = enc._discover_bold(_DEFAULT_SUB)
        assert BoldKey(ses=None, run="1") in bold_metas

    def test_no_files_raises(self, tree: BIDSTree):
        enc = make_encoding(tree, ["mfcc"])
        with pytest.raises(FileNotFoundError, match="No BOLD files"):
            enc._discover_bold(_DEFAULT_SUB)

    def test_duplicate_bold_raises(self, tree: BIDSTree):
        tree.add_bold(run="1")
        tree.add_bold(run="1", subdir="sub")
        enc = make_encoding(tree, ["mfcc"])
        with pytest.raises(ValueError, match="Duplicate BOLD"):
            enc._discover_bold(_DEFAULT_SUB)

    def test_cross_session_runs_distinguished(self, tree: BIDSTree):
        tree.add_bold(ses="1", run="1")
        tree.add_bold(ses="2", run="1")
        enc = make_encoding(tree, ["mfcc"])
        bold_metas = enc._discover_bold(_DEFAULT_SUB)
        assert BoldKey(ses="1", run="1") in bold_metas
        assert BoldKey(ses="2", run="1") in bold_metas

    def test_bold_siblings_excluded(self, tree: BIDSTree):
        tree.add_bold(run="1")
        tree.add_bold_siblings(run="1")
        enc = make_encoding(tree, ["mfcc"])
        bold_metas = enc._discover_bold(_DEFAULT_SUB)
        assert list(bold_metas.keys()) == [BoldKey(ses=None, run="1")]

    def test_wrong_space_bold_excluded(self, tree: BIDSTree):
        tree.add_bold(run="1", space="MNI152NLin6Asym")
        tree.add_bold(run="1", space="T1w")
        enc = make_encoding(tree, ["mfcc"], bold_space="MNI152NLin6Asym")
        bold_metas = enc._discover_bold(_DEFAULT_SUB)
        assert list(bold_metas.keys()) == [BoldKey(ses=None, run="1")]

    def test_inconsistent_tr_raises(self, tree: BIDSTree):
        tree.add_bold(run="1", tr=2.0)
        tree.add_bold(run="2", tr=1.5)
        enc = make_encoding(tree, ["mfcc"])
        with pytest.raises(ValueError, match="Inconsistent repetition times"):
            enc._discover_bold(_DEFAULT_SUB)

    def test_task_invariance_violation_raises(self, tree: BIDSTree):
        tree.add_bold(task="rest", run="1")
        tree.add_bold(task="conv", run="2")
        enc = make_encoding(tree, ["mfcc"])
        with pytest.raises(ValueError, match="task"):
            enc._discover_bold(_DEFAULT_SUB)

    def test_acq_invariance_violation_raises(self, tree: BIDSTree):
        for acq, run in (("hi", "1"), ("lo", "2")):
            stem = (
                f"sub-{_DEFAULT_SUB}_task-{_DEFAULT_TASK}_acq-{acq}_run-{run}_"
                f"space-{_DEFAULT_SPACE}_desc-preproc_bold"
            )
            (tree.bold_dir / f"{stem}.nii.gz").touch()
            (tree.bold_dir / f"{stem}.json").write_text('{"RepetitionTime": 2.0}')
        enc = make_encoding(tree, ["mfcc"])
        with pytest.raises(ValueError, match="acq"):
            enc._discover_bold(_DEFAULT_SUB)

    def test_no_events_gives_no_partitioning(self, tree: BIDSTree):
        tree.add_bold(run="1", tr=2.0)
        enc = make_encoding(tree, ["mfcc"])
        bold_metas = enc._discover_bold(_DEFAULT_SUB)
        meta = bold_metas[BoldKey(ses=None, run="1")]
        assert meta.partitioning is None

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
        bold_metas = enc._discover_bold(_DEFAULT_SUB)
        meta = bold_metas[BoldKey(ses=None, run="1")]
        assert meta.partitioning is not None
        assert meta.partitioning.entity == "block"
        assert meta.partitioning.slices["1"] == slice(0, 50)
        assert meta.partitioning.slices["2"] == slice(50, 100)

    def test_leaf_entity_is_finest_granularity(self, tree: BIDSTree):
        tree.add_bold(run="1", tr=2.0)
        tree.add_events(
            run="1",
            rows=[
                {"trial_type": "block-1", "onset": 0.0, "duration": 200.0},
                {"trial_type": "trial-1", "onset": 0.0, "duration": 100.0},
                {"trial_type": "trial-2", "onset": 100.0, "duration": 100.0},
            ],
        )
        enc = make_encoding(tree, ["mfcc"])
        bold_metas = enc._discover_bold(_DEFAULT_SUB)
        meta = bold_metas[BoldKey(ses=None, run="1")]
        assert meta.partitioning is not None
        assert meta.partitioning.entity == "trial"

    def test_tied_granularity_raises(self, tree: BIDSTree):
        tree.add_bold(run="1", tr=2.0)
        tree.add_events(
            run="1",
            rows=[
                {"trial_type": "block-1", "onset": 0.0, "duration": 100.0},
                {"trial_type": "block-2", "onset": 100.0, "duration": 100.0},
                {"trial_type": "segment-a", "onset": 0.0, "duration": 100.0},
                {"trial_type": "segment-b", "onset": 100.0, "duration": 100.0},
            ],
        )
        enc = make_encoding(tree, ["mfcc"])
        with pytest.raises(ValueError, match="identical granularity"):
            enc._discover_bold(_DEFAULT_SUB)

    def test_hyphen_free_trial_type_ignored(self, tree: BIDSTree):
        tree.add_bold(run="1", tr=2.0)
        events_path = (
            tree.bold_dir / f"sub-{_DEFAULT_SUB}_task-{_DEFAULT_TASK}_run-1_events.tsv"
        )
        events_path.write_text("trial_type\tonset\tduration\nrest\t0.0\t100.0\n")
        enc = make_encoding(tree, ["mfcc"])
        bold_metas = enc._discover_bold(_DEFAULT_SUB)
        meta = bold_metas[BoldKey(ses=None, run="1")]
        assert meta.partitioning is None

    def test_non_tiling_kv_entity_raises(self, tree: BIDSTree):
        tree.add_bold(run="1", tr=2.0)
        tree.add_events(
            run="1",
            rows=[
                {"trial_type": "block-1", "onset": 0.0, "duration": 100.0},
                {"trial_type": "block-2", "onset": 120.0, "duration": 80.0},
            ],
        )
        enc = make_encoding(tree, ["mfcc"])
        with pytest.raises(ValueError, match="do not partition the run"):
            enc._discover_bold(_DEFAULT_SUB)

    def test_kv_entity_not_starting_at_zero_raises(self, tree: BIDSTree):
        tree.add_bold(run="1", tr=2.0)
        tree.add_events(
            run="1",
            rows=[{"trial_type": "block-1", "onset": 10.0, "duration": 90.0}],
        )
        enc = make_encoding(tree, ["mfcc"])
        with pytest.raises(ValueError, match="do not partition the run"):
            enc._discover_bold(_DEFAULT_SUB)

    def test_partial_tiling_raises(self, tree: BIDSTree):
        tree.add_bold(run="1", tr=2.0)
        tree.add_events(
            run="1",
            rows=[
                {"trial_type": "block-1", "onset": 0.0, "duration": 100.0},
                {"trial_type": "block-2", "onset": 100.0, "duration": 100.0},
                {"trial_type": "trial-1", "onset": 0.0, "duration": 100.0},
            ],
        )
        enc = make_encoding(tree, ["mfcc"])
        with pytest.raises(ValueError, match="do not partition the run"):
            enc._discover_bold(_DEFAULT_SUB)

    def test_tiling_with_flat_labels_passes(self, tree: BIDSTree):
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
        bold_metas = enc._discover_bold(_DEFAULT_SUB)
        meta = bold_metas[BoldKey(ses=None, run="1")]
        assert meta.partitioning is not None
        assert meta.partitioning.entity == "block"

    def test_runs_disagree_on_partition_entity_raises(self, tree: BIDSTree):
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
        with pytest.raises(ValueError, match="disagree on partition entity"):
            enc._discover_bold(_DEFAULT_SUB)

    def test_mixed_partitioned_and_unpartitioned_raises(self, tree: BIDSTree):
        tree.add_bold(run="1", tr=2.0)
        tree.add_bold(run="2", tr=2.0)  # no events → unpartitioned
        tree.add_events(
            run="1",
            rows=[
                {"trial_type": "block-1", "onset": 0.0, "duration": 100.0},
                {"trial_type": "block-2", "onset": 100.0, "duration": 100.0},
            ],
        )
        enc = make_encoding(tree, ["mfcc"])
        with pytest.raises(ValueError, match="disagree on partition entity"):
            enc._discover_bold(_DEFAULT_SUB)

    def test_all_unpartitioned_passes(self, tree: BIDSTree):
        tree.add_bold(run="1", tr=2.0)
        tree.add_bold(run="2", tr=2.0)
        enc = make_encoding(tree, ["mfcc"])
        bold_metas = enc._discover_bold(_DEFAULT_SUB)
        assert all(meta.partitioning is None for meta in bold_metas.values())


class TestValidateAlignment:
    def test_valid_alignment_passes(self, tree: BIDSTree):
        tree.add_feature("mfcc", run="1")
        tree.add_bold(run="1")
        enc = make_encoding(tree, ["mfcc"])
        feature_paths = enc._discover_features(_DEFAULT_SUB)
        bold_metas = enc._discover_bold(_DEFAULT_SUB)
        enc._validate_alignment(feature_paths, bold_metas)

    def test_cross_file_task_invariance_raises(self, tree: BIDSTree):
        tree.add_feature("mfcc", task="rest", run="1")
        tree.add_bold(task="conv", run="1")
        enc = make_encoding(tree, ["mfcc"])
        feature_paths = enc._discover_features(_DEFAULT_SUB)
        bold_metas = enc._discover_bold(_DEFAULT_SUB)
        with pytest.raises(ValueError, match="task"):
            enc._validate_alignment(feature_paths, bold_metas)

    def test_bold_without_features_raises(self, tree: BIDSTree):
        tree.add_feature("mfcc", run="1")
        tree.add_bold(run="1")
        tree.add_bold(run="2")
        enc = make_encoding(tree, ["mfcc"])
        feature_paths = enc._discover_features(_DEFAULT_SUB)
        bold_metas = enc._discover_bold(_DEFAULT_SUB)
        with pytest.raises(FileNotFoundError, match="No feature files found for BOLD"):
            enc._validate_alignment(feature_paths, bold_metas)

    def test_features_without_bold_raises(self, tree: BIDSTree):
        tree.add_feature("mfcc", run="1")
        tree.add_feature("mfcc", run="2")
        tree.add_bold(run="1")
        enc = make_encoding(tree, ["mfcc"])
        feature_paths = enc._discover_features(_DEFAULT_SUB)
        bold_metas = enc._discover_bold(_DEFAULT_SUB)
        with pytest.raises(FileNotFoundError, match="No BOLD file found for features"):
            enc._validate_alignment(feature_paths, bold_metas)

    def test_multiple_bold_gaps_reports_count(self, tree: BIDSTree):
        for run in ("1", "2", "3"):
            tree.add_feature("mfcc", run=run)
            tree.add_bold(run=run)
        tree.add_bold(run="4")
        tree.add_bold(run="5")
        enc = make_encoding(tree, ["mfcc"])
        feature_paths = enc._discover_features(_DEFAULT_SUB)
        bold_metas = enc._discover_bold(_DEFAULT_SUB)
        with pytest.raises(FileNotFoundError, match="other coverage gaps"):
            enc._validate_alignment(feature_paths, bold_metas)

    def test_multiple_feature_gaps_reports_count(self, tree: BIDSTree):
        for run in ("1", "2", "3"):
            tree.add_feature("mfcc", run=run)
            tree.add_bold(run=run)
        tree.add_feature("mfcc", run="4")
        tree.add_feature("mfcc", run="5")
        enc = make_encoding(tree, ["mfcc"])
        feature_paths = enc._discover_features(_DEFAULT_SUB)
        bold_metas = enc._discover_bold(_DEFAULT_SUB)
        with pytest.raises(FileNotFoundError, match="other coverage gaps"):
            enc._validate_alignment(feature_paths, bold_metas)

    def test_partitioned_valid_cells_pass(self, tree: BIDSTree):
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
        feature_paths = enc._discover_features(_DEFAULT_SUB)
        bold_metas = enc._discover_bold(_DEFAULT_SUB)
        enc._validate_alignment(feature_paths, bold_metas)

    def test_cell_missing_partition_entity_raises(self, tree: BIDSTree):
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
        feature_paths = enc._discover_features(_DEFAULT_SUB)
        bold_metas = enc._discover_bold(_DEFAULT_SUB)
        with pytest.raises(ValueError, match="missing partition entity"):
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
        feature_paths = enc._discover_features(_DEFAULT_SUB)
        bold_metas = enc._discover_bold(_DEFAULT_SUB)
        with pytest.raises(ValueError, match="not found in events"):
            enc._validate_alignment(feature_paths, bold_metas)

    def test_unpartitioned_run_multiple_cells_raises(self, tree: BIDSTree):
        tree.add_bold(run="1", tr=2.0)  # no events → unpartitioned
        tree.add_feature("mfcc", run="1", trial="1")
        tree.add_feature("mfcc", run="1", trial="2")
        enc = make_encoding(tree, ["mfcc"])
        feature_paths = enc._discover_features(_DEFAULT_SUB)
        bold_metas = enc._discover_bold(_DEFAULT_SUB)
        with pytest.raises(ValueError, match="unpartitioned but has 2 feature cells"):
            enc._validate_alignment(feature_paths, bold_metas)

    def test_apparent_partition_entity_without_events_passes_silently(
        self, tree: BIDSTree
    ):
        # Blind spot: trial-1 on filename without events.tsv is indistinguishable
        # from a descriptive tag — mismatch surfaces only as row-count errors later
        tree.add_bold(run="1", tr=2.0)
        tree.add_feature("mfcc", run="1", trial="1")
        enc = make_encoding(tree, ["mfcc"])
        feature_paths = enc._discover_features(_DEFAULT_SUB)
        bold_metas = enc._discover_bold(_DEFAULT_SUB)
        enc._validate_alignment(feature_paths, bold_metas)  # does not raise
