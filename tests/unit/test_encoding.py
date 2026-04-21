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
        partition: str | None = None,
        subdir: str | None = None,
    ) -> Path:
        entities = self._identity_entities(sub, ses, task, run)
        if partition is not None:
            entities["partition"] = partition
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
        partitions: dict[str, tuple[float, float]] | None = None,
    ) -> Path:
        entities = self._identity_entities(sub, ses, task, run)
        rows = []
        if partitions:
            for name, (onset, duration) in partitions.items():
                rows.append(f"partition-{name}\t{onset}\t{duration}")
        events_path = self.bold_dir / f"{self._stem(entities)}_events.tsv"
        events_path.write_text("trial_type\tonset\tduration\n" + "\n".join(rows))
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

    def test_unknown_entity_in_filters_raises(self, tree: BIDSTree):
        with pytest.raises(ValueError, match="Unknown bids_filters entity"):
            make_encoding(tree, ["mfcc"], bids_filters=["xyz-foo"])

    def test_invalid_bold_space_raises(self, tree: BIDSTree):
        with pytest.raises(ValueError, match="Unsupported BOLD data space"):
            make_encoding(tree, ["mfcc"], bold_space="notaspace")


class TestDiscoverFeatures:
    def test_returns_expected_keys(self, tree: BIDSTree):
        tree.add_feature("mfcc", run="1")
        enc = make_encoding(tree, ["mfcc"])
        feature_paths = enc._discover_features(_DEFAULT_SUB)
        expected = FeatureKey(
            cell=CellKey(ses=None, run="1", partition=None),
            feature="mfcc",
        )
        assert expected in feature_paths

    def test_no_files_raises(self, tree: BIDSTree):
        enc = make_encoding(tree, ["mfcc"])
        with pytest.raises(FileNotFoundError, match="mfcc"):
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

    def test_partition_filter_skips_unrequested(self, tree: BIDSTree):
        tree.add_feature("mfcc", partition="a")
        tree.add_feature("mfcc", partition="b")
        enc = make_encoding(tree, ["mfcc"], bids_filters=["partition-a"])
        feature_paths = enc._discover_features(_DEFAULT_SUB)
        assert {k.cell.partition for k in feature_paths} == {"a"}

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
        expected = FeatureKey(
            cell=CellKey(ses=None, run="1", partition=None),
            feature="mfcc",
        )
        assert expected in feature_paths


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

    def test_inconsistent_tr_raises(self, tree: BIDSTree):
        tree.add_bold(run="1", tr=2.0)
        tree.add_bold(run="2", tr=1.5)
        enc = make_encoding(tree, ["mfcc"])
        with pytest.raises(ValueError, match="Inconsistent repetition times"):
            enc._discover_bold(_DEFAULT_SUB)

    def test_partition_slices_parsed(self, tree: BIDSTree):
        tree.add_bold(run="1", tr=2.0)
        tree.add_events(run="1", partitions={"a": (0.0, 100.0), "b": (100.0, 100.0)})
        enc = make_encoding(tree, ["mfcc"])
        bold_metas = enc._discover_bold(_DEFAULT_SUB)
        meta = bold_metas[BoldKey(ses=None, run="1")]
        assert meta.partitions["a"] == slice(0, 50)
        assert meta.partitions["b"] == slice(50, 100)

    def test_non_contiguous_partitions_raises(self, tree: BIDSTree):
        tree.add_bold(run="1", tr=2.0)
        tree.add_events(run="1", partitions={"a": (0.0, 100.0), "b": (120.0, 80.0)})
        enc = make_encoding(tree, ["mfcc"])
        with pytest.raises(ValueError, match="not contiguous"):
            enc._discover_bold(_DEFAULT_SUB)

    def test_partitions_not_starting_at_zero_raises(self, tree: BIDSTree):
        tree.add_bold(run="1", tr=2.0)
        tree.add_events(run="1", partitions={"a": (10.0, 90.0)})
        enc = make_encoding(tree, ["mfcc"])
        with pytest.raises(ValueError, match="start at TR 0"):
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

    def test_undeclared_partition_raises(self, tree: BIDSTree):
        tree.add_feature("mfcc", run="1", partition="a")
        tree.add_bold(run="1")
        enc = make_encoding(tree, ["mfcc"])
        feature_paths = enc._discover_features(_DEFAULT_SUB)
        bold_metas = enc._discover_bold(_DEFAULT_SUB)
        with pytest.raises(ValueError, match="Partition.*not found in events"):
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
