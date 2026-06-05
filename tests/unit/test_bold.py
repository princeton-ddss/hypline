import pytest

from hypline.bids import BIDSPath
from hypline.bold import get_repetition_time, load_bold_meta, resolve_bold_image
from hypline.layout import BIDSLayout

from .conftest import DEFAULT_BOLD_N_TRS, HEADER_TR, BIDSTree, minimal_nifti_gz

SUB = "001"
TASK = "conv"
SPACE = "MNI152NLin6Asym"


_SEGMENT_ROWS = [
    {"trial_type": "block-1", "onset": 0.0, "duration": 100.0},
    {"trial_type": "block-2", "onset": 100.0, "duration": 100.0},
]


class TestGetRepetitionTime:
    """TR resolves by priority: raw sidecar -> any fmriprep .nii.gz
    (sidecar over header) -> raw header."""

    def test_raw_sidecar_preferred(self, tree: BIDSTree):
        # Sidecar says 2.0, header says 1.0 -> sidecar wins
        bold_path = tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0)
        tr = get_repetition_time(BIDSLayout(tree.root), BIDSPath(bold_path))
        assert tr == 2.0

    def test_fmriprep_sidecar_when_raw_absent(self, tree: BIDSTree):
        bold_path = tree.add_bold(
            sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, write_raw=False
        )
        # No raw tree, so the fmriprep sidecar resolves it
        tr = get_repetition_time(BIDSLayout(tree.root), BIDSPath(bold_path))
        assert tr == 2.0

    def test_fmriprep_header_when_no_sidecar(self, tree: BIDSTree):
        bold_path = tree.add_bold(
            sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, write_raw=False
        )
        # Drop the fmriprep sidecar too: TR falls back to the header
        bold_path.with_name(bold_path.name.rsplit(".nii.gz", 1)[0] + ".json").unlink()
        tr = get_repetition_time(BIDSLayout(tree.root), BIDSPath(bold_path))
        assert tr == HEADER_TR

    def test_raw_header_last_resort(self, tree: BIDSTree):
        # Surface bids, raw sidecar and all fmriprep .nii.gz gone, but a raw
        # volumetric .nii.gz survives: TR resolves from its header
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0)
        surface_path = tree._add_fmriprep(
            sub=SUB,
            ses=None,
            task=TASK,
            run="1",
            suffix="bold",
            ext=".func.gii",
            extra_entities={"space": "fsaverage6", "desc": "preproc", "hemi": "L"},
        )
        raw_dir = tree.raw_func_dir(sub=SUB)
        (raw_dir / f"sub-{SUB}_task-{TASK}_run-1_bold.json").unlink()
        for f in tree.func_dir(sub=SUB).glob("*_bold.nii.gz"):
            f.unlink()
            f.with_name(f.name.rsplit(".nii.gz", 1)[0] + ".json").unlink()
        tr = get_repetition_time(BIDSLayout(tree.root), BIDSPath(surface_path))
        assert tr == HEADER_TR

    def test_sibling_fmriprep_nii_when_bids_is_surface(self, tree: BIDSTree):
        # Surface .func.gii has no header TR; falls through to the
        # sibling volumetric .nii.gz for the same run
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, write_raw=False)
        surface_path = tree._add_fmriprep(
            sub=SUB,
            ses=None,
            task=TASK,
            run="1",
            suffix="bold",
            ext=".func.gii",
            extra_entities={"space": "fsaverage6", "desc": "preproc", "hemi": "L"},
        )
        tr = get_repetition_time(BIDSLayout(tree.root), BIDSPath(surface_path))
        assert tr == 2.0

    def test_no_source_raises(self, tree: BIDSTree):
        bold_path = tree.add_bold(
            sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, write_raw=False
        )
        # Remove the fmriprep BOLD and its sidecar: nothing left to resolve from
        bold_path.with_name(bold_path.name.rsplit(".nii.gz", 1)[0] + ".json").unlink()
        bold_path.unlink()
        with pytest.raises(ValueError, match="Cannot resolve TR"):
            get_repetition_time(BIDSLayout(tree.root), BIDSPath(bold_path))


class TestResolveBoldImage:
    """Resolves an on-disk BOLD `.nii.gz` for a run: fmriprep -> raw -> raise."""

    def test_fmriprep_preferred(self, tree: BIDSTree):
        # Both present: the fmriprep BOLD wins over the raw image
        bold_path = tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1")
        feature_path = tree.add_feature(sub=SUB, task=TASK, run="1", kind="phonemic")
        resolved = resolve_bold_image(BIDSLayout(tree.root), BIDSPath(feature_path))
        assert resolved.path == bold_path

    def test_raw_when_fmriprep_absent(self, tree: BIDSTree):
        # Only the raw image exists: resolution falls back to it
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", write_raw=True)
        bold_path = tree.func_dir(sub=SUB).glob("*_bold.nii.gz").__next__()
        bold_path.unlink()
        feature_path = tree.add_feature(sub=SUB, task=TASK, run="1", kind="phonemic")
        resolved = resolve_bold_image(BIDSLayout(tree.root), BIDSPath(feature_path))
        raw_bold = (
            tree.raw_func_dir(sub=SUB) / f"sub-{SUB}_task-{TASK}_run-1_bold.nii.gz"
        )
        assert resolved.path == raw_bold

    def test_neither_raises(self, tree: BIDSTree):
        # No BOLD image of any kind: n_trs is unresolvable
        feature_path = tree.add_feature(sub=SUB, task=TASK, run="1", kind="phonemic")
        with pytest.raises(FileNotFoundError, match="No BOLD image found"):
            resolve_bold_image(BIDSLayout(tree.root), BIDSPath(feature_path))


class TestLoadBoldMeta:
    def test_non_bold_input_raises(self, tree: BIDSTree):
        feature_path = tree.add_feature(sub=SUB, task=TASK, run="1", kind="phonemic")
        with pytest.raises(ValueError, match="Expected a BOLD file"):
            load_bold_meta(BIDSLayout(tree.root), BIDSPath(feature_path))

    def test_missing_task_entity_raises(self, tree: BIDSTree):
        bold_path = tree.add_bold(sub=SUB, space=SPACE, run="1")
        with pytest.raises(ValueError, match="missing required 'task' entity"):
            load_bold_meta(BIDSLayout(tree.root), BIDSPath(bold_path))

    def test_derivative_n_trs_mismatches_raw_raises(self, tree: BIDSTree):
        bold_path = tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1")
        bold_path.write_bytes(minimal_nifti_gz(n_trs=8))
        with pytest.raises(ValueError, match="raw-relative"):
            load_bold_meta(BIDSLayout(tree.root), BIDSPath(bold_path))

    def test_derivative_n_trs_matches_raw_passes(self, tree: BIDSTree):
        bold_path = tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1")
        meta = load_bold_meta(BIDSLayout(tree.root), BIDSPath(bold_path))
        assert meta.n_trs == DEFAULT_BOLD_N_TRS

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

    def test_overlapping_segments_raises(self, tree: BIDSTree):
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
        with pytest.raises(ValueError, match="collides with a BIDS-reserved entity"):
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
