from pathlib import Path

import pytest

from hypline.bids import BIDSPath
from hypline.layout import BIDSLayout

from .conftest import HyplineBIDSTree

SUB = "001"
SES = "1"
TASK = "conv"
SPACE = "MNI152NLin6Asym"


class TestBIDSLayoutConstruction:
    def test_raises_if_root_missing(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            BIDSLayout(tmp_path / "nonexistent")

    def test_does_not_raise_if_features_absent(self, tmp_path: Path):
        (tmp_path / "stimuli").mkdir()
        BIDSLayout(tmp_path)


class TestFindStimuli:
    def test_returns_matching_files(self, layout_tree: HyplineBIDSTree):
        layout_tree.add_stimulus(kind="audio", ext=".wav", sub=SUB, task=TASK)
        layout = BIDSLayout(layout_tree.root)
        results = layout.find.stimuli(sub=SUB, kind="audio", ext=".wav")
        assert len(results) == 1
        assert results[0].sub == SUB

    def test_returns_empty_when_sub_absent(self, layout_tree: HyplineBIDSTree):
        layout = BIDSLayout(layout_tree.root)
        results = layout.find.stimuli(sub="999", kind="audio", ext=".wav")
        assert results == []

    def test_ses_filter_via_bids_filters(self, layout_tree: HyplineBIDSTree):
        layout_tree.add_stimulus(kind="audio", ext=".wav", sub=SUB, ses="1", task=TASK)
        layout_tree.add_stimulus(kind="audio", ext=".wav", sub=SUB, ses="2", task=TASK)
        layout = BIDSLayout(layout_tree.root)
        results = layout.find.stimuli(
            sub=SUB, kind="audio", ext=".wav", bids_filters=["ses-1"]
        )
        assert len(results) == 1
        assert results[0].ses == "1"

    def test_arbitrary_bids_filter(self, layout_tree: HyplineBIDSTree):
        layout_tree.add_stimulus(kind="audio", ext=".wav", sub=SUB, task="conv")
        layout_tree.add_stimulus(kind="audio", ext=".wav", sub=SUB, task="rest")
        layout = BIDSLayout(layout_tree.root)
        results = layout.find.stimuli(
            sub=SUB, kind="audio", ext=".wav", bids_filters=["task-conv"]
        )
        assert len(results) == 1

    def test_rejects_reserved_sub_filter(self, layout_tree: HyplineBIDSTree):
        layout = BIDSLayout(layout_tree.root)
        with pytest.raises(ValueError, match="sub"):
            layout.find.stimuli(
                sub=SUB, kind="audio", ext=".wav", bids_filters=["sub-001"]
            )

    def test_rejects_reserved_stim_filter(self, layout_tree: HyplineBIDSTree):
        layout = BIDSLayout(layout_tree.root)
        with pytest.raises(ValueError, match="stim"):
            layout.find.stimuli(
                sub=SUB, kind="audio", ext=".wav", bids_filters=["stim-audio"]
            )

    def test_empty_if_area_absent(self, tmp_path: Path):
        layout = BIDSLayout(tmp_path)
        assert layout.find.stimuli(sub=SUB, kind="audio", ext=".wav") == []

    def test_run_filter_via_bids_filters(self, layout_tree: HyplineBIDSTree):
        layout_tree.add_stimulus(kind="audio", ext=".wav", sub=SUB, task=TASK, run="1")
        layout_tree.add_stimulus(kind="audio", ext=".wav", sub=SUB, task=TASK, run="2")
        layout = BIDSLayout(layout_tree.root)
        results = layout.find.stimuli(
            sub=SUB, kind="audio", ext=".wav", bids_filters=["run-1"]
        )
        assert len(results) == 1
        assert results[0].entities.get("run") == "1"

    def test_cross_session_aggregation_sorted(self, layout_tree: HyplineBIDSTree):
        layout_tree.add_stimulus(kind="audio", ext=".wav", sub=SUB, ses="2", task=TASK)
        layout_tree.add_stimulus(kind="audio", ext=".wav", sub=SUB, ses="1", task=TASK)
        layout = BIDSLayout(layout_tree.root)
        results = layout.find.stimuli(sub=SUB, kind="audio", ext=".wav")
        assert len(results) == 2
        assert results == sorted(results)

    def test_multiple_ses_filters(self, layout_tree: HyplineBIDSTree):
        layout_tree.add_stimulus(kind="audio", ext=".wav", sub=SUB, ses="1", task=TASK)
        layout_tree.add_stimulus(kind="audio", ext=".wav", sub=SUB, ses="2", task=TASK)
        layout_tree.add_stimulus(kind="audio", ext=".wav", sub=SUB, ses="3", task=TASK)
        layout = BIDSLayout(layout_tree.root)
        results = layout.find.stimuli(
            sub=SUB, kind="audio", ext=".wav", bids_filters=["ses-1", "ses-2"]
        )
        assert len(results) == 2
        assert {r.ses for r in results} == {"1", "2"}


class TestFindFeatures:
    def test_translates_kind_to_feature_entity(self, layout_tree: HyplineBIDSTree):
        layout_tree.add_feature(kind="phonemic", sub=SUB, task=TASK)
        layout = BIDSLayout(layout_tree.root)
        results = layout.find.features(sub=SUB, kind="phonemic")
        assert len(results) == 1
        assert results[0].entities.get("feature") == "phonemic"

    def test_ses_filter_via_bids_filters(self, layout_tree: HyplineBIDSTree):
        layout_tree.add_feature(kind="phonemic", sub=SUB, ses="1", task=TASK)
        layout_tree.add_feature(kind="phonemic", sub=SUB, ses="2", task=TASK)
        layout = BIDSLayout(layout_tree.root)
        results = layout.find.features(sub=SUB, kind="phonemic", bids_filters=["ses-1"])
        assert len(results) == 1
        assert results[0].ses == "1"

    def test_arbitrary_bids_filter(self, layout_tree: HyplineBIDSTree):
        layout_tree.add_feature(kind="phonemic", sub=SUB, task=TASK, desc="gpt3")
        layout_tree.add_feature(kind="phonemic", sub=SUB, task=TASK)
        layout = BIDSLayout(layout_tree.root)
        results = layout.find.features(
            sub=SUB, kind="phonemic", bids_filters=["desc-gpt3"]
        )
        assert len(results) == 1
        assert results[0].entities.get("desc") == "gpt3"

    def test_rejects_reserved_sub_filter(self, layout_tree: HyplineBIDSTree):
        layout = BIDSLayout(layout_tree.root)
        with pytest.raises(ValueError, match="sub"):
            layout.find.features(sub=SUB, kind="phonemic", bids_filters=["sub-001"])

    def test_rejects_reserved_feature_filter(self, layout_tree: HyplineBIDSTree):
        layout = BIDSLayout(layout_tree.root)
        with pytest.raises(ValueError, match="feature"):
            layout.find.features(
                sub=SUB, kind="phonemic", bids_filters=["feature-phonemic"]
            )

    def test_returns_empty_when_sub_absent(self, layout_tree: HyplineBIDSTree):
        layout = BIDSLayout(layout_tree.root)
        assert layout.find.features(sub="999", kind="phonemic") == []

    def test_empty_if_area_absent(self, tmp_path: Path):
        layout = BIDSLayout(tmp_path)
        assert layout.find.features(sub=SUB, kind="phonemic") == []

    def test_run_filter_via_bids_filters(self, layout_tree: HyplineBIDSTree):
        layout_tree.add_feature(kind="phonemic", sub=SUB, task=TASK, run="1")
        layout_tree.add_feature(kind="phonemic", sub=SUB, task=TASK, run="2")
        layout = BIDSLayout(layout_tree.root)
        results = layout.find.features(sub=SUB, kind="phonemic", bids_filters=["run-1"])
        assert len(results) == 1
        assert results[0].entities.get("run") == "1"

    def test_cross_session_aggregation_sorted(self, layout_tree: HyplineBIDSTree):
        layout_tree.add_feature(kind="phonemic", sub=SUB, ses="2", task=TASK)
        layout_tree.add_feature(kind="phonemic", sub=SUB, ses="1", task=TASK)
        layout = BIDSLayout(layout_tree.root)
        results = layout.find.features(sub=SUB, kind="phonemic")
        assert len(results) == 2
        assert results == sorted(results)

    def test_multiple_ses_filters(self, layout_tree: HyplineBIDSTree):
        layout_tree.add_feature(kind="phonemic", sub=SUB, ses="1", task=TASK)
        layout_tree.add_feature(kind="phonemic", sub=SUB, ses="2", task=TASK)
        layout_tree.add_feature(kind="phonemic", sub=SUB, ses="3", task=TASK)
        layout = BIDSLayout(layout_tree.root)
        results = layout.find.features(
            sub=SUB, kind="phonemic", bids_filters=["ses-1", "ses-2"]
        )
        assert len(results) == 2
        assert {r.ses for r in results} == {"1", "2"}


class TestFindFmriprep:
    def test_returns_bold_files(self, layout_tree: HyplineBIDSTree):
        layout_tree.add_fmriprep_bold(space=SPACE, sub=SUB, task=TASK)
        layout = BIDSLayout(layout_tree.root)
        results = layout.find.fmriprep(sub=SUB, suffix="bold", ext=".nii.gz")
        assert len(results) == 1
        assert results[0].entities.get("space") == SPACE
        assert results[0].entities.get("desc") == "preproc"

    def test_skips_non_bold_suffix(self, layout_tree: HyplineBIDSTree):
        layout_tree.add_fmriprep_bold(space=SPACE, sub=SUB, task=TASK)
        layout_tree.add_fmriprep(
            suffix="mask", ext=".nii.gz", sub=SUB, task=TASK, space=SPACE, desc="brain"
        )
        layout = BIDSLayout(layout_tree.root)
        results = layout.find.fmriprep(sub=SUB, suffix="bold", ext=".nii.gz")
        assert len(results) == 1
        assert results[0].path.name.endswith("_bold.nii.gz")

    def test_ses_filter_via_bids_filters(self, layout_tree: HyplineBIDSTree):
        layout_tree.add_fmriprep_bold(space=SPACE, sub=SUB, ses="1", task=TASK)
        layout_tree.add_fmriprep_bold(space=SPACE, sub=SUB, ses="2", task=TASK)
        layout = BIDSLayout(layout_tree.root)
        results = layout.find.fmriprep(
            sub=SUB, suffix="bold", ext=".nii.gz", bids_filters=["ses-1"]
        )
        assert len(results) == 1
        assert results[0].ses == "1"

    def test_space_filter_via_bids_filters(self, layout_tree: HyplineBIDSTree):
        layout_tree.add_fmriprep_bold(space=SPACE, sub=SUB, task=TASK)
        layout_tree.add_fmriprep_bold(space="T1w", sub=SUB, task="rest")
        layout = BIDSLayout(layout_tree.root)
        results = layout.find.fmriprep(
            sub=SUB, suffix="bold", ext=".nii.gz", bids_filters=[f"space-{SPACE}"]
        )
        assert len(results) == 1
        assert results[0].entities.get("space") == SPACE

    def test_desc_filter_via_bids_filters(self, layout_tree: HyplineBIDSTree):
        layout_tree.add_fmriprep_bold(space=SPACE, desc="clean", sub=SUB, task=TASK)
        layout_tree.add_fmriprep_bold(space=SPACE, sub=SUB, task="rest")
        layout = BIDSLayout(layout_tree.root)
        results = layout.find.fmriprep(
            sub=SUB, suffix="bold", ext=".nii.gz", bids_filters=["desc-clean"]
        )
        assert len(results) == 1
        assert results[0].entities.get("desc") == "clean"

    def test_run_filter_via_bids_filters(self, layout_tree: HyplineBIDSTree):
        layout_tree.add_fmriprep_bold(space=SPACE, sub=SUB, task=TASK, run="1")
        layout_tree.add_fmriprep_bold(space=SPACE, sub=SUB, task=TASK, run="2")
        layout = BIDSLayout(layout_tree.root)
        results = layout.find.fmriprep(
            sub=SUB, suffix="bold", ext=".nii.gz", bids_filters=["run-1"]
        )
        assert len(results) == 1
        assert results[0].entities.get("run") == "1"

    def test_rejects_reserved_sub_filter(self, layout_tree: HyplineBIDSTree):
        layout = BIDSLayout(layout_tree.root)
        with pytest.raises(ValueError, match="sub"):
            layout.find.fmriprep(
                sub=SUB, suffix="bold", ext=".nii.gz", bids_filters=["sub-001"]
            )

    def test_returns_empty_when_sub_absent(self, layout_tree: HyplineBIDSTree):
        layout = BIDSLayout(layout_tree.root)
        assert layout.find.fmriprep(sub="999", suffix="bold", ext=".nii.gz") == []

    def test_empty_if_fmriprep_absent(self, tmp_path: Path):
        layout = BIDSLayout(tmp_path)
        assert layout.find.fmriprep(sub=SUB, suffix="bold", ext=".nii.gz") == []

    def test_cross_session_aggregation_sorted(self, layout_tree: HyplineBIDSTree):
        layout_tree.add_fmriprep_bold(space=SPACE, sub=SUB, ses="2", task=TASK)
        layout_tree.add_fmriprep_bold(space=SPACE, sub=SUB, ses="1", task=TASK)
        layout = BIDSLayout(layout_tree.root)
        results = layout.find.fmriprep(sub=SUB, suffix="bold", ext=".nii.gz")
        assert len(results) == 2
        assert results == sorted(results)


class TestBuildStimulus:
    def test_derives_output_path(self, tmp_path: Path):
        layout = BIDSLayout(tmp_path)
        source = BIDSPath(f"sub-{SUB}_task-{TASK}_stim-audio.wav")
        out = layout.build.stimulus(kind="transcript", source=source, ext=".csv")
        assert out.entities.get("stim") == "transcript"
        assert out.path.name.endswith(".csv")
        assert "stimuli" in out.path.parts
        assert f"sub-{SUB}" in out.path.parts
        assert "transcript" in out.path.parts

    def test_applies_entity_override(self, tmp_path: Path):
        layout = BIDSLayout(tmp_path)
        source = BIDSPath(f"sub-{SUB}_task-{TASK}_stim-audio.wav")
        out = layout.build.stimulus(
            kind="transcript", source=source, ext=".csv", run="02"
        )
        assert out.entities.get("run") == "02"

    def test_raises_if_source_missing_sub(self, tmp_path: Path):
        layout = BIDSLayout(tmp_path)
        source = BIDSPath(f"ses-{SES}_task-{TASK}_stim-audio.wav")
        with pytest.raises(ValueError, match="sub"):
            layout.build.stimulus(kind="transcript", source=source, ext=".csv")

    def test_omits_ses_dir_when_source_has_no_ses(self, tmp_path: Path):
        layout = BIDSLayout(tmp_path)
        source = BIDSPath(f"sub-{SUB}_task-{TASK}_stim-audio.wav")
        out = layout.build.stimulus(kind="transcript", source=source, ext=".csv")
        assert "stimuli" in out.path.parts
        assert f"sub-{SUB}" in out.path.parts
        assert not any(p.startswith("ses-") for p in out.path.parts)
        assert "transcript" in out.path.parts

    def test_includes_ses_dir_when_source_has_ses(self, tmp_path: Path):
        layout = BIDSLayout(tmp_path)
        source = BIDSPath(f"sub-{SUB}_ses-{SES}_task-{TASK}_stim-audio.wav")
        out = layout.build.stimulus(kind="transcript", source=source, ext=".csv")
        assert "stimuli" in out.path.parts
        assert f"sub-{SUB}" in out.path.parts
        assert f"ses-{SES}" in out.path.parts
        assert "transcript" in out.path.parts

    def test_invalid_extension_raises(self, tmp_path: Path):
        layout = BIDSLayout(tmp_path)
        source = BIDSPath(f"sub-{SUB}_task-{TASK}_stim-audio.wav")
        with pytest.raises(ValueError, match="extension"):
            layout.build.stimulus(kind="transcript", source=source, ext="csv")

    def test_invalid_override_value_raises(self, tmp_path: Path):
        layout = BIDSLayout(tmp_path)
        source = BIDSPath(f"sub-{SUB}_task-{TASK}_stim-audio.wav")
        with pytest.raises(ValueError, match="Invalid BIDS entity"):
            layout.build.stimulus(
                kind="transcript", source=source, ext=".csv", run="BAD!"
            )


class TestBuildFeature:
    def test_derives_output_path(self, tmp_path: Path):
        layout = BIDSLayout(tmp_path)
        source = BIDSPath(f"sub-{SUB}_task-{TASK}_stim-transcript.csv")
        out = layout.build.feature(source=source, kind="phonemic")
        assert out.entities.get("feature") == "phonemic"
        assert out.path.suffix == ".parquet"
        assert "features" in out.path.parts
        assert f"sub-{SUB}" in out.path.parts
        assert "phonemic" in out.path.parts

    def test_applies_entity_override(self, tmp_path: Path):
        layout = BIDSLayout(tmp_path)
        source = BIDSPath(f"sub-{SUB}_task-{TASK}_stim-transcript.csv")
        out = layout.build.feature(source=source, kind="phonemic", desc="gpt3")
        assert out.entities.get("desc") == "gpt3"

    def test_raises_if_source_missing_sub(self, tmp_path: Path):
        layout = BIDSLayout(tmp_path)
        source = BIDSPath(f"ses-{SES}_task-{TASK}_stim-transcript.csv")
        with pytest.raises(ValueError, match="sub"):
            layout.build.feature(source=source, kind="phonemic")

    def test_omits_ses_dir_when_source_has_no_ses(self, tmp_path: Path):
        layout = BIDSLayout(tmp_path)
        source = BIDSPath(f"sub-{SUB}_task-{TASK}_stim-transcript.csv")
        out = layout.build.feature(source=source, kind="phonemic")
        assert "features" in out.path.parts
        assert f"sub-{SUB}" in out.path.parts
        assert not any(p.startswith("ses-") for p in out.path.parts)
        assert "phonemic" in out.path.parts

    def test_includes_ses_dir_when_source_has_ses(self, tmp_path: Path):
        layout = BIDSLayout(tmp_path)
        source = BIDSPath(f"sub-{SUB}_ses-{SES}_task-{TASK}_stim-transcript.csv")
        out = layout.build.feature(source=source, kind="phonemic")
        assert "features" in out.path.parts
        assert f"sub-{SUB}" in out.path.parts
        assert f"ses-{SES}" in out.path.parts
        assert "phonemic" in out.path.parts

    def test_invalid_override_value_raises(self, tmp_path: Path):
        layout = BIDSLayout(tmp_path)
        source = BIDSPath(f"sub-{SUB}_task-{TASK}_stim-transcript.csv")
        with pytest.raises(ValueError, match="Invalid BIDS entity"):
            layout.build.feature(source=source, kind="phonemic", desc="BAD!")


class TestListSubjects:
    def test_subjects_stimuli(self, layout_tree: HyplineBIDSTree):
        layout_tree.add_stimulus(kind="audio", ext=".wav", sub=SUB, task=TASK)
        layout_tree.add_stimulus(kind="audio", ext=".wav", sub="002", task=TASK)
        layout = BIDSLayout(layout_tree.root)
        assert layout.list.subjects("stimuli") == [SUB, "002"]

    def test_subjects_features(self, layout_tree: HyplineBIDSTree):
        layout_tree.add_feature(kind="phonemic", sub=SUB, task=TASK)
        layout = BIDSLayout(layout_tree.root)
        assert layout.list.subjects("features") == [SUB]

    def test_subjects_fmriprep(self, layout_tree: HyplineBIDSTree):
        layout_tree.add_fmriprep_bold(space=SPACE, sub=SUB, task=TASK)
        layout = BIDSLayout(layout_tree.root)
        assert layout.list.subjects("fmriprep") == [SUB]

    def test_subjects_skips_non_sub_dirs(self, layout_tree: HyplineBIDSTree):
        layout_tree.add_fmriprep_bold(space=SPACE, sub=SUB, task=TASK)
        non_sub_dir = layout_tree.root / "derivatives" / "fmriprep" / "logs"
        non_sub_dir.mkdir(parents=True, exist_ok=True)
        layout = BIDSLayout(layout_tree.root)
        assert layout.list.subjects("fmriprep") == [SUB]

    def test_subjects_empty_if_area_absent(self, tmp_path: Path):
        layout = BIDSLayout(tmp_path)
        assert layout.list.subjects("stimuli") == []


class TestListSessions:
    def test_sessions_stimuli(self, layout_tree: HyplineBIDSTree):
        layout_tree.add_stimulus(kind="audio", ext=".wav", sub=SUB, ses="1", task=TASK)
        layout_tree.add_stimulus(kind="audio", ext=".wav", sub=SUB, ses="2", task=TASK)
        layout = BIDSLayout(layout_tree.root)
        assert layout.list.sessions(sub=SUB, area="stimuli") == ["1", "2"]

    def test_sessions_fmriprep(self, layout_tree: HyplineBIDSTree):
        layout_tree.add_fmriprep_bold(space=SPACE, sub=SUB, ses=SES, task=TASK)
        layout = BIDSLayout(layout_tree.root)
        assert layout.list.sessions(sub=SUB, area="fmriprep") == [SES]

    def test_sessions_empty_if_sub_absent(self, tmp_path: Path):
        layout = BIDSLayout(tmp_path)
        assert layout.list.sessions(sub=SUB, area="stimuli") == []

    def test_sessions_features(self, layout_tree: HyplineBIDSTree):
        layout_tree.add_feature(kind="phonemic", sub=SUB, ses="1", task=TASK)
        layout_tree.add_feature(kind="phonemic", sub=SUB, ses="2", task=TASK)
        layout = BIDSLayout(layout_tree.root)
        assert layout.list.sessions(sub=SUB, area="features") == ["1", "2"]

    def test_sessions_skips_non_ses_dirs(self, layout_tree: HyplineBIDSTree):
        layout_tree.add_fmriprep_bold(space=SPACE, sub=SUB, ses=SES, task=TASK)
        layout_tree.add_fmriprep_bold(space=SPACE, sub=SUB, task=TASK)
        layout = BIDSLayout(layout_tree.root)
        assert layout.list.sessions(sub=SUB, area="fmriprep") == [SES]

    def test_sessions_empty_when_subject_has_no_ses(self, layout_tree: HyplineBIDSTree):
        layout_tree.add_stimulus(kind="audio", ext=".wav", sub=SUB, task=TASK)
        layout = BIDSLayout(layout_tree.root)
        assert layout.list.sessions(sub=SUB, area="stimuli") == []
