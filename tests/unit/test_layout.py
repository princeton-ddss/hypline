import json
from pathlib import Path

import pytest

from hypline._version import __version__
from hypline.bids import BIDSPath
from hypline.layout import BIDSLayout

from .conftest import BIDSTree

SUB = "001"
DYAD = "101"
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
    def test_returns_matching_files(self, tree: BIDSTree):
        tree.add_stimulus(kind="audio", ext=".wav", dyad=DYAD, task=TASK)
        layout = BIDSLayout(tree.root)
        results = layout.find.stimuli(dyad=DYAD, kind="audio", ext=".wav")
        assert len(results) == 1
        assert results[0].dyad == DYAD

    def test_ses_filter_via_bids_filters(self, tree: BIDSTree):
        tree.add_stimulus(kind="audio", ext=".wav", dyad=DYAD, ses="1", task=TASK)
        tree.add_stimulus(kind="audio", ext=".wav", dyad=DYAD, ses="2", task=TASK)
        layout = BIDSLayout(tree.root)
        results = layout.find.stimuli(
            dyad=DYAD, kind="audio", ext=".wav", bids_filters=["ses-1"]
        )
        assert len(results) == 1
        assert results[0].ses == "1"

    def test_arbitrary_bids_filter(self, tree: BIDSTree):
        tree.add_stimulus(kind="audio", ext=".wav", dyad=DYAD, task="conv")
        tree.add_stimulus(kind="audio", ext=".wav", dyad=DYAD, task="rest")
        layout = BIDSLayout(tree.root)
        results = layout.find.stimuli(
            dyad=DYAD, kind="audio", ext=".wav", bids_filters=["task-conv"]
        )
        assert len(results) == 1

    def test_rejects_reserved_dyad_filter(self, tree: BIDSTree):
        layout = BIDSLayout(tree.root)
        with pytest.raises(ValueError, match="dyad"):
            layout.find.stimuli(
                dyad=DYAD, kind="audio", ext=".wav", bids_filters=["dyad-101"]
            )

    def test_rejects_reserved_stim_filter(self, tree: BIDSTree):
        layout = BIDSLayout(tree.root)
        with pytest.raises(ValueError, match="stim"):
            layout.find.stimuli(
                dyad=DYAD, kind="audio", ext=".wav", bids_filters=["stim-audio"]
            )

    def test_run_filter_via_bids_filters(self, tree: BIDSTree):
        tree.add_stimulus(kind="audio", ext=".wav", dyad=DYAD, task=TASK, run="1")
        tree.add_stimulus(kind="audio", ext=".wav", dyad=DYAD, task=TASK, run="2")
        layout = BIDSLayout(tree.root)
        results = layout.find.stimuli(
            dyad=DYAD, kind="audio", ext=".wav", bids_filters=["run-1"]
        )
        assert len(results) == 1
        assert results[0].entities.get("run") == "1"

    def test_cross_session_aggregation_sorted(self, tree: BIDSTree):
        tree.add_stimulus(kind="audio", ext=".wav", dyad=DYAD, ses="2", task=TASK)
        tree.add_stimulus(kind="audio", ext=".wav", dyad=DYAD, ses="1", task=TASK)
        layout = BIDSLayout(tree.root)
        results = layout.find.stimuli(dyad=DYAD, kind="audio", ext=".wav")
        assert len(results) == 2
        assert results == sorted(results)

    def test_multiple_ses_filters(self, tree: BIDSTree):
        tree.add_stimulus(kind="audio", ext=".wav", dyad=DYAD, ses="1", task=TASK)
        tree.add_stimulus(kind="audio", ext=".wav", dyad=DYAD, ses="2", task=TASK)
        tree.add_stimulus(kind="audio", ext=".wav", dyad=DYAD, ses="3", task=TASK)
        layout = BIDSLayout(tree.root)
        results = layout.find.stimuli(
            dyad=DYAD, kind="audio", ext=".wav", bids_filters=["ses-1", "ses-2"]
        )
        assert len(results) == 2
        assert {r.ses for r in results} == {"1", "2"}

    def test_raises_if_area_absent(self, tmp_path: Path):
        layout = BIDSLayout(tmp_path)
        with pytest.raises(FileNotFoundError, match="stimuli"):
            layout.find.stimuli(dyad=DYAD, kind="audio", ext=".wav")

    def test_raises_when_dyad_absent(self, tree: BIDSTree):
        tree.add_stimulus(kind="audio", ext=".wav", dyad=DYAD, task=TASK)
        layout = BIDSLayout(tree.root)
        with pytest.raises(FileNotFoundError, match="dyad-999"):
            layout.find.stimuli(dyad="999", kind="audio", ext=".wav")

    def test_kind_dir_absent_lists_siblings(self, tree: BIDSTree):
        tree.add_stimulus(kind="transcript", ext=".csv", dyad=DYAD, task=TASK)
        layout = BIDSLayout(tree.root)
        with pytest.raises(FileNotFoundError, match="transcript"):
            layout.find.stimuli(dyad=DYAD, kind="audio", ext=".wav")

    def test_extension_mismatch_message(self, tree: BIDSTree):
        tree.add_stimulus(kind="audio", ext=".mp3", dyad=DYAD, task=TASK)
        layout = BIDSLayout(tree.root)
        with pytest.raises(FileNotFoundError, match=r"\.mp3"):
            layout.find.stimuli(dyad=DYAD, kind="audio", ext=".wav")

    def test_filter_mismatch_message(self, tree: BIDSTree):
        tree.add_stimulus(kind="audio", ext=".wav", dyad=DYAD, task="rest")
        layout = BIDSLayout(tree.root)
        with pytest.raises(FileNotFoundError, match="task-conv"):
            layout.find.stimuli(
                dyad=DYAD, kind="audio", ext=".wav", bids_filters=["task-conv"]
            )

    def test_session_missing_lists_available(self, tree: BIDSTree):
        tree.add_stimulus(kind="audio", ext=".wav", dyad=DYAD, ses="1", task=TASK)
        layout = BIDSLayout(tree.root)
        with pytest.raises(FileNotFoundError, match="ses-1|available"):
            layout.find.stimuli(
                dyad=DYAD, kind="audio", ext=".wav", bids_filters=["ses-2"]
            )

    def test_raises_when_task_missing(self, tree: BIDSTree):
        path = (
            tree.stimuli_dir / f"dyad-{DYAD}" / "audio" / f"dyad-{DYAD}_stim-audio.wav"
        )
        path.parent.mkdir(parents=True)
        path.touch()
        layout = BIDSLayout(tree.root)
        with pytest.raises(ValueError, match="task"):
            layout.find.stimuli(dyad=DYAD, kind="audio", ext=".wav")

    def test_descriptive_filter_via_events_metadata(self, tree: BIDSTree):
        # Filter `cond-R` matches only the trial whose sidecar metadata says cond=R
        tree.add_participants({SUB: DYAD})
        tree.add_stimulus(
            kind="audio",
            ext=".wav",
            dyad=DYAD,
            task=TASK,
            run="1",
            extra_entities={"trial": "1"},
        )
        tree.add_stimulus(
            kind="audio",
            ext=".wav",
            dyad=DYAD,
            task=TASK,
            run="1",
            extra_entities={"trial": "2"},
        )
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=[
                {"trial_type": "trial-1", "onset": 0.0, "duration": 10.0},
                {"trial_type": "trial-2", "onset": 10.0, "duration": 10.0},
            ],
            sidecar_json={
                "trial_type": {
                    "Levels": {
                        "trial-1": {"metadata": {"cond": "R"}},
                        "trial-2": {"metadata": {"cond": "L"}},
                    }
                }
            },
        )
        layout = BIDSLayout(tree.root)
        results = layout.find.stimuli(
            dyad=DYAD, kind="audio", ext=".wav", bids_filters=["cond-R"]
        )
        assert len(results) == 1
        assert results[0].entities["trial"] == "1"

    def test_descriptive_filter_no_match_raises(self, tree: BIDSTree):
        tree.add_participants({SUB: DYAD})
        tree.add_stimulus(
            kind="audio",
            ext=".wav",
            dyad=DYAD,
            task=TASK,
            run="1",
            extra_entities={"trial": "1"},
        )
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=[{"trial_type": "trial-1", "onset": 0.0, "duration": 10.0}],
            sidecar_json={
                "trial_type": {"Levels": {"trial-1": {"metadata": {"cond": "R"}}}}
            },
        )
        layout = BIDSLayout(tree.root)
        with pytest.raises(FileNotFoundError, match="cond-L"):
            layout.find.stimuli(
                dyad=DYAD, kind="audio", ext=".wav", bids_filters=["cond-L"]
            )

    def test_descriptive_filter_or_within_key(self, tree: BIDSTree):
        tree.add_participants({SUB: DYAD})
        for trial in ("1", "2", "3"):
            tree.add_stimulus(
                kind="audio",
                ext=".wav",
                dyad=DYAD,
                task=TASK,
                run="1",
                extra_entities={"trial": trial},
            )
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=[
                {"trial_type": f"trial-{i}", "onset": (i - 1) * 10.0, "duration": 10.0}
                for i in (1, 2, 3)
            ],
            sidecar_json={
                "trial_type": {
                    "Levels": {
                        "trial-1": {"metadata": {"cond": "R"}},
                        "trial-2": {"metadata": {"cond": "L"}},
                        "trial-3": {"metadata": {"cond": "R"}},
                    }
                }
            },
        )
        layout = BIDSLayout(tree.root)
        results = layout.find.stimuli(
            dyad=DYAD, kind="audio", ext=".wav", bids_filters=["cond-R", "cond-L"]
        )
        assert len(results) == 3

    def test_structural_filter_mismatch_uses_tiered_diagnostic(self, tree: BIDSTree):
        # Asserts on `_diagnose_lookup`'s message, not the descriptive helper's
        tree.add_stimulus(
            kind="audio",
            ext=".wav",
            dyad=DYAD,
            task=TASK,
            run="1",
        )
        layout = BIDSLayout(tree.root)
        with pytest.raises(FileNotFoundError) as exc:
            layout.find.stimuli(
                dyad=DYAD,
                kind="audio",
                ext=".wav",
                bids_filters=["run-99"],
            )
        assert "descriptive filters" not in str(exc.value)
        assert "run-99" in str(exc.value)


class TestFindFeatures:
    def test_translates_kind_to_feature_entity(self, tree: BIDSTree):
        tree.add_feature(kind="phonemic", dyad=DYAD, task=TASK)
        layout = BIDSLayout(tree.root)
        results = layout.find.features(dyad=DYAD, kind="phonemic")
        assert len(results) == 1
        assert results[0].entities.get("feat") == "phonemic"

    def test_ses_filter_via_bids_filters(self, tree: BIDSTree):
        tree.add_feature(kind="phonemic", dyad=DYAD, ses="1", task=TASK)
        tree.add_feature(kind="phonemic", dyad=DYAD, ses="2", task=TASK)
        layout = BIDSLayout(tree.root)
        results = layout.find.features(
            dyad=DYAD, kind="phonemic", bids_filters=["ses-1"]
        )
        assert len(results) == 1
        assert results[0].ses == "1"

    def test_desc_none_excludes_variants(self, tree: BIDSTree):
        tree.add_feature(kind="phonemic", dyad=DYAD, task=TASK, desc="gpt3")
        tree.add_feature(kind="phonemic", dyad=DYAD, task=TASK)
        layout = BIDSLayout(tree.root)
        results = layout.find.features(dyad=DYAD, kind="phonemic")
        assert len(results) == 1
        assert results[0].entities.get("desc") is None

    def test_desc_label_selects_variant(self, tree: BIDSTree):
        tree.add_feature(kind="phonemic", dyad=DYAD, task=TASK, desc="gpt3")
        tree.add_feature(kind="phonemic", dyad=DYAD, task=TASK)
        layout = BIDSLayout(tree.root)
        results = layout.find.features(dyad=DYAD, kind="phonemic", desc="gpt3")
        assert len(results) == 1
        assert results[0].entities.get("desc") == "gpt3"

    def test_desc_star_returns_all_variants(self, tree: BIDSTree):
        # Variant folder holds a cell absent from the canonical folder
        tree.add_feature(kind="phonemic", dyad=DYAD, task=TASK, run="1")
        tree.add_feature(kind="phonemic", dyad=DYAD, task=TASK, run="2", desc="gpt3")
        layout = BIDSLayout(tree.root)
        results = layout.find.features(dyad=DYAD, kind="phonemic", desc="*")
        descs = {r.entities.get("desc") for r in results}
        assert descs == {None, "gpt3"}

    def test_rejects_reserved_desc_filter(self, tree: BIDSTree):
        layout = BIDSLayout(tree.root)
        with pytest.raises(ValueError, match="desc"):
            layout.find.features(dyad=DYAD, kind="phonemic", bids_filters=["desc-gpt3"])

    def test_rejects_reserved_dyad_filter(self, tree: BIDSTree):
        layout = BIDSLayout(tree.root)
        with pytest.raises(ValueError, match="dyad"):
            layout.find.features(dyad=DYAD, kind="phonemic", bids_filters=["dyad-101"])

    def test_rejects_reserved_feature_filter(self, tree: BIDSTree):
        layout = BIDSLayout(tree.root)
        with pytest.raises(ValueError, match="feat"):
            layout.find.features(
                dyad=DYAD, kind="phonemic", bids_filters=["feat-phonemic"]
            )

    def test_run_filter_via_bids_filters(self, tree: BIDSTree):
        tree.add_feature(kind="phonemic", dyad=DYAD, task=TASK, run="1")
        tree.add_feature(kind="phonemic", dyad=DYAD, task=TASK, run="2")
        layout = BIDSLayout(tree.root)
        results = layout.find.features(
            dyad=DYAD, kind="phonemic", bids_filters=["run-1"]
        )
        assert len(results) == 1
        assert results[0].entities.get("run") == "1"

    def test_cross_session_aggregation_sorted(self, tree: BIDSTree):
        tree.add_feature(kind="phonemic", dyad=DYAD, ses="2", task=TASK)
        tree.add_feature(kind="phonemic", dyad=DYAD, ses="1", task=TASK)
        layout = BIDSLayout(tree.root)
        results = layout.find.features(dyad=DYAD, kind="phonemic")
        assert len(results) == 2
        assert results == sorted(results)

    def test_multiple_ses_filters(self, tree: BIDSTree):
        tree.add_feature(kind="phonemic", dyad=DYAD, ses="1", task=TASK)
        tree.add_feature(kind="phonemic", dyad=DYAD, ses="2", task=TASK)
        tree.add_feature(kind="phonemic", dyad=DYAD, ses="3", task=TASK)
        layout = BIDSLayout(tree.root)
        results = layout.find.features(
            dyad=DYAD, kind="phonemic", bids_filters=["ses-1", "ses-2"]
        )
        assert len(results) == 2
        assert {r.ses for r in results} == {"1", "2"}

    def test_raises_if_area_absent(self, tmp_path: Path):
        layout = BIDSLayout(tmp_path)
        with pytest.raises(FileNotFoundError, match="features"):
            layout.find.features(dyad=DYAD, kind="phonemic")

    def test_raises_when_dyad_absent(self, tree: BIDSTree):
        tree.add_feature(kind="phonemic", dyad=DYAD, task=TASK)
        layout = BIDSLayout(tree.root)
        with pytest.raises(FileNotFoundError, match="dyad-999"):
            layout.find.features(dyad="999", kind="phonemic")

    def test_raises_when_task_missing(self, tree: BIDSTree):
        path = (
            tree.features_dir
            / f"dyad-{DYAD}"
            / "phonemic"
            / f"dyad-{DYAD}_feat-phonemic.parquet"
        )
        path.parent.mkdir(parents=True)
        path.touch()
        layout = BIDSLayout(tree.root)
        with pytest.raises(ValueError, match="task"):
            layout.find.features(dyad=DYAD, kind="phonemic")

    def test_structural_filter_mismatch_uses_tiered_diagnostic(self, tree: BIDSTree):
        # Asserts on `_diagnose_lookup`'s message, not the descriptive helper's
        tree.add_feature(
            kind="phonemic",
            dyad=DYAD,
            task=TASK,
            run="1",
        )
        layout = BIDSLayout(tree.root)
        with pytest.raises(FileNotFoundError) as exc:
            layout.find.features(
                dyad=DYAD,
                kind="phonemic",
                bids_filters=["run-99"],
            )
        assert "descriptive filters" not in str(exc.value)
        assert "run-99" in str(exc.value)


class TestFindConfounds:
    def test_translates_kind_to_confound_entity(self, tree: BIDSTree):
        tree.add_confound(kind="phonemic", dyad=DYAD, task=TASK)
        layout = BIDSLayout(tree.root)
        results = layout.find.confounds(dyad=DYAD, kind="phonemic")
        assert len(results) == 1
        assert results[0].entities.get("conf") == "phonemic"

    def test_ses_filter_via_bids_filters(self, tree: BIDSTree):
        tree.add_confound(kind="phonemic", dyad=DYAD, ses="1", task=TASK)
        tree.add_confound(kind="phonemic", dyad=DYAD, ses="2", task=TASK)
        layout = BIDSLayout(tree.root)
        results = layout.find.confounds(
            dyad=DYAD, kind="phonemic", bids_filters=["ses-1"]
        )
        assert len(results) == 1
        assert results[0].ses == "1"

    def test_desc_none_excludes_variants(self, tree: BIDSTree):
        tree.add_confound(kind="phonemic", dyad=DYAD, task=TASK, desc="onset")
        tree.add_confound(kind="phonemic", dyad=DYAD, task=TASK)
        layout = BIDSLayout(tree.root)
        results = layout.find.confounds(dyad=DYAD, kind="phonemic")
        assert len(results) == 1
        assert results[0].entities.get("desc") is None

    def test_desc_label_selects_variant(self, tree: BIDSTree):
        tree.add_confound(kind="phonemic", dyad=DYAD, task=TASK, desc="onset")
        tree.add_confound(kind="phonemic", dyad=DYAD, task=TASK)
        layout = BIDSLayout(tree.root)
        results = layout.find.confounds(dyad=DYAD, kind="phonemic", desc="onset")
        assert len(results) == 1
        assert results[0].entities.get("desc") == "onset"

    def test_desc_star_returns_all_variants(self, tree: BIDSTree):
        tree.add_confound(kind="phonemic", dyad=DYAD, task=TASK, run="1")
        tree.add_confound(kind="phonemic", dyad=DYAD, task=TASK, run="2", desc="onset")
        layout = BIDSLayout(tree.root)
        results = layout.find.confounds(dyad=DYAD, kind="phonemic", desc="*")
        descs = {r.entities.get("desc") for r in results}
        assert descs == {None, "onset"}

    def test_rejects_reserved_desc_filter(self, tree: BIDSTree):
        layout = BIDSLayout(tree.root)
        with pytest.raises(ValueError, match="desc"):
            layout.find.confounds(
                dyad=DYAD, kind="phonemic", bids_filters=["desc-onset"]
            )

    def test_raises_when_area_absent(self, tmp_path):
        layout = BIDSLayout(tmp_path)
        with pytest.raises(FileNotFoundError, match="confounds"):
            layout.find.confounds(dyad=DYAD, kind="phonemic")

    def test_raises_when_task_missing(self, tree: BIDSTree):
        path = (
            tree.confounds_dir
            / f"dyad-{DYAD}"
            / "phonemic"
            / f"dyad-{DYAD}_conf-phonemic.parquet"
        )
        path.parent.mkdir(parents=True)
        path.touch()
        layout = BIDSLayout(tree.root)
        with pytest.raises(ValueError, match="task"):
            layout.find.confounds(dyad=DYAD, kind="phonemic")


class TestFindNuisance:
    def test_translates_kind_to_nuisance_entity(self, tree: BIDSTree):
        tree.add_nuisance(kind="physio", sub=SUB, task=TASK)
        layout = BIDSLayout(tree.root)
        results = layout.find.nuisance(sub=SUB, kind="physio")
        assert len(results) == 1
        assert results[0].entities.get("nuis") == "physio"
        assert results[0].suffix == "timeseries"
        assert results[0].ext == ".tsv"

    def test_ses_filter_via_bids_filters(self, tree: BIDSTree):
        tree.add_nuisance(kind="physio", sub=SUB, ses="1", task=TASK)
        tree.add_nuisance(kind="physio", sub=SUB, ses="2", task=TASK)
        layout = BIDSLayout(tree.root)
        results = layout.find.nuisance(sub=SUB, kind="physio", bids_filters=["ses-1"])
        assert len(results) == 1
        assert results[0].ses == "1"

    def test_desc_none_excludes_variants(self, tree: BIDSTree):
        tree.add_nuisance(kind="physio", sub=SUB, task=TASK, desc="v1")
        tree.add_nuisance(kind="physio", sub=SUB, task=TASK)
        layout = BIDSLayout(tree.root)
        results = layout.find.nuisance(sub=SUB, kind="physio")
        assert len(results) == 1
        assert results[0].entities.get("desc") is None

    def test_desc_label_selects_variant(self, tree: BIDSTree):
        tree.add_nuisance(kind="physio", sub=SUB, task=TASK, desc="v1")
        tree.add_nuisance(kind="physio", sub=SUB, task=TASK)
        layout = BIDSLayout(tree.root)
        results = layout.find.nuisance(sub=SUB, kind="physio", desc="v1")
        assert len(results) == 1
        assert results[0].entities.get("desc") == "v1"

    def test_desc_star_returns_all_variants(self, tree: BIDSTree):
        tree.add_nuisance(kind="physio", sub=SUB, task=TASK, run="1")
        tree.add_nuisance(kind="physio", sub=SUB, task=TASK, run="2", desc="v1")
        layout = BIDSLayout(tree.root)
        results = layout.find.nuisance(sub=SUB, kind="physio", desc="*")
        descs = {r.entities.get("desc") for r in results}
        assert descs == {None, "v1"}

    def test_rejects_reserved_desc_filter(self, tree: BIDSTree):
        layout = BIDSLayout(tree.root)
        with pytest.raises(ValueError, match="desc"):
            layout.find.nuisance(sub=SUB, kind="physio", bids_filters=["desc-v1"])

    def test_rejects_reserved_nuis_filter(self, tree: BIDSTree):
        layout = BIDSLayout(tree.root)
        with pytest.raises(ValueError, match="nuis"):
            layout.find.nuisance(sub=SUB, kind="physio", bids_filters=["nuis-physio"])

    def test_raises_when_area_absent(self, tmp_path):
        layout = BIDSLayout(tmp_path)
        with pytest.raises(FileNotFoundError, match="nuisance"):
            layout.find.nuisance(sub=SUB, kind="physio")

    def test_raises_when_task_missing(self, tree: BIDSTree):
        path = (
            tree.nuisance_dir
            / f"sub-{SUB}"
            / "physio"
            / f"sub-{SUB}_nuis-physio_timeseries.tsv"
        )
        path.parent.mkdir(parents=True)
        path.touch()
        layout = BIDSLayout(tree.root)
        with pytest.raises(ValueError, match="task"):
            layout.find.nuisance(sub=SUB, kind="physio")

    def test_skips_non_timeseries_suffix(self, tree: BIDSTree):
        tree.add_nuisance(kind="physio", sub=SUB, task=TASK)
        stray = (
            tree.nuisance_dir
            / f"sub-{SUB}"
            / "physio"
            / f"sub-{SUB}_task-{TASK}_nuis-physio_bold.tsv"
        )
        stray.touch()
        layout = BIDSLayout(tree.root)
        results = layout.find.nuisance(sub=SUB, kind="physio")
        assert len(results) == 1
        assert results[0].suffix == "timeseries"

    def test_run_identity_resolves_to_single_file(self, tree: BIDSTree):
        tree.add_nuisance(kind="physio", sub=SUB, task=TASK, run="1")
        tree.add_nuisance(kind="physio", sub=SUB, task=TASK, run="2")
        layout = BIDSLayout(tree.root)
        results = layout.find.nuisance(
            sub=SUB, kind="physio", bids_filters=["task-conv", "run-1"]
        )
        assert len(results) == 1
        assert results[0].run == "1"


class TestFindFmriprep:
    def test_returns_bold_files(self, tree: BIDSTree):
        tree.add_bold(space=SPACE, sub=SUB, task=TASK)
        layout = BIDSLayout(tree.root)
        results = layout.find.fmriprep(sub=SUB, suffix="bold", ext=".nii.gz")
        assert len(results) == 1
        assert results[0].entities.get("space") == SPACE
        assert results[0].entities.get("desc") == "preproc"

    def test_skips_non_bold_suffix(self, tree: BIDSTree):
        tree.add_bold(space=SPACE, sub=SUB, task=TASK)
        tree.add_brain_mask(sub=SUB, task=TASK, space=SPACE)
        layout = BIDSLayout(tree.root)
        results = layout.find.fmriprep(sub=SUB, suffix="bold", ext=".nii.gz")
        assert len(results) == 1
        assert results[0].path.name.endswith("_bold.nii.gz")

    def test_ses_filter_via_bids_filters(self, tree: BIDSTree):
        tree.add_bold(space=SPACE, sub=SUB, ses="1", task=TASK)
        tree.add_bold(space=SPACE, sub=SUB, ses="2", task=TASK)
        layout = BIDSLayout(tree.root)
        results = layout.find.fmriprep(
            sub=SUB, suffix="bold", ext=".nii.gz", bids_filters=["ses-1"]
        )
        assert len(results) == 1
        assert results[0].ses == "1"

    def test_space_filter_via_bids_filters(self, tree: BIDSTree):
        tree.add_bold(space=SPACE, sub=SUB, task=TASK)
        tree.add_bold(space="T1w", sub=SUB, task="rest")
        layout = BIDSLayout(tree.root)
        results = layout.find.fmriprep(
            sub=SUB, suffix="bold", ext=".nii.gz", bids_filters=[f"space-{SPACE}"]
        )
        assert len(results) == 1
        assert results[0].entities.get("space") == SPACE

    def test_desc_filter_via_bids_filters(self, tree: BIDSTree):
        tree.add_bold(space=SPACE, desc="alt", area="fmriprep", sub=SUB, task=TASK)
        tree.add_bold(space=SPACE, sub=SUB, task="rest")
        layout = BIDSLayout(tree.root)
        results = layout.find.fmriprep(
            sub=SUB, suffix="bold", ext=".nii.gz", bids_filters=["desc-alt"]
        )
        assert len(results) == 1
        assert results[0].entities.get("desc") == "alt"

    def test_run_filter_via_bids_filters(self, tree: BIDSTree):
        tree.add_bold(space=SPACE, sub=SUB, task=TASK, run="1")
        tree.add_bold(space=SPACE, sub=SUB, task=TASK, run="2")
        layout = BIDSLayout(tree.root)
        results = layout.find.fmriprep(
            sub=SUB, suffix="bold", ext=".nii.gz", bids_filters=["run-1"]
        )
        assert len(results) == 1
        assert results[0].entities.get("run") == "1"

    def test_rejects_reserved_sub_filter(self, tree: BIDSTree):
        layout = BIDSLayout(tree.root)
        with pytest.raises(ValueError, match="sub"):
            layout.find.fmriprep(
                sub=SUB, suffix="bold", ext=".nii.gz", bids_filters=["sub-001"]
            )

    def test_cross_session_aggregation_sorted(self, tree: BIDSTree):
        tree.add_bold(space=SPACE, sub=SUB, ses="2", task=TASK)
        tree.add_bold(space=SPACE, sub=SUB, ses="1", task=TASK)
        layout = BIDSLayout(tree.root)
        results = layout.find.fmriprep(sub=SUB, suffix="bold", ext=".nii.gz")
        assert len(results) == 2
        assert results == sorted(results)

    def test_raises_if_area_absent(self, tmp_path: Path):
        layout = BIDSLayout(tmp_path)
        with pytest.raises(FileNotFoundError, match="fmriprep"):
            layout.find.fmriprep(sub=SUB, suffix="bold", ext=".nii.gz")

    def test_raises_when_sub_absent(self, tree: BIDSTree):
        tree.add_bold(space=SPACE, sub=SUB, task=TASK)
        layout = BIDSLayout(tree.root)
        with pytest.raises(FileNotFoundError, match="sub-999"):
            layout.find.fmriprep(sub="999", suffix="bold", ext=".nii.gz")

    def test_suffix_mismatch_message(self, tree: BIDSTree):
        tree.add_bold(space=SPACE, sub=SUB, task=TASK)
        layout = BIDSLayout(tree.root)
        with pytest.raises(FileNotFoundError, match="boldref"):
            layout.find.fmriprep(sub=SUB, suffix="boldref", ext=".nii.gz")

    def test_structural_filter_mismatch_uses_tiered_diagnostic(self, tree: BIDSTree):
        # Asserts on `_diagnose_lookup`'s message, not the descriptive helper's
        tree.add_bold(space=SPACE, sub=SUB, task=TASK)
        layout = BIDSLayout(tree.root)
        with pytest.raises(FileNotFoundError) as exc:
            layout.find.fmriprep(
                sub=SUB,
                suffix="bold",
                ext=".nii.gz",
                bids_filters=["space-Bogus"],
            )
        assert "descriptive filters" not in str(exc.value)
        assert "space-Bogus" in str(exc.value)

    def test_descriptive_filter_via_task_escape_hatch(self, tree: BIDSTree):
        # fmriprep BOLD has no segment entity; `task-<value>` escape hatch
        # attaches run-level metadata for the descriptive filter to resolve against
        tree.add_bold(space=SPACE, sub=SUB, task=TASK, run="1")
        tree.add_bold(space=SPACE, sub=SUB, task=TASK, run="2")
        for run, cond in [("1", "R"), ("2", "L")]:
            tree.add_events(
                sub=SUB,
                task=TASK,
                run=run,
                rows=[{"trial_type": f"task-{TASK}", "onset": 0.0, "duration": 100.0}],
                sidecar_json={
                    "trial_type": {
                        "Levels": {f"task-{TASK}": {"metadata": {"cond": cond}}}
                    }
                },
            )
        layout = BIDSLayout(tree.root)
        results = layout.find.fmriprep(
            sub=SUB, suffix="bold", ext=".nii.gz", bids_filters=["cond-R"]
        )
        assert len(results) == 1
        assert results[0].entities.get("run") == "1"


class TestPathRaw:
    def test_derives_output_path(self, tmp_path: Path):
        layout = BIDSLayout(tmp_path)
        source = BIDSPath(f"sub-{SUB}_task-{TASK}_bold.nii.gz")
        out = layout.path.raw(source=source, suffix="events", ext=".tsv")
        assert out.entities.get("sub") == SUB
        assert out.entities.get("task") == TASK
        assert out.path.name.endswith("_events.tsv")
        assert "func" in out.path.parts
        assert f"sub-{SUB}" in out.path.parts

    def test_omits_ses_dir_when_source_has_no_ses(self, tmp_path: Path):
        layout = BIDSLayout(tmp_path)
        source = BIDSPath(f"sub-{SUB}_task-{TASK}_bold.nii.gz")
        out = layout.path.raw(source=source, suffix="events", ext=".tsv")
        assert not any(p.startswith("ses-") for p in out.path.parts)

    def test_includes_ses_dir_when_source_has_ses(self, tmp_path: Path):
        layout = BIDSLayout(tmp_path)
        source = BIDSPath(f"sub-{SUB}_ses-{SES}_task-{TASK}_bold.nii.gz")
        out = layout.path.raw(source=source, suffix="events", ext=".tsv")
        assert f"ses-{SES}" in out.path.parts

    def test_strips_non_identity_entities(self, tmp_path: Path):
        # fmriprep source carries space/desc — must not appear in raw output
        layout = BIDSLayout(tmp_path)
        source = BIDSPath(
            f"sub-{SUB}_task-{TASK}_space-{SPACE}_desc-preproc_bold.nii.gz"
        )
        out = layout.path.raw(source=source, suffix="events", ext=".tsv")
        assert out.entities.get("space") is None
        assert out.entities.get("desc") is None

    def test_run_entity_preserved(self, tmp_path: Path):
        layout = BIDSLayout(tmp_path)
        source = BIDSPath(f"sub-{SUB}_task-{TASK}_run-1_bold.nii.gz")
        out = layout.path.raw(source=source, suffix="events", ext=".tsv")
        assert out.entities.get("run") == "1"

    def test_compound_extension(self, tmp_path: Path):
        layout = BIDSLayout(tmp_path)
        source = BIDSPath(f"sub-{SUB}_task-{TASK}_bold.nii.gz")
        out = layout.path.raw(source=source, suffix="physio", ext=".tsv.gz")
        assert out.path.name.endswith("_physio.tsv.gz")

    def test_invalid_suffix_raises(self, tmp_path: Path):
        layout = BIDSLayout(tmp_path)
        source = BIDSPath(f"sub-{SUB}_task-{TASK}_bold.nii.gz")
        with pytest.raises(ValueError, match="suffix"):
            layout.path.raw(source=source, suffix="bad!", ext=".tsv")

    def test_invalid_extension_raises(self, tmp_path: Path):
        layout = BIDSLayout(tmp_path)
        source = BIDSPath(f"sub-{SUB}_task-{TASK}_bold.nii.gz")
        with pytest.raises(ValueError, match="extension"):
            layout.path.raw(source=source, suffix="events", ext="tsv")


class TestPathDenoised:
    def test_roots_under_hypline_derivatives_func(self, tmp_path: Path):
        layout = BIDSLayout(tmp_path)
        source = BIDSPath(
            f"sub-{SUB}_task-{TASK}_space-{SPACE}_desc-preproc_bold.nii.gz"
        )
        out = layout.path.denoised(source=source)
        assert out.path.parent == (
            tmp_path / "derivatives" / "hypline" / f"sub-{SUB}" / "func"
        )

    def test_swaps_preproc_to_denoised(self, tmp_path: Path):
        layout = BIDSLayout(tmp_path)
        source = BIDSPath(
            f"sub-{SUB}_task-{TASK}_space-{SPACE}_desc-preproc_bold.nii.gz"
        )
        out = layout.path.denoised(source=source)
        assert out.entities.get("desc") == "denoised"

    def test_preserves_volume_suffix_and_ext(self, tmp_path: Path):
        layout = BIDSLayout(tmp_path)
        source = BIDSPath(
            f"sub-{SUB}_task-{TASK}_space-{SPACE}_desc-preproc_bold.nii.gz"
        )
        out = layout.path.denoised(source=source)
        assert out.suffix == "bold"
        assert out.path.name.endswith("_bold.nii.gz")

    def test_preserves_space_and_hemi_for_surface(self, tmp_path: Path):
        # `hemi` is in neither the identity nor variant-descriptor sets; a
        # filter-to-known-set deriver would drop it and collide L/R. Preserve-all
        # keeps them distinct.
        layout = BIDSLayout(tmp_path)
        source = BIDSPath(
            f"sub-{SUB}_task-{TASK}_space-fsaverage6_hemi-L_desc-preproc_bold.func.gii"
        )
        out = layout.path.denoised(source=source)
        assert out.entities.get("space") == "fsaverage6"
        assert out.entities.get("hemi") == "L"
        assert out.entities.get("desc") == "denoised"
        assert out.path.name.endswith("_bold.func.gii")

    def test_omits_ses_dir_when_source_has_no_ses(self, tmp_path: Path):
        layout = BIDSLayout(tmp_path)
        source = BIDSPath(f"sub-{SUB}_task-{TASK}_desc-preproc_bold.nii.gz")
        out = layout.path.denoised(source=source)
        assert not any(p.startswith("ses-") for p in out.path.parts)

    def test_includes_ses_dir_when_source_has_ses(self, tmp_path: Path):
        layout = BIDSLayout(tmp_path)
        source = BIDSPath(f"sub-{SUB}_ses-{SES}_task-{TASK}_desc-preproc_bold.nii.gz")
        out = layout.path.denoised(source=source)
        assert f"ses-{SES}" in out.path.parts

    def test_run_entity_preserved(self, tmp_path: Path):
        layout = BIDSLayout(tmp_path)
        source = BIDSPath(f"sub-{SUB}_task-{TASK}_run-1_desc-preproc_bold.nii.gz")
        out = layout.path.denoised(source=source)
        assert out.entities.get("run") == "1"


class TestBidsUri:
    def test_renders_area_relative_uri(self, tmp_path: Path):
        layout = BIDSLayout(tmp_path)
        fmriprep = tmp_path / "derivatives" / "fmriprep" / f"sub-{SUB}" / "func"
        bold = BIDSPath(fmriprep / f"sub-{SUB}_task-{TASK}_desc-preproc_bold.nii.gz")
        assert layout.bids_uri(bold, area="fmriprep") == (
            f"bids:fmriprep:sub-{SUB}/func/"
            f"sub-{SUB}_task-{TASK}_desc-preproc_bold.nii.gz"
        )

    def test_raises_when_bold_outside_area(self, tmp_path: Path):
        layout = BIDSLayout(tmp_path)
        bold = BIDSPath(tmp_path / "elsewhere" / f"sub-{SUB}_bold.nii.gz")
        with pytest.raises(ValueError):
            layout.bids_uri(bold, area="fmriprep")


class TestStampDatasetDescription:
    def _read(self, tmp_path: Path) -> dict:
        path = tmp_path / "derivatives" / "hypline" / "dataset_description.json"
        return json.loads(path.read_text())

    def test_writes_minimal_compliant_header_with_derived_links(self, tmp_path: Path):
        layout = BIDSLayout(tmp_path)
        layout.stamp_dataset_description(area="hypline", sources=["fmriprep"])
        # link is derived from the area roots, not hand-typed
        assert self._read(tmp_path) == {
            "Name": "hypline",
            "BIDSVersion": "1.9.0",
            "DatasetType": "derivative",
            "GeneratedBy": [{"Name": "hypline", "Version": __version__}],
            "DatasetLinks": {"fmriprep": "../fmriprep"},
        }

    def test_omits_dataset_links_when_no_sources(self, tmp_path: Path):
        layout = BIDSLayout(tmp_path)
        layout.stamp_dataset_description(area="hypline", sources=[])
        assert "DatasetLinks" not in self._read(tmp_path)

    def test_creates_parent_dirs(self, tmp_path: Path):
        layout = BIDSLayout(tmp_path)
        layout.stamp_dataset_description(area="hypline", sources=["fmriprep"])
        assert (
            tmp_path / "derivatives" / "hypline" / "dataset_description.json"
        ).exists()

    def test_leaves_existing_untouched(self, tmp_path: Path):
        layout = BIDSLayout(tmp_path)
        path = tmp_path / "derivatives" / "hypline" / "dataset_description.json"
        path.parent.mkdir(parents=True)
        path.write_text('{"Name": "preexisting"}')
        layout.stamp_dataset_description(area="hypline", sources=["fmriprep"])
        assert json.loads(path.read_text()) == {"Name": "preexisting"}


class TestPathStimulus:
    def test_derives_output_path(self, tmp_path: Path):
        layout = BIDSLayout(tmp_path)
        source = BIDSPath(f"dyad-{DYAD}_task-{TASK}_stim-audio.wav")
        out = layout.path.stimulus(kind="transcript", source=source, ext=".csv")
        assert out.entities.get("stim") == "transcript"
        assert out.path.name.endswith(".csv")
        assert "stimuli" in out.path.parts
        assert f"dyad-{DYAD}" in out.path.parts
        assert "transcript" in out.path.parts

    def test_omits_ses_dir_when_source_has_no_ses(self, tmp_path: Path):
        layout = BIDSLayout(tmp_path)
        source = BIDSPath(f"dyad-{DYAD}_task-{TASK}_stim-audio.wav")
        out = layout.path.stimulus(kind="transcript", source=source, ext=".csv")
        assert "stimuli" in out.path.parts
        assert f"dyad-{DYAD}" in out.path.parts
        assert not any(p.startswith("ses-") for p in out.path.parts)
        assert "transcript" in out.path.parts

    def test_includes_ses_dir_when_source_has_ses(self, tmp_path: Path):
        layout = BIDSLayout(tmp_path)
        source = BIDSPath(f"dyad-{DYAD}_ses-{SES}_task-{TASK}_stim-audio.wav")
        out = layout.path.stimulus(kind="transcript", source=source, ext=".csv")
        assert "stimuli" in out.path.parts
        assert f"dyad-{DYAD}" in out.path.parts
        assert f"ses-{SES}" in out.path.parts
        assert "transcript" in out.path.parts

    def test_invalid_extension_raises(self, tmp_path: Path):
        layout = BIDSLayout(tmp_path)
        source = BIDSPath(f"dyad-{DYAD}_task-{TASK}_stim-audio.wav")
        with pytest.raises(ValueError, match="extension"):
            layout.path.stimulus(kind="transcript", source=source, ext="csv")

    def test_source_with_desc_raises(self, tmp_path: Path):
        layout = BIDSLayout(tmp_path)
        source = BIDSPath(f"dyad-{DYAD}_task-{TASK}_stim-audio_desc-v2.wav")
        with pytest.raises(ValueError, match="no variants"):
            layout.path.stimulus(kind="transcript", source=source, ext=".csv")


class TestPathFeature:
    def test_derives_output_path(self, tmp_path: Path):
        layout = BIDSLayout(tmp_path)
        source = BIDSPath(f"dyad-{DYAD}_task-{TASK}_stim-transcript.csv")
        out = layout.path.feature(source=source, kind="phonemic")
        assert out.entities.get("feat") == "phonemic"
        assert out.path.suffix == ".parquet"
        assert "features" in out.path.parts
        assert f"dyad-{DYAD}" in out.path.parts
        assert "phonemic" in out.path.parts

    def test_applies_entity_override(self, tmp_path: Path):
        layout = BIDSLayout(tmp_path)
        source = BIDSPath(f"dyad-{DYAD}_task-{TASK}_stim-transcript.csv")
        out = layout.path.feature(source=source, kind="phonemic", desc="gpt3")
        assert out.entities.get("desc") == "gpt3"
        assert out.path.name.endswith("_feat-phonemic_desc-gpt3.parquet")
        assert "phonemic-gpt3" in out.path.parts
        assert "phonemic" not in out.path.parts

    def test_omits_ses_dir_when_source_has_no_ses(self, tmp_path: Path):
        layout = BIDSLayout(tmp_path)
        source = BIDSPath(f"dyad-{DYAD}_task-{TASK}_stim-transcript.csv")
        out = layout.path.feature(source=source, kind="phonemic")
        assert "features" in out.path.parts
        assert f"dyad-{DYAD}" in out.path.parts
        assert not any(p.startswith("ses-") for p in out.path.parts)
        assert "phonemic" in out.path.parts

    def test_includes_ses_dir_when_source_has_ses(self, tmp_path: Path):
        layout = BIDSLayout(tmp_path)
        source = BIDSPath(f"dyad-{DYAD}_ses-{SES}_task-{TASK}_stim-transcript.csv")
        out = layout.path.feature(source=source, kind="phonemic")
        assert "features" in out.path.parts
        assert f"dyad-{DYAD}" in out.path.parts
        assert f"ses-{SES}" in out.path.parts
        assert "phonemic" in out.path.parts

    def test_invalid_override_value_raises(self, tmp_path: Path):
        layout = BIDSLayout(tmp_path)
        source = BIDSPath(f"dyad-{DYAD}_task-{TASK}_stim-transcript.csv")
        with pytest.raises(ValueError, match="Invalid BIDS entity"):
            layout.path.feature(source=source, kind="phonemic", desc="BAD!")


class TestPathConfound:
    def test_derives_output_path(self, tmp_path: Path):
        layout = BIDSLayout(tmp_path)
        source = BIDSPath(f"dyad-{DYAD}_task-{TASK}_run-1_feat-phonemic.parquet")
        out = layout.path.confound(source=source, kind="phonemic")
        assert out.entities.get("conf") == "phonemic"
        assert out.entities.get("desc") is None
        assert out.path.suffix == ".parquet"
        assert "confounds" in out.path.parts
        assert f"dyad-{DYAD}" in out.path.parts
        assert "phonemic" in out.path.parts

    def test_sets_desc_entity(self, tmp_path: Path):
        layout = BIDSLayout(tmp_path)
        source = BIDSPath(f"dyad-{DYAD}_task-{TASK}_run-1_feat-phonemic.parquet")
        out = layout.path.confound(source=source, kind="phonemic", desc="onset")
        assert out.entities.get("conf") == "phonemic"
        assert out.entities.get("desc") == "onset"
        assert "desc-onset" in out.path.name

    def test_omits_ses_dir_when_source_has_no_ses(self, tmp_path: Path):
        layout = BIDSLayout(tmp_path)
        source = BIDSPath(f"dyad-{DYAD}_task-{TASK}_run-1_feat-phonemic.parquet")
        out = layout.path.confound(source=source, kind="phonemic", desc="onset")
        assert "confounds" in out.path.parts
        assert f"dyad-{DYAD}" in out.path.parts
        assert not any(p.startswith("ses-") for p in out.path.parts)
        assert "phonemic-onset" in out.path.parts

    def test_includes_ses_dir_when_source_has_ses(self, tmp_path: Path):
        layout = BIDSLayout(tmp_path)
        source = BIDSPath(
            f"dyad-{DYAD}_ses-{SES}_task-{TASK}_run-1_feat-phonemic.parquet"
        )
        out = layout.path.confound(source=source, kind="phonemic", desc="rate")
        assert "confounds" in out.path.parts
        assert f"dyad-{DYAD}" in out.path.parts
        assert f"ses-{SES}" in out.path.parts
        assert "phonemic-rate" in out.path.parts

    def test_invalid_desc_value_raises(self, tmp_path: Path):
        layout = BIDSLayout(tmp_path)
        source = BIDSPath(f"dyad-{DYAD}_task-{TASK}_run-1_feat-phonemic.parquet")
        with pytest.raises(ValueError, match="Invalid BIDS entity"):
            layout.path.confound(source=source, kind="phonemic", desc="BAD!")


class TestListSubjects:
    def test_subjects_fmriprep(self, tree: BIDSTree):
        tree.add_bold(space=SPACE, sub=SUB, task=TASK)
        layout = BIDSLayout(tree.root)
        assert layout.list.subjects(area="fmriprep") == [SUB]

    def test_subjects_skips_non_sub_dirs(self, tree: BIDSTree):
        tree.add_bold(space=SPACE, sub=SUB, task=TASK)
        non_sub_dir = tree.root / "derivatives" / "fmriprep" / "logs"
        non_sub_dir.mkdir(parents=True, exist_ok=True)
        layout = BIDSLayout(tree.root)
        assert layout.list.subjects(area="fmriprep") == [SUB]

    def test_subjects_empty_if_area_absent(self, tmp_path: Path):
        layout = BIDSLayout(tmp_path)
        assert layout.list.subjects(area="stimuli") == []


class TestListDyads:
    def test_dyads_stimuli(self, tree: BIDSTree):
        (tree.stimuli_dir / "dyad-001").mkdir(parents=True)
        (tree.stimuli_dir / "dyad-002").mkdir(parents=True)
        layout = BIDSLayout(tree.root)
        assert layout.list.dyads(area="stimuli") == ["001", "002"]

    def test_dyads_features(self, tree: BIDSTree):
        tree.add_feature(kind="phonemic", dyad=DYAD, task=TASK)
        layout = BIDSLayout(tree.root)
        assert layout.list.dyads(area="features") == [DYAD]

    def test_dyads_skips_non_dyad_dirs(self, tree: BIDSTree):
        (tree.stimuli_dir / "dyad-001").mkdir(parents=True)
        (tree.stimuli_dir / "sub-001").mkdir(parents=True)
        layout = BIDSLayout(tree.root)
        assert layout.list.dyads(area="stimuli") == ["001"]

    def test_dyads_empty_if_area_absent(self, tmp_path: Path):
        layout = BIDSLayout(tmp_path)
        assert layout.list.dyads(area="stimuli") == []


class TestParticipants:
    def test_dyad_of(self, tree: BIDSTree):
        tree.add_participants({"001": "001", "002": "001", "003": "002"})
        layout = BIDSLayout(tree.root)
        assert layout.dyad_of("001") == "001"
        assert layout.dyad_of("003") == "002"

    def test_subjects_of_sorted(self, tree: BIDSTree):
        tree.add_participants({"002": "001", "001": "001"})
        layout = BIDSLayout(tree.root)
        assert layout.subjects_of("001") == ["001", "002"]

    def test_dyad_of_unknown_sub_raises(self, tree: BIDSTree):
        tree.add_participants({"001": "001"})
        layout = BIDSLayout(tree.root)
        with pytest.raises(KeyError, match="sub-999"):
            layout.dyad_of("999")

    def test_subjects_of_unknown_dyad_raises(self, tree: BIDSTree):
        tree.add_participants({"001": "001"})
        layout = BIDSLayout(tree.root)
        with pytest.raises(KeyError, match="dyad-999"):
            layout.subjects_of("999")

    def test_missing_file_raises(self, tmp_path: Path):
        layout = BIDSLayout(tmp_path)
        with pytest.raises(FileNotFoundError, match="participants.tsv"):
            layout.dyad_of("001")

    def test_duplicate_participant_id_raises(self, tree: BIDSTree):
        path = tree.root / "participants.tsv"
        path.write_text(
            "participant_id\tdyad_id\nsub-001\tdyad-001\nsub-001\tdyad-002\n"
        )
        layout = BIDSLayout(tree.root)
        with pytest.raises(ValueError, match="Duplicate participant_id"):
            layout.dyad_of("001")

    def test_missing_column_raises(self, tree: BIDSTree):
        path = tree.root / "participants.tsv"
        path.write_text("participant_id\nsub-001\n")
        layout = BIDSLayout(tree.root)
        with pytest.raises(ValueError, match="dyad_id"):
            layout.dyad_of("001")

    def test_mapping_read_once(self, tree: BIDSTree):
        tree.add_participants({"001": "001"})
        layout = BIDSLayout(tree.root)
        assert layout.dyad_of("001") == "001"
        (tree.root / "participants.tsv").unlink()
        assert layout.dyad_of("001") == "001"
