import numpy as np
import polars as pl
import pytest

from hypline.confounds import PhonemicConfound
from hypline.confounds._utils import pick_timing_source
from hypline.io import read_confound, read_confound_metadata
from hypline.layout import BIDSLayout

from ..conftest import DEFAULT_BOLD_N_TRS, BIDSTree

SUB = "001"
TASK = "story"
TR = 2.0


def _add_phonemic(
    tree: BIDSTree,
    *,
    start_times: list[float],
    phonemes: list[str | None],
    sub: str = SUB,
    ses: str | None = None,
    task: str = TASK,
    run: str = "1",
    desc: str | None = None,
    extra_entities: dict[str, str] | None = None,
):
    """Write a phonemic feature mirroring `PhonemicFeature.generate`:
    one row per phoneme, multiple rows may share a `start_time`.
    """
    assert len(start_times) == len(phonemes)

    df = pl.DataFrame(
        {
            "start_time": start_times,
            "phoneme": phonemes,
            "feature": [[0.0]] * len(start_times),
        },
        schema={
            "start_time": pl.Float64,
            "phoneme": pl.Utf8,
            "feature": pl.Array(pl.Float64, 1),
        },
    )

    return tree.add_feature(
        sub=sub,
        ses=ses,
        task=task,
        run=run,
        kind="phonemic",
        df=df,
        desc=desc,
        extra_entities=extra_entities,
    )


class TestPhonemicConfoundGenerate:
    def test_writes_onset_and_rate_files(self, tree: BIDSTree):
        _add_phonemic(tree, start_times=[0.0, 0.0, 0.0], phonemes=["K", "AE", "T"])
        tree.add_bold(sub=SUB, task=TASK, run="1", space="MNI152NLin6Asym", tr=TR)
        layout = BIDSLayout(tree.root)

        PhonemicConfound(bids_root=tree.root).generate(SUB)

        onset = layout.path.confound(
            source=layout.find.features(sub=SUB, kind="phonemic")[0],
            kind="phonemic",
            desc="onset",
        )
        rate = layout.path.confound(
            source=layout.find.features(sub=SUB, kind="phonemic")[0],
            kind="phonemic",
            desc="rate",
        )
        assert onset.path.exists()
        assert rate.path.exists()

    def test_onset_marks_tr_with_any_event(self, tree: BIDSTree):
        # Events at 0.5s → TR 0, 5.0s → TR 2
        _add_phonemic(
            tree,
            start_times=[0.5, 0.5, 0.5, 5.0, 5.0, 5.0],
            phonemes=["K", "AE", "T", "D", "AO", "G"],
        )
        tree.add_bold(sub=SUB, task=TASK, run="1", space="MNI152NLin6Asym", tr=TR)
        layout = BIDSLayout(tree.root)

        PhonemicConfound(bids_root=tree.root).generate(SUB)

        onset_path = layout.path.confound(
            source=layout.find.features(sub=SUB, kind="phonemic")[0],
            kind="phonemic",
            desc="onset",
        )
        df = read_confound(onset_path.path)
        confound = np.array(df.get_column("confound").to_list()).ravel()
        expected = np.zeros(DEFAULT_BOLD_N_TRS)
        expected[[0, 2]] = 1.0
        np.testing.assert_array_equal(confound, expected)

    def test_rate_counts_phoneme_rows_per_tr(self, tree: BIDSTree):
        # 3 phonemes at TR 0, 3 phonemes at TR 2
        _add_phonemic(
            tree,
            start_times=[0.5, 0.5, 0.5, 5.0, 5.0, 5.0],
            phonemes=["K", "AE", "T", "D", "AO", "G"],
        )
        tree.add_bold(sub=SUB, task=TASK, run="1", space="MNI152NLin6Asym", tr=TR)
        layout = BIDSLayout(tree.root)

        PhonemicConfound(bids_root=tree.root).generate(SUB)

        rate_path = layout.path.confound(
            source=layout.find.features(sub=SUB, kind="phonemic")[0],
            kind="phonemic",
            desc="rate",
        )
        df = read_confound(rate_path.path)
        confound = np.array(df.get_column("confound").to_list()).ravel()
        expected = np.zeros(DEFAULT_BOLD_N_TRS)
        expected[0] = 3.0
        expected[2] = 3.0
        np.testing.assert_array_equal(confound, expected)

    def test_records_tr_method_in_metadata(self, tree: BIDSTree):
        _add_phonemic(tree, start_times=[0.0], phonemes=["K"])
        tree.add_bold(sub=SUB, task=TASK, run="1", space="MNI152NLin6Asym", tr=TR)
        layout = BIDSLayout(tree.root)

        PhonemicConfound(bids_root=tree.root).generate(SUB)

        feat = layout.find.features(sub=SUB, kind="phonemic")[0]
        onset_meta = read_confound_metadata(
            layout.path.confound(source=feat, kind="phonemic", desc="onset").path
        )
        rate_meta = read_confound_metadata(
            layout.path.confound(source=feat, kind="phonemic", desc="rate").path
        )
        assert onset_meta["tr_method"] == "any"
        assert rate_meta["tr_method"] == "count"
        assert onset_meta["repetition_time"] == TR
        assert onset_meta["n_trs"] == DEFAULT_BOLD_N_TRS

    def test_start_time_grid_starts_at_zero_and_uses_tr_spacing(self, tree: BIDSTree):
        _add_phonemic(tree, start_times=[0.0], phonemes=["K"])
        tree.add_bold(sub=SUB, task=TASK, run="1", space="MNI152NLin6Asym", tr=TR)
        layout = BIDSLayout(tree.root)

        PhonemicConfound(bids_root=tree.root).generate(SUB)

        feat = layout.find.features(sub=SUB, kind="phonemic")[0]
        df = read_confound(
            layout.path.confound(source=feat, kind="phonemic", desc="onset").path
        )
        start = df.get_column("start_time").to_numpy()
        np.testing.assert_allclose(start, np.arange(DEFAULT_BOLD_N_TRS) * TR)

    def test_uses_segment_n_trs_for_segmented_runs(self, tree: BIDSTree):
        # Two trials: trial-1 covers TRs 0..2 (0–6s), trial-2 covers TRs 3..5
        _add_phonemic(
            tree,
            start_times=[0.0, 0.0],
            phonemes=["K", "AE"],
            extra_entities={"trial": "1"},
        )
        _add_phonemic(
            tree,
            start_times=[0.0, 0.0],
            phonemes=["D", "AO"],
            extra_entities={"trial": "2"},
        )
        tree.add_bold(sub=SUB, task=TASK, run="1", space="MNI152NLin6Asym", tr=TR)
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=[
                {"trial_type": "trial-1", "onset": 0.0, "duration": 6.0},
                {"trial_type": "trial-2", "onset": 6.0, "duration": 6.0},
            ],
        )
        layout = BIDSLayout(tree.root)

        PhonemicConfound(bids_root=tree.root).generate(SUB)

        for trial in ("1", "2"):
            feat = next(
                f
                for f in layout.find.features(sub=SUB, kind="phonemic")
                if f.entities.get("trial") == trial
            )
            meta = read_confound_metadata(
                layout.path.confound(source=feat, kind="phonemic", desc="onset").path
            )
            assert meta["n_trs"] == 3

    def test_unsegmented_run_uses_bold_n_trs(self, tree: BIDSTree):
        _add_phonemic(tree, start_times=[0.0], phonemes=["K"])
        tree.add_bold(sub=SUB, task=TASK, run="1", space="MNI152NLin6Asym", tr=TR)
        layout = BIDSLayout(tree.root)

        PhonemicConfound(bids_root=tree.root).generate(SUB)

        feat = layout.find.features(sub=SUB, kind="phonemic")[0]
        meta = read_confound_metadata(
            layout.path.confound(source=feat, kind="phonemic", desc="onset").path
        )
        assert meta["n_trs"] == DEFAULT_BOLD_N_TRS

    def test_desc_variants_share_single_confound_set(self, tree: BIDSTree):
        # Two `desc-*` variants must collapse to one confound set
        _add_phonemic(
            tree,
            start_times=[0.5, 0.5, 5.0],
            phonemes=["K", "AE", "T"],
            desc="v1",
        )
        _add_phonemic(
            tree,
            start_times=[0.5, 0.5, 5.0],
            phonemes=["K", "AE", "T"],
            desc="v2",
        )
        tree.add_bold(sub=SUB, task=TASK, run="1", space="MNI152NLin6Asym", tr=TR)

        PhonemicConfound(bids_root=tree.root).generate(SUB)

        sub_dir = tree.root / "confounds" / f"sub-{SUB}"
        onset_files = sorted((sub_dir / "phonemic-onset").glob("*desc-onset*.parquet"))
        rate_files = sorted((sub_dir / "phonemic-rate").glob("*desc-rate*.parquet"))
        assert len(onset_files) == 1
        assert len(rate_files) == 1

    def test_picks_lexicographically_first_desc(self, tree: BIDSTree):
        # Bare folder (desc=None) sorts before any labeled variant
        _add_phonemic(tree, start_times=[0.5], phonemes=["K"], desc="gpt3")
        _add_phonemic(tree, start_times=[0.5], phonemes=["K"], desc=None)
        layout = BIDSLayout(tree.root)
        files = layout.find.features(sub=SUB, kind="phonemic", desc="*")
        kept = pick_timing_source(files)
        assert len(kept) == 1
        assert kept[0].entities.get("desc") is None

    def test_raises_when_segment_value_unknown(self, tree: BIDSTree):
        _add_phonemic(
            tree,
            start_times=[0.0],
            phonemes=["K"],
            extra_entities={"trial": "9"},
        )
        tree.add_bold(sub=SUB, task=TASK, run="1", space="MNI152NLin6Asym", tr=TR)
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=[
                {"trial_type": "trial-1", "onset": 0.0, "duration": 6.0},
            ],
        )

        with pytest.raises(ValueError, match="trial-9"):
            PhonemicConfound(bids_root=tree.root).generate(SUB)
