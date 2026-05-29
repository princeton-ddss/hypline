import polars as pl
import pytest

from hypline.bids import BIDSPath
from hypline.denoise import Denoiser
from hypline.enums import VolumeSpace

from .conftest import DEFAULT_BOLD_N_TRS, BIDSTree

VOLUME_SPACE = VolumeSpace.MNI_152_NLIN_2009_C_ASYM


def _denoiser(tree: BIDSTree, confounds: list[str], **kwargs) -> Denoiser:
    return Denoiser(
        bids_root=tree.root,
        space=VOLUME_SPACE,
        confounds=confounds,
        **kwargs,
    )


def _array_df(rows: list[list[float]]) -> pl.DataFrame:
    """Build a confound DataFrame with a fixed-width `confound` array column."""
    width = len(rows[0])
    return pl.DataFrame(
        {
            "start_time": [float(i) * 2.0 for i in range(len(rows))],
            "confound": rows,
        },
        schema={"start_time": pl.Float64, "confound": pl.Array(pl.Float64, width)},
    )


class TestLoadConfounds:
    def _add_bold(self, tree: BIDSTree) -> BIDSPath:
        bold_path = tree.add_bold(sub="01", task="A", run="1", space=VOLUME_SPACE.value)
        return BIDSPath(bold_path)

    def test_load_width_one(self, tree: BIDSTree):
        bold = self._add_bold(tree)
        tree.add_confound(
            sub="01", task="A", run="1", kind="motion", df=_array_df([[10.0], [20.0]])
        )
        denoiser = _denoiser(tree, ["motion"])
        loaded = denoiser._load_confounds(bold)
        assert loaded.equals(pl.DataFrame({"motion_0": [10.0, 20.0]}))

    def test_load_width_gt_one(self, tree: BIDSTree):
        bold = self._add_bold(tree)
        tree.add_confound(
            sub="01",
            task="A",
            run="1",
            kind="motion",
            df=_array_df([[10.0, 11.0], [20.0, 21.0]]),
        )
        denoiser = _denoiser(tree, ["motion"])
        loaded = denoiser._load_confounds(bold)
        assert loaded.columns == ["motion_0", "motion_1"]
        assert loaded["motion_1"].to_list() == [11.0, 21.0]

    def test_load_multiple_variants_same_kind(self, tree: BIDSTree):
        bold = self._add_bold(tree)
        tree.add_confound(
            sub="01",
            task="A",
            run="1",
            kind="phonemic",
            desc="rate",
            df=_array_df([[1.0], [2.0]]),
        )
        tree.add_confound(
            sub="01",
            task="A",
            run="1",
            kind="phonemic",
            desc="onset",
            df=_array_df([[3.0], [4.0]]),
        )
        denoiser = _denoiser(
            tree, ["phonemic-rate", "phonemic-onset"]
        )
        loaded = denoiser._load_confounds(bold)
        assert loaded.columns == ["phonemic-rate_0", "phonemic-onset_0"]

    def test_load_concats_distinct_kinds(self, tree: BIDSTree):
        bold = self._add_bold(tree)
        tree.add_confound(
            sub="01", task="A", run="1", kind="motion", df=_array_df([[10.0], [20.0]])
        )
        tree.add_confound(
            sub="01",
            task="A",
            run="1",
            kind="fmriprep",
            desc="minimal",
            df=_array_df([[1.0], [2.0]]),
        )
        denoiser = _denoiser(
            tree, ["motion", "fmriprep-minimal"]
        )
        loaded = denoiser._load_confounds(bold)
        assert loaded.columns == ["motion_0", "fmriprep-minimal_0"]

    def test_duplicate_ref_loaded_once(self, tree: BIDSTree):
        bold = self._add_bold(tree)
        tree.add_confound(
            sub="01", task="A", run="1", kind="motion", df=_array_df([[10.0], [20.0]])
        )
        denoiser = _denoiser(tree, ["motion", "motion"])
        loaded = denoiser._load_confounds(bold)
        assert loaded.columns == ["motion_0"]

    def test_missing_file(self, tree: BIDSTree):
        bold = self._add_bold(tree)
        denoiser = _denoiser(tree, ["motion"])
        with pytest.raises(FileNotFoundError):
            denoiser._load_confounds(bold)

    def test_bundle_row_mismatch(self, tree: BIDSTree):
        bold = self._add_bold(tree)
        tree.add_confound(
            sub="01", task="A", run="1", kind="motion", df=_array_df([[10.0], [20.0]])
        )
        tree.add_confound(
            sub="01",
            task="A",
            run="1",
            kind="fmriprep",
            desc="minimal",
            df=_array_df([[1.0], [2.0], [3.0]]),
        )
        denoiser = _denoiser(
            tree, ["motion", "fmriprep-minimal"]
        )
        with pytest.raises(ValueError, match="Unequal number of rows"):
            denoiser._load_confounds(bold)


class TestDenoise:
    def _confound_df(self, n: int = DEFAULT_BOLD_N_TRS) -> pl.DataFrame:
        return _array_df([[float(i)] for i in range(n)])

    def test_volume_writes_desc_clean(self, tree: BIDSTree):
        tree.add_bold(sub="01", task="A", run="1", space=VOLUME_SPACE.value)
        tree.add_confound(
            sub="01", task="A", run="1", kind="motion", df=self._confound_df()
        )
        denoiser = _denoiser(tree, ["motion"])

        denoiser.denoise("01")

        func_dir = tree.func_dir(sub="01")
        clean = (
            func_dir
            / f"sub-01_task-A_run-1_space-{VOLUME_SPACE.value}_desc-clean_bold.nii.gz"
        )
        assert clean.exists()

        import nibabel as nib
        from nibabel.nifti1 import Nifti1Image

        img = nib.load(clean)
        assert isinstance(img, Nifti1Image)
        assert img.shape[-1] == DEFAULT_BOLD_N_TRS

    def test_tr_count_mismatch_raises(self, tree: BIDSTree):
        tree.add_bold(sub="01", task="A", run="1", space=VOLUME_SPACE.value)
        tree.add_confound(
            sub="01",
            task="A",
            run="1",
            kind="motion",
            df=self._confound_df(DEFAULT_BOLD_N_TRS - 1),
        )
        denoiser = _denoiser(tree, ["motion"])
        with pytest.raises(ValueError, match="Unequal number of TRs"):
            denoiser.denoise("01")

    def test_denoise_ignores_existing_desc_clean(self, tree: BIDSTree):
        tree.add_bold(sub="01", task="A", run="1", space=VOLUME_SPACE.value)
        clean_path = tree.func_dir(sub="01") / (
            f"sub-01_task-A_run-1_space-{VOLUME_SPACE.value}_desc-clean_bold.nii.gz"
        )
        # Sentinel sits at the output path; if denoise treated desc-clean as an
        # input (re-cleaning), it would choke loading this instead of desc-preproc
        clean_path.write_bytes(b"sentinel")
        tree.add_confound(
            sub="01", task="A", run="1", kind="motion", df=self._confound_df()
        )
        denoiser = _denoiser(tree, ["motion"])

        denoiser.denoise("01")

        import nibabel as nib
        from nibabel.nifti1 import Nifti1Image

        # Valid NIfTI means the sentinel was overwritten by real clean output
        img = nib.load(clean_path)
        assert isinstance(img, Nifti1Image)
        assert img.shape[-1] == DEFAULT_BOLD_N_TRS

    def test_bids_filters_reach_finder(self, tree: BIDSTree):
        tree.add_bold(sub="01", task="A", run="1", space=VOLUME_SPACE.value)
        tree.add_bold(sub="01", task="A", run="2", space=VOLUME_SPACE.value)
        tree.add_confound(
            sub="01", task="A", run="1", kind="motion", df=self._confound_df()
        )
        tree.add_confound(
            sub="01", task="A", run="2", kind="motion", df=self._confound_df()
        )
        denoiser = _denoiser(
            tree, ["motion"], bids_filters=["task-A", "run-2"]
        )

        denoiser.denoise("01")

        func_dir = tree.func_dir(sub="01")
        clean_run2 = (
            func_dir
            / f"sub-01_task-A_run-2_space-{VOLUME_SPACE.value}_desc-clean_bold.nii.gz"
        )
        clean_run1 = (
            func_dir
            / f"sub-01_task-A_run-1_space-{VOLUME_SPACE.value}_desc-clean_bold.nii.gz"
        )
        assert clean_run2.exists()
        assert not clean_run1.exists()

    @pytest.mark.parametrize("reserved", ["desc-foo", "space-fsaverage6", "sub-02"])
    def test_reserved_filter_rejected(self, tree: BIDSTree, reserved: str):
        with pytest.raises(ValueError, match="bids_filters cannot contain"):
            _denoiser(tree, ["motion"], bids_filters=[reserved])
