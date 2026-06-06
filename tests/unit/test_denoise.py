import polars as pl
import pytest

from hypline.bids import BIDSPath
from hypline.denoise import Denoiser
from hypline.enums import VolumeSpace

from .conftest import DEFAULT_BOLD_N_TRS, BIDSTree

VOLUME_SPACE = VolumeSpace.MNI_152_NLIN_2009_C_ASYM


def _denoiser(
    tree: BIDSTree,
    columns: list[str] | None = None,
    compcor: list[str] | None = None,
    **kwargs,
) -> Denoiser:
    return Denoiser(
        bids_root=tree.root,
        space=VOLUME_SPACE.value,
        columns=columns or [],
        compcor=compcor or [],
        **kwargs,
    )


def _tsv_df(n: int = DEFAULT_BOLD_N_TRS) -> pl.DataFrame:
    """A confounds tsv frame with literal, group, and CompCor columns."""
    return pl.DataFrame(
        {
            "trans_x": [float(i) for i in range(n)],
            "cosine00": [float(i) for i in range(n)],
            "cosine01": [float(i) * 2 for i in range(n)],
            "a_comp_cor_00": [float(i) * 3 for i in range(n)],
            "a_comp_cor_01": [float(i) * 4 for i in range(n)],
        }
    )


def _tsv_meta() -> dict:
    """JSON sidecar describing two aCompCor CSF components (for compcor tests)."""
    return {
        "a_comp_cor_00": {
            "Method": "aCompCor",
            "Mask": "CSF",
            "Retained": True,
            "SingularValue": 2.0,
        },
        "a_comp_cor_01": {
            "Method": "aCompCor",
            "Mask": "CSF",
            "Retained": True,
            "SingularValue": 1.0,
        },
    }


def _add_run(
    tree: BIDSTree,
    *,
    sub: str = "01",
    task: str = "A",
    run: str = "1",
    df: pl.DataFrame | None = None,
    meta: dict | None = None,
) -> BIDSPath:
    """Lay down a preproc bold plus its matching confounds tsv; return the bold path."""
    bold_path = tree.add_bold(sub=sub, task=task, run=run, space=VOLUME_SPACE.value)
    tree.add_confounds_timeseries(
        sub=sub,
        task=task,
        run=run,
        df=_tsv_df() if df is None else df,
        metadata=_tsv_meta() if meta is None else meta,
    )
    return BIDSPath(bold_path)


class TestLoadNuisance:
    """Path resolution + delegation; selection logic lives in test_fmriprep.py."""

    def test_resolves_and_delegates(self, tree: BIDSTree):
        bold = _add_run(tree)
        loaded = _denoiser(tree, ["trans_x"])._load_nuisance(bold)
        assert loaded.shape == (DEFAULT_BOLD_N_TRS, 1)
        assert loaded[:, 0].tolist() == [float(i) for i in range(DEFAULT_BOLD_N_TRS)]

    def test_no_tsv_raises(self, tree: BIDSTree):
        bold = tree.add_bold(sub="01", task="A", run="1", space=VOLUME_SPACE.value)
        with pytest.raises(FileNotFoundError):
            _denoiser(tree, ["trans_x"])._load_nuisance(BIDSPath(bold))

    def test_duplicate_tsv_raises(self, tree: BIDSTree):
        bold = _add_run(tree)
        # A second confounds tsv differing only by a benign entity still matches
        # the run's structural filters, so the finder returns two candidates
        canonical = tree.add_confounds_timeseries(
            sub="01", task="A", run="1", df=_tsv_df(), metadata=_tsv_meta()
        )
        sibling = BIDSPath(canonical).with_entity("conf", "x").path
        sibling.write_text(canonical.read_text())

        with pytest.raises(ValueError, match="Expected one confounds tsv"):
            _denoiser(tree, ["trans_x"])._load_nuisance(bold)


class TestConstructor:
    def test_empty_selection_rejected(self, tree: BIDSTree):
        with pytest.raises(ValueError, match="at least one of columns or compcor"):
            _denoiser(tree)

    @pytest.mark.parametrize("reserved", ["desc-foo", "space-fsaverage6", "sub-02"])
    def test_reserved_filter_rejected(self, tree: BIDSTree, reserved: str):
        with pytest.raises(ValueError, match="bids_filters cannot contain"):
            _denoiser(tree, ["trans_x"], bids_filters=[reserved])

    def test_invalid_space_rejected(self, tree: BIDSTree):
        with pytest.raises(ValueError, match="Unsupported BOLD data space"):
            Denoiser(
                bids_root=tree.root,
                space="not-a-space",
                columns=["trans_x"],
                compcor=[],
            )


class TestDenoise:
    def test_volume_writes_desc_clean(self, tree: BIDSTree):
        _add_run(tree)
        _denoiser(tree, ["trans_x"]).denoise("01")

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
        # Confounds tsv one TR shorter than the bold the fixture writes
        _add_run(tree, df=_tsv_df(DEFAULT_BOLD_N_TRS - 1))
        with pytest.raises(ValueError, match="Unequal number of TRs"):
            _denoiser(tree, ["trans_x"]).denoise("01")

    def test_existing_desc_clean_skipped(self, tree: BIDSTree):
        _add_run(tree)
        clean_path = tree.func_dir(sub="01") / (
            f"sub-01_task-A_run-1_space-{VOLUME_SPACE.value}_desc-clean_bold.nii.gz"
        )
        clean_path.write_bytes(b"sentinel")

        _denoiser(tree, ["trans_x"]).denoise("01")

        # Sentinel survives untouched: the existing output was skipped
        assert clean_path.read_bytes() == b"sentinel"

    def test_force_overwrites_desc_clean(self, tree: BIDSTree):
        _add_run(tree)
        clean_path = tree.func_dir(sub="01") / (
            f"sub-01_task-A_run-1_space-{VOLUME_SPACE.value}_desc-clean_bold.nii.gz"
        )
        # Sentinel sits at the output path; if denoise treated desc-clean as an
        # input (re-cleaning), it would choke loading this instead of desc-preproc
        clean_path.write_bytes(b"sentinel")

        _denoiser(tree, ["trans_x"], force=True).denoise("01")

        import nibabel as nib
        from nibabel.nifti1 import Nifti1Image

        # Valid NIfTI means the sentinel was overwritten by real clean output
        img = nib.load(clean_path)
        assert isinstance(img, Nifti1Image)
        assert img.shape[-1] == DEFAULT_BOLD_N_TRS

    def test_bids_filters_reach_finder(self, tree: BIDSTree):
        _add_run(tree, run="1")
        _add_run(tree, run="2")
        _denoiser(tree, ["trans_x"], bids_filters=["task-A", "run-2"]).denoise("01")

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
