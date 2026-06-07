import polars as pl
import pytest

from hypline.bids import BIDSPath
from hypline.denoise import Denoiser
from hypline.enums import VolumeSpace

from .conftest import DEFAULT_BOLD_N_TRS, BIDSTree

VOLUME_SPACE = VolumeSpace.MNI_152_NLIN_2009_C_ASYM


def _denoiser(
    tree: BIDSTree,
    *,
    columns: list[str] | None = None,
    compcor: list[str] | None = None,
    custom_sources: list[str] | None = None,
    custom_columns: list[str] | None = None,
    **kwargs,
) -> Denoiser:
    return Denoiser(
        bids_root=tree.root,
        space=VOLUME_SPACE,
        columns=columns or [],
        compcor=compcor or [],
        custom_sources=custom_sources or [],
        custom_columns=custom_columns or [],
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
    bold_path = tree.add_bold(sub=sub, task=task, run=run, space=VOLUME_SPACE)
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
        loaded = _denoiser(tree, columns=["trans_x"])._load_nuisance(bold)
        assert loaded.shape == (DEFAULT_BOLD_N_TRS, 1)
        assert loaded[:, 0].tolist() == [float(i) for i in range(DEFAULT_BOLD_N_TRS)]

    def test_no_tsv_raises(self, tree: BIDSTree):
        bold = tree.add_bold(sub="01", task="A", run="1", space=VOLUME_SPACE)
        with pytest.raises(FileNotFoundError):
            _denoiser(tree, columns=["trans_x"])._load_nuisance(BIDSPath(bold))

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
            _denoiser(tree, columns=["trans_x"])._load_nuisance(bold)


def _nuis_df(n: int = DEFAULT_BOLD_N_TRS, **cols: float) -> pl.DataFrame:
    """A wide nuisance TSV frame; each kwarg is a column scaled by its value."""
    return pl.DataFrame(
        {name: [float(i) * scale for i in range(n)] for name, scale in cols.items()}
    )


class TestCustomNuisance:
    def test_custom_only(self, tree: BIDSTree):
        bold = _add_run(tree)
        tree.add_nuisance(
            sub="01",
            task="A",
            run="1",
            kind="physio",
            df=_nuis_df(physio0=1.0, physio1=2.0),
        )
        loaded = _denoiser(
            tree, custom_sources=["physio"], custom_columns=["physio1"]
        )._load_nuisance(bold)
        assert loaded.shape == (DEFAULT_BOLD_N_TRS, 1)
        assert loaded[:, 0].tolist() == [
            float(i) * 2 for i in range(DEFAULT_BOLD_N_TRS)
        ]

    def test_combined_with_fmriprep(self, tree: BIDSTree):
        bold = _add_run(tree)
        tree.add_nuisance(
            sub="01", task="A", run="1", kind="physio", df=_nuis_df(physio0=1.0)
        )
        loaded = _denoiser(
            tree,
            columns=["trans_x"],
            custom_sources=["physio"],
            custom_columns=["physio0"],
        )._load_nuisance(bold)
        assert loaded.shape == (DEFAULT_BOLD_N_TRS, 2)

    def test_selects_across_concat(self, tree: BIDSTree):
        bold = _add_run(tree)
        tree.add_nuisance(
            sub="01", task="A", run="1", kind="physio", df=_nuis_df(physio0=1.0)
        )
        tree.add_nuisance(
            sub="01", task="A", run="1", kind="resp", df=_nuis_df(resp0=3.0)
        )
        loaded = _denoiser(
            tree,
            custom_sources=["physio", "resp"],
            custom_columns=["resp0", "physio0"],
        )._load_nuisance(bold)
        assert loaded.shape == (DEFAULT_BOLD_N_TRS, 2)
        # selection order follows --custom-columns, not source order
        assert loaded[:, 0].tolist() == [
            float(i) * 3 for i in range(DEFAULT_BOLD_N_TRS)
        ]

    def test_kind_desc_ref(self, tree: BIDSTree):
        bold = _add_run(tree)
        tree.add_nuisance(
            sub="01",
            task="A",
            run="1",
            kind="physio",
            desc="v1",
            df=_nuis_df(physio0=1.0),
        )
        loaded = _denoiser(
            tree, custom_sources=["physio-v1"], custom_columns=["physio0"]
        )._load_nuisance(bold)
        assert loaded.shape == (DEFAULT_BOLD_N_TRS, 1)
        assert loaded[:, 0].tolist() == [float(i) for i in range(DEFAULT_BOLD_N_TRS)]

    def test_collision_across_sources_raises(self, tree: BIDSTree):
        bold = _add_run(tree)
        tree.add_nuisance(
            sub="01", task="A", run="1", kind="physio", df=_nuis_df(shared=1.0)
        )
        tree.add_nuisance(
            sub="01", task="A", run="1", kind="resp", df=_nuis_df(shared=2.0)
        )
        with pytest.raises(ValueError, match="Duplicate custom nuisance column"):
            _denoiser(
                tree, custom_sources=["physio", "resp"], custom_columns=["shared"]
            )._load_nuisance(bold)

    def test_collision_with_fmriprep_raises(self, tree: BIDSTree):
        bold = _add_run(tree)
        tree.add_nuisance(
            sub="01", task="A", run="1", kind="physio", df=_nuis_df(trans_x=1.0)
        )
        with pytest.raises(ValueError, match="collision across channels"):
            _denoiser(
                tree,
                columns=["trans_x"],
                custom_sources=["physio"],
                custom_columns=["trans_x"],
            )._load_nuisance(bold)

    def test_height_mismatch_across_sources_raises(self, tree: BIDSTree):
        bold = _add_run(tree)
        tree.add_nuisance(
            sub="01", task="A", run="1", kind="physio", df=_nuis_df(physio0=1.0)
        )
        tree.add_nuisance(
            sub="01",
            task="A",
            run="1",
            kind="resp",
            df=_nuis_df(n=DEFAULT_BOLD_N_TRS - 1, resp0=1.0),
        )
        with pytest.raises(ValueError, match="row count mismatch"):
            _denoiser(
                tree,
                custom_sources=["physio", "resp"],
                custom_columns=["physio0", "resp0"],
            )._load_nuisance(bold)

    def test_missing_column_raises(self, tree: BIDSTree):
        bold = _add_run(tree)
        tree.add_nuisance(
            sub="01", task="A", run="1", kind="physio", df=_nuis_df(physio0=1.0)
        )
        with pytest.raises(ValueError, match="missing from sources"):
            _denoiser(
                tree, custom_sources=["physio"], custom_columns=["nope"]
            )._load_nuisance(bold)

    def test_no_match_raises(self, tree: BIDSTree):
        bold = _add_run(tree)
        # A physio file for a *different* run; the kind folder exists but no file
        # matches this run's structural filters → finder raises FileNotFoundError
        tree.add_nuisance(
            sub="01", task="A", run="2", kind="physio", df=_nuis_df(physio0=1.0)
        )
        with pytest.raises(FileNotFoundError):
            _denoiser(
                tree, custom_sources=["physio"], custom_columns=["physio0"]
            )._load_nuisance(bold)


class TestConstructor:
    def test_empty_selection_rejected(self, tree: BIDSTree):
        with pytest.raises(ValueError, match="at least one of columns, compcor"):
            _denoiser(tree)

    def test_custom_pairing_required(self, tree: BIDSTree):
        with pytest.raises(ValueError, match="must be given together"):
            _denoiser(tree, custom_sources=["physio"])
        with pytest.raises(ValueError, match="must be given together"):
            _denoiser(tree, columns=["trans_x"], custom_columns=["physio0"])

    @pytest.mark.parametrize("reserved", ["desc-foo", "space-fsaverage6", "sub-02"])
    def test_reserved_filter_rejected(self, tree: BIDSTree, reserved: str):
        with pytest.raises(ValueError, match="bids_filters cannot contain"):
            _denoiser(tree, columns=["trans_x"], bids_filters=[reserved])

    def test_invalid_space_rejected(self, tree: BIDSTree):
        with pytest.raises(ValueError, match="Unsupported BOLD data space"):
            Denoiser(
                bids_root=tree.root,
                space="not-a-space",
                columns=["trans_x"],
                compcor=[],
                custom_sources=[],
                custom_columns=[],
            )


class TestDenoise:
    def test_volume_writes_desc_denoised(self, tree: BIDSTree):
        _add_run(tree)
        _denoiser(tree, columns=["trans_x"]).denoise("01")

        func_dir = tree.denoised_func_dir(sub="01")
        denoised = (
            func_dir
            / f"sub-01_task-A_run-1_space-{VOLUME_SPACE}_desc-denoised_bold.nii.gz"
        )
        assert denoised.exists()

        import nibabel as nib
        from nibabel.nifti1 import Nifti1Image

        img = nib.load(denoised)
        assert isinstance(img, Nifti1Image)
        assert img.shape[-1] == DEFAULT_BOLD_N_TRS

    def test_tr_count_mismatch_raises(self, tree: BIDSTree):
        # Confounds tsv one TR shorter than the bold the fixture writes
        _add_run(tree, df=_tsv_df(DEFAULT_BOLD_N_TRS - 1))
        with pytest.raises(ValueError, match="Unequal number of TRs"):
            _denoiser(tree, columns=["trans_x"]).denoise("01")

    def test_existing_desc_denoised_skipped(self, tree: BIDSTree):
        _add_run(tree)
        denoised_dir = tree.denoised_func_dir(sub="01")
        denoised_dir.mkdir(parents=True, exist_ok=True)
        denoised_path = denoised_dir / (
            f"sub-01_task-A_run-1_space-{VOLUME_SPACE}_desc-denoised_bold.nii.gz"
        )
        denoised_path.write_bytes(b"sentinel")

        _denoiser(tree, columns=["trans_x"]).denoise("01")

        # Sentinel survives untouched: the existing output was skipped
        assert denoised_path.read_bytes() == b"sentinel"

    def test_force_overwrites_desc_denoised(self, tree: BIDSTree):
        _add_run(tree)
        denoised_dir = tree.denoised_func_dir(sub="01")
        denoised_dir.mkdir(parents=True, exist_ok=True)
        denoised_path = denoised_dir / (
            f"sub-01_task-A_run-1_space-{VOLUME_SPACE}_desc-denoised_bold.nii.gz"
        )
        # Sentinel sits at the output path; if denoise treated desc-denoised as an
        # input (re-denoising), it would choke loading this instead of desc-preproc
        denoised_path.write_bytes(b"sentinel")

        _denoiser(tree, columns=["trans_x"], force=True).denoise("01")

        import nibabel as nib
        from nibabel.nifti1 import Nifti1Image

        # Valid NIfTI means the sentinel was overwritten by real denoised output
        img = nib.load(denoised_path)
        assert isinstance(img, Nifti1Image)
        assert img.shape[-1] == DEFAULT_BOLD_N_TRS

    def test_bids_filters_reach_finder(self, tree: BIDSTree):
        _add_run(tree, run="1")
        _add_run(tree, run="2")
        _denoiser(tree, columns=["trans_x"], bids_filters=["task-A", "run-2"]).denoise(
            "01"
        )

        func_dir = tree.denoised_func_dir(sub="01")
        denoised_run2 = (
            func_dir
            / f"sub-01_task-A_run-2_space-{VOLUME_SPACE}_desc-denoised_bold.nii.gz"
        )
        denoised_run1 = (
            func_dir
            / f"sub-01_task-A_run-1_space-{VOLUME_SPACE}_desc-denoised_bold.nii.gz"
        )
        assert denoised_run2.exists()
        assert not denoised_run1.exists()
