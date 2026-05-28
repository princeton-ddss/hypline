from pathlib import Path

import polars as pl
import pytest
from pydantic import TypeAdapter

from hypline.bids import BIDSPath
from hypline.denoise import (
    CompCorMask,
    CompCorMethod,
    ConfoundMetadata,
    Denoiser,
    ModelSpec,
)
from hypline.enums import VolumeSpace

from .conftest import DEFAULT_BOLD_N_TRS, BIDSTree

VOLUME_SPACE = VolumeSpace.MNI_152_NLIN_2009_C_ASYM


@pytest.fixture(scope="session")
def confounds_meta() -> dict[str, ConfoundMetadata]:
    path = Path(__file__).parents[1] / "data" / "confounds_timeseries.json"
    return TypeAdapter(dict[str, ConfoundMetadata]).validate_json(path.read_text())


def _denoiser(tree: BIDSTree, model_spec: ModelSpec, **kwargs) -> Denoiser:
    return Denoiser(
        model_spec,
        bids_root=tree.root,
        space=VOLUME_SPACE,
        **kwargs,
    )


class TestSelectComps:
    @pytest.mark.parametrize(
        "method, n_comps, mask, expected_output",
        [
            (CompCorMethod.ANATOMICAL, 1, CompCorMask.CSF, ["a_comp_cor_00"]),
            (
                CompCorMethod.ANATOMICAL,
                3,
                CompCorMask.CSF,
                ["a_comp_cor_00", "a_comp_cor_01", "a_comp_cor_02"],
            ),
            (
                CompCorMethod.ANATOMICAL,
                10,
                CompCorMask.CSF,
                [f"a_comp_cor_0{i}" for i in range(10)],
            ),
            (
                CompCorMethod.ANATOMICAL,
                0.3,
                CompCorMask.CSF,
                [f"a_comp_cor_0{i}" for i in range(5)],
            ),
            (
                CompCorMethod.ANATOMICAL,
                3,
                CompCorMask.WM,
                ["a_comp_cor_12", "a_comp_cor_13", "a_comp_cor_14"],
            ),
            (
                CompCorMethod.ANATOMICAL,
                0.1,
                CompCorMask.WM,
                ["a_comp_cor_12", "a_comp_cor_13"],
            ),
            (
                CompCorMethod.ANATOMICAL,
                3,
                CompCorMask.COMBINED,
                ["a_comp_cor_100", "a_comp_cor_101", "a_comp_cor_102"],
            ),
            (
                CompCorMethod.ANATOMICAL,
                0.1,
                CompCorMask.COMBINED,
                ["a_comp_cor_100", "a_comp_cor_101"],
            ),
            (CompCorMethod.TEMPORAL, 1, None, ["t_comp_cor_00"]),
            (
                CompCorMethod.TEMPORAL,
                3,
                None,
                ["t_comp_cor_00", "t_comp_cor_01", "t_comp_cor_02"],
            ),
            (
                CompCorMethod.TEMPORAL,
                10,
                None,
                ["t_comp_cor_00", "t_comp_cor_01", "t_comp_cor_02"],
            ),
            (CompCorMethod.TEMPORAL, 0.4, None, ["t_comp_cor_00", "t_comp_cor_01"]),
            (
                CompCorMethod.TEMPORAL,
                0.4,
                CompCorMask.CSF,  # Expected to be ignored
                ["t_comp_cor_00", "t_comp_cor_01"],
            ),
        ],
    )
    def test_select_comps(
        self,
        tree: BIDSTree,
        confounds_meta: dict[str, ConfoundMetadata],
        method: CompCorMethod,
        n_comps: int | float,
        mask: CompCorMask | None,
        expected_output: list[str],
    ):
        denoiser = _denoiser(tree, ModelSpec(confounds=["x"]))
        output = denoiser._select_comps(
            confounds_meta=confounds_meta,
            method=method,
            n_comps=n_comps,
            mask=mask,
        )
        assert output == expected_output

    def test_invalid_n_comps(
        self, tree: BIDSTree, confounds_meta: dict[str, ConfoundMetadata]
    ):
        denoiser = _denoiser(tree, ModelSpec(confounds=["x"]))
        with pytest.raises(AssertionError, match="`n_comps` must be positive"):
            denoiser._select_comps(
                confounds_meta=confounds_meta,
                method=CompCorMethod.TEMPORAL,
                n_comps=-1,
                mask=None,
            )

    def test_missing_mask_for_acompcor(
        self, tree: BIDSTree, confounds_meta: dict[str, ConfoundMetadata]
    ):
        denoiser = _denoiser(tree, ModelSpec(confounds=["x"]))
        with pytest.raises(AssertionError, match="Mask must be specified for aCompCor"):
            denoiser._select_comps(
                confounds_meta=confounds_meta,
                method=CompCorMethod.ANATOMICAL,
                n_comps=1,
                mask=None,
            )

    def test_unsupported_method(
        self, tree: BIDSTree, confounds_meta: dict[str, ConfoundMetadata]
    ):
        denoiser = _denoiser(tree, ModelSpec(confounds=["x"]))
        with pytest.raises(
            ValueError, match=f"Unsupported CompCor method: {CompCorMethod.MEAN}"
        ):
            denoiser._select_comps(
                confounds_meta=confounds_meta,
                method=CompCorMethod.MEAN,
                n_comps=1,
                mask=None,
            )


class TestExtractConfounds:
    @pytest.fixture
    def confounds_df(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "global_signal": [1, 2],
                "csf": [3, 4],
                "white_matter": [5, 6],
                "cosine00": [7, 8],
                "cosine01": [9, 10],
                "cosine02": [11, 12],
                "motion_outlier00": [13, 14],
                "motion_outlier01": [15, 16],
                "motion_outlier02": [17, 18],
            }
        )

    @pytest.mark.parametrize(
        "model_confounds, expected_output_confounds",
        [
            (["global_signal"], ["global_signal"]),
            (["global_signal", "csf"], ["global_signal", "csf"]),
            (
                ["global_signal", "csf", "white_matter"],
                ["global_signal", "csf", "white_matter"],
            ),
            (
                ["global_signal", "csf", "cosine"],
                ["global_signal", "csf", "cosine00", "cosine01", "cosine02"],
            ),
            (
                ["global_signal", "csf", "motion_outlier"],
                [
                    "global_signal",
                    "csf",
                    "motion_outlier00",
                    "motion_outlier01",
                    "motion_outlier02",
                ],
            ),
        ],
    )
    def test_extract_confounds(
        self,
        tree: BIDSTree,
        confounds_df: pl.DataFrame,
        model_confounds: list[str],
        expected_output_confounds: list[str],
    ):
        denoiser = _denoiser(tree, ModelSpec(confounds=model_confounds))
        extracted_df = denoiser._extract_confounds(confounds_df, {})
        assert extracted_df.equals(confounds_df[expected_output_confounds])

    def test_extract_nonexisting_confounds(
        self, tree: BIDSTree, confounds_df: pl.DataFrame
    ):
        denoiser = _denoiser(tree, ModelSpec(confounds=["a"]))
        with pytest.raises(
            ValueError, match="Model confounds missing from confound data"
        ):
            denoiser._extract_confounds(confounds_df, {})


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
    def _add_bold(self, tree: BIDSTree, *, confounds: pl.DataFrame) -> BIDSPath:
        bold_path = tree.add_bold(sub="01", task="A", run="1", space=VOLUME_SPACE.value)
        tree.add_confounds_timeseries(sub="01", task="A", run="1", df=confounds)
        return BIDSPath(bold_path)

    def test_load_standard_confounds(self, tree: BIDSTree):
        confounds = pl.DataFrame({"X": [1, 2], "Y": [3, 4]})
        bold = self._add_bold(tree, confounds=confounds)
        denoiser = _denoiser(tree, ModelSpec(confounds=["X"]))
        loaded = denoiser._load_confounds(bold)
        assert loaded.equals(pl.DataFrame({"X": [1, 2]}))

    def test_load_custom_confound_width_one(self, tree: BIDSTree):
        bold = self._add_bold(tree, confounds=pl.DataFrame({"X": [1.0, 2.0]}))
        tree.add_confound(
            sub="01", task="A", run="1", kind="motion", df=_array_df([[10.0], [20.0]])
        )
        denoiser = _denoiser(
            tree, ModelSpec(confounds=["X"], custom_confounds=["motion"])
        )
        loaded = denoiser._load_confounds(bold)
        assert loaded.equals(pl.DataFrame({"X": [1.0, 2.0], "motion_0": [10.0, 20.0]}))

    def test_load_custom_confound_width_gt_one(self, tree: BIDSTree):
        bold = self._add_bold(tree, confounds=pl.DataFrame({"X": [1.0, 2.0]}))
        tree.add_confound(
            sub="01",
            task="A",
            run="1",
            kind="motion",
            df=_array_df([[10.0, 11.0], [20.0, 21.0]]),
        )
        denoiser = _denoiser(
            tree, ModelSpec(confounds=["X"], custom_confounds=["motion"])
        )
        loaded = denoiser._load_confounds(bold)
        assert loaded.columns == ["X", "motion_0", "motion_1"]
        assert loaded["motion_1"].to_list() == [11.0, 21.0]

    def test_load_multiple_variants_same_kind(self, tree: BIDSTree):
        bold = self._add_bold(tree, confounds=pl.DataFrame({"X": [1.0, 2.0]}))
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
            tree,
            ModelSpec(
                confounds=["X"], custom_confounds=["phonemic-rate", "phonemic-onset"]
            ),
        )
        loaded = denoiser._load_confounds(bold)
        assert loaded.columns == ["X", "phonemic-rate_0", "phonemic-onset_0"]

    def test_custom_confound_file_missing(self, tree: BIDSTree):
        bold = self._add_bold(tree, confounds=pl.DataFrame({"X": [1.0, 2.0]}))
        denoiser = _denoiser(
            tree, ModelSpec(confounds=["X"], custom_confounds=["motion"])
        )
        with pytest.raises(FileNotFoundError):
            denoiser._load_confounds(bold)

    def test_custom_confound_row_mismatch(self, tree: BIDSTree):
        bold = self._add_bold(tree, confounds=pl.DataFrame({"X": [1.0, 2.0]}))
        tree.add_confound(
            sub="01",
            task="A",
            run="1",
            kind="motion",
            df=_array_df([[10.0], [20.0], [30.0]]),
        )
        denoiser = _denoiser(
            tree, ModelSpec(confounds=["X"], custom_confounds=["motion"])
        )
        with pytest.raises(ValueError, match="Unequal number of rows"):
            denoiser._load_confounds(bold)


class TestDenoise:
    def _confounds_df(self, n: int = DEFAULT_BOLD_N_TRS) -> pl.DataFrame:
        return pl.DataFrame({"X": [float(i) for i in range(n)]})

    def test_volume_writes_desc_clean(self, tree: BIDSTree):
        tree.add_bold(sub="01", task="A", run="1", space=VOLUME_SPACE.value)
        tree.add_confounds_timeseries(
            sub="01", task="A", run="1", df=self._confounds_df()
        )
        denoiser = _denoiser(tree, ModelSpec(confounds=["X"]))

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
        tree.add_confounds_timeseries(
            sub="01", task="A", run="1", df=self._confounds_df(DEFAULT_BOLD_N_TRS - 1)
        )
        denoiser = _denoiser(tree, ModelSpec(confounds=["X"]))
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
        tree.add_confounds_timeseries(
            sub="01", task="A", run="1", df=self._confounds_df()
        )
        denoiser = _denoiser(tree, ModelSpec(confounds=["X"]))

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
        tree.add_confounds_timeseries(
            sub="01", task="A", run="1", df=self._confounds_df()
        )
        tree.add_confounds_timeseries(
            sub="01", task="A", run="2", df=self._confounds_df()
        )
        denoiser = _denoiser(
            tree, ModelSpec(confounds=["X"]), bids_filters=["task-A", "run-2"]
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
            _denoiser(tree, ModelSpec(confounds=["X"]), bids_filters=[reserved])
