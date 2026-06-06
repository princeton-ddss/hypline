from pathlib import Path

import polars as pl
import pytest
from pydantic import TypeAdapter

from hypline.confounds.fmriprep import FmriprepConfound
from hypline.enums import VolumeSpace
from hypline.fmriprep import (
    CompCorMask,
    CompCorMethod,
    CompCorOptions,
    ConfoundMetadata,
    _select_comps,
)
from hypline.io import read_confound, read_confound_metadata

from ..conftest import DEFAULT_BOLD_N_TRS, BIDSTree

VOLUME_SPACE = VolumeSpace.MNI_152_NLIN_2009_C_ASYM


@pytest.fixture(scope="session")
def confounds_meta() -> dict[str, ConfoundMetadata]:
    path = Path(__file__).parents[2] / "data" / "confounds_timeseries.json"
    return TypeAdapter(dict[str, ConfoundMetadata]).validate_json(path.read_text())


class TestValidateCompcor:
    """The mask-iff-aCompCor invariant enforced at FmriprepConfound construction."""

    def _build(self, tree: BIDSTree, options: CompCorOptions) -> FmriprepConfound:
        return FmriprepConfound(
            bids_root=tree.root, desc="x", columns=[], compcor=[options]
        )

    def test_method_none_rejected(self, tree: BIDSTree):
        with pytest.raises(ValueError, match="method must be set"):
            self._build(tree, CompCorOptions(method=None))

    def test_acompcor_requires_mask(self, tree: BIDSTree):
        with pytest.raises(ValueError, match="aCompCor requires a mask"):
            self._build(
                tree, CompCorOptions(method=CompCorMethod.ANATOMICAL, mask=None)
            )

    def test_tcompcor_rejects_mask(self, tree: BIDSTree):
        with pytest.raises(ValueError, match="tCompCor must not carry a mask"):
            self._build(
                tree,
                CompCorOptions(method=CompCorMethod.TEMPORAL, mask=CompCorMask.CSF),
            )

    def test_desc_must_be_alphanumeric(self, tree: BIDSTree):
        with pytest.raises(ValueError, match="desc must be alphanumeric"):
            FmriprepConfound(
                bids_root=tree.root, desc="bad-desc", columns=["trans_x"], compcor=[]
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
        confounds_meta: dict[str, ConfoundMetadata],
        method: CompCorMethod,
        n_comps: int | float,
        mask: CompCorMask | None,
        expected_output: list[str],
    ):
        output = _select_comps(
            confounds_meta=confounds_meta,
            method=method,
            n_comps=n_comps,
            mask=mask,
        )
        assert output == expected_output

    def test_invalid_n_comps(self, confounds_meta: dict[str, ConfoundMetadata]):
        with pytest.raises(AssertionError, match="`n_comps` must be positive"):
            _select_comps(
                confounds_meta=confounds_meta,
                method=CompCorMethod.TEMPORAL,
                n_comps=-1,
                mask=None,
            )

    def test_missing_mask_for_acompcor(
        self, confounds_meta: dict[str, ConfoundMetadata]
    ):
        with pytest.raises(AssertionError, match="Mask must be specified for aCompCor"):
            _select_comps(
                confounds_meta=confounds_meta,
                method=CompCorMethod.ANATOMICAL,
                n_comps=1,
                mask=None,
            )

    def test_unsupported_method(self, confounds_meta: dict[str, ConfoundMetadata]):
        with pytest.raises(
            ValueError, match=f"Unsupported CompCor method: {CompCorMethod.MEAN}"
        ):
            _select_comps(
                confounds_meta=confounds_meta,
                method=CompCorMethod.MEAN,
                n_comps=1,
                mask=None,
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


class TestGenerate:
    def _setup(self, tree: BIDSTree) -> None:
        tree.add_bold(sub="01", task="A", run="1", space=VOLUME_SPACE.value)
        tree.add_confounds_timeseries(
            sub="01", task="A", run="1", df=_tsv_df(), metadata=_tsv_meta()
        )

    def _output(self, tree: BIDSTree):
        matches = list(
            (tree.confounds_dir / "sub-01" / "fmriprep-min").glob("*.parquet")
        )
        assert len(matches) == 1
        return matches[0]

    def test_columns_only(self, tree: BIDSTree):
        self._setup(tree)
        FmriprepConfound(
            bids_root=tree.root,
            desc="min",
            columns=["trans_x", "cosine"],
            compcor=[],
        ).generate("01")

        out = self._output(tree)
        block = read_confound(out).get_column("confound")
        dtype = block.dtype
        assert isinstance(dtype, pl.Array)
        assert dtype.size == 3  # trans_x + cosine00 + cosine01
        meta = read_confound_metadata(out)
        assert meta["_confound_dim_labels"] == ["trans_x", "cosine00", "cosine01"]

    def test_compcor_only(self, tree: BIDSTree):
        self._setup(tree)
        FmriprepConfound(
            bids_root=tree.root,
            desc="min",
            columns=[],
            compcor=[
                CompCorOptions(
                    method=CompCorMethod.ANATOMICAL, n_comps=2, mask=CompCorMask.CSF
                )
            ],
        ).generate("01")

        meta = read_confound_metadata(self._output(tree))
        assert meta["_confound_dim_labels"] == ["a_comp_cor_00", "a_comp_cor_01"]

    def test_columns_then_compcor_order(self, tree: BIDSTree):
        self._setup(tree)
        FmriprepConfound(
            bids_root=tree.root,
            desc="min",
            columns=["trans_x"],
            compcor=[
                CompCorOptions(
                    method=CompCorMethod.ANATOMICAL, n_comps=1, mask=CompCorMask.CSF
                )
            ],
        ).generate("01")

        meta = read_confound_metadata(self._output(tree))
        assert meta["_confound_dim_labels"] == ["trans_x", "a_comp_cor_00"]

    def test_existing_output_skipped(self, tree: BIDSTree):
        self._setup(tree)
        FmriprepConfound(
            bids_root=tree.root, desc="min", columns=["trans_x"], compcor=[]
        ).generate("01")
        out = self._output(tree)
        out.write_bytes(b"sentinel")

        FmriprepConfound(
            bids_root=tree.root, desc="min", columns=["trans_x"], compcor=[]
        ).generate("01")
        assert out.read_bytes() == b"sentinel"

    def test_force_overwrites_output(self, tree: BIDSTree):
        self._setup(tree)
        FmriprepConfound(
            bids_root=tree.root, desc="min", columns=["trans_x"], compcor=[]
        ).generate("01")
        out = self._output(tree)
        out.write_bytes(b"sentinel")

        FmriprepConfound(
            bids_root=tree.root, desc="min", columns=["trans_x"], compcor=[], force=True
        ).generate("01")
        assert out.read_bytes() != b"sentinel"
        assert read_confound(out).height == DEFAULT_BOLD_N_TRS

    def test_missing_column_raises(self, tree: BIDSTree):
        self._setup(tree)
        confound = FmriprepConfound(
            bids_root=tree.root, desc="min", columns=["nonexistent"], compcor=[]
        )
        with pytest.raises(ValueError, match="missing from tsv"):
            confound.generate("01")
