import polars as pl
import pytest

from hypline.fmriprep import (
    CompCorMask,
    CompCorMethod,
    CompCorOptions,
    ConfoundMetadata,
    _select_comps,
    parse_compcor,
    read_fmriprep_confounds,
    select_fmriprep_columns,
)

N_ROWS = 10


def _df(n: int = N_ROWS) -> pl.DataFrame:
    """A confounds frame with literal, group, and CompCor columns."""
    return pl.DataFrame(
        {
            "trans_x": [float(i) for i in range(n)],
            "cosine00": [float(i) for i in range(n)],
            "cosine01": [float(i) * 2 for i in range(n)],
            "a_comp_cor_00": [float(i) * 3 for i in range(n)],
            "a_comp_cor_01": [float(i) * 4 for i in range(n)],
        }
    )


def _meta() -> dict[str, ConfoundMetadata]:
    """Sidecar metadata for two retained aCompCor CSF components."""
    return {
        "a_comp_cor_00": ConfoundMetadata(
            Method=CompCorMethod.ANATOMICAL,
            Mask=CompCorMask.CSF,
            Retained=True,
            SingularValue=2.0,
        ),
        "a_comp_cor_01": ConfoundMetadata(
            Method=CompCorMethod.ANATOMICAL,
            Mask=CompCorMask.CSF,
            Retained=True,
            SingularValue=1.0,
        ),
    }


class TestParseCompcor:
    def test_empty_yields_empty(self):
        assert parse_compcor([]) == []

    def test_acompcor_with_mask(self):
        (options,) = parse_compcor(["a:CSF:5"])
        assert options.method is CompCorMethod.ANATOMICAL
        assert options.mask is CompCorMask.CSF
        assert options.n_comps == 5

    def test_tcompcor_empty_mask(self):
        (options,) = parse_compcor(["t::10"])
        assert options.method is CompCorMethod.TEMPORAL
        assert options.mask is None
        assert options.n_comps == 10

    def test_variance_fraction_n(self):
        (options,) = parse_compcor(["t::0.5"])
        assert options.n_comps == 0.5

    def test_multiple_tokens_order(self):
        a, t = parse_compcor(["a:WM:3", "t::2"])
        assert a.method is CompCorMethod.ANATOMICAL
        assert t.method is CompCorMethod.TEMPORAL

    @pytest.mark.parametrize(
        "token, match",
        [
            ("a:CSF", "3 colon-separated fields"),
            ("a:CSF:5:6", "3 colon-separated fields"),
            ("x:CSF:5", "method must be 'a' or 't'"),
            ("a::5", "aCompCor requires a mask"),
            ("a:BAD:5", "mask must be one of"),
            ("t:CSF:5", "tCompCor is not mask-restricted"),
            ("a:CSF:", "missing n"),
            ("a:CSF:abc", "n must be a number"),
            ("a:CSF:0", "n must be positive"),
            ("a:CSF:-1", "n must be positive"),
        ],
    )
    def test_invalid_tokens(self, token: str, match: str):
        with pytest.raises(ValueError, match=match):
            parse_compcor([token])


class TestSelectComps:
    def test_acompcor_top_n(self):
        comps = _select_comps(
            _meta(), CompCorMethod.ANATOMICAL, n_comps=2, mask=CompCorMask.CSF
        )
        # Descending SingularValue: 00 (2.0) before 01 (1.0), not name order
        assert comps == ["a_comp_cor_00", "a_comp_cor_01"]

    def test_acompcor_mask_filters(self):
        meta = _meta()
        meta["a_comp_cor_01"] = ConfoundMetadata(
            Method=CompCorMethod.ANATOMICAL,
            Mask=CompCorMask.WM,
            Retained=True,
            SingularValue=1.0,
        )
        comps = _select_comps(
            meta, CompCorMethod.ANATOMICAL, n_comps=5, mask=CompCorMask.CSF
        )
        assert comps == ["a_comp_cor_00"]

    def test_retained_false_excluded(self):
        meta = _meta()
        meta["a_comp_cor_01"] = ConfoundMetadata(
            Method=CompCorMethod.ANATOMICAL,
            Mask=CompCorMask.CSF,
            Retained=False,
            SingularValue=1.0,
        )
        comps = _select_comps(
            meta, CompCorMethod.ANATOMICAL, n_comps=5, mask=CompCorMask.CSF
        )
        assert comps == ["a_comp_cor_00"]

    def test_variance_fraction_selects_fewest(self):
        meta = {
            "a_comp_cor_00": ConfoundMetadata(
                Method=CompCorMethod.ANATOMICAL,
                Mask=CompCorMask.CSF,
                Retained=True,
                SingularValue=2.0,
                CumulativeVarianceExplained=0.4,
            ),
            "a_comp_cor_01": ConfoundMetadata(
                Method=CompCorMethod.ANATOMICAL,
                Mask=CompCorMask.CSF,
                Retained=True,
                SingularValue=1.0,
                CumulativeVarianceExplained=0.6,
            ),
        }
        # 0.5 threshold reached only after the second component
        comps = _select_comps(
            meta, CompCorMethod.ANATOMICAL, n_comps=0.5, mask=CompCorMask.CSF
        )
        assert comps == ["a_comp_cor_00", "a_comp_cor_01"]

    def test_fewer_available_than_requested(self):
        comps = _select_comps(
            _meta(), CompCorMethod.ANATOMICAL, n_comps=5, mask=CompCorMask.CSF
        )
        assert comps == ["a_comp_cor_00", "a_comp_cor_01"]

    def test_tcompcor_ignores_mask(self):
        meta = {
            "t_comp_cor_00": ConfoundMetadata(
                Method=CompCorMethod.TEMPORAL,
                Retained=True,
                SingularValue=1.0,
            ),
        }
        comps = _select_comps(
            meta, CompCorMethod.TEMPORAL, n_comps=1, mask=CompCorMask.CSF
        )
        assert comps == ["t_comp_cor_00"]

    def test_non_positive_n_comps_rejected(self):
        with pytest.raises(AssertionError, match="`n_comps` must be positive"):
            _select_comps(_meta(), CompCorMethod.TEMPORAL, n_comps=-1, mask=None)

    def test_acompcor_without_mask_rejected(self):
        with pytest.raises(AssertionError, match="Mask must be specified for aCompCor"):
            _select_comps(_meta(), CompCorMethod.ANATOMICAL, n_comps=1, mask=None)

    def test_unsupported_method_rejected(self):
        with pytest.raises(
            ValueError, match=f"Unsupported CompCor method: {CompCorMethod.MEAN}"
        ):
            _select_comps(_meta(), CompCorMethod.MEAN, n_comps=1, mask=None)


class TestSelectColumns:
    def test_literal_columns(self):
        names = select_fmriprep_columns(_df(), _meta(), columns=["trans_x"], compcor=[])
        assert names == ["trans_x"]

    def test_group_prefix_expands(self):
        # cosine expands to cosine00 + cosine01 in column order
        names = select_fmriprep_columns(_df(), _meta(), columns=["cosine"], compcor=[])
        assert names == ["cosine00", "cosine01"]

    def test_columns_then_compcor_order(self):
        options = CompCorOptions(
            method=CompCorMethod.ANATOMICAL, n_comps=1, mask=CompCorMask.CSF
        )
        names = select_fmriprep_columns(
            _df(), _meta(), columns=["trans_x"], compcor=[options]
        )
        assert names == ["trans_x", "a_comp_cor_00"]

    def test_missing_column_raises(self):
        with pytest.raises(ValueError, match="missing from tsv"):
            select_fmriprep_columns(
                _df(), _meta(), columns=["nonexistent"], compcor=[]
            )


class TestReadConfounds:
    def test_reads_frame_and_meta(self, tmp_path):
        path = tmp_path / "sub-01_task-A_run-1_desc-confounds_timeseries.tsv"
        path.write_text(_df().write_csv(separator="\t"))
        path.with_suffix(".json").write_text(
            '{"a_comp_cor_00": {"Method": "aCompCor", "Mask": "CSF", '
            '"Retained": true, "SingularValue": 2.0}}'
        )

        df, meta = read_fmriprep_confounds(path)
        assert df.shape == (N_ROWS, 5)
        assert meta["a_comp_cor_00"].Method is CompCorMethod.ANATOMICAL

    def test_leading_na_backfilled(self, tmp_path):
        # fmriprep writes literal `n/a` (not an empty cell) for leading nulls;
        # it must parse as null and backfill, not become a string or NaN regressor
        path = tmp_path / "sub-01_task-A_run-1_desc-confounds_timeseries.tsv"
        df = _df().with_columns(
            pl.Series("trans_x", [None] + [float(i) for i in range(1, N_ROWS)])
        )
        path.write_text(df.write_csv(separator="\t", null_value="n/a"))
        path.with_suffix(".json").write_text("{}")

        out, _ = read_fmriprep_confounds(path)
        assert out["trans_x"][0] == 1.0
