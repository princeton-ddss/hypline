from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import pytest

from hypline.features.utils import (
    Downsample,
    read_feature,
    resample_feature,
    save_feature,
)


@pytest.fixture()
def bids_path(tmp_path: Path) -> Path:
    return tmp_path / "sub-01_ses-1_feature-mfcc_bold.parquet"


@pytest.fixture()
def sample_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "start_time": [0.0, 0.5, 1.0],
            "feature": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        },
        schema={"start_time": pl.Float64, "feature": pl.Array(pl.Float64, 3)},
    )


class TestSaveFeature:
    def test_roundtrip(self, bids_path: Path, sample_df: pl.DataFrame):
        save_feature(sample_df, bids_path)
        df, meta = read_feature(bids_path)
        assert df.equals(sample_df)

    def test_creates_parent_dirs(self, tmp_path: Path, sample_df: pl.DataFrame):
        path = tmp_path / "a" / "b" / "sub-01_feature-mfcc_bold.parquet"
        save_feature(sample_df, path)
        assert path.exists()

    def test_list_column_cast_to_array(self, bids_path: Path):
        df = pl.DataFrame(
            {"start_time": [0.0, 0.5], "feature": [[1.0, 2.0], [3.0, 4.0]]},
            schema={"start_time": pl.Float64, "feature": pl.List(pl.Float64)},
        )
        save_feature(df, bids_path)
        loaded, _ = read_feature(bids_path)
        assert loaded.get_column("feature").dtype == pl.Array(pl.Float64, 2)

    def test_int_list_cast_to_float64(self, bids_path: Path):
        df = pl.DataFrame(
            {"start_time": [0.0, 0.5], "feature": [[1, 2], [3, 4]]},
            schema={"start_time": pl.Float64, "feature": pl.Array(pl.Int64, 2)},
        )
        save_feature(df, bids_path)
        loaded, _ = read_feature(bids_path)
        assert loaded.get_column("feature").dtype == pl.Array(pl.Float64, 2)

    def test_metadata_stored_in_footer(self, bids_path: Path, sample_df: pl.DataFrame):
        save_feature(sample_df, bids_path, metadata={"key": "value", "foo": "bar"})
        _, meta = read_feature(bids_path)
        assert meta["key"] == "value"
        assert meta["foo"] == "bar"

    def test_no_metadata(self, bids_path: Path, sample_df: pl.DataFrame):
        save_feature(sample_df, bids_path)
        _, meta = read_feature(bids_path)
        assert meta == {}

    def test_missing_required_columns(self, bids_path: Path):
        df = pl.DataFrame({"onset": [0.0, 1.0]})
        with pytest.raises(ValueError, match="missing required columns"):
            save_feature(df, bids_path)

    def test_missing_start_time_column(self, bids_path: Path):
        df = pl.DataFrame(
            {"feature": [[1.0, 2.0], [3.0, 4.0]]},
            schema={"feature": pl.Array(pl.Float64, 2)},
        )
        with pytest.raises(ValueError, match="missing required columns"):
            save_feature(df, bids_path)

    def test_non_numeric_start_time(self, bids_path: Path):
        df = pl.DataFrame(
            {"start_time": ["1.2", "3.5"], "feature": [[1.0, 2.0], [3.0, 4.0]]},
            schema={"start_time": pl.String, "feature": pl.Array(pl.Float64, 2)},
        )
        with pytest.raises(ValueError, match="must be a numeric type"):
            save_feature(df, bids_path)

    def test_unsupported_feature_dtype(self, bids_path: Path):
        df = pl.DataFrame({"start_time": [0.0, 0.5], "feature": ["a", "b"]})
        with pytest.raises(ValueError, match="must be an Array or List type"):
            save_feature(df, bids_path)

    def test_missing_feature_entity_in_path(
        self, tmp_path: Path, sample_df: pl.DataFrame
    ):
        path = tmp_path / "sub-01_ses-1_bold.parquet"
        with pytest.raises(ValueError, match="must contain a 'feature' entity"):
            save_feature(sample_df, path)


class TestReadFeature:
    def test_missing_feature_entity(self, tmp_path: Path):
        path = tmp_path / "sub-01_bold.parquet"
        with pytest.raises(ValueError, match="must contain a 'feature' entity"):
            read_feature(path)

    def test_missing_required_columns(self, tmp_path: Path):
        path = tmp_path / "sub-01_feature-mfcc_bold.parquet"
        table = pl.DataFrame({"onset": [0.0]}).to_arrow()
        pq.write_table(table, path)
        with pytest.raises(ValueError, match="missing required columns"):
            read_feature(path)

    def test_missing_start_time_column(self, tmp_path: Path):
        path = tmp_path / "sub-01_feature-mfcc_bold.parquet"
        df = pl.DataFrame(
            {"feature": [[1.0, 2.0], [3.0, 4.0]]},
            schema={"feature": pl.Array(pl.Float64, 2)},
        )
        pq.write_table(df.to_arrow(), path)
        with pytest.raises(ValueError, match="missing required columns"):
            read_feature(path)

    def test_non_numeric_start_time(self, tmp_path: Path):
        path = tmp_path / "sub-01_feature-mfcc_bold.parquet"
        df = pl.DataFrame(
            {"start_time": ["1.2", "3.5"], "feature": [[1.0, 2.0], [3.0, 4.0]]},
            schema={"start_time": pl.String, "feature": pl.Array(pl.Float64, 2)},
        )
        pq.write_table(df.to_arrow(), path)
        with pytest.raises(ValueError, match="must be a numeric type"):
            read_feature(path)

    def test_int_list_cast_to_float64_array(self, tmp_path: Path):
        path = tmp_path / "sub-01_feature-mfcc_bold.parquet"
        df = pl.DataFrame(
            {"start_time": [0.0, 0.5], "feature": [[1, 2], [3, 4]]},
            schema={"start_time": pl.Float64, "feature": pl.List(pl.Int64)},
        )
        pq.write_table(df.to_arrow(), path)
        loaded, _ = read_feature(path)
        assert loaded.get_column("feature").dtype == pl.Array(pl.Float64, 2)

    def test_metadata_roundtrip(self, bids_path: Path, sample_df: pl.DataFrame):
        save_feature(sample_df, bids_path, metadata={"sr": "16000"})
        _, meta = read_feature(bids_path)
        assert meta["sr"] == "16000"


def _make_df(start_times: list[float], features: list[list[float]]) -> pl.DataFrame:
    dim = len(features[0])
    return pl.DataFrame(
        {"start_time": start_times, "feature": features},
        schema={"start_time": pl.Float64, "feature": pl.Array(pl.Float64, dim)},
    )


class TestResampleFeature:
    def test_pass_through_when_already_tr_aligned(self):
        tr = 2.0
        df = _make_df([0.0, 2.0, 4.0], [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = resample_feature(df, n_trs=3, repetition_time=tr, method="mean")
        assert np.allclose(result, [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    def test_downsample_mean_multiple_events_per_tr(self):
        tr = 2.0
        # Two events in TR 0 (0.0, 0.5), one in TR 1 (2.0)
        df = _make_df(
            [0.0, 0.5, 2.0],
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        )
        result = resample_feature(df, n_trs=2, repetition_time=tr, method="mean")
        assert np.allclose(result[0], [2.0, 3.0])  # mean of [1,2] and [3,4]
        assert np.allclose(result[1], [5.0, 6.0])

    def test_bin_boundary_event_at_exact_tr_goes_to_next_tr(self):
        tr = 2.0
        # Event at start=2.0 should go to TR 1, not TR 0
        df = _make_df([2.0], [[1.0, 2.0]])
        result = resample_feature(df, n_trs=2, repetition_time=tr, method="mean")
        assert np.allclose(result[0], [0.0, 0.0])  # TR 0 empty
        assert np.allclose(result[1], [1.0, 2.0])  # TR 1 has the event

    def test_event_at_onset_zero_goes_to_tr_zero(self):
        tr = 2.0
        df = _make_df([0.0], [[1.0, 2.0]])
        result = resample_feature(df, n_trs=2, repetition_time=tr, method="mean")
        assert np.allclose(result[0], [1.0, 2.0])

    def test_events_outside_tr_range_are_ignored(self):
        tr = 2.0
        # Event at start=10.0 is beyond n_trs=2 (covers 0-4s)
        df = _make_df([0.0, 10.0], [[1.0, 2.0], [9.0, 9.0]])
        result = resample_feature(df, n_trs=2, repetition_time=tr, method="mean")
        assert np.allclose(result[0], [1.0, 2.0])
        assert np.allclose(result[1], [0.0, 0.0])

    def test_empty_tr_bins_are_zero(self):
        tr = 2.0
        df = _make_df([0.0], [[1.0, 2.0]])
        result = resample_feature(df, n_trs=3, repetition_time=tr, method="mean")
        assert np.allclose(result[1], [0.0, 0.0])
        assert np.allclose(result[2], [0.0, 0.0])

    def test_output_shape(self):
        tr = 2.0
        df = _make_df([0.0, 0.5, 2.0], [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = resample_feature(df, n_trs=2, repetition_time=tr, method="mean")
        assert result.shape == (2, 2)

    def test_output_dtype_is_float64(self):
        df = _make_df([0.0, 2.0], [[1.0, 2.0], [3.0, 4.0]])
        result = resample_feature(df, n_trs=2, repetition_time=2.0, method="mean")
        assert result.dtype == np.float64

    def test_string_method_is_accepted(self):
        df = _make_df([0.0, 2.0], [[1.0], [2.0]])
        result = resample_feature(df, n_trs=2, repetition_time=2.0, method="mean")
        assert result.shape == (2, 1)

    def test_downsample_enum_method_is_accepted(self):
        df = _make_df([0.0, 2.0], [[1.0], [2.0]])
        result = resample_feature(
            df, n_trs=2, repetition_time=2.0, method=Downsample.MEAN
        )
        assert result.shape == (2, 1)

    def test_invalid_method_raises(self):
        df = _make_df([0.0], [[1.0]])
        with pytest.raises(ValueError):
            resample_feature(df, n_trs=1, repetition_time=2.0, method="invalid")

    def test_pass_through_for_phase_shifted_tr_aligned_data(self):
        # Data sampled at TR cadence but offset (e.g. mid-TR) triggers pass-through
        tr = 2.0
        df = _make_df([0.5, 2.5], [[1.0, 2.0], [3.0, 4.0]])
        result = resample_feature(df, n_trs=2, repetition_time=tr, method="mean")
        assert np.allclose(result, [[1.0, 2.0], [3.0, 4.0]])

    def test_single_tr_pass_through(self):
        # n_trs=1 means len(intervals)==0; should fall through to binning
        tr = 2.0
        df = _make_df([0.0], [[1.0, 2.0]])
        result = resample_feature(df, n_trs=1, repetition_time=tr, method="mean")
        assert np.allclose(result[0], [1.0, 2.0])

    def test_n_trs_zero_raises(self):
        df = _make_df([0.0, 2.0], [[1.0], [2.0]])
        with pytest.raises(ValueError, match="n_trs must be positive"):
            resample_feature(df, n_trs=0, repetition_time=2.0, method="mean")

    def test_empty_input_returns_zero_filled_output(self):
        df = pl.DataFrame(
            {"start_time": [], "feature": []},
            schema={"start_time": pl.Float64, "feature": pl.Array(pl.Float64, 2)},
        )
        result = resample_feature(df, n_trs=3, repetition_time=2.0, method="mean")
        assert result.shape == (3, 2)
        assert np.allclose(result, np.zeros((3, 2)))
