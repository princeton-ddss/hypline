import json
import shutil
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import pytest

from hypline.bids import BIDSPath
from hypline.features._utils import (
    Downsample,
    read_feature,
    read_feature_metadata,
    resample_feature,
    save_feature,
)


@pytest.fixture()
def feature_path(tmp_path: Path) -> Path:
    return tmp_path / "sub-01_ses-1_feature-mfcc.parquet"


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
    def test_roundtrip(self, feature_path: Path, sample_df: pl.DataFrame):
        save_feature(sample_df, feature_path)
        df = read_feature(feature_path)
        assert df.equals(sample_df)

    def test_creates_parent_dirs(self, tmp_path: Path, sample_df: pl.DataFrame):
        path = tmp_path / "a" / "b" / "sub-01_feature-mfcc.parquet"
        save_feature(sample_df, path)
        assert path.exists()

    def test_list_column_cast_to_array(self, feature_path: Path):
        df = pl.DataFrame(
            {"start_time": [0.0, 0.5], "feature": [[1.0, 2.0], [3.0, 4.0]]},
            schema={"start_time": pl.Float64, "feature": pl.List(pl.Float64)},
        )
        save_feature(df, feature_path)
        loaded = read_feature(feature_path)
        assert loaded.get_column("feature").dtype == pl.Array(pl.Float64, 2)

    def test_int_list_cast_to_float64(self, feature_path: Path):
        df = pl.DataFrame(
            {"start_time": [0.0, 0.5], "feature": [[1, 2], [3, 4]]},
            schema={"start_time": pl.Float64, "feature": pl.Array(pl.Int64, 2)},
        )
        save_feature(df, feature_path)
        loaded = read_feature(feature_path)
        assert loaded.get_column("feature").dtype == pl.Array(pl.Float64, 2)

    def test_metadata_stored_in_footer(
        self, feature_path: Path, sample_df: pl.DataFrame
    ):
        save_feature(sample_df, feature_path, metadata={"key": "value", "foo": "bar"})
        meta = read_feature_metadata(feature_path)
        assert meta["key"] == "value"
        assert meta["foo"] == "bar"

    def test_list_valued_metadata_roundtrip(self, feature_path, sample_df):
        save_feature(sample_df, feature_path, metadata={"dim_labels": ["a", "b", "c"]})
        meta = read_feature_metadata(feature_path)
        assert meta["dim_labels"] == ["a", "b", "c"]

    def test_caller_metadata_absent(self, feature_path: Path, sample_df: pl.DataFrame):
        save_feature(sample_df, feature_path)
        meta = read_feature_metadata(feature_path)
        assert meta["feature_name"] == "mfcc"
        assert "hypline_version" in meta

    def test_reserved_key_feature_name_raises(
        self, feature_path: Path, sample_df: pl.DataFrame
    ):
        with pytest.raises(ValueError, match="reserved keys"):
            save_feature(sample_df, feature_path, metadata={"feature_name": "mfcc"})

    def test_reserved_key_hypline_version_raises(
        self, feature_path: Path, sample_df: pl.DataFrame
    ):
        with pytest.raises(ValueError, match="reserved keys"):
            save_feature(sample_df, feature_path, metadata={"hypline_version": "0.0.0"})

    def test_missing_required_columns(self, feature_path: Path):
        df = pl.DataFrame({"onset": [0.0, 1.0]})
        with pytest.raises(ValueError, match="missing required columns"):
            save_feature(df, feature_path)

    def test_missing_start_time_column(self, feature_path: Path):
        df = pl.DataFrame(
            {"feature": [[1.0, 2.0], [3.0, 4.0]]},
            schema={"feature": pl.Array(pl.Float64, 2)},
        )
        with pytest.raises(ValueError, match="missing required columns"):
            save_feature(df, feature_path)

    def test_non_numeric_start_time(self, feature_path: Path):
        df = pl.DataFrame(
            {"start_time": ["1.2", "3.5"], "feature": [[1.0, 2.0], [3.0, 4.0]]},
            schema={"start_time": pl.String, "feature": pl.Array(pl.Float64, 2)},
        )
        with pytest.raises(ValueError, match="must be a numeric type"):
            save_feature(df, feature_path)

    def test_unsupported_feature_dtype(self, feature_path: Path):
        df = pl.DataFrame({"start_time": [0.0, 0.5], "feature": ["a", "b"]})
        with pytest.raises(ValueError, match="must be an Array or List type"):
            save_feature(df, feature_path)

    def test_missing_feature_entity_in_path(
        self, tmp_path: Path, sample_df: pl.DataFrame
    ):
        path = tmp_path / "sub-01_ses-1_bold.parquet"
        with pytest.raises(ValueError, match="must contain a 'feature' entity"):
            save_feature(sample_df, path)

    def test_non_parquet_extension(self, tmp_path: Path, sample_df: pl.DataFrame):
        path = tmp_path / "sub-01_feature-mfcc_bold.tsv"
        with pytest.raises(ValueError, match=".parquet extension"):
            save_feature(sample_df, path)

    def test_rejects_bids_suffix(self, tmp_path: Path, sample_df: pl.DataFrame):
        path = tmp_path / "sub-01_feature-mfcc_bold.parquet"
        with pytest.raises(ValueError, match="must not have a BIDS suffix"):
            save_feature(sample_df, path)


class TestReadFeatureMetadata:
    def test_missing_feature_entity(self, tmp_path: Path):
        path = tmp_path / "sub-01.parquet"
        with pytest.raises(ValueError, match="must contain a 'feature' entity"):
            read_feature_metadata(path)

    def test_rejects_bids_suffix(self, tmp_path: Path):
        path = tmp_path / "sub-01_feature-mfcc_bold.parquet"
        with pytest.raises(ValueError, match="must not have a BIDS suffix"):
            read_feature_metadata(path)

    def test_no_hypline_metadata_raises(
        self, feature_path: Path, sample_df: pl.DataFrame
    ):
        pq.write_table(sample_df.to_arrow(), feature_path)
        with pytest.raises(ValueError, match="no hypline metadata"):
            read_feature_metadata(feature_path)

    def test_feature_name_mismatch_raises(
        self, tmp_path: Path, sample_df: pl.DataFrame
    ):
        src = tmp_path / "sub-01_feature-phonemic.parquet"
        dst = tmp_path / "sub-01_feature-mfcc.parquet"
        save_feature(sample_df, src)
        shutil.copy(src, dst)
        with pytest.raises(ValueError, match="does not match path entity"):
            read_feature_metadata(dst)

    def test_returns_metadata_without_loading_data(
        self, feature_path: Path, sample_df: pl.DataFrame
    ):
        save_feature(sample_df, feature_path, metadata={"sr": "16000"})
        meta = read_feature_metadata(feature_path)
        assert meta["sr"] == "16000"
        assert meta["feature_name"] == "mfcc"
        assert "hypline_version" in meta


def _write_raw_feature(df: pl.DataFrame, path: Path) -> None:
    """Write a parquet with hypline metadata, bypassing save_feature validation."""
    feature_name = BIDSPath(path).entities["feature"]
    table = df.to_arrow().replace_schema_metadata(
        {b"hypline": json.dumps({"feature_name": feature_name}).encode()}
    )
    pq.write_table(table, path)


class TestReadFeature:
    def test_missing_feature_entity(self, tmp_path: Path):
        path = tmp_path / "sub-01_bold.parquet"
        with pytest.raises(ValueError, match="must contain a 'feature' entity"):
            read_feature(path)

    def test_non_parquet_extension(self, tmp_path: Path):
        path = tmp_path / "sub-01_feature-mfcc_bold.tsv"
        with pytest.raises(ValueError, match=".parquet extension"):
            read_feature(path)

    def test_rejects_bids_suffix(self, tmp_path: Path):
        path = tmp_path / "sub-01_feature-mfcc_bold.parquet"
        with pytest.raises(ValueError, match="must not have a BIDS suffix"):
            read_feature(path)

    def test_missing_required_columns(self, feature_path: Path):
        df = pl.DataFrame({"onset": [0.0]})
        _write_raw_feature(df, feature_path)
        with pytest.raises(ValueError, match="missing required columns"):
            read_feature(feature_path)

    def test_missing_start_time_column(self, feature_path: Path):
        df = pl.DataFrame(
            {"feature": [[1.0, 2.0], [3.0, 4.0]]},
            schema={"feature": pl.Array(pl.Float64, 2)},
        )
        _write_raw_feature(df, feature_path)
        with pytest.raises(ValueError, match="missing required columns"):
            read_feature(feature_path)

    def test_non_numeric_start_time(self, feature_path: Path):
        df = pl.DataFrame(
            {"start_time": ["1.2", "3.5"], "feature": [[1.0, 2.0], [3.0, 4.0]]},
            schema={"start_time": pl.String, "feature": pl.Array(pl.Float64, 2)},
        )
        _write_raw_feature(df, feature_path)
        with pytest.raises(ValueError, match="must be a numeric type"):
            read_feature(feature_path)

    def test_int_list_cast_to_float64_array(self, feature_path: Path):
        df = pl.DataFrame(
            {"start_time": [0.0, 0.5], "feature": [[1, 2], [3, 4]]},
            schema={"start_time": pl.Float64, "feature": pl.List(pl.Int64)},
        )
        _write_raw_feature(df, feature_path)
        loaded = read_feature(feature_path)
        assert loaded.get_column("feature").dtype == pl.Array(pl.Float64, 2)

    def test_metadata_roundtrip(self, feature_path: Path, sample_df: pl.DataFrame):
        save_feature(sample_df, feature_path, metadata={"sr": "16000"})
        meta = read_feature_metadata(feature_path)
        assert meta["sr"] == "16000"
        assert "feature_name" in meta
        assert "hypline_version" in meta

    def test_raw_parquet_without_hypline_metadata_raises(
        self, feature_path: Path, sample_df: pl.DataFrame
    ):
        pq.write_table(sample_df.to_arrow(), feature_path)
        with pytest.raises(ValueError, match="no hypline metadata"):
            read_feature(feature_path)

    def test_feature_name_mismatch_raises(
        self, tmp_path: Path, sample_df: pl.DataFrame
    ):
        src = tmp_path / "sub-01_feature-phonemic.parquet"
        dst = tmp_path / "sub-01_feature-mfcc.parquet"
        save_feature(sample_df, src)
        shutil.copy(src, dst)
        with pytest.raises(ValueError, match="does not match path entity"):
            read_feature(dst)


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
