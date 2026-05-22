import json
import shutil
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
import pytest

from hypline.bids import BIDSPath
from hypline.io import (
    read_feature,
    read_feature_metadata,
    save_feature,
)


@pytest.fixture()
def feature_path(tmp_path: Path) -> Path:
    return tmp_path / "sub-01_ses-1_feat-mfcc.parquet"


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
        path = tmp_path / "a" / "b" / "sub-01_feat-mfcc.parquet"
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
        assert meta["feature_dim"] == 3
        assert "hypline_version" in meta

    @pytest.mark.parametrize(
        "key",
        ["feature_name", "hypline_version", "feature_dim"],
    )
    def test_reserved_keys_raise(
        self, feature_path: Path, sample_df: pl.DataFrame, key: str
    ):
        with pytest.raises(ValueError, match="reserved keys"):
            save_feature(sample_df, feature_path, metadata={key: "x"})

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
        with pytest.raises(ValueError, match="must contain a 'feat' entity"):
            save_feature(sample_df, path)

    def test_non_parquet_extension(self, tmp_path: Path, sample_df: pl.DataFrame):
        path = tmp_path / "sub-01_feat-mfcc_bold.tsv"
        with pytest.raises(ValueError, match=".parquet extension"):
            save_feature(sample_df, path)

    def test_rejects_bids_suffix(self, tmp_path: Path, sample_df: pl.DataFrame):
        path = tmp_path / "sub-01_feat-mfcc_bold.parquet"
        with pytest.raises(ValueError, match="must not have a BIDS suffix"):
            save_feature(sample_df, path)


class TestReadFeatureMetadata:
    def test_missing_feature_entity(self, tmp_path: Path):
        path = tmp_path / "sub-01.parquet"
        with pytest.raises(ValueError, match="must contain a 'feat' entity"):
            read_feature_metadata(path)

    def test_rejects_bids_suffix(self, tmp_path: Path):
        path = tmp_path / "sub-01_feat-mfcc_bold.parquet"
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
        src = tmp_path / "sub-01_feat-phonemic.parquet"
        dst = tmp_path / "sub-01_feat-mfcc.parquet"
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
    feature_name = BIDSPath(path).entities["feat"]
    table = df.to_arrow().replace_schema_metadata(
        {b"hypline": json.dumps({"feature_name": feature_name}).encode()}
    )
    pq.write_table(table, path)


class TestReadFeature:
    def test_missing_feature_entity(self, tmp_path: Path):
        path = tmp_path / "sub-01_bold.parquet"
        with pytest.raises(ValueError, match="must contain a 'feat' entity"):
            read_feature(path)

    def test_non_parquet_extension(self, tmp_path: Path):
        path = tmp_path / "sub-01_feat-mfcc_bold.tsv"
        with pytest.raises(ValueError, match=".parquet extension"):
            read_feature(path)

    def test_rejects_bids_suffix(self, tmp_path: Path):
        path = tmp_path / "sub-01_feat-mfcc_bold.parquet"
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
        src = tmp_path / "sub-01_feat-phonemic.parquet"
        dst = tmp_path / "sub-01_feat-mfcc.parquet"
        save_feature(sample_df, src)
        shutil.copy(src, dst)
        with pytest.raises(ValueError, match="does not match path entity"):
            read_feature(dst)
