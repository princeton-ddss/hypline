from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
import pytest

from hypline.featuregen.utils import read_feature, save_feature


@pytest.fixture()
def bids_path(tmp_path: Path) -> Path:
    return tmp_path / "sub-01_ses-1_feature-mfcc_bold.parquet"


@pytest.fixture()
def sample_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "onset": [0.0, 0.5, 1.0],
            "feature": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        },
        schema={"onset": pl.Float64, "feature": pl.Array(pl.Float64, 3)},
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
            {"feature": [[1.0, 2.0], [3.0, 4.0]]},
            schema={"feature": pl.List(pl.Float64)},
        )
        save_feature(df, bids_path)
        loaded, _ = read_feature(bids_path)
        assert loaded.get_column("feature").dtype == pl.Array(pl.Float64, 2)

    def test_int_list_cast_to_float64(self, bids_path: Path):
        df = pl.DataFrame(
            {"feature": [[1, 2], [3, 4]]},
            schema={"feature": pl.Array(pl.Int64, 2)},
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

    def test_missing_feature_column(self, bids_path: Path):
        df = pl.DataFrame({"onset": [0.0, 1.0]})
        with pytest.raises(ValueError, match="must contain a 'feature' column"):
            save_feature(df, bids_path)

    def test_unsupported_feature_dtype(self, bids_path: Path):
        df = pl.DataFrame({"feature": ["a", "b"]})
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

    def test_missing_feature_column(self, tmp_path: Path):
        path = tmp_path / "sub-01_feature-mfcc_bold.parquet"
        table = pl.DataFrame({"onset": [0.0]}).to_arrow()
        pq.write_table(table, path)
        with pytest.raises(ValueError, match="must contain a 'feature' column"):
            read_feature(path)

    def test_wrong_feature_dtype(self, tmp_path: Path):
        path = tmp_path / "sub-01_feature-mfcc_bold.parquet"
        df = pl.DataFrame(
            {"feature": [[1, 2], [3, 4]]},
            schema={"feature": pl.Array(pl.Int64, 2)},
        )
        pq.write_table(df.to_arrow(), path)
        with pytest.raises(ValueError, match="must be an Array\\(Float64\\)"):
            read_feature(path)

    def test_metadata_roundtrip(self, bids_path: Path, sample_df: pl.DataFrame):
        save_feature(sample_df, bids_path, metadata={"sr": "16000"})
        _, meta = read_feature(bids_path)
        assert meta["sr"] == "16000"
