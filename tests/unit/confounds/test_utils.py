import json
import shutil
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
import pytest

from hypline.bids import BIDSPath
from hypline.confounds._utils import (
    read_confound,
    read_confound_metadata,
    save_confound,
)


@pytest.fixture()
def confound_path(tmp_path: Path) -> Path:
    return tmp_path / "sub-01_ses-1_conf-phonemic_desc-onset.parquet"


@pytest.fixture()
def sample_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "start_time": [0.0, 2.0, 4.0],
            "confound": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        },
        schema={"start_time": pl.Float64, "confound": pl.Array(pl.Float64, 2)},
    )


class TestSaveConfound:
    def test_roundtrip(self, confound_path: Path, sample_df: pl.DataFrame):
        save_confound(sample_df, confound_path, repetition_time=2.0, tr_method="mean")
        df = read_confound(confound_path)
        assert df.equals(sample_df)

    def test_creates_parent_dirs(self, tmp_path: Path, sample_df: pl.DataFrame):
        path = tmp_path / "a" / "b" / "sub-01_conf-phonemic_desc-onset.parquet"
        save_confound(sample_df, path, repetition_time=2.0, tr_method="mean")
        assert path.exists()

    def test_list_column_cast_to_array(self, confound_path: Path):
        df = pl.DataFrame(
            {"start_time": [0.0, 2.0], "confound": [[1.0, 2.0], [3.0, 4.0]]},
            schema={"start_time": pl.Float64, "confound": pl.List(pl.Float64)},
        )
        save_confound(df, confound_path, repetition_time=2.0, tr_method="mean")
        loaded = read_confound(confound_path)
        assert loaded.get_column("confound").dtype == pl.Array(pl.Float64, 2)

    def test_int_list_cast_to_float64(self, confound_path: Path):
        df = pl.DataFrame(
            {"start_time": [0.0, 2.0], "confound": [[1, 2], [3, 4]]},
            schema={"start_time": pl.Float64, "confound": pl.Array(pl.Int64, 2)},
        )
        save_confound(df, confound_path, repetition_time=2.0, tr_method="mean")
        loaded = read_confound(confound_path)
        assert loaded.get_column("confound").dtype == pl.Array(pl.Float64, 2)

    def test_single_row_allowed(self, confound_path: Path):
        df = pl.DataFrame(
            {"start_time": [0.0], "confound": [[1.0, 2.0]]},
            schema={"start_time": pl.Float64, "confound": pl.Array(pl.Float64, 2)},
        )
        save_confound(df, confound_path, repetition_time=2.0, tr_method="mean")
        meta = read_confound_metadata(confound_path)
        assert meta["repetition_time"] == 2.0
        assert meta["n_trs"] == 1

    def test_metadata_stored_in_footer(
        self, confound_path: Path, sample_df: pl.DataFrame
    ):
        save_confound(
            sample_df,
            confound_path,
            repetition_time=2.0,
            tr_method="mean",
            metadata={"key": "value"},
        )
        meta = read_confound_metadata(confound_path)
        assert meta["key"] == "value"

    def test_auto_metadata_injected(
        self, confound_path: Path, sample_df: pl.DataFrame
    ):
        save_confound(sample_df, confound_path, repetition_time=2.0, tr_method="sum")
        meta = read_confound_metadata(confound_path)
        assert meta["confound_kind"] == "phonemic"
        assert meta["confound_variant"] == "onset"
        assert meta["tr_method"] == "sum"
        assert meta["repetition_time"] == 2.0
        assert meta["n_trs"] == 3
        assert meta["confound_dim"] == 2
        assert "hypline_version" in meta

    def test_tr_method_none_allowed(
        self, confound_path: Path, sample_df: pl.DataFrame
    ):
        save_confound(sample_df, confound_path, repetition_time=2.0, tr_method=None)
        meta = read_confound_metadata(confound_path)
        assert meta["tr_method"] is None

    @pytest.mark.parametrize(
        "key", ["confound_kind", "confound_variant", "hypline_version", "tr_method",
                "repetition_time", "n_trs", "confound_dim"],
    )
    def test_reserved_keys_raise(
        self, confound_path: Path, sample_df: pl.DataFrame, key: str
    ):
        with pytest.raises(ValueError, match="reserved keys"):
            save_confound(
                sample_df,
                confound_path,
                repetition_time=2.0,
                tr_method="mean",
                metadata={key: "x"},
            )

    def test_missing_required_columns(self, confound_path: Path):
        df = pl.DataFrame({"onset": [0.0, 2.0]})
        with pytest.raises(ValueError, match="missing required columns"):
            save_confound(df, confound_path, repetition_time=2.0, tr_method="mean")

    def test_non_numeric_start_time(self, confound_path: Path):
        df = pl.DataFrame(
            {"start_time": ["0.0", "2.0"], "confound": [[1.0], [2.0]]},
            schema={"start_time": pl.String, "confound": pl.Array(pl.Float64, 1)},
        )
        with pytest.raises(ValueError, match="must be a numeric type"):
            save_confound(df, confound_path, repetition_time=2.0, tr_method="mean")

    def test_empty_dataframe_raises(self, confound_path: Path):
        df = pl.DataFrame(
            {"start_time": [], "confound": []},
            schema={"start_time": pl.Float64, "confound": pl.Array(pl.Float64, 1)},
        )
        with pytest.raises(ValueError, match="at least one row"):
            save_confound(df, confound_path, repetition_time=2.0, tr_method="mean")

    def test_unsupported_confound_dtype(self, confound_path: Path):
        df = pl.DataFrame({"start_time": [0.0, 2.0], "confound": ["a", "b"]})
        with pytest.raises(ValueError, match="must be an Array or List type"):
            save_confound(df, confound_path, repetition_time=2.0, tr_method="mean")

    def test_start_time_must_begin_at_zero(self, confound_path: Path):
        df = pl.DataFrame(
            {"start_time": [2.0, 4.0], "confound": [[1.0], [2.0]]},
            schema={"start_time": pl.Float64, "confound": pl.Array(pl.Float64, 1)},
        )
        with pytest.raises(ValueError, match="must begin at 0.0"):
            save_confound(df, confound_path, repetition_time=2.0, tr_method="mean")

    def test_intervals_must_match_repetition_time(self, confound_path: Path):
        df = pl.DataFrame(
            {"start_time": [0.0, 1.0, 2.0], "confound": [[1.0], [2.0], [3.0]]},
            schema={"start_time": pl.Float64, "confound": pl.Array(pl.Float64, 1)},
        )
        with pytest.raises(ValueError, match="intervals must equal repetition_time"):
            save_confound(df, confound_path, repetition_time=2.0, tr_method="mean")

    def test_non_uniform_intervals_raise(self, confound_path: Path):
        df = pl.DataFrame(
            {"start_time": [0.0, 2.0, 5.0], "confound": [[1.0], [2.0], [3.0]]},
            schema={"start_time": pl.Float64, "confound": pl.Array(pl.Float64, 1)},
        )
        with pytest.raises(ValueError, match="intervals must equal repetition_time"):
            save_confound(df, confound_path, repetition_time=2.0, tr_method="mean")

    def test_missing_conf_entity_in_path(
        self, tmp_path: Path, sample_df: pl.DataFrame
    ):
        path = tmp_path / "sub-01_ses-1_bold.parquet"
        with pytest.raises(ValueError, match="must contain a 'conf' entity"):
            save_confound(sample_df, path, repetition_time=2.0, tr_method="mean")

    def test_non_parquet_extension(self, tmp_path: Path, sample_df: pl.DataFrame):
        path = tmp_path / "sub-01_conf-phonemic_desc-onset.tsv"
        with pytest.raises(ValueError, match=".parquet extension"):
            save_confound(sample_df, path, repetition_time=2.0, tr_method="mean")

    def test_rejects_bids_suffix(self, tmp_path: Path, sample_df: pl.DataFrame):
        path = tmp_path / "sub-01_conf-phonemic_desc-onset_bold.parquet"
        with pytest.raises(ValueError, match="must not have a BIDS suffix"):
            save_confound(sample_df, path, repetition_time=2.0, tr_method="mean")

    def test_desc_optional(self, tmp_path: Path, sample_df: pl.DataFrame):
        path = tmp_path / "sub-01_ses-1_conf-phonemic.parquet"
        save_confound(sample_df, path, repetition_time=2.0, tr_method="mean")
        meta = read_confound_metadata(path)
        assert meta["confound_kind"] == "phonemic"
        assert meta["confound_variant"] is None
        loaded = read_confound(path)
        assert loaded.equals(sample_df)


class TestReadConfoundMetadata:
    def test_missing_conf_entity(self, tmp_path: Path):
        path = tmp_path / "sub-01.parquet"
        with pytest.raises(ValueError, match="must contain a 'conf' entity"):
            read_confound_metadata(path)

    def test_no_hypline_metadata_raises(
        self, confound_path: Path, sample_df: pl.DataFrame
    ):
        pq.write_table(sample_df.to_arrow(), confound_path)
        with pytest.raises(ValueError, match="no hypline metadata"):
            read_confound_metadata(confound_path)

    def test_confound_kind_mismatch_raises(
        self, tmp_path: Path, sample_df: pl.DataFrame
    ):
        src = tmp_path / "sub-01_conf-phonemic_desc-onset.parquet"
        dst = tmp_path / "sub-01_conf-other_desc-onset.parquet"
        save_confound(sample_df, src, repetition_time=2.0, tr_method="mean")
        shutil.copy(src, dst)
        with pytest.raises(ValueError, match="confound_kind metadata"):
            read_confound_metadata(dst)

    def test_confound_variant_mismatch_raises(
        self, tmp_path: Path, sample_df: pl.DataFrame
    ):
        src = tmp_path / "sub-01_conf-phonemic_desc-onset.parquet"
        dst = tmp_path / "sub-01_conf-phonemic_desc-rate.parquet"
        save_confound(sample_df, src, repetition_time=2.0, tr_method="mean")
        shutil.copy(src, dst)
        with pytest.raises(ValueError, match="confound_variant metadata"):
            read_confound_metadata(dst)


def _write_raw_confound(df: pl.DataFrame, path: Path, **meta_overrides) -> None:
    """Write a parquet with hypline metadata, bypassing save_confound validation."""
    bids = BIDSPath(path)
    metadata = {
        "confound_kind": bids.entities["conf"],
        "confound_variant": bids.entities.get("desc"),
        "repetition_time": 2.0,
        **meta_overrides,
    }
    table = df.to_arrow().replace_schema_metadata(
        {b"hypline": json.dumps(metadata).encode()}
    )
    pq.write_table(table, path)


class TestReadConfound:
    def test_missing_conf_entity(self, tmp_path: Path):
        path = tmp_path / "sub-01_bold.parquet"
        with pytest.raises(ValueError, match="must contain a 'conf' entity"):
            read_confound(path)

    def test_raw_parquet_without_hypline_metadata_raises(
        self, confound_path: Path, sample_df: pl.DataFrame
    ):
        pq.write_table(sample_df.to_arrow(), confound_path)
        with pytest.raises(ValueError, match="no hypline metadata"):
            read_confound(confound_path)

    def test_validates_stored_tr_against_row_spacing(
        self, confound_path: Path, sample_df: pl.DataFrame
    ):
        _write_raw_confound(sample_df, confound_path, repetition_time=1.0)
        with pytest.raises(ValueError, match="intervals must equal repetition_time"):
            read_confound(confound_path)

    def test_confound_kind_mismatch_raises(
        self, tmp_path: Path, sample_df: pl.DataFrame
    ):
        src = tmp_path / "sub-01_conf-phonemic_desc-onset.parquet"
        dst = tmp_path / "sub-01_conf-other_desc-onset.parquet"
        save_confound(sample_df, src, repetition_time=2.0, tr_method="mean")
        shutil.copy(src, dst)
        with pytest.raises(ValueError, match="confound_kind metadata"):
            read_confound(dst)

    def test_confound_variant_mismatch_raises(
        self, tmp_path: Path, sample_df: pl.DataFrame
    ):
        src = tmp_path / "sub-01_conf-phonemic_desc-onset.parquet"
        dst = tmp_path / "sub-01_conf-phonemic_desc-rate.parquet"
        save_confound(sample_df, src, repetition_time=2.0, tr_method="mean")
        shutil.copy(src, dst)
        with pytest.raises(ValueError, match="confound_variant metadata"):
            read_confound(dst)
