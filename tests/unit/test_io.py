import json
import shutil
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
import pytest
from loguru import logger

from hypline.bids import BIDSPath
from hypline.io import (
    read_confound,
    read_confound_metadata,
    read_feature,
    read_feature_metadata,
    read_nuisance,
    save_confound,
    save_feature,
    skip_existing,
    write_confound,
    write_feature,
)


@pytest.fixture()
def feature_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "start_time": [0.0, 0.5, 1.0],
            "feature": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        },
        schema={"start_time": pl.Float64, "feature": pl.Array(pl.Float64, 2)},
    )


@pytest.fixture()
def confound_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "start_time": [0.0, 2.0, 4.0],
            "confound": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        },
        schema={"start_time": pl.Float64, "confound": pl.Array(pl.Float64, 2)},
    )


@pytest.fixture()
def feature_path(tmp_path: Path) -> Path:
    return tmp_path / "sub-01_ses-1_feat-mfcc.parquet"


@pytest.fixture()
def confound_path(tmp_path: Path) -> Path:
    return tmp_path / "sub-01_ses-1_conf-phonemic_desc-onset.parquet"


def _write_raw_feature(df: pl.DataFrame, path: Path) -> None:
    """Write a parquet with hypline metadata, bypassing write_feature validation."""
    feature_name = BIDSPath(path).entities["feat"]
    table = df.to_arrow().replace_schema_metadata(
        {b"hypline": json.dumps({"feature_name": feature_name}).encode()}
    )
    pq.write_table(table, path)


def _write_raw_confound(df: pl.DataFrame, path: Path, **meta_overrides) -> None:
    """Write a parquet with hypline metadata, bypassing write_confound validation."""
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


# ---------------------------------------------------------------------------
# Path-based: feature
# ---------------------------------------------------------------------------


class TestSkipExisting:
    def test_missing_path_not_skipped(self, tmp_path: Path):
        assert skip_existing(tmp_path / "absent.parquet", force=False) is False

    def test_existing_path_skipped(self, tmp_path: Path):
        path = tmp_path / "out.parquet"
        path.write_bytes(b"sentinel")
        assert skip_existing(path, force=False) is True
        assert path.read_bytes() == b"sentinel"

    def test_force_overrides_existing(self, tmp_path: Path):
        path = tmp_path / "out.parquet"
        path.write_bytes(b"sentinel")
        assert skip_existing(path, force=True) is False

    def test_logs_info_on_skip(self, tmp_path: Path):
        path = tmp_path / "out.parquet"
        path.write_bytes(b"sentinel")
        messages: list[str] = []
        sink_id = logger.add(messages.append, level="INFO", format="{message}")
        try:
            skip_existing(path, force=False)
        finally:
            logger.remove(sink_id)
        assert any("out.parquet" in m for m in messages)


class TestWriteFeature:
    def test_roundtrip(self, feature_path: Path, feature_df: pl.DataFrame):
        write_feature(feature_df, feature_path)
        df = read_feature(feature_path)
        assert df.equals(feature_df)

    def test_creates_parent_dirs(self, tmp_path: Path, feature_df: pl.DataFrame):
        path = tmp_path / "a" / "b" / "sub-01_feat-mfcc.parquet"
        write_feature(feature_df, path)
        assert path.exists()

    def test_list_column_cast_to_array(self, feature_path: Path):
        df = pl.DataFrame(
            {"start_time": [0.0, 0.5], "feature": [[1.0, 2.0], [3.0, 4.0]]},
            schema={"start_time": pl.Float64, "feature": pl.List(pl.Float64)},
        )
        write_feature(df, feature_path)
        loaded = read_feature(feature_path)
        assert loaded.get_column("feature").dtype == pl.Array(pl.Float64, 2)

    def test_int_list_cast_to_float64(self, feature_path: Path):
        df = pl.DataFrame(
            {"start_time": [0.0, 0.5], "feature": [[1, 2], [3, 4]]},
            schema={"start_time": pl.Float64, "feature": pl.Array(pl.Int64, 2)},
        )
        write_feature(df, feature_path)
        loaded = read_feature(feature_path)
        assert loaded.get_column("feature").dtype == pl.Array(pl.Float64, 2)

    def test_caller_metadata_absent(self, feature_path: Path, feature_df: pl.DataFrame):
        write_feature(feature_df, feature_path)
        meta = read_feature_metadata(feature_path)
        assert meta["feature_name"] == "mfcc"
        assert meta["feature_dim"] == 2
        assert "hypline_version" in meta

    def test_metadata_stored_in_footer(
        self, feature_path: Path, feature_df: pl.DataFrame
    ):
        write_feature(feature_df, feature_path, metadata={"key": "value", "foo": "bar"})
        meta = read_feature_metadata(feature_path)
        assert meta["key"] == "value"
        assert meta["foo"] == "bar"

    def test_list_valued_metadata_roundtrip(self, feature_path, feature_df):
        write_feature(feature_df, feature_path, metadata={"dim_labels": ["a", "b"]})
        meta = read_feature_metadata(feature_path)
        assert meta["dim_labels"] == ["a", "b"]

    @pytest.mark.parametrize(
        "key",
        ["feature_name", "hypline_version", "feature_dim"],
    )
    def test_reserved_keys_raise(
        self, feature_path: Path, feature_df: pl.DataFrame, key: str
    ):
        with pytest.raises(ValueError, match="reserved keys"):
            write_feature(feature_df, feature_path, metadata={key: "x"})

    def test_missing_required_columns(self, feature_path: Path):
        df = pl.DataFrame({"onset": [0.0, 1.0]})
        with pytest.raises(ValueError, match="missing required columns"):
            write_feature(df, feature_path)

    def test_missing_start_time_column(self, feature_path: Path):
        df = pl.DataFrame(
            {"feature": [[1.0, 2.0], [3.0, 4.0]]},
            schema={"feature": pl.Array(pl.Float64, 2)},
        )
        with pytest.raises(ValueError, match="missing required columns"):
            write_feature(df, feature_path)

    def test_non_numeric_start_time(self, feature_path: Path):
        df = pl.DataFrame(
            {"start_time": ["1.2", "3.5"], "feature": [[1.0, 2.0], [3.0, 4.0]]},
            schema={"start_time": pl.String, "feature": pl.Array(pl.Float64, 2)},
        )
        with pytest.raises(ValueError, match="must be a numeric type"):
            write_feature(df, feature_path)

    def test_unsupported_feature_dtype(self, feature_path: Path):
        df = pl.DataFrame({"start_time": [0.0, 0.5], "feature": ["a", "b"]})
        with pytest.raises(ValueError, match="must be an Array or List type"):
            write_feature(df, feature_path)

    def test_missing_feature_entity_in_path(
        self, tmp_path: Path, feature_df: pl.DataFrame
    ):
        path = tmp_path / "sub-01_ses-1_bold.parquet"
        with pytest.raises(ValueError, match="must contain a 'feat' entity"):
            write_feature(feature_df, path)

    def test_non_parquet_extension(self, tmp_path: Path, feature_df: pl.DataFrame):
        path = tmp_path / "sub-01_feat-mfcc_bold.tsv"
        with pytest.raises(ValueError, match=".parquet extension"):
            write_feature(feature_df, path)

    def test_rejects_bids_suffix(self, tmp_path: Path, feature_df: pl.DataFrame):
        path = tmp_path / "sub-01_feat-mfcc_bold.parquet"
        with pytest.raises(ValueError, match="must not have a BIDS suffix"):
            write_feature(feature_df, path)


class TestReadFeature:
    def test_int_list_cast_to_float64_array(self, feature_path: Path):
        df = pl.DataFrame(
            {"start_time": [0.0, 0.5], "feature": [[1, 2], [3, 4]]},
            schema={"start_time": pl.Float64, "feature": pl.List(pl.Int64)},
        )
        _write_raw_feature(df, feature_path)
        loaded = read_feature(feature_path)
        assert loaded.get_column("feature").dtype == pl.Array(pl.Float64, 2)

    def test_metadata_roundtrip(self, feature_path: Path, feature_df: pl.DataFrame):
        write_feature(feature_df, feature_path, metadata={"sr": "16000"})
        meta = read_feature_metadata(feature_path)
        assert meta["sr"] == "16000"
        assert "feature_name" in meta
        assert "hypline_version" in meta

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

    def test_raw_parquet_without_hypline_metadata_raises(
        self, feature_path: Path, feature_df: pl.DataFrame
    ):
        pq.write_table(feature_df.to_arrow(), feature_path)
        with pytest.raises(ValueError, match="no hypline metadata"):
            read_feature(feature_path)

    def test_feature_name_mismatch_raises(
        self, tmp_path: Path, feature_df: pl.DataFrame
    ):
        src = tmp_path / "sub-01_feat-phonemic.parquet"
        dst = tmp_path / "sub-01_feat-mfcc.parquet"
        write_feature(feature_df, src)
        shutil.copy(src, dst)
        with pytest.raises(ValueError, match="does not match path entity"):
            read_feature(dst)

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


class TestReadFeatureMetadata:
    def test_returns_metadata_without_loading_data(
        self, feature_path: Path, feature_df: pl.DataFrame
    ):
        write_feature(feature_df, feature_path, metadata={"sr": "16000"})
        meta = read_feature_metadata(feature_path)
        assert meta["sr"] == "16000"
        assert meta["feature_name"] == "mfcc"
        assert "hypline_version" in meta

    def test_no_hypline_metadata_raises(
        self, feature_path: Path, feature_df: pl.DataFrame
    ):
        pq.write_table(feature_df.to_arrow(), feature_path)
        with pytest.raises(ValueError, match="no hypline metadata"):
            read_feature_metadata(feature_path)

    def test_feature_name_mismatch_raises(
        self, tmp_path: Path, feature_df: pl.DataFrame
    ):
        src = tmp_path / "sub-01_feat-phonemic.parquet"
        dst = tmp_path / "sub-01_feat-mfcc.parquet"
        write_feature(feature_df, src)
        shutil.copy(src, dst)
        with pytest.raises(ValueError, match="does not match path entity"):
            read_feature_metadata(dst)

    def test_missing_feature_entity(self, tmp_path: Path):
        path = tmp_path / "sub-01.parquet"
        with pytest.raises(ValueError, match="must contain a 'feat' entity"):
            read_feature_metadata(path)

    def test_rejects_bids_suffix(self, tmp_path: Path):
        path = tmp_path / "sub-01_feat-mfcc_bold.parquet"
        with pytest.raises(ValueError, match="must not have a BIDS suffix"):
            read_feature_metadata(path)


# ---------------------------------------------------------------------------
# Path-based: confound
# ---------------------------------------------------------------------------


class TestWriteConfound:
    def test_roundtrip(self, confound_path: Path, confound_df: pl.DataFrame):
        write_confound(
            confound_df, confound_path, repetition_time=2.0, tr_method="mean"
        )
        df = read_confound(confound_path)
        assert df.equals(confound_df)

    def test_creates_parent_dirs(self, tmp_path: Path, confound_df: pl.DataFrame):
        path = tmp_path / "a" / "b" / "sub-01_conf-phonemic_desc-onset.parquet"
        write_confound(confound_df, path, repetition_time=2.0, tr_method="mean")
        assert path.exists()

    def test_list_column_cast_to_array(self, confound_path: Path):
        df = pl.DataFrame(
            {"start_time": [0.0, 2.0], "confound": [[1.0, 2.0], [3.0, 4.0]]},
            schema={"start_time": pl.Float64, "confound": pl.List(pl.Float64)},
        )
        write_confound(df, confound_path, repetition_time=2.0, tr_method="mean")
        loaded = read_confound(confound_path)
        assert loaded.get_column("confound").dtype == pl.Array(pl.Float64, 2)

    def test_int_list_cast_to_float64(self, confound_path: Path):
        df = pl.DataFrame(
            {"start_time": [0.0, 2.0], "confound": [[1, 2], [3, 4]]},
            schema={"start_time": pl.Float64, "confound": pl.Array(pl.Int64, 2)},
        )
        write_confound(df, confound_path, repetition_time=2.0, tr_method="mean")
        loaded = read_confound(confound_path)
        assert loaded.get_column("confound").dtype == pl.Array(pl.Float64, 2)

    def test_single_row_allowed(self, confound_path: Path):
        df = pl.DataFrame(
            {"start_time": [0.0], "confound": [[1.0, 2.0]]},
            schema={"start_time": pl.Float64, "confound": pl.Array(pl.Float64, 2)},
        )
        write_confound(df, confound_path, repetition_time=2.0, tr_method="mean")
        meta = read_confound_metadata(confound_path)
        assert meta["repetition_time"] == 2.0
        assert meta["n_trs"] == 1

    def test_desc_optional(self, tmp_path: Path, confound_df: pl.DataFrame):
        path = tmp_path / "sub-01_ses-1_conf-phonemic.parquet"
        write_confound(confound_df, path, repetition_time=2.0, tr_method="mean")
        meta = read_confound_metadata(path)
        assert meta["confound_kind"] == "phonemic"
        assert meta["confound_variant"] is None
        loaded = read_confound(path)
        assert loaded.equals(confound_df)

    def test_auto_metadata_injected(
        self, confound_path: Path, confound_df: pl.DataFrame
    ):
        write_confound(confound_df, confound_path, repetition_time=2.0, tr_method="sum")
        meta = read_confound_metadata(confound_path)
        assert meta["confound_kind"] == "phonemic"
        assert meta["confound_variant"] == "onset"
        assert meta["tr_method"] == "sum"
        assert meta["repetition_time"] == 2.0
        assert meta["n_trs"] == 3
        assert meta["confound_dim"] == 2
        assert "hypline_version" in meta

    def test_metadata_stored_in_footer(
        self, confound_path: Path, confound_df: pl.DataFrame
    ):
        write_confound(
            confound_df,
            confound_path,
            repetition_time=2.0,
            tr_method="mean",
            metadata={"key": "value"},
        )
        meta = read_confound_metadata(confound_path)
        assert meta["key"] == "value"

    def test_tr_method_none_allowed(
        self, confound_path: Path, confound_df: pl.DataFrame
    ):
        write_confound(confound_df, confound_path, repetition_time=2.0, tr_method=None)
        meta = read_confound_metadata(confound_path)
        assert meta["tr_method"] is None

    @pytest.mark.parametrize(
        "key",
        [
            "confound_kind",
            "confound_variant",
            "hypline_version",
            "tr_method",
            "repetition_time",
            "n_trs",
            "confound_dim",
        ],
    )
    def test_reserved_keys_raise(
        self, confound_path: Path, confound_df: pl.DataFrame, key: str
    ):
        with pytest.raises(ValueError, match="reserved keys"):
            write_confound(
                confound_df,
                confound_path,
                repetition_time=2.0,
                tr_method="mean",
                metadata={key: "x"},
            )

    def test_missing_required_columns(self, confound_path: Path):
        df = pl.DataFrame({"onset": [0.0, 2.0]})
        with pytest.raises(ValueError, match="missing required columns"):
            write_confound(df, confound_path, repetition_time=2.0, tr_method="mean")

    def test_non_numeric_start_time(self, confound_path: Path):
        df = pl.DataFrame(
            {"start_time": ["0.0", "2.0"], "confound": [[1.0], [2.0]]},
            schema={"start_time": pl.String, "confound": pl.Array(pl.Float64, 1)},
        )
        with pytest.raises(ValueError, match="must be a numeric type"):
            write_confound(df, confound_path, repetition_time=2.0, tr_method="mean")

    def test_empty_dataframe_raises(self, confound_path: Path):
        df = pl.DataFrame(
            {"start_time": [], "confound": []},
            schema={"start_time": pl.Float64, "confound": pl.Array(pl.Float64, 1)},
        )
        with pytest.raises(ValueError, match="at least one row"):
            write_confound(df, confound_path, repetition_time=2.0, tr_method="mean")

    def test_unsupported_confound_dtype(self, confound_path: Path):
        df = pl.DataFrame({"start_time": [0.0, 2.0], "confound": ["a", "b"]})
        with pytest.raises(ValueError, match="must be an Array or List type"):
            write_confound(df, confound_path, repetition_time=2.0, tr_method="mean")

    def test_start_time_must_begin_at_zero(self, confound_path: Path):
        df = pl.DataFrame(
            {"start_time": [2.0, 4.0], "confound": [[1.0], [2.0]]},
            schema={"start_time": pl.Float64, "confound": pl.Array(pl.Float64, 1)},
        )
        with pytest.raises(ValueError, match="must begin at 0.0"):
            write_confound(df, confound_path, repetition_time=2.0, tr_method="mean")

    def test_intervals_must_match_repetition_time(self, confound_path: Path):
        df = pl.DataFrame(
            {"start_time": [0.0, 1.0, 2.0], "confound": [[1.0], [2.0], [3.0]]},
            schema={"start_time": pl.Float64, "confound": pl.Array(pl.Float64, 1)},
        )
        with pytest.raises(ValueError, match="intervals must equal repetition_time"):
            write_confound(df, confound_path, repetition_time=2.0, tr_method="mean")

    def test_non_uniform_intervals_raise(self, confound_path: Path):
        df = pl.DataFrame(
            {"start_time": [0.0, 2.0, 5.0], "confound": [[1.0], [2.0], [3.0]]},
            schema={"start_time": pl.Float64, "confound": pl.Array(pl.Float64, 1)},
        )
        with pytest.raises(ValueError, match="intervals must equal repetition_time"):
            write_confound(df, confound_path, repetition_time=2.0, tr_method="mean")

    def test_nan_confound_value_raises(self, confound_path: Path):
        df = pl.DataFrame(
            {"start_time": [0.0, 2.0], "confound": [[1.0], [float("nan")]]},
            schema={"start_time": pl.Float64, "confound": pl.Array(pl.Float64, 1)},
        )
        with pytest.raises(ValueError, match="non-finite values"):
            write_confound(df, confound_path, repetition_time=2.0, tr_method="mean")

    def test_inf_confound_value_raises(self, confound_path: Path):
        df = pl.DataFrame(
            {"start_time": [0.0, 2.0], "confound": [[1.0], [float("inf")]]},
            schema={"start_time": pl.Float64, "confound": pl.Array(pl.Float64, 1)},
        )
        with pytest.raises(ValueError, match="non-finite values"):
            write_confound(df, confound_path, repetition_time=2.0, tr_method="mean")

    def test_missing_conf_entity_in_path(
        self, tmp_path: Path, confound_df: pl.DataFrame
    ):
        path = tmp_path / "sub-01_ses-1_bold.parquet"
        with pytest.raises(ValueError, match="must contain a 'conf' entity"):
            write_confound(confound_df, path, repetition_time=2.0, tr_method="mean")

    def test_non_parquet_extension(self, tmp_path: Path, confound_df: pl.DataFrame):
        path = tmp_path / "sub-01_conf-phonemic_desc-onset.tsv"
        with pytest.raises(ValueError, match=".parquet extension"):
            write_confound(confound_df, path, repetition_time=2.0, tr_method="mean")

    def test_rejects_bids_suffix(self, tmp_path: Path, confound_df: pl.DataFrame):
        path = tmp_path / "sub-01_conf-phonemic_desc-onset_bold.parquet"
        with pytest.raises(ValueError, match="must not have a BIDS suffix"):
            write_confound(confound_df, path, repetition_time=2.0, tr_method="mean")


class TestReadConfound:
    def test_raw_parquet_without_hypline_metadata_raises(
        self, confound_path: Path, confound_df: pl.DataFrame
    ):
        pq.write_table(confound_df.to_arrow(), confound_path)
        with pytest.raises(ValueError, match="no hypline metadata"):
            read_confound(confound_path)

    def test_validates_stored_tr_against_row_spacing(
        self, confound_path: Path, confound_df: pl.DataFrame
    ):
        _write_raw_confound(confound_df, confound_path, repetition_time=1.0)
        with pytest.raises(ValueError, match="intervals must equal repetition_time"):
            read_confound(confound_path)

    def test_non_finite_confound_value_raises(
        self, confound_path: Path, confound_df: pl.DataFrame
    ):
        corrupt = confound_df.with_columns(
            pl.Series("confound", [[1.0, 2.0], [3.0, 4.0], [5.0, float("nan")]]).cast(
                pl.Array(pl.Float64, 2)
            )
        )
        _write_raw_confound(corrupt, confound_path)
        with pytest.raises(ValueError, match="non-finite values"):
            read_confound(confound_path)

    def test_confound_kind_mismatch_raises(
        self, tmp_path: Path, confound_df: pl.DataFrame
    ):
        src = tmp_path / "sub-01_conf-phonemic_desc-onset.parquet"
        dst = tmp_path / "sub-01_conf-other_desc-onset.parquet"
        write_confound(confound_df, src, repetition_time=2.0, tr_method="mean")
        shutil.copy(src, dst)
        with pytest.raises(ValueError, match="confound_kind metadata"):
            read_confound(dst)

    def test_confound_variant_mismatch_raises(
        self, tmp_path: Path, confound_df: pl.DataFrame
    ):
        src = tmp_path / "sub-01_conf-phonemic_desc-onset.parquet"
        dst = tmp_path / "sub-01_conf-phonemic_desc-rate.parquet"
        write_confound(confound_df, src, repetition_time=2.0, tr_method="mean")
        shutil.copy(src, dst)
        with pytest.raises(ValueError, match="confound_variant metadata"):
            read_confound(dst)

    def test_missing_conf_entity(self, tmp_path: Path):
        path = tmp_path / "sub-01_bold.parquet"
        with pytest.raises(ValueError, match="must contain a 'conf' entity"):
            read_confound(path)


class TestReadConfoundMetadata:
    def test_no_hypline_metadata_raises(
        self, confound_path: Path, confound_df: pl.DataFrame
    ):
        pq.write_table(confound_df.to_arrow(), confound_path)
        with pytest.raises(ValueError, match="no hypline metadata"):
            read_confound_metadata(confound_path)

    def test_confound_kind_mismatch_raises(
        self, tmp_path: Path, confound_df: pl.DataFrame
    ):
        src = tmp_path / "sub-01_conf-phonemic_desc-onset.parquet"
        dst = tmp_path / "sub-01_conf-other_desc-onset.parquet"
        write_confound(confound_df, src, repetition_time=2.0, tr_method="mean")
        shutil.copy(src, dst)
        with pytest.raises(ValueError, match="confound_kind metadata"):
            read_confound_metadata(dst)

    def test_confound_variant_mismatch_raises(
        self, tmp_path: Path, confound_df: pl.DataFrame
    ):
        src = tmp_path / "sub-01_conf-phonemic_desc-onset.parquet"
        dst = tmp_path / "sub-01_conf-phonemic_desc-rate.parquet"
        write_confound(confound_df, src, repetition_time=2.0, tr_method="mean")
        shutil.copy(src, dst)
        with pytest.raises(ValueError, match="confound_variant metadata"):
            read_confound_metadata(dst)

    def test_missing_conf_entity(self, tmp_path: Path):
        path = tmp_path / "sub-01.parquet"
        with pytest.raises(ValueError, match="must contain a 'conf' entity"):
            read_confound_metadata(path)


# ---------------------------------------------------------------------------
# Path-based: nuisance
# ---------------------------------------------------------------------------


def _write_nuisance_tsv(df: pl.DataFrame, path: Path) -> None:
    path.write_text(df.write_csv(separator="\t"))


@pytest.fixture()
def nuisance_path(tmp_path: Path) -> Path:
    return tmp_path / "sub-01_task-conv_nuis-physio_timeseries.tsv"


@pytest.fixture()
def nuisance_df() -> pl.DataFrame:
    return pl.DataFrame({"physio123": [0.0, 1.0, 2.0], "physio345": [3.0, 4.0, 5.0]})


class TestReadNuisance:
    def test_roundtrip(self, nuisance_path: Path, nuisance_df: pl.DataFrame):
        _write_nuisance_tsv(nuisance_df, nuisance_path)
        loaded = read_nuisance(nuisance_path)
        assert loaded.equals(nuisance_df)

    def test_desc_variant_path(self, tmp_path: Path, nuisance_df: pl.DataFrame):
        path = tmp_path / "sub-01_nuis-physio_desc-v1_timeseries.tsv"
        _write_nuisance_tsv(nuisance_df, path)
        assert read_nuisance(path).equals(nuisance_df)

    def test_single_row_allowed(self, nuisance_path: Path):
        _write_nuisance_tsv(pl.DataFrame({"reg0": [1.0]}), nuisance_path)
        assert read_nuisance(nuisance_path).height == 1

    def test_missing_nuis_entity_raises(
        self, tmp_path: Path, nuisance_df: pl.DataFrame
    ):
        path = tmp_path / "sub-01_conf-physio_timeseries.tsv"
        _write_nuisance_tsv(nuisance_df, path)
        with pytest.raises(ValueError, match="nuis"):
            read_nuisance(path)

    def test_wrong_extension_raises(self, tmp_path: Path, nuisance_df: pl.DataFrame):
        path = tmp_path / "sub-01_nuis-physio_timeseries.csv"
        _write_nuisance_tsv(nuisance_df, path)
        with pytest.raises(ValueError, match=".tsv"):
            read_nuisance(path)

    def test_missing_suffix_raises(self, tmp_path: Path, nuisance_df: pl.DataFrame):
        path = tmp_path / "sub-01_nuis-physio.tsv"
        _write_nuisance_tsv(nuisance_df, path)
        with pytest.raises(ValueError, match="timeseries"):
            read_nuisance(path)

    def test_empty_rows_raise(self, nuisance_path: Path):
        empty = pl.DataFrame({"reg0": pl.Series([], dtype=pl.Float64)})
        _write_nuisance_tsv(empty, nuisance_path)
        with pytest.raises(ValueError, match="no rows"):
            read_nuisance(nuisance_path)

    def test_non_numeric_column_raises(self, nuisance_path: Path):
        _write_nuisance_tsv(
            pl.DataFrame({"reg0": [1.0, 2.0], "label": ["a", "b"]}), nuisance_path
        )
        with pytest.raises(ValueError, match="numeric"):
            read_nuisance(nuisance_path)

    def test_nan_value_raises(self, nuisance_path: Path):
        _write_nuisance_tsv(pl.DataFrame({"reg0": [0.0, float("nan")]}), nuisance_path)
        with pytest.raises(ValueError, match="non-finite"):
            read_nuisance(nuisance_path)

    def test_inf_value_raises(self, nuisance_path: Path):
        _write_nuisance_tsv(pl.DataFrame({"reg0": [0.0, float("inf")]}), nuisance_path)
        with pytest.raises(ValueError, match="non-finite"):
            read_nuisance(nuisance_path)


# ---------------------------------------------------------------------------
# Entity-based: feature
# ---------------------------------------------------------------------------


class TestSaveFeature:
    def test_places_under_canonical_path(
        self, tmp_path: Path, feature_df: pl.DataFrame
    ):
        path = save_feature(
            feature_df,
            bids_root=tmp_path,
            sub="001",
            ses="1",
            task="conv",
            run="1",
            feat="llm",
        )
        expected = (
            tmp_path
            / "features"
            / "sub-001"
            / "ses-1"
            / "llm"
            / "sub-001_ses-1_task-conv_run-1_feat-llm.parquet"
        )
        assert path == expected
        assert expected.exists()

    def test_with_desc_uses_variant_subdir(
        self, tmp_path: Path, feature_df: pl.DataFrame
    ):
        path = save_feature(
            feature_df,
            bids_root=tmp_path,
            sub="001",
            task="conv",
            feat="llm",
            desc="gpt2v10",
        )
        expected = (
            tmp_path
            / "features"
            / "sub-001"
            / "llm-gpt2v10"
            / "sub-001_task-conv_feat-llm_desc-gpt2v10.parquet"
        )
        assert path == expected
        assert expected.exists()

    def test_with_custom_entities(self, tmp_path: Path, feature_df: pl.DataFrame):
        path = save_feature(
            feature_df,
            bids_root=tmp_path,
            sub="001",
            task="conv",
            feat="llm",
            cond="G",
            trial="3",
        )
        assert path.name == "sub-001_task-conv_cond-G_trial-3_feat-llm.parquet"
        assert path.exists()

    def test_roundtrip_with_read_feature(
        self, tmp_path: Path, feature_df: pl.DataFrame
    ):
        path = save_feature(
            feature_df,
            bids_root=tmp_path,
            sub="001",
            task="conv",
            feat="llm",
        )
        loaded = read_feature(path)
        assert loaded.equals(feature_df)

    def test_read_metadata(self, tmp_path: Path, feature_df: pl.DataFrame):
        path = save_feature(
            feature_df,
            bids_root=tmp_path,
            sub="001",
            task="conv",
            feat="llm",
            metadata={"model": "gpt2"},
        )
        meta = read_feature_metadata(path)
        assert meta["model"] == "gpt2"
        assert meta["feature_name"] == "llm"
        assert meta["feature_dim"] == 2

    def test_requires_sub(self, tmp_path: Path, feature_df: pl.DataFrame):
        with pytest.raises(TypeError):
            save_feature(feature_df, bids_root=tmp_path, feat="llm")  # type: ignore[call-arg]

    def test_rejects_unsupported_entity(self, tmp_path: Path, feature_df: pl.DataFrame):
        with pytest.raises(ValueError, match="not supported"):
            save_feature(feature_df, bids_root=tmp_path, sub="001", feat="llm", acq="x")


# ---------------------------------------------------------------------------
# Entity-based: confound
# ---------------------------------------------------------------------------


class TestSaveConfound:
    def test_places_under_canonical_path(
        self, tmp_path: Path, confound_df: pl.DataFrame
    ):
        path = save_confound(
            confound_df,
            bids_root=tmp_path,
            sub="001",
            ses="1",
            task="conv",
            run="1",
            conf="phonemic",
            repetition_time=2.0,
            tr_method="any",
        )
        expected = (
            tmp_path
            / "confounds"
            / "sub-001"
            / "ses-1"
            / "phonemic"
            / "sub-001_ses-1_task-conv_run-1_conf-phonemic.parquet"
        )
        assert path == expected
        assert expected.exists()

    def test_with_desc_uses_variant_subdir(
        self, tmp_path: Path, confound_df: pl.DataFrame
    ):
        path = save_confound(
            confound_df,
            bids_root=tmp_path,
            sub="001",
            task="conv",
            conf="phonemic",
            desc="onset",
            repetition_time=2.0,
            tr_method="any",
        )
        expected = (
            tmp_path
            / "confounds"
            / "sub-001"
            / "phonemic-onset"
            / "sub-001_task-conv_conf-phonemic_desc-onset.parquet"
        )
        assert path == expected
        assert expected.exists()

    def test_roundtrip_with_read_confound(
        self, tmp_path: Path, confound_df: pl.DataFrame
    ):
        path = save_confound(
            confound_df,
            bids_root=tmp_path,
            sub="001",
            task="conv",
            conf="phonemic",
            repetition_time=2.0,
            tr_method="any",
        )
        loaded = read_confound(path)
        assert loaded.equals(confound_df)

    def test_read_metadata(self, tmp_path: Path, confound_df: pl.DataFrame):
        path = save_confound(
            confound_df,
            bids_root=tmp_path,
            sub="001",
            task="conv",
            conf="phonemic",
            repetition_time=2.0,
            tr_method="any",
            metadata={"note": "x"},
        )
        meta = read_confound_metadata(path)
        assert meta["note"] == "x"
        assert meta["confound_kind"] == "phonemic"
        assert meta["repetition_time"] == 2.0
        assert meta["tr_method"] == "any"
        assert meta["n_trs"] == 3
