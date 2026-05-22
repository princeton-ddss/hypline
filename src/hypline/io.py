"""Parquet I/O primitives for hypline feature and confound files."""

import json
import os
from typing import Any, cast

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from hypline import __version__
from hypline.bids import BIDSPath


def _validate_bids_parquet_path(
    path: str | os.PathLike[str],
    *,
    required_entity: str,
    kind: str,
) -> BIDSPath:
    bids = BIDSPath(path)
    if required_entity not in bids.entities:
        raise ValueError(f"BIDS path must contain a {required_entity!r} entity")
    if bids.ext != ".parquet":
        raise ValueError(
            f"{kind.capitalize()} path must have .parquet extension, got {bids.ext!r}"
        )
    if bids.suffix:
        raise ValueError(
            f"{kind.capitalize()} path must not have a BIDS suffix, got {bids.suffix!r}"
        )
    return bids


def _coerce_array_column(df: pl.DataFrame, column: str) -> tuple[pl.DataFrame, int]:
    col = df.get_column(column)
    if isinstance(col.dtype, pl.Array):
        width = col.dtype.size
    elif isinstance(col.dtype, pl.List):
        if df.height == 0:
            raise ValueError(
                f"cannot infer width of {column!r} List column from empty DataFrame"
            )
        widths = col.list.len()
        if widths.n_unique() != 1:
            raise ValueError(
                f"{column!r} List column has ragged widths: "
                f"{sorted(set(widths.to_list()))}"
            )
        width = widths[0]
    else:
        raise ValueError(f"{column!r} column must be an Array or List type")
    df = df.with_columns(pl.col(column).cast(pl.Array(pl.Float64, width)))
    return df, width


def _write_with_hypline_metadata(
    df: pl.DataFrame,
    path: BIDSPath,
    metadata: dict[str, Any],
) -> None:
    table = df.to_arrow()
    existing = table.schema.metadata or {}
    merged = {**existing, b"hypline": json.dumps(metadata).encode()}
    table = table.replace_schema_metadata(merged)
    path.path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path.path)


def _read_hypline_metadata(raw_metadata: dict) -> dict[str, Any]:
    blob = raw_metadata.get(b"hypline")
    if not blob:
        raise ValueError("file has no hypline metadata")
    return cast(dict[str, Any], json.loads(blob))


def _check_reserved(metadata: dict[str, Any] | None, reserved: set[str]) -> None:
    if metadata and reserved & metadata.keys():
        raise ValueError(
            f"metadata must not contain reserved keys: {reserved & metadata.keys()}"
        )


def _normalize_feature_df(df: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    missing = {"start_time", "feature"} - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")

    if not df.get_column("start_time").dtype.is_numeric():
        raise ValueError("'start_time' column must be a numeric type")

    return _coerce_array_column(df, "feature")


def _parse_feature_metadata(raw_metadata: dict, bids: BIDSPath) -> dict[str, Any]:
    metadata = _read_hypline_metadata(raw_metadata)
    if metadata.get("feature_name") != bids.entities["feat"]:
        raise ValueError(
            f"feature_name metadata {metadata.get('feature_name')!r} "
            f"does not match path entity {bids.entities['feat']!r}"
        )
    return metadata


def save_feature(
    df: pl.DataFrame,
    path: str | os.PathLike[str],
    *,
    metadata: dict[str, Any] | None = None,
):
    """Save a feature DataFrame to a BIDS Parquet file.

    Reserved metadata keys (auto-injected): `feature_name`,
    `hypline_version`, `feature_dim`.
    """
    bids = _validate_bids_parquet_path(path, required_entity="feat", kind="feature")
    _check_reserved(metadata, {"hypline_version", "feature_name", "feature_dim"})

    df, dim = _normalize_feature_df(df)

    auto_metadata = {
        "hypline_version": __version__,
        "feature_name": bids.entities["feat"],
        "feature_dim": dim,
    }
    _write_with_hypline_metadata(df, bids, {**auto_metadata, **(metadata or {})})


def read_feature(path: str | os.PathLike[str]) -> pl.DataFrame:
    """Read a feature DataFrame written by `save_feature`."""
    bids = _validate_bids_parquet_path(path, required_entity="feat", kind="feature")
    table = pq.read_table(bids.path)
    _parse_feature_metadata(table.schema.metadata or {}, bids)
    df, _ = _normalize_feature_df(pl.DataFrame(pl.from_arrow(table)))
    return df


def read_feature_metadata(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read footer metadata from a feature file without loading data."""
    bids = _validate_bids_parquet_path(path, required_entity="feat", kind="feature")
    raw_metadata = pq.read_metadata(bids.path).metadata
    return _parse_feature_metadata(raw_metadata or {}, bids)


def _normalize_confound_df(
    df: pl.DataFrame,
    repetition_time: float,
) -> tuple[pl.DataFrame, int]:
    missing = {"start_time", "confound"} - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")

    start_time = df.get_column("start_time")
    if not start_time.dtype.is_numeric():
        raise ValueError("'start_time' column must be a numeric type")

    if df.height == 0:
        raise ValueError("DataFrame must have at least one row")

    start_arr = start_time.to_numpy()
    if start_arr[0] != 0.0:
        raise ValueError(f"'start_time' must begin at 0.0, got {start_arr[0]}")
    if len(start_arr) > 1:
        intervals = np.diff(start_arr)
        if not np.allclose(intervals, repetition_time):
            raise ValueError(
                f"'start_time' intervals must equal repetition_time "
                f"{repetition_time}, got {intervals.tolist()}"
            )

    return _coerce_array_column(df, "confound")


def _parse_confound_metadata(raw_metadata: dict, bids: BIDSPath) -> dict[str, Any]:
    metadata = _read_hypline_metadata(raw_metadata)
    if metadata.get("confound_kind") != bids.entities["conf"]:
        raise ValueError(
            f"confound_kind metadata {metadata.get('confound_kind')!r} "
            f"does not match path entity {bids.entities['conf']!r}"
        )
    desc = bids.entities.get("desc")
    if metadata.get("confound_variant") != desc:
        raise ValueError(
            f"confound_variant metadata {metadata.get('confound_variant')!r} "
            f"does not match path entity {desc!r}"
        )
    return metadata


def save_confound(
    df: pl.DataFrame,
    path: str | os.PathLike[str],
    *,
    repetition_time: float,
    tr_method: str | None,
    metadata: dict[str, Any] | None = None,
):
    """Save a confound DataFrame to a BIDS Parquet file.

    `path` may carry an optional `desc` entity discriminating individually-
    selectable regressors within the kind (e.g., `desc-onset`).

    `repetition_time` must be passed explicitly: a single-row DataFrame
    carries no spacing, and inferring TR from row spacing would silently
    disagree with the BOLD's true TR.

    `tr_method` labels how TR-aligned rows were produced (downsampling /
    upsampling method, or a marker for native-TR computation; pass `None`
    if not applicable). Must be equal across files sharing the same
    `(conf, desc)` pair — downstream consistency checks rely on it.

    Reserved metadata keys (auto-injected): `confound_kind`,
    `confound_variant`, `hypline_version`, `tr_method`, `repetition_time`,
    `n_trs`, `confound_dim`.
    """
    bids = _validate_bids_parquet_path(path, required_entity="conf", kind="confound")
    _check_reserved(
        metadata,
        {
            "hypline_version",
            "confound_kind",
            "confound_variant",
            "tr_method",
            "repetition_time",
            "n_trs",
            "confound_dim",
        },
    )

    df, dim = _normalize_confound_df(df, repetition_time)

    auto_metadata = {
        "hypline_version": __version__,
        "confound_kind": bids.entities["conf"],
        "confound_variant": bids.entities.get("desc"),
        "tr_method": tr_method,
        "repetition_time": repetition_time,
        "n_trs": df.height,
        "confound_dim": dim,
    }
    _write_with_hypline_metadata(df, bids, {**auto_metadata, **(metadata or {})})


def read_confound(path: str | os.PathLike[str]) -> pl.DataFrame:
    """Read a confound DataFrame written by `save_confound`."""
    bids = _validate_bids_parquet_path(path, required_entity="conf", kind="confound")
    table = pq.read_table(bids.path)
    metadata = _parse_confound_metadata(table.schema.metadata or {}, bids)
    df, _ = _normalize_confound_df(
        pl.DataFrame(pl.from_arrow(table)), metadata["repetition_time"]
    )
    return df


def read_confound_metadata(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read footer metadata from a confound file without loading data."""
    bids = _validate_bids_parquet_path(path, required_entity="conf", kind="confound")
    raw_metadata = pq.read_metadata(bids.path).metadata
    return _parse_confound_metadata(raw_metadata or {}, bids)
