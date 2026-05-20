import json
import os
from typing import Any, cast

import polars as pl
import pyarrow.parquet as pq

from hypline import __version__
from hypline.bids import BIDSPath


def _validate_feature_path(path: str | os.PathLike[str]) -> BIDSPath:
    bids = BIDSPath(path)
    if "feat" not in bids.entities:
        raise ValueError("BIDS path must contain a 'feat' entity")
    if bids.ext != ".parquet":
        raise ValueError(f"Feature path must have .parquet extension, got {bids.ext!r}")
    if bids.suffix:
        raise ValueError(
            f"Feature path must not have a BIDS suffix, got {bids.suffix!r}"
        )
    return bids


def _normalize_feature_df(df: pl.DataFrame) -> pl.DataFrame:
    missing = {"start_time", "feature"} - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")

    if not df.get_column("start_time").dtype.is_numeric():
        raise ValueError("'start_time' column must be a numeric type")

    feature_col = df.get_column("feature")
    if isinstance(feature_col.dtype, pl.Array):
        width = feature_col.dtype.size
    elif isinstance(feature_col.dtype, pl.List):
        width = len(feature_col[0])
    else:
        raise ValueError("'feature' column must be an Array or List type")

    df = df.with_columns(pl.col("feature").cast(pl.Array(pl.Float64, width)))

    return df


def save_feature(
    df: pl.DataFrame,
    path: str | os.PathLike[str],
    *,
    metadata: dict[str, Any] | None = None,
):
    """Save a feature DataFrame to a BIDS-compliant Parquet file.

    `feature` is normalized to `Array(Float64)` before writing. Parent
    directories are created automatically. `feature_name`,
    `hypline_version`, and `feature_dim` are injected into the Parquet
    footer automatically.

    Parameters
    ----------
    df
        DataFrame with `start_time` and `feature` columns.
        `feature` must be an Array or List type.
    path
        BIDS-compliant path to a `.parquet` feature file. Must contain
        a `feat` entity (e.g., `feat-mfcc`).
    metadata
        Optional metadata merged into the Parquet footer. Must not
        contain a reserved key (`feature_name`, `hypline_version`,
        `feature_dim`).

    Raises
    ------
    ValueError
        If required columns are missing, `feature` dtype is unsupported,
        the path lacks a `feat` entity, or `metadata` contains a
        reserved key.
    """
    bids = _validate_feature_path(path)

    reserved = {"hypline_version", "feature_name", "feature_dim"}
    if metadata and reserved & metadata.keys():
        raise ValueError(
            f"metadata must not contain reserved keys: {reserved & metadata.keys()}"
        )

    df = _normalize_feature_df(df)
    dim = cast(pl.Array, df.get_column("feature").dtype).size

    auto_metadata = {
        "hypline_version": __version__,
        "feature_name": bids.entities["feat"],
        "feature_dim": dim,
    }
    metadata = {**auto_metadata, **(metadata or {})}

    table = df.to_arrow()
    existing = table.schema.metadata or {}
    merged = {**existing, b"hypline": json.dumps(metadata).encode()}
    table = table.replace_schema_metadata(merged)

    bids.path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, bids.path)


def _parse_feature_metadata(raw_metadata: dict, bids: BIDSPath) -> dict[str, Any]:
    blob = raw_metadata.get(b"hypline")
    if not blob:
        raise ValueError(
            "file has no hypline metadata; feature files "
            "must be written via save_feature"
        )

    metadata = json.loads(blob)
    if metadata.get("feature_name") != bids.entities["feat"]:
        raise ValueError(
            f"feature_name metadata {metadata.get('feature_name')!r} "
            f"does not match path entity {bids.entities['feat']!r}"
        )

    return metadata


def read_feature(path: str | os.PathLike[str]) -> pl.DataFrame:
    """Read a feature DataFrame written by `save_feature`.

    Parameters
    ----------
    path
        BIDS-compliant path to a `.parquet` feature file. Must contain
        a `feat` entity (e.g., `feat-mfcc`).

    Returns
    -------
    pl.DataFrame
        DataFrame with `start_time` and `feature` (`Array(Float64)`) columns.

    Raises
    ------
    ValueError
        If the path lacks a `feat` entity, a `start_time` or `feature`
        column is missing, the `feature` column is not `Array(Float64)`, or
        `feature_name` metadata does not match the path entity.
    """
    bids = _validate_feature_path(path)
    table = pq.read_table(bids.path)
    _parse_feature_metadata(table.schema.metadata or {}, bids)  # for validation
    return _normalize_feature_df(pl.DataFrame(pl.from_arrow(table)))


def read_feature_metadata(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read metadata from a feature file without loading the data.

    Parameters
    ----------
    path
        BIDS-compliant path to a `.parquet` feature file.

    Returns
    -------
    dict[str, Any]
        User metadata plus auto-injected `feature_name`, `hypline_version`,
        and `feature_dim`.

    Raises
    ------
    ValueError
        If the path lacks a `feat` entity, the file has no hypline
        metadata, or `feature_name` does not match the path entity.
    """
    bids = _validate_feature_path(path)
    raw_metadata = pq.read_metadata(bids.path).metadata
    return _parse_feature_metadata(raw_metadata or {}, bids)
