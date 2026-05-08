import json
import os
from enum import StrEnum
from typing import Any

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from hypline import __version__
from hypline.bids import BIDSPath


class Downsample(StrEnum):
    MEAN = "mean"


def resample_feature(
    feature_df: pl.DataFrame,
    *,
    n_trs: int,
    repetition_time: float,
    method: str | Downsample,
) -> np.ndarray:
    """Resample a feature DataFrame to TR-level resolution.

    If rows already align to TRs (count matches and intervals equal
    repetition_time), the feature values are passed through unchanged.
    Otherwise, each row is assigned to a TR bin by start_time and
    aggregated per method. Returns an array of shape (n_trs, feature_dim).

    NOTE: Assumes each event's duration is shorter than the TR. Events
    spanning multiple TRs would be misassigned; revisit when end_time is
    added to the feature schema.

    Parameters
    ----------
    feature_df : pl.DataFrame
        Feature DataFrame with `start_time` (numeric) and `feature`
        (Array or List of floats) columns.
    n_trs : int
        Number of TRs in the output.
    repetition_time : float
        TR duration in seconds.
    method : str or Downsample
        Aggregation strategy for rows that fall in the same TR bin.
        Only applied when input is not already TR-aligned. String values
        are coerced to `Downsample`; invalid values raise `ValueError`.

    Raises
    ------
    ValueError
        If `n_trs` is not positive, or `method` is an invalid string.

    Returns
    -------
    np.ndarray
        Array of shape `(n_trs, feature_dim)` with TR-aligned feature values.
    """
    if n_trs <= 0:
        raise ValueError(f"n_trs must be positive, got {n_trs}")

    method = Downsample(method)
    start_times = feature_df.get_column("start_time").to_numpy()
    feature_col = feature_df.get_column("feature")
    feature_dim = feature_col.dtype.size  # type: ignore[union-attr]
    features = (
        np.vstack(feature_col.to_list())
        if len(start_times) > 0
        else np.empty((0, feature_dim))
    )

    # Pass through if already at TR level
    if len(start_times) == n_trs:
        intervals = np.diff(start_times)
        if len(intervals) > 0 and np.allclose(intervals, repetition_time):
            return features

    result = np.zeros((n_trs, feature_dim), dtype=np.float64)
    bins = np.floor(start_times / repetition_time).astype(int)

    if method is Downsample.MEAN:
        counts = np.zeros(n_trs, dtype=int)
        for i, b in enumerate(bins):
            if 0 <= b < n_trs:
                result[b] += features[i]
                counts[b] += 1
        nonzero = counts > 0
        result[nonzero] /= counts[nonzero, np.newaxis]

    return result


def _validate_feature_path(path: str | os.PathLike[str]) -> BIDSPath:
    bids = BIDSPath(path)
    if "feature" not in bids.entities:
        raise ValueError("BIDS path must contain a 'feature' entity")
    if bids.extension != ".parquet":
        raise ValueError(
            f"Feature path must have .parquet extension, got {bids.extension!r}"
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
    directories are created automatically.

    `feature_name` and `hypline_version` are injected into the Parquet
    footer automatically; raise if caller supplies either key.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with `start_time` and `feature` columns.
        `feature` must be Array or List of numerics.
    path : str or os.PathLike
        Must be a valid BIDS path with a `feature` entity (e.g., `feature-mfcc`).
    metadata : dict[str, Any] | None
        Optional metadata merged into the Parquet footer.

    Raises
    ------
    ValueError
        If required columns are missing, `feature` dtype is unsupported,
        the path lacks a `feature` entity, or `metadata` contains a
        reserved key (`feature_name`, `hypline_version`).
    """
    bids = _validate_feature_path(path)

    reserved = {"feature_name", "hypline_version"}
    if metadata and reserved & metadata.keys():
        raise ValueError(
            f"metadata must not contain reserved keys: {reserved & metadata.keys()}"
        )
    auto_metadata = {
        "feature_name": bids.entities["feature"],
        "hypline_version": __version__,
    }
    metadata = {**(metadata or {}), **auto_metadata}

    df = _normalize_feature_df(df)
    table = df.to_arrow()
    existing = table.schema.metadata or {}
    merged = {**existing, b"hypline": json.dumps(metadata).encode()}
    table = table.replace_schema_metadata(merged)

    bids.path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, bids.path)


def read_feature(path: str | os.PathLike[str]) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Read a feature DataFrame from a BIDS-compliant Parquet file.

    Validates that the file contains a `feature` column stored as
    `Array(Float64)`, the canonical format produced by `save_feature`.

    Parameters
    ----------
    path : str or os.PathLike
        Path to the Parquet file. Must be a valid BIDS path containing
        a `feature` entity (e.g., `feature-mfcc`).

    Returns
    -------
    tuple[pl.DataFrame, dict[str, Any]]
        The loaded DataFrame and a dict of metadata from the Parquet
        file footer.

    Raises
    ------
    ValueError
        If the path lacks a `feature` entity, the file is missing a
        `start_time` or `feature` column, the `feature` column is not
        `Array(Float64)`, or `feature_name` metadata does not match the
        path entity (indicates file was not written via `save_feature`).
    """
    bids = _validate_feature_path(path)

    table = pq.read_table(bids.path)
    df = pl.DataFrame(pl.from_arrow(table))
    df = _normalize_feature_df(df)

    raw_meta = table.schema.metadata or {}
    hypline_blob = raw_meta.get(b"hypline")
    if not hypline_blob:
        raise ValueError(
            "file has no hypline metadata; feature files "
            "must be written via save_feature"
        )

    metadata = json.loads(hypline_blob)
    if metadata.get("feature_name") != bids.entities["feature"]:
        raise ValueError(
            f"feature_name metadata {metadata.get('feature_name')!r} "
            f"does not match path entity {bids.entities['feature']!r}"
        )

    return df, metadata
