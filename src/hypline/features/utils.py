import json
import os
from enum import StrEnum
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

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


def _validate_bids_feature_path(path: Path) -> BIDSPath:
    bids_path = BIDSPath(path)
    if "feature" not in bids_path.entities:
        raise ValueError("BIDS path must contain a 'feature' entity")
    return bids_path


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
    metadata: dict[str, str] | None = None,
):
    """Save a feature DataFrame to a BIDS-compliant Parquet file.

    The DataFrame must contain `start_time` and `feature` columns. The
    `feature` column must be of Array or List type.
    The column is normalized to `Array(Float64)` before writing. Parent
    directories are created automatically if they do not exist.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with a `feature` column (Array or List of numerics)
        and any additional metadata columns.
    path : str or os.PathLike
        Output file path. Must be a valid BIDS path containing a
        `feature` entity (e.g., `feature-mfcc`).
    metadata : dict[str, str] | None
        Optional key-value metadata to store in the Parquet file footer.

    Raises
    ------
    ValueError
        If the DataFrame is missing a `start_time` or `feature` column,
        the `feature` column has an unsupported dtype, or the path lacks
        a `feature` entity.
    """
    path = Path(path)
    _validate_bids_feature_path(path)

    df = _normalize_feature_df(df)
    table = df.to_arrow()
    if metadata:
        existing = table.schema.metadata or {}
        merged = {**existing, b"hypline": json.dumps(metadata).encode()}
        table = table.replace_schema_metadata(merged)

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def read_feature(path: str | os.PathLike[str]) -> tuple[pl.DataFrame, dict[str, str]]:
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
    tuple[pl.DataFrame, dict[str, str]]
        The loaded DataFrame and a dict of metadata from the Parquet
        file footer.

    Raises
    ------
    ValueError
        If the path lacks a `feature` entity, the file is missing a
        `start_time` or `feature` column, or the `feature` column is
        not `Array(Float64)`.
    """
    path = Path(path)
    _validate_bids_feature_path(path)

    table = pq.read_table(path)
    df = pl.DataFrame(pl.from_arrow(table))
    df = _normalize_feature_df(df)

    raw_meta = table.schema.metadata or {}
    hypline_blob = raw_meta.get(b"hypline")
    metadata = json.loads(hypline_blob) if hypline_blob else {}

    return df, metadata
