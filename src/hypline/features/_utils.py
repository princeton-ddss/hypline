import json
import os
from typing import Any, Literal, get_args

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from hypline import __version__
from hypline.bids import BIDSPath

DownsampleMethod = Literal["mean", "sum", "max", "min", "any", "count"]
FeatureDownsampleMethod = Literal["mean", "sum"]

# Public Encoding-facing methods must be a subset of all methods
if not set(get_args(FeatureDownsampleMethod)) <= set(get_args(DownsampleMethod)):
    raise RuntimeError("FeatureDownsampleMethod must be a subset of DownsampleMethod")


def stack_array_column(col: pl.Series) -> np.ndarray:
    """Stack a Polars `Array`-dtype column into a 2-D NumPy array.

    Handles the empty-column case by returning a `(0, dim)` array with
    `dim` recovered from the column's Array dtype.
    """
    if len(col) > 0:
        return np.vstack(col.to_list())
    dim = col.dtype.size  # type: ignore[union-attr]
    return np.empty((0, dim), dtype=np.float64)


def downsample(
    values: np.ndarray,
    *,
    start_times: np.ndarray,
    n_trs: int,
    repetition_time: float,
    method: DownsampleMethod,
) -> np.ndarray:
    """Downsample an event-level array to TR resolution.

    If `start_times` already form a TR-cadence grid (count equals `n_trs`
    and uniform spacing equals `repetition_time`), `values` is returned
    as a copy unchanged. Otherwise each row is binned by
    `floor(start_time / repetition_time)` and aggregated.

    Parameters
    ----------
    values
        Shape `(n_events,)` or `(n_events, dim)`. Ignored when
        `method` is `"any"` or `"count"`, but a correctly shaped array
        is still required.
    start_times
        Shape `(n_events,)`, seconds from the start of the source file.
    n_trs
        Number of TR bins in the output.
    repetition_time
        TR duration in seconds.
    method
        Aggregation method: `"mean"`, `"sum"`, `"max"`, `"min"`,
        `"any"` (1 if any event falls in the bin, else 0), or
        `"count"` (number of events per bin; `values` is ignored).

    Raises
    ------
    ValueError
        If `n_trs` is not positive or `method` is unrecognized.

    Returns
    -------
    np.ndarray
        Shape `(n_trs,)` or `(n_trs, dim)`, matching the input
        dimensionality. `method="any"` and `method="count"` always
        return shape `(n_trs,)`. Empty bins are `0.0` for every method,
        including `max`/`min` — callers cannot distinguish "empty" from
        "true zero."

    Notes
    -----
    Assumes each event's duration ≤ TR. Events spanning multiple TRs are
    misassigned by start-time binning.
    """
    if n_trs <= 0:
        raise ValueError(f"n_trs must be positive, got {n_trs}")
    if method not in get_args(DownsampleMethod):
        raise ValueError(f"Unrecognized method: {method!r}")

    squeeze = values.ndim == 1
    if squeeze:
        values = values[:, np.newaxis]
    dim = values.shape[1]

    # Pass through if already at TR level
    if len(start_times) == n_trs:
        intervals = np.diff(start_times)
        if len(intervals) > 0 and np.allclose(intervals, repetition_time):
            out = values.astype(np.float64, copy=True)
            return out.squeeze(axis=1) if squeeze else out

    bins = np.floor(start_times / repetition_time).astype(int)
    mask = (bins >= 0) & (bins < n_trs)
    valid_bins = bins[mask]
    valid_values = values[mask]

    # `any` and `count` are bin-level; ignore `dim` and return 1-D
    if method == "any":
        result_1d = np.zeros(n_trs, dtype=np.float64)
        result_1d[np.unique(valid_bins)] = 1.0
        return result_1d
    if method == "count":
        counts = np.zeros(n_trs, dtype=np.intp)
        np.add.at(counts, valid_bins, 1)
        return counts.astype(np.float64)

    if method == "mean":
        result = np.zeros((n_trs, dim), dtype=np.float64)
        counts = np.zeros(n_trs, dtype=np.intp)
        np.add.at(result, valid_bins, valid_values)
        np.add.at(counts, valid_bins, 1)
        nonzero = counts > 0
        result[nonzero] /= counts[nonzero, np.newaxis]
    elif method == "sum":
        result = np.zeros((n_trs, dim), dtype=np.float64)
        np.add.at(result, valid_bins, valid_values)
    elif method == "max":
        result = np.full((n_trs, dim), -np.inf, dtype=np.float64)
        np.maximum.at(result, valid_bins, valid_values)
        result[result == -np.inf] = 0.0
    elif method == "min":
        result = np.full((n_trs, dim), np.inf, dtype=np.float64)
        np.minimum.at(result, valid_bins, valid_values)
        result[result == np.inf] = 0.0
    else:
        raise NotImplementedError(f"Unhandled method: {method}")

    return result.squeeze(axis=1) if squeeze else result


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
    directories are created automatically. `feature_name` and
    `hypline_version` are injected into the Parquet footer automatically.

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
        contain `feature_name` or `hypline_version`.

    Raises
    ------
    ValueError
        If required columns are missing, `feature` dtype is unsupported,
        the path lacks a `feat` entity, or `metadata` contains a
        reserved key (`feature_name`, `hypline_version`).
    """
    bids = _validate_feature_path(path)

    reserved = {"feature_name", "hypline_version"}
    if metadata and reserved & metadata.keys():
        raise ValueError(
            f"metadata must not contain reserved keys: {reserved & metadata.keys()}"
        )
    auto_metadata = {
        "feature_name": bids.entities["feat"],
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
        User metadata plus auto-injected `feature_name` and `hypline_version`.

    Raises
    ------
    ValueError
        If the path lacks a `feat` entity, the file has no hypline
        metadata, or `feature_name` does not match the path entity.
    """
    bids = _validate_feature_path(path)
    raw_metadata = pq.read_metadata(bids.path).metadata
    return _parse_feature_metadata(raw_metadata or {}, bids)
