import json
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

from hypline.bids import BIDSPath


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
    path: Path,
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
    path : Path
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
    _validate_bids_feature_path(path)

    df = _normalize_feature_df(df)
    table = df.to_arrow()
    if metadata:
        existing = table.schema.metadata or {}
        merged = {**existing, b"hypline": json.dumps(metadata).encode()}
        table = table.replace_schema_metadata(merged)

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def read_feature(path: Path) -> tuple[pl.DataFrame, dict[str, str]]:
    """Read a feature DataFrame from a BIDS-compliant Parquet file.

    Validates that the file contains a `feature` column stored as
    `Array(Float64)`, the canonical format produced by `save_feature`.

    Parameters
    ----------
    path : Path
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
    _validate_bids_feature_path(path)

    table = pq.read_table(path)
    df = pl.DataFrame(pl.from_arrow(table))
    df = _normalize_feature_df(df)

    raw_meta = table.schema.metadata or {}
    hypline_blob = raw_meta.get(b"hypline")
    metadata = json.loads(hypline_blob) if hypline_blob else {}

    return df, metadata
