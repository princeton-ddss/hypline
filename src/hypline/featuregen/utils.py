import json
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

from hypline.bids import BIDSPath


def save_feature(
    df: pl.DataFrame,
    path: Path,
    *,
    metadata: dict[str, str] | None = None,
):
    """Save a feature DataFrame to a BIDS-compliant Parquet file.

    The DataFrame must contain a `feature` column of Array or List type.
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
        If the DataFrame is missing a `feature` column, the column has
        an unsupported dtype, or the path lacks a `feature` entity.
    """
    if "feature" not in df.columns:
        raise ValueError("DataFrame must contain a 'feature' column")

    bids_path = BIDSPath(path)

    if "feature" not in bids_path.entities:
        raise ValueError("BIDS path must contain a 'feature' entity")

    feature_col = df.get_column("feature")
    if isinstance(feature_col.dtype, pl.Array):
        width = feature_col.dtype.size
    elif isinstance(feature_col.dtype, pl.List):
        width = len(feature_col[0])
    else:
        raise ValueError("'feature' column must be an Array or List type")

    df = df.with_columns(pl.col("feature").cast(pl.Array(pl.Float64, width)))

    path.parent.mkdir(parents=True, exist_ok=True)

    table = df.to_arrow()
    if metadata:
        existing = table.schema.metadata or {}
        merged = {**existing, b"hypline": json.dumps(metadata).encode()}
        table = table.replace_schema_metadata(merged)
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
        `feature` column, or the column is not `Array(Float64)`.
    """
    bids_path = BIDSPath(path)

    if "feature" not in bids_path.entities:
        raise ValueError("BIDS path must contain a 'feature' entity")

    table = pq.read_table(path)
    raw_meta = table.schema.metadata or {}
    hypline_blob = raw_meta.get(b"hypline")
    metadata = json.loads(hypline_blob) if hypline_blob else {}

    df = pl.DataFrame(pl.from_arrow(table))

    if "feature" not in df.columns:
        raise ValueError("DataFrame must contain a 'feature' column")

    feature_col = df.get_column("feature")
    if (
        not isinstance(feature_col.dtype, pl.Array)
        or feature_col.dtype.inner != pl.Float64
    ):
        raise ValueError("'feature' column must be an Array(Float64) type")

    return df, metadata
