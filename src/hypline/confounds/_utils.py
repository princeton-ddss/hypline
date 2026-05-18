import json
import os
from typing import Any, cast

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from hypline import __version__
from hypline.bids import BIDSPath


def _validate_confound_path(path: str | os.PathLike[str]) -> BIDSPath:
    bids = BIDSPath(path)
    if "conf" not in bids.entities:
        raise ValueError("BIDS path must contain a 'conf' entity")
    if bids.ext != ".parquet":
        raise ValueError(
            f"Confound path must have .parquet extension, got {bids.ext!r}"
        )
    if bids.suffix:
        raise ValueError(
            f"Confound path must not have a BIDS suffix, got {bids.suffix!r}"
        )
    return bids


def _normalize_confound_df(df: pl.DataFrame, repetition_time: float) -> pl.DataFrame:
    missing = {"start_time", "confound"} - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")

    start_time = df.get_column("start_time")
    if not start_time.dtype.is_numeric():
        raise ValueError("'start_time' column must be a numeric type")

    if df.height == 0:
        raise ValueError("DataFrame must have at least one row")

    confound_col = df.get_column("confound")
    if isinstance(confound_col.dtype, pl.Array):
        dim = confound_col.dtype.size
    elif isinstance(confound_col.dtype, pl.List):
        dim = len(confound_col[0])
    else:
        raise ValueError("'confound' column must be an Array or List type")

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

    df = df.with_columns(pl.col("confound").cast(pl.Array(pl.Float64, dim)))

    return df


def save_confound(
    df: pl.DataFrame,
    path: str | os.PathLike[str],
    *,
    repetition_time: float,
    tr_method: str | None,
    metadata: dict[str, Any] | None = None,
):
    """Save a confound DataFrame to a BIDS-compliant Parquet file.

    `confound` is normalized to `Array(Float64)` before writing. Parent
    directories are created automatically. `confound_kind`, `confound_variant`,
    `hypline_version`, `tr_method`, `repetition_time`, `n_trs`, and `confound_dim`
    are injected into the Parquet footer automatically.

    Parameters
    ----------
    df
        DataFrame with `start_time` and `confound` columns. `confound`
        must be an Array or List type. `start_time` must begin at 0.0
        and intervals must equal `repetition_time` (TR-aligned).
    path
        BIDS-compliant path to a `.parquet` confound file. Must contain
        a `conf` entity (e.g., `conf-phonemic`). May optionally contain
        a `desc` entity (e.g., `desc-onset`) discriminating individually-
        selectable regressors within the kind.
    repetition_time
        TR of the target BOLD acquisition, in seconds. Required because
        a single-row DataFrame carries no spacing, and inferring TR from
        row spacing would silently disagree with the BOLD's true TR.
    tr_method
        Free-form label for how TR-aligned rows were produced (e.g., a
        downsampling or upsampling method, or a marker for native-TR
        computation). Pass `None` if not applicable. Must be equal across
        files sharing the same `(conf, desc)` pair for consistency checks
        to pass.
    metadata
        Optional metadata merged into the Parquet footer. Must not
        contain reserved keys.

    Raises
    ------
    ValueError
        If required columns are missing, `confound` dtype is unsupported,
        `start_time` intervals do not equal `repetition_time`, the path
        lacks a `conf` entity, or `metadata` contains a reserved key.
    """
    bids = _validate_confound_path(path)

    reserved = {
        "confound_kind",
        "confound_variant",
        "hypline_version",
        "tr_method",
        "repetition_time",
        "n_trs",
        "confound_dim",
    }
    if metadata and reserved & metadata.keys():
        raise ValueError(
            f"metadata must not contain reserved keys: {reserved & metadata.keys()}"
        )

    df = _normalize_confound_df(df, repetition_time)

    n_trs = df.height
    dim = cast(pl.Array, df.get_column("confound").dtype).size

    auto_metadata = {
        "confound_kind": bids.entities["conf"],
        "confound_variant": bids.entities.get("desc"),
        "hypline_version": __version__,
        "tr_method": tr_method,
        "repetition_time": repetition_time,
        "n_trs": n_trs,
        "confound_dim": dim,
    }
    metadata = {**(metadata or {}), **auto_metadata}

    table = df.to_arrow()
    existing = table.schema.metadata or {}
    merged = {**existing, b"hypline": json.dumps(metadata).encode()}
    table = table.replace_schema_metadata(merged)

    bids.path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, bids.path)


def _parse_confound_metadata(raw_metadata: dict, bids: BIDSPath) -> dict[str, Any]:
    blob = raw_metadata.get(b"hypline")
    if not blob:
        raise ValueError(
            "file has no hypline metadata; confound files "
            "must be written via save_confound"
        )

    metadata = json.loads(blob)
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


def read_confound(path: str | os.PathLike[str]) -> pl.DataFrame:
    """Read a confound DataFrame written by `save_confound`.

    Parameters
    ----------
    path
        BIDS-compliant path to a `.parquet` confound file. Must contain
        a `conf` entity (e.g., `conf-phonemic`); `desc` is optional.

    Returns
    -------
    pl.DataFrame
        DataFrame with `start_time` and `confound` (`Array(Float64)`) columns.

    Raises
    ------
    ValueError
        If the path lacks a `conf` entity, a required column is missing,
        the `confound` column is not Array/List, `start_time` is not
        TR-aligned, or `confound_kind`/`confound_variant` metadata does not
        match the path entities.
    """
    bids = _validate_confound_path(path)
    table = pq.read_table(bids.path)
    metadata = _parse_confound_metadata(table.schema.metadata or {}, bids)
    return _normalize_confound_df(
        pl.DataFrame(pl.from_arrow(table)), metadata["repetition_time"]
    )


def read_confound_metadata(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read metadata from a confound file without loading the data.

    Parameters
    ----------
    path
        BIDS-compliant path to a `.parquet` confound file.

    Returns
    -------
    dict[str, Any]
        User metadata plus auto-injected `confound_kind`, `confound_variant`,
        `hypline_version`, `tr_method`, `repetition_time`, `n_trs`, and `confound_dim`.

    Raises
    ------
    ValueError
        If the path lacks a `conf` entity, the file has no hypline
        metadata, or `confound_kind`/`confound_variant` does not match the
        path entities.
    """
    bids = _validate_confound_path(path)
    raw_metadata = pq.read_metadata(bids.path).metadata
    return _parse_confound_metadata(raw_metadata or {}, bids)
