"""I/O for hypline feature and confound files.

Public API (re-exported at `hypline.*`): saves are entity-based
(`save_feature(df, bids_root, *, sub, feat, ...)`) because the canonical
output path is layout-derived; reads are path-based (`read_feature(path)`)
because users typically already have a file in hand.

Internal: `write_feature` / `write_confound` take an explicit path and back
the `save_*` wrappers. Used by in-package generators driven by
`BIDSLayout.path.*` (which already know the target path). Not re-exported.
"""

import json
import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from hypline._version import __version__
from hypline.bids import BIDSPath
from hypline.layout import Area, kind_subdir

__all__ = [
    "read_confound",
    "read_confound_metadata",
    "read_feature",
    "read_feature_metadata",
    "save_confound",
    "save_feature",
]


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


def _derive_parquet_path(
    bids_root: str | os.PathLike[str],
    *,
    area: Area,
    category_entity: str,
    kind: str,
    entities: dict[str, str],
) -> Path:
    """Resolve the canonical Parquet path for a derived file.

    Builds the BIDS filename via `BIDSPath.from_entities` (which enforces
    entity ordering and validation), then places it under the canonical
    `<area>/sub-XX/[ses-YY/]<kind>[-<desc>]/` subdirectory.
    """
    bp = BIDSPath.from_entities(
        ext=".parquet",
        **{category_entity: kind},
        **entities,
    )
    out_dir = kind_subdir(
        Path(bids_root),
        area,
        sub=bp.entities["sub"],
        ses=bp.entities.get("ses"),
        kind=kind,
        desc=bp.entities.get("desc"),
    )
    return out_dir / bp.path.name


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


# --------------------------------------------------------------------------- #
# Feature
# --------------------------------------------------------------------------- #


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


def write_feature(
    df: pl.DataFrame,
    path: str | os.PathLike[str],
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write a feature DataFrame to a BIDS Parquet file.

    Reserved metadata keys (auto-injected): `feature_name`,
    `hypline_version`, `feature_dim`. Keys prefixed with `_` are exempt
    from cross-file equality checks at encoding time.
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
    """Read a feature DataFrame from a hypline-format Parquet file.

    Validates that the path carries a `feat` entity, `.parquet` extension,
    and no BIDS suffix; checks the footer's `feature_name` matches the
    `feat` entity; normalizes the `feature` column to fixed-width
    `Array(Float64)`.

    Parameters
    ----------
    path
        Path to a hypline feature Parquet file.

    Returns
    -------
    pl.DataFrame
        The stored feature DataFrame.

    Raises
    ------
    ValueError
        If the path is not a valid hypline feature path, the footer lacks
        a `hypline` metadata blob, or `feature_name` disagrees with the
        path entity.
    """
    bids = _validate_bids_parquet_path(path, required_entity="feat", kind="feature")
    table = pq.read_table(bids.path)
    _parse_feature_metadata(table.schema.metadata or {}, bids)
    df, _ = _normalize_feature_df(pl.DataFrame(pl.from_arrow(table)))
    return df


def read_feature_metadata(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read footer metadata from a hypline feature file without loading data.

    Parameters
    ----------
    path
        Path to a hypline feature Parquet file.

    Returns
    -------
    dict
        The `hypline` JSON blob from the Parquet footer, including the
        auto-injected `feature_name`, `feature_dim`, and `hypline_version`
        keys alongside any caller-supplied keys.

    Raises
    ------
    ValueError
        Same conditions as `read_feature`.
    """
    bids = _validate_bids_parquet_path(path, required_entity="feat", kind="feature")
    raw_metadata = pq.read_metadata(bids.path).metadata
    return _parse_feature_metadata(raw_metadata or {}, bids)


def save_feature(
    df: pl.DataFrame,
    *,
    bids_root: str | os.PathLike[str],
    sub: str,
    feat: str,
    metadata: dict[str, Any] | None = None,
    **entities: str,
) -> Path:
    """Save a feature DataFrame to its canonical layout location.

    Parameters
    ----------
    df
        Must contain `start_time` (numeric, source-relative seconds) and
        `feature` (Array or List of equal width per row). The `feature`
        column is normalized to fixed-width `Array(Float64)` on write;
        other columns are preserved.
    bids_root
        Project root; the file lands under
        `<bids_root>/features/sub-<sub>/[ses-<ses>/]<feat>[-<desc>]/`.
    sub, feat
        Required subject and feature-kind labels.
    metadata
        Extra keys merged into the Parquet footer's `hypline` blob. The
        reserved keys `feature_name`, `feature_dim`, and `hypline_version`
        are auto-injected and must not be supplied. Keys prefixed with `_`
        are exempt from cross-file equality checks at encoding time.
    **entities
        Additional BIDS entities (`ses`, `task`, `run`, `desc`, custom
        descriptors). Entity ordering and validation are handled by
        `BIDSPath`.

    Returns
    -------
    Path
        Resolved output path.

    Raises
    ------
    ValueError
        If required columns are missing, `feature` widths are ragged, or
        `metadata` supplies a reserved key.
    """
    path = _derive_parquet_path(
        bids_root,
        area="features",
        category_entity="feat",
        kind=feat,
        entities={"sub": sub, **entities},
    )
    write_feature(df, path, metadata=metadata)
    return path


# --------------------------------------------------------------------------- #
# Confound
# --------------------------------------------------------------------------- #


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


def write_confound(
    df: pl.DataFrame,
    path: str | os.PathLike[str],
    *,
    repetition_time: float,
    tr_method: str | None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write a confound DataFrame to a BIDS Parquet file.

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
    `n_trs`, `confound_dim`. Keys prefixed with `_` are exempt from
    cross-file equality checks at encoding time.
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
    """Read a confound DataFrame from a hypline-format Parquet file.

    Validates that the path carries a `conf` entity, `.parquet` extension,
    and no BIDS suffix; checks that the footer's `confound_kind` and
    `confound_variant` match the path's `conf` and `desc` entities;
    normalizes the `confound` column to fixed-width `Array(Float64)`.

    Parameters
    ----------
    path
        Path to a hypline confound Parquet file.

    Returns
    -------
    pl.DataFrame
        The stored confound DataFrame.

    Raises
    ------
    ValueError
        If the path is not a valid hypline confound path, the footer lacks
        a `hypline` metadata blob, or `confound_kind`/`confound_variant`
        disagrees with the path entities.
    """
    bids = _validate_bids_parquet_path(path, required_entity="conf", kind="confound")
    table = pq.read_table(bids.path)
    metadata = _parse_confound_metadata(table.schema.metadata or {}, bids)
    df, _ = _normalize_confound_df(
        pl.DataFrame(pl.from_arrow(table)), metadata["repetition_time"]
    )
    return df


def read_confound_metadata(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read footer metadata from a hypline confound file without loading data.

    Parameters
    ----------
    path
        Path to a hypline confound Parquet file.

    Returns
    -------
    dict
        The `hypline` JSON blob from the Parquet footer, including the
        auto-injected `confound_kind`, `confound_variant`, `tr_method`,
        `repetition_time`, `n_trs`, `confound_dim`, and `hypline_version`
        keys alongside any caller-supplied keys.

    Raises
    ------
    ValueError
        Same conditions as `read_confound`.
    """
    bids = _validate_bids_parquet_path(path, required_entity="conf", kind="confound")
    raw_metadata = pq.read_metadata(bids.path).metadata
    return _parse_confound_metadata(raw_metadata or {}, bids)


def save_confound(
    df: pl.DataFrame,
    *,
    bids_root: str | os.PathLike[str],
    sub: str,
    conf: str,
    repetition_time: float,
    tr_method: str | None,
    metadata: dict[str, Any] | None = None,
    **entities: str,
) -> Path:
    """Save a confound DataFrame to its canonical layout location.

    Parameters
    ----------
    df
        Must contain `start_time` (numeric, beginning at `0.0` with
        intervals equal to `repetition_time`) and `confound` (Array or
        List of equal width per row). The `confound` column is normalized
        to fixed-width `Array(Float64)` on write.
    bids_root
        Project root; the file lands under
        `<bids_root>/confounds/sub-<sub>/[ses-<ses>/]<conf>[-<desc>]/`.
    sub, conf
        Required subject and confound-kind labels.
    repetition_time
        TR of the target BOLD acquisition, in seconds. Must be passed
        explicitly: a single-row DataFrame carries no spacing, and
        inferring TR from row spacing would silently disagree with the
        BOLD's true TR.
    tr_method
        Free-form label for how TR-aligned rows were produced
        (downsampling/upsampling method, or a marker for native-TR
        computation). Pass `None` if not applicable. Recorded verbatim;
        must be equal across files sharing the same `(conf, desc)` pair
        for downstream consistency checks to pass.
    metadata
        Extra keys merged into the Parquet footer's `hypline` blob.
        Reserved keys (`confound_kind`, `confound_variant`, `tr_method`,
        `repetition_time`, `n_trs`, `confound_dim`, `hypline_version`)
        are auto-injected and must not be supplied. Keys prefixed with
        `_` are exempt from cross-file equality checks.
    **entities
        Additional BIDS entities (`ses`, `task`, `run`, the optional
        `desc` variant tag, custom descriptors).

    Returns
    -------
    Path
        Resolved output path.

    Raises
    ------
    ValueError
        If required columns are missing, `start_time` is not TR-aligned,
        `confound` widths are ragged, or `metadata` supplies a reserved
        key.
    """
    path = _derive_parquet_path(
        bids_root,
        area="confounds",
        category_entity="conf",
        kind=conf,
        entities={"sub": sub, **entities},
    )
    write_confound(
        df,
        path,
        repetition_time=repetition_time,
        tr_method=tr_method,
        metadata=metadata,
    )
    return path
