import os
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl
from pydantic import BaseModel

from hypline.bids import BIDSPath, validate_bids_entities, validate_entity_invariance
from hypline.bold import (
    BOLD_EXTENSIONS,
    get_repetition_time,
    load_events,
    parse_bold_space,
)
from hypline.enums import Device
from hypline.features.utils import Downsample, read_feature, resample_feature
from hypline.utils import find_files, validate_dirs

# Entities provided via dedicated arguments — not allowed in bids_filters.
_RESERVED_ENTITIES = frozenset(("sub", "space", "feature"))

# Allowed bids_filters entities per target. Strict whitelist — unknown entities
# are rejected so users get a clear error rather than silent mismatches.
_BOLD_FILTER_ENTITIES = frozenset(
    ("ses", "task", "run", "desc", "res", "den", "echo", "acq", "ce", "rec", "dir")
)
_FEATURE_FILTER_ENTITIES = frozenset(("ses", "task", "run", "partition"))


class BoldKey(NamedTuple):
    ses: str | None
    run: str | None


class CellKey(NamedTuple):
    ses: str | None
    run: str | None
    partition: str | None


class FeatureKey(NamedTuple):
    cell: CellKey
    feature: str


class BoldMeta(NamedTuple):
    path: Path
    repetition_time: float
    partitions: dict[str, slice]


class EncodingConfig(BaseModel):
    device: Device = Device.CPU


@dataclass(frozen=True)
class TrainingData:
    """Assembled feature and BOLD arrays ready for regression.

    X and Y share the same row axis: X[row_slices[cell_key]] and
    Y[row_slices[cell_key]] together give the feature matrix and BOLD
    response for that cell. col_slices indexes into the column axis of X,
    mapping each feature name to its contiguous block of columns.
    """

    X: np.ndarray
    Y: np.ndarray
    row_slices: dict[CellKey, slice]
    col_slices: dict[str, slice]


def _format_loc(**entities: str | None) -> str:
    """Format BIDS entities as 'k1=v1, k2=v2', skipping None values."""
    return ", ".join(f"{k}={v}" for k, v in entities.items() if v is not None)


def _load_bold_array(path: Path) -> np.ndarray:
    """Load a BOLD file into a 2D array of shape (n_trs, n_voxels)."""
    import nibabel as nib

    img = nib.load(path)
    data = np.asarray(img.dataobj)  # type: ignore
    if isinstance(img, nib.Nifti1Image):
        return data.reshape(-1, data.shape[-1]).T
    elif isinstance(img, nib.GiftiImage):
        return np.column_stack([d.data for d in img.darrays]).T
    else:
        raise ValueError(f"Unsupported image format: {type(img).__name__}")


class Encoding:
    def __init__(
        self,
        config: EncodingConfig,
        *,
        features_dir: str | os.PathLike[str],
        bold_dir: str | os.PathLike[str],
        output_dir: str | os.PathLike[str],
        features: list[str],
        bold_space: str,
        downsample: str | Downsample = "mean",
        bids_filters: list[str] | None = None,
    ):
        import torch

        if config.device is Device.CUDA and not torch.cuda.is_available():
            raise RuntimeError("CUDA is requested but not available")
        self.config = config

        validate_dirs(features_dir, bold_dir, output_dir)
        self.features_dir = Path(features_dir)
        self.bold_dir = Path(bold_dir)
        self.output_dir = Path(output_dir)

        if not features:
            raise ValueError("features must be a non-empty list")
        if len(features) != len(set(features)):
            dupes = sorted({f for f in features if features.count(f) > 1})
            raise ValueError(f"Duplicate entries in features: {dupes}")
        self.features = features

        self.bold_space = parse_bold_space(bold_space)
        self.downsample = Downsample(downsample)

        bids_filters = list(bids_filters or [])
        validate_bids_entities(*bids_filters)
        allowed = _BOLD_FILTER_ENTITIES | _FEATURE_FILTER_ENTITIES
        for entity in bids_filters:
            key = entity.split("-", 1)[0]
            if key in _RESERVED_ENTITIES:
                raise ValueError(
                    f"bids_filters cannot contain {key!r} "
                    "— use the dedicated argument instead"
                )
            if key not in allowed:
                raise ValueError(
                    f"Unknown bids_filters entity {key!r}. Allowed: {sorted(allowed)}"
                )
        self.bids_filters = bids_filters

    def train(self, sub_id: str):
        feature_paths = self._discover_features(sub_id)
        bold_metas = self._discover_bold(sub_id)
        self._validate_alignment(feature_paths, bold_metas)
        data = self._build_xy(feature_paths, bold_metas)

        # TODO: modeling (banded ridge regression) goes here
        data

    def _discover_features(self, sub_id: str) -> dict[FeatureKey, Path]:
        """Discover and validate feature file paths for a subject.

        Scans the features directory by BIDS filename alone — no feature data is read.
        Partition filtering from bids_filters is applied here; files referencing
        unrequested partitions are silently skipped. Duplicate files for the same
        (cell, feature) pair raise immediately.

        Returns a flat dict mapping each (cell, feature) pair to its path.
        Every cell is guaranteed to have all requested features — a missing feature
        at any cell raises rather than silently producing an incomplete matrix.
        """

        requested_partitions = {
            entity.removeprefix("partition-")
            for entity in self.bids_filters
            if entity.startswith("partition-")
        }

        feature_paths: dict[FeatureKey, Path] = {}
        for feature_name in self.features:
            feature_filters = [
                f"sub-{sub_id}",
                f"feature-{feature_name}",
                *(
                    entity
                    for entity in self.bids_filters
                    if entity.split("-", 1)[0] in _FEATURE_FILTER_ENTITIES
                ),
            ]
            feature_files = find_files(
                self.features_dir,
                ends_with=".parquet",
                recursive=True,
                bids_filters=feature_filters,
            )
            if not feature_files:
                raise FileNotFoundError(
                    f"No feature files found for sub={sub_id}, feature={feature_name} "
                    f"in {self.features_dir}"
                )

            for feature_file in feature_files:
                bids = BIDSPath(feature_file)
                partition_key = bids.entities.get("partition")

                # Skip files whose partition was not requested
                if (
                    len(requested_partitions) > 0
                    and partition_key is not None
                    and partition_key not in requested_partitions
                ):
                    continue

                cell_key = CellKey(
                    ses=bids.entities.get("ses"),
                    run=bids.entities.get("run"),
                    partition=partition_key,
                )
                feature_key = FeatureKey(cell=cell_key, feature=feature_name)

                if feature_key in feature_paths:
                    loc = _format_loc(
                        sub=sub_id,
                        ses=cell_key.ses,
                        run=cell_key.run,
                        partition=cell_key.partition,
                    )
                    raise ValueError(
                        f"Multiple feature files for feature={feature_name}, {loc}:\n"
                        f"  {feature_paths[feature_key]}\n  {feature_file}"
                    )

                feature_paths[feature_key] = feature_file

        # Validate: all features present at every cell
        expected = {
            FeatureKey(cell_key, feature_name)
            for cell_key in {feature_key.cell for feature_key in feature_paths}
            for feature_name in self.features
        }
        missing = expected - feature_paths.keys()
        if missing:
            feature_key = next(iter(missing))
            loc = _format_loc(
                sub=sub_id,
                ses=feature_key.cell.ses,
                run=feature_key.cell.run,
                partition=feature_key.cell.partition,
            )
            msg = f"Missing feature={feature_key.feature} at {loc}"
            if len(missing) > 1:
                msg += f" ({len(missing) - 1} other coverage gaps exist)"
            raise FileNotFoundError(msg)

        feature_bids = [BIDSPath(path) for path in feature_paths.values()]
        try:
            validate_entity_invariance(feature_bids, ("task",))
        except ValueError as e:
            raise ValueError(f"{e} (subject {sub_id})") from e

        return feature_paths

    def _discover_bold(self, sub_id: str) -> dict[BoldKey, BoldMeta]:
        """Discover BOLD files and load their metadata for a subject.

        Scans the BOLD directory by filename without loading image arrays. TR is
        read from the sidecar JSON, falling back to the image header. Partition
        slices are precomputed from the colocated events TSV when present, and
        validated to be contiguous and zero-starting — gaps or offsets raise here
        rather than producing a silently misaligned Y.

        Returns a dict mapping each BoldKey to its BoldMeta, with all runs
        guaranteed to share the same TR and BOLD-level entities.
        """

        bold_ext = BOLD_EXTENSIONS[type(self.bold_space)]
        bold_filters = [
            f"sub-{sub_id}",
            f"space-{self.bold_space}",
            *(
                entity
                for entity in self.bids_filters
                if entity.split("-", 1)[0] in _BOLD_FILTER_ENTITIES
            ),
        ]
        bold_files = find_files(
            self.bold_dir,
            ends_with=f"_bold{bold_ext}",
            recursive=True,
            bids_filters=bold_filters,
        )
        if not bold_files:
            raise FileNotFoundError(
                f"No BOLD files found for sub={sub_id}, space={self.bold_space} "
                f"in {self.bold_dir}"
            )

        bold_metas: dict[BoldKey, BoldMeta] = {}
        for bold_file in bold_files:
            bids = BIDSPath(bold_file)
            bold_key = BoldKey(
                ses=bids.entities.get("ses"),
                run=bids.entities.get("run"),
            )
            if bold_key in bold_metas:
                loc = _format_loc(sub=sub_id, ses=bold_key.ses, run=bold_key.run)
                raise ValueError(
                    f"Duplicate BOLD file at {loc}:\n"
                    f"  {bold_metas[bold_key].path}\n  {bold_file}"
                )
            repetition_time = get_repetition_time(bold_file)
            events = load_events(bold_file)
            partitions: dict[str, slice] = {}
            if events is not None:
                # Partitions are encoded as trial_type="partition-<name>" by convention
                for row in events.filter(
                    pl.col("trial_type").str.starts_with("partition-")
                ).iter_rows(named=True):
                    name = row["trial_type"].removeprefix("partition-")
                    onset_tr = round(row["onset"] / repetition_time)
                    n_trs = round(row["duration"] / repetition_time)
                    partitions[name] = slice(onset_tr, onset_tr + n_trs)
            if partitions:
                loc = _format_loc(sub=sub_id, ses=bold_key.ses, run=bold_key.run)
                sorted_slices = sorted(partitions.values(), key=lambda s: s.start)
                if sorted_slices[0].start != 0:
                    raise ValueError(f"Partitions do not start at TR 0 for {loc}")
                for a, b in zip(sorted_slices, sorted_slices[1:]):
                    if a.stop != b.start:
                        raise ValueError(
                            f"Partitions are not contiguous (gap or overlap between "
                            f"TR {a.stop} and {b.start}) for {loc}"
                        )
            bold_metas[bold_key] = BoldMeta(
                path=bold_file,
                repetition_time=repetition_time,
                partitions=partitions,
            )

        bold_bids = [BIDSPath(meta.path) for meta in bold_metas.values()]
        try:
            validate_entity_invariance(bold_bids, ("task", "acq", "ce", "rec", "dir"))
        except ValueError as e:
            raise ValueError(f"{e} (subject {sub_id})") from e

        repetition_times = {meta.repetition_time for meta in bold_metas.values()}
        if len(repetition_times) > 1:
            raise ValueError(
                f"Inconsistent repetition times (TRs) across BOLD files for "
                f"subject {sub_id}: {repetition_times}"
            )

        return bold_metas

    def _validate_alignment(
        self,
        feature_paths: dict[FeatureKey, Path],
        bold_metas: dict[BoldKey, BoldMeta],
    ) -> None:
        """Validate that feature and BOLD files are mutually consistent.

        Checks sub/task invariance across all files, bidirectional ses/run coverage
        between features and BOLD, and that every partition referenced by a
        feature file is declared in the corresponding events file.
        """
        feature_bids = [BIDSPath(path) for path in feature_paths.values()]
        bold_bids = [BIDSPath(meta.path) for meta in bold_metas.values()]
        sub_id = (feature_bids or bold_bids)[0].entities.get("sub")
        try:
            validate_entity_invariance(feature_bids + bold_bids, ("sub", "task"))
        except ValueError as e:
            raise ValueError(f"{e} (subject {sub_id})") from e

        # Every BOLD file must have feature coverage
        cell_keys = {feature_key.cell for feature_key in feature_paths}
        covered_bold_keys = {BoldKey(key.ses, key.run) for key in cell_keys}
        bold_without_features = bold_metas.keys() - covered_bold_keys
        if bold_without_features:
            bold_key = next(iter(bold_without_features))
            loc = _format_loc(sub=sub_id, ses=bold_key.ses, run=bold_key.run)
            msg = f"No feature files found for BOLD at {loc or f'sub={sub_id}'}"
            if len(bold_without_features) > 1:
                msg += f" ({len(bold_without_features) - 1} other coverage gaps exist)"
            raise FileNotFoundError(msg)

        # Every feature file must have a matching BOLD file
        features_without_bold = covered_bold_keys - bold_metas.keys()
        if features_without_bold:
            bold_key = next(iter(features_without_bold))
            loc = _format_loc(sub=sub_id, ses=bold_key.ses, run=bold_key.run)
            msg = f"No BOLD file found for features at {loc or f'sub={sub_id}'}"
            if len(features_without_bold) > 1:
                msg += f" ({len(features_without_bold) - 1} other coverage gaps exist)"
            raise FileNotFoundError(msg)

        # Every partitioned cell must reference a partition declared in events
        for cell_key in cell_keys:
            if cell_key.partition is None:
                continue
            bold_meta = bold_metas[BoldKey(cell_key.ses, cell_key.run)]
            if cell_key.partition not in bold_meta.partitions:
                loc = _format_loc(sub=sub_id, ses=cell_key.ses, run=cell_key.run)
                raise ValueError(
                    f"Partition {cell_key.partition!r} not found in events for {loc}"
                )

    def _build_xy(
        self,
        feature_paths: dict[FeatureKey, Path],
        bold_metas: dict[BoldKey, BoldMeta],
    ) -> TrainingData:
        """Assemble X and Y matrices for regression from feature files and BOLD arrays.

        Cells are sorted deterministically so row positions in X and Y are stable
        across runs. Column layout is derived from the first cell and assumed
        invariant — all cells must yield the same feature dimensionality.

        Partition boundaries are validated against actual BOLD array length before
        any data is assembled; a mismatch raises early rather than producing a
        silently truncated Y.
        """
        bold_arrays: dict[BoldKey, np.ndarray] = {
            key: _load_bold_array(meta.path) for key, meta in bold_metas.items()
        }

        for bold_key, bold_meta in bold_metas.items():
            if bold_meta.partitions:
                bold_data = bold_arrays[bold_key]
                last_tr = max(slice.stop for slice in bold_meta.partitions.values())
                if last_tr != bold_data.shape[0]:
                    loc = _format_loc(ses=bold_key.ses, run=bold_key.run)
                    raise ValueError(
                        f"Partitions cover {last_tr} TRs but BOLD has "
                        f"{bold_data.shape[0]} TRs for {loc}"
                    )

        # None sorts before any value; empty string is a stable tiebreaker
        def _sort_key(k: CellKey) -> tuple:
            return (
                k.ses is not None,
                k.ses or "",
                k.run is not None,
                k.run or "",
                k.partition is not None,
                k.partition or "",
            )

        cell_keys = sorted(
            {feature_key.cell for feature_key in feature_paths}, key=_sort_key
        )

        X_parts: list[np.ndarray] = []
        Y_parts: list[np.ndarray] = []
        row_slices: dict[CellKey, slice] = {}
        col_slices: dict[str, slice] = {}
        row_offset = 0
        col_offset = 0
        col_slices_initialized = False

        for cell_key in cell_keys:
            bold_key = BoldKey(cell_key.ses, cell_key.run)
            bold_meta = bold_metas[bold_key]
            bold_data = bold_arrays[bold_key]

            # Construct Y for the given cell
            if cell_key.partition is None:
                onset_tr, n_trs = 0, bold_data.shape[0]
            else:
                part = bold_meta.partitions[cell_key.partition]
                onset_tr, n_trs = part.start, part.stop - part.start
            row_slices[cell_key] = slice(row_offset, row_offset + n_trs)
            row_offset += n_trs
            Y_parts.append(bold_data[onset_tr : onset_tr + n_trs])

            # Construct X for the given cell
            feature_arrays: list[np.ndarray] = []
            for feature_name in self.features:
                df, _ = read_feature(feature_paths[FeatureKey(cell_key, feature_name)])
                arr = resample_feature(
                    df,
                    n_trs=n_trs,
                    repetition_time=bold_meta.repetition_time,
                    method=self.downsample,
                )
                feature_arrays.append(arr)
            if not col_slices_initialized:
                for feature_name, arr in zip(self.features, feature_arrays):
                    n_cols = arr.shape[1]
                    col_slices[feature_name] = slice(col_offset, col_offset + n_cols)
                    col_offset += n_cols
                col_slices_initialized = True  # col slices are invariant across cells
            X_parts.append(np.hstack(feature_arrays))

        X = np.concatenate(X_parts, axis=0)
        Y = np.concatenate(Y_parts, axis=0)

        return TrainingData(X=X, Y=Y, row_slices=row_slices, col_slices=col_slices)
