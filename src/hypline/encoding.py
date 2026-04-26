import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, NamedTuple

import numpy as np
import polars as pl
from pydantic import BaseModel

from hypline.bids import (
    BIDS_ENTITY_RE,
    BIDSPath,
    validate_bids_entities,
    validate_entity_invariance,
)
from hypline.bold import (
    BOLD_EXTENSIONS,
    get_repetition_time,
    load_events,
    parse_bold_space,
)
from hypline.enums import Device
from hypline.features.utils import Downsample, read_feature, resample_feature
from hypline.utils import find_files, validate_dirs

# Entities provided via dedicated arguments — not allowed in bids_filters
_RESERVED_ENTITIES = frozenset(("sub", "space", "feature"))

# Entities valid for filtering both feature and BOLD files
_COMMON_FILTER_ENTITIES = frozenset(("ses", "task", "run"))

# Entities valid for BOLD files only — stripped from feature-side filters
_BOLD_EXCLUSIVE_ENTITIES = frozenset(
    ("desc", "res", "den", "echo", "acq", "ce", "rec", "dir")
)

# Entities valid for BOLD file filtering — union of common and BOLD-exclusive
_BOLD_FILTER_ENTITIES = _COMMON_FILTER_ENTITIES | _BOLD_EXCLUSIVE_ENTITIES


class BoldKey(NamedTuple):
    ses: str | None
    run: str | None


class CellKey:
    """Open-schema key identifying a single feature time window.

    Entities in EXCLUDE are rejected — they are invariant within a training call
    (sub, task), belong to a different axis (feature), or are BOLD-only (space).
    Equality and hashing are order-independent.
    """

    EXCLUDE: frozenset[str] = frozenset(("sub", "task", "space", "feature"))
    __slots__ = ("_entities",)

    def __init__(self, **entities: str) -> None:
        invalid = frozenset(entities) & self.EXCLUDE
        if invalid:
            raise ValueError(f"CellKey does not accept entities: {sorted(invalid)}")
        self._entities: dict[str, str] = dict(entities)

    def __getitem__(self, key: str) -> str:
        return self._entities[key]

    def get(self, key: str, default: str | None = None) -> str | None:
        return self._entities.get(key, default)

    def keys(self) -> frozenset[str]:
        return frozenset(self._entities)

    def items(self) -> Iterator[tuple[str, str]]:
        return iter(self._entities.items())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CellKey):
            return NotImplemented
        return self._entities == other._entities

    def __hash__(self) -> int:
        return hash(frozenset(self._entities.items()))

    def __repr__(self) -> str:
        pairs = ", ".join(f"{k}={v!r}" for k, v in sorted(self._entities.items()))
        return f"CellKey({pairs})"


class FeatureKey(NamedTuple):
    cell: CellKey
    feature: str


class Partitioning(NamedTuple):
    entity: str
    slices: dict[str, slice]  # entity value → TR-slice


class BoldMeta(NamedTuple):
    path: Path
    repetition_time: float
    partitioning: Partitioning | None


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


def _tiles(slices: list[slice]) -> bool:
    """Return True if slices start at 0, are non-overlapping, and are contiguous."""
    ordered = sorted(slices, key=lambda s: s.start)
    if ordered[0].start != 0:
        return False
    return all(a.stop == b.start for a, b in zip(ordered, ordered[1:]))


def _mean_slice_duration(slices: dict[str, slice]) -> float:
    return sum(s.stop - s.start for s in slices.values()) / len(slices)


def _infer_partitioning(
    events: pl.DataFrame | None,
    repetition_time: float,
) -> Partitioning | None:
    """Identify the partition entity and its TR-slices from events.tsv.

    Convention: BIDS key-value trial_type rows (e.g., `block-1`, `trial-A`)
    declare partitions — non-overlapping time windows that tile the run.
    Non-partition annotations must use flat (non-key-value) trial_type labels
    (e.g., `rest`, `fixation`). Every BIDS key-value entity found in events must
    tile the run; partial tiling is a design error and raises.

    Tiling = values start at onset=0, are non-overlapping, contiguous, and cover the
    full events span. The partition entity is the tiling entity with the smallest
    average slice duration; ties raise.

    Returns None when there are no BIDS key-value events (unpartitioned run).
    """
    if events is None:
        return None

    partition_rows = events.filter(
        pl.col("trial_type").str.contains(BIDS_ENTITY_RE.pattern)
        & (pl.col("duration") > 0.0)
    )

    if partition_rows.is_empty():
        return None

    slices_by_entity: dict[str, dict[str, slice]] = {}
    for row in partition_rows.iter_rows(named=True):
        entity_name, entity_value = row["trial_type"].split("-", 1)
        onset_tr = round(row["onset"] / repetition_time)
        n_trs = round(row["duration"] / repetition_time)
        entity_slices = slices_by_entity.setdefault(entity_name, {})
        entity_slices[entity_value] = slice(onset_tr, onset_tr + n_trs)

    events_span_trs = round(
        partition_rows.select((pl.col("onset") + pl.col("duration")).max()).item()
        / repetition_time
    )

    tiling = {
        entity_name: entity_slices
        for entity_name, entity_slices in slices_by_entity.items()
        if _tiles(list(entity_slices.values()))
        and max(s.stop for s in entity_slices.values()) == events_span_trs
    }

    non_tiling = sorted(slices_by_entity.keys() - tiling.keys())
    if non_tiling:
        raise ValueError(
            f"BIDS entities {non_tiling} in events.tsv do not partition the run "
            f"cleanly. Values must be unique and slices must cover the run "
            f"contiguously from onset 0. Check for duplicate values (e.g. trial IDs "
            f"reset per block), gaps, overlaps, or partial run coverage. Use flat "
            f"trial_type labels (e.g. 'rest') for non-partition annotations."
        )

    # Partition entity is the tiling entity with the finest granularity
    avg_durations = {name: _mean_slice_duration(s) for name, s in tiling.items()}
    min_avg = min(avg_durations.values())
    tied = [name for name, avg in avg_durations.items() if avg == min_avg]
    if len(tied) > 1:
        raise ValueError(
            f"entities {tied[0]!r} and {tied[1]!r} both tile the run at identical "
            f"granularity — remove the redundant one"
        )
    partition_entity = tied[0]

    return Partitioning(entity=partition_entity, slices=tiling[partition_entity])


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
        for entity in bids_filters:
            key = entity.split("-", 1)[0]
            if key in _RESERVED_ENTITIES:
                raise ValueError(
                    f"bids_filters cannot contain {key!r} "
                    "— use the dedicated argument instead"
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
        bids_filters are routed to find_files after stripping BOLD-exclusive entities.
        Duplicate files for the same (cell, feature) pair raise immediately.

        Returns a flat dict mapping each (cell, feature) pair to its path.
        Every cell is guaranteed to have all requested features — a missing feature
        at any cell raises rather than silently producing an incomplete matrix.
        All files are guaranteed to share the same cell schema (entity key set).
        A filter entity absent from all discovered files raises ValueError before
        FileNotFoundError — catches typos and mismatched filter entities early.
        """
        feature_filters = [
            entity
            for entity in self.bids_filters
            if entity.split("-", 1)[0] not in _BOLD_EXCLUSIVE_ENTITIES
        ]

        feature_paths: dict[FeatureKey, Path] = {}
        for feature_name in self.features:
            feature_files = find_files(
                self.features_dir,
                ends_with=".parquet",
                recursive=True,
                bids_filters=[
                    f"sub-{sub_id}",
                    f"feature-{feature_name}",
                    *feature_filters,
                ],
            )
            if not feature_files:
                # Check whether a filter entity typo caused the empty result
                unfiltered = find_files(
                    self.features_dir,
                    ends_with=".parquet",
                    recursive=True,
                    bids_filters=[f"sub-{sub_id}", f"feature-{feature_name}"],
                )
                if unfiltered:
                    unfiltered_schema = frozenset(
                        key for path in unfiltered for key in BIDSPath(path).entities
                    )
                    for entity in feature_filters:
                        key = entity.split("-", 1)[0]
                        if key not in unfiltered_schema:
                            raise ValueError(
                                f"bids_filters entity {key!r} not found on any "
                                f"feature={feature_name} file for sub={sub_id}"
                            )
                raise FileNotFoundError(
                    f"No matching feature files found for sub={sub_id}, "
                    f"feature={feature_name} in {self.features_dir}"
                )

            for feature_file in feature_files:
                bids = BIDSPath(feature_file)
                cell_key = CellKey(
                    **{
                        key: val
                        for key, val in bids.entities.items()
                        if key not in CellKey.EXCLUDE
                    }
                )
                feature_key = FeatureKey(cell=cell_key, feature=feature_name)
                if feature_key in feature_paths:
                    loc = _format_loc(sub=sub_id, **dict(cell_key.items()))
                    raise ValueError(
                        f"Multiple feature files for feature={feature_name}, {loc}:\n"
                        f"  {feature_paths[feature_key]}\n  {feature_file}"
                    )
                feature_paths[feature_key] = feature_file

        # Validate: all files share the same entity key set
        schema: frozenset[str] | None = None
        schema_path: Path | None = None
        for feature_key, path in feature_paths.items():
            file_schema = feature_key.cell.keys()
            if schema is None:
                schema, schema_path = file_schema, path
            elif file_schema != schema:
                raise ValueError(
                    f"Inconsistent feature file schemas:\n  {schema_path}\n  {path}"
                )

        # Validate: task entity is invariant across all files
        feature_bids = [BIDSPath(path) for path in feature_paths.values()]
        try:
            validate_entity_invariance(feature_bids, ("task",))
        except ValueError as e:
            raise ValueError(f"{e} (subject {sub_id})") from e

        # Validate: all features present at every cell
        expected = {
            FeatureKey(cell_key, feature_name)
            for cell_key in {feature_key.cell for feature_key in feature_paths}
            for feature_name in self.features
        }
        missing = expected - feature_paths.keys()
        if missing:
            feature_key = next(iter(missing))
            loc = _format_loc(sub=sub_id, **dict(feature_key.cell.items()))
            msg = f"Missing feature={feature_key.feature} at {loc}"
            if len(missing) > 1:
                msg += f" ({len(missing) - 1} other coverage gaps exist)"
            raise FileNotFoundError(msg)

        return feature_paths

    def _discover_bold(self, sub_id: str) -> dict[BoldKey, BoldMeta]:
        """Discover BOLD files and load their metadata for a subject.

        Scans the BOLD directory by filename without loading image arrays. TR is
        read from the sidecar JSON, falling back to the image header. Partitioning
        is inferred from the colocated events TSV when present — raises if events
        contain BIDS entities that fail to tile the run. All runs are guaranteed to
        share the same TR, BOLD-level entity invariants, and partition entity (or all
        unpartitioned).
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
            try:
                partitioning = _infer_partitioning(events, repetition_time)
            except ValueError as e:
                loc = _format_loc(sub=sub_id, ses=bold_key.ses, run=bold_key.run)
                raise ValueError(f"{e} for {loc}") from e
            bold_metas[bold_key] = BoldMeta(
                path=bold_file,
                repetition_time=repetition_time,
                partitioning=partitioning,
            )

        # Validate: acquisition entities are invariant across all runs
        bold_bids = [BIDSPath(meta.path) for meta in bold_metas.values()]
        try:
            validate_entity_invariance(bold_bids, ("task", "acq", "ce", "rec", "dir"))
        except ValueError as e:
            raise ValueError(f"{e} (subject {sub_id})") from e

        # Validate: TR is invariant across all runs
        repetition_times = {meta.repetition_time for meta in bold_metas.values()}
        if len(repetition_times) > 1:
            raise ValueError(
                f"Inconsistent repetition times (TRs) across BOLD files for "
                f"subject {sub_id}: {repetition_times}"
            )

        # Validate: partition entity is invariant across all runs (or all unpartitioned)
        partition_entities = {
            meta.partitioning.entity if meta.partitioning is not None else None
            for meta in bold_metas.values()
        }
        if len(partition_entities) > 1:
            run_labels = sorted(
                f"{meta.path.name} ("
                f"{meta.partitioning.entity if meta.partitioning else 'unpartitioned'})"
                for meta in bold_metas.values()
            )
            raise ValueError(
                f"BOLD runs disagree on partition entity for subject {sub_id}:\n  "
                + "\n  ".join(run_labels)
            )

        return bold_metas

    def _validate_alignment(
        self,
        feature_paths: dict[FeatureKey, Path],
        bold_metas: dict[BoldKey, BoldMeta],
    ) -> None:
        """Validate that feature and BOLD files are mutually consistent.

        Checks sub/task invariance across all files, bidirectional ses/run coverage
        between features and BOLD, and that feature cells align with each run's
        partitioning: for partitioned runs, every cell must carry the partition entity
        with a value declared in events; for unpartitioned runs, exactly one cell per
        run is allowed.

        Notes
        -----
        When all BOLD runs are unpartitioned, extra entities on feature filenames
        beyond ses/run are accepted as descriptive tags. If those entities were intended
        as partition keys but events.tsv is absent or contains no BIDS key-value rows,
        the misalignment is not detectable here and will surface only as unexpected
        row counts in the assembled X/Y matrices.
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
        covered_bold_keys = {
            BoldKey(key.get("ses"), key.get("run")) for key in cell_keys
        }
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

        # Validate feature cells against each run's partitioning
        cells_by_bold_key: dict[BoldKey, list[CellKey]] = {}
        for cell_key in cell_keys:
            bold_key = BoldKey(cell_key.get("ses"), cell_key.get("run"))
            cells_by_bold_key.setdefault(bold_key, []).append(cell_key)

        for bold_key, bold_meta in bold_metas.items():
            run_cells = cells_by_bold_key[bold_key]
            loc = _format_loc(sub=sub_id, ses=bold_key.ses, run=bold_key.run)

            if bold_meta.partitioning is None:
                if len(run_cells) > 1:
                    raise ValueError(
                        f"Run is unpartitioned but has {len(run_cells)} feature cells "
                        f"at {loc} — provide an events.tsv with tiling entities to "
                        f"partition the run"
                    )
            else:
                entity = bold_meta.partitioning.entity
                for cell_key in run_cells:
                    value = cell_key.get(entity)
                    if value is None:
                        raise ValueError(
                            f"Feature cell {cell_key!r} at {loc} is missing partition "
                            f"entity {entity!r} declared in events"
                        )
                    if value not in bold_meta.partitioning.slices:
                        raise ValueError(
                            f"Partition value {entity}-{value} at {loc} not found in "
                            f"events — valid values: "
                            f"{sorted(bold_meta.partitioning.slices)}"
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

        Partition slice coverage is validated against actual BOLD array length before
        any data is assembled; a mismatch raises early rather than producing a
        silently truncated Y.
        """
        bold_arrays: dict[BoldKey, np.ndarray] = {
            key: _load_bold_array(meta.path) for key, meta in bold_metas.items()
        }

        for bold_key, bold_meta in bold_metas.items():
            if bold_meta.partitioning is None:
                continue
            expected = max(s.stop for s in bold_meta.partitioning.slices.values())
            actual = bold_arrays[bold_key].shape[0]
            if expected != actual:
                first_bold = next(iter(bold_metas.values()))
                sub_id = BIDSPath(first_bold.path).entities.get("sub")
                loc = _format_loc(sub=sub_id, ses=bold_key.ses, run=bold_key.run)
                raise ValueError(
                    f"Partition slices cover {expected} TRs "
                    f"but BOLD has {actual} for {loc}"
                )

        # None sorts before any value; empty string is a stable tiebreaker for ses/run
        def _sort_key(k: CellKey) -> tuple:
            ses = k.get("ses")
            run = k.get("run")
            rest = sorted(val for key, val in k.items() if key not in ("ses", "run"))
            return (ses is not None, ses or "", run is not None, run or "", *rest)

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
            bold_key = BoldKey(cell_key.get("ses"), cell_key.get("run"))
            bold_meta = bold_metas[bold_key]
            bold_data = bold_arrays[bold_key]

            # Construct Y for the given cell
            if bold_meta.partitioning is None:
                onset_tr, n_trs = 0, bold_data.shape[0]
            else:
                partition_value = cell_key[bold_meta.partitioning.entity]
                tr_slice = bold_meta.partitioning.slices[partition_value]
                onset_tr, n_trs = tr_slice.start, tr_slice.stop - tr_slice.start
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
