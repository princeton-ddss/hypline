import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, NamedTuple

import numpy as np
from pydantic import BaseModel

from hypline.bids import (
    BIDSPath,
    validate_bids_entities,
    validate_entity_invariance,
)
from hypline.bold import (
    BOLD_EXTENSIONS,
    BoldMeta,
    load_bold_meta,
    parse_bold_space,
)
from hypline.enums import Device
from hypline.features.utils import Downsample, read_feature, resample_feature
from hypline.utils import find_files, validate_dirs

# Entities provided via dedicated arguments — not allowed in bids_filters
_RESERVED_ENTITIES = frozenset(("sub", "space", "feature"))


class BoldKey(NamedTuple):
    ses: str | None
    run: str | None


class CellKey:
    """Open-schema key identifying a single feature time window.

    EXCLUDE defines which entities must never appear on a cell key:
    - sub, task, acq, ce, rec, dir: invariant across a training call
    - desc, res, den, echo: image-variant entities (BOLD derivatives only)
    - space, feature: orthogonal axes — handled by dedicated arguments

    Equality and hashing are order-independent.
    """

    EXCLUDE: frozenset[str] = frozenset(
        (
            "sub",
            "task",
            "acq",
            "ce",
            "rec",
            "dir",
            "desc",
            "res",
            "den",
            "echo",
            "space",
            "feature",
        )
    )
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
        feature_paths = self._resolve_cell_keys(feature_paths, bold_metas)
        feature_paths, bold_metas = self._apply_filters(feature_paths, bold_metas)
        self._validate_coverage(feature_paths, bold_metas)
        data = self._build_xy(feature_paths, bold_metas)

        # TODO: modeling (banded ridge regression) goes here
        data

    def _discover_features(self, sub_id: str) -> dict[FeatureKey, Path]:
        """Discover and validate feature file paths for a subject.

        Scans the features directory by BIDS filename alone — no feature data is read.
        Only sub and feature are used to filter find_files; bids_filters are applied
        post-enrichment in _apply_filters. Duplicate files for the same (cell, feature)
        pair raise immediately.

        Returns a flat dict mapping each (cell, feature) pair to its path.
        Every cell is guaranteed to have all requested features — a missing feature
        at any cell raises rather than silently producing an incomplete matrix.
        All files are guaranteed to share the same cell schema (entity key set).
        """
        feature_paths: dict[FeatureKey, Path] = {}
        for feature_name in self.features:
            feature_files = find_files(
                self.features_dir,
                ends_with=".parquet",
                recursive=True,
                bids_filters=[f"sub-{sub_id}", f"feature-{feature_name}"],
            )
            if not feature_files:
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
        read from the sidecar JSON, falling back to the image header. Segmentation
        is inferred from the colocated events TSV when present. All runs are guaranteed
        to share the same TR, BOLD-level entity invariants, and segment entity (or all
        unsegmented).
        """
        bold_ext = BOLD_EXTENSIONS[type(self.bold_space)]
        bold_files = find_files(
            self.bold_dir,
            ends_with=f"_bold{bold_ext}",
            recursive=True,
            bids_filters=[f"sub-{sub_id}", f"space-{self.bold_space}"],
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
                    f"  {bold_metas[bold_key].bids.path}\n  {bold_file}"
                )
            try:
                bold_metas[bold_key] = load_bold_meta(bold_file)
            except ValueError as e:
                loc = _format_loc(sub=sub_id, ses=bold_key.ses, run=bold_key.run)
                raise ValueError(f"{e} for {loc}") from e

        # Validate: acquisition entities are invariant across all runs
        bold_bids = [meta.bids for meta in bold_metas.values()]
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

        # Validate: segment entity is invariant across all runs (or all unsegmented)
        segment_entities = {
            meta.segments[0].entity if meta.segments else None
            for meta in bold_metas.values()
        }
        if len(segment_entities) > 1:
            run_labels = sorted(
                f"{meta.bids.path.name} ("
                f"{meta.segments[0].entity if meta.segments else 'unsegmented'})"
                for meta in bold_metas.values()
            )
            raise ValueError(
                f"BOLD runs disagree on segment entity for subject {sub_id}:\n  "
                + "\n  ".join(run_labels)
            )

        # Validate: segment metadata schema is invariant across segmented runs
        segmented_metas = [meta for meta in bold_metas.values() if meta.segments]
        metadata_key_sets = {
            frozenset(seg.metadata) for meta in segmented_metas for seg in meta.segments
        }
        if len(metadata_key_sets) > 1:
            run_labels = sorted(
                f"{meta.bids.path.name} "
                f"({sorted(meta.segments[0].metadata) or 'no metadata'})"
                for meta in segmented_metas
            )
            raise ValueError(
                f"BOLD runs disagree on segment metadata schema for subject "
                f"{sub_id}:\n  " + "\n  ".join(run_labels)
            )

        return bold_metas

    def _resolve_cell_keys(
        self,
        feature_paths: dict[FeatureKey, Path],
        bold_metas: dict[BoldKey, BoldMeta],
    ) -> dict[FeatureKey, Path]:
        """Validate and resolve feature CellKeys against BOLD segment metadata.

        For each feature cell, locates the matching BOLD run and segment, then
        merges segment.metadata into the CellKey. Filename entities beyond ses, run,
        and the segment entity are rejected unless they echo a metadata key from
        events.json — descriptive metadata must live in events.json, not filenames.

        Invariant: _discover_bold guarantees all segments share the same metadata
        schema across runs, so resolved cells always end up with a uniform key set.
        """
        cell_keys_by_bold_key: dict[BoldKey, set[CellKey]] = {}
        for feature_key in feature_paths:
            bold_key = BoldKey(feature_key.cell.get("ses"), feature_key.cell.get("run"))
            cell_keys_by_bold_key.setdefault(bold_key, set()).add(feature_key.cell)

        orphan_bold_keys = cell_keys_by_bold_key.keys() - bold_metas.keys()
        if orphan_bold_keys:
            bold_key = next(iter(orphan_bold_keys))
            loc = _format_loc(
                ses=bold_key.ses, run=bold_key.run, space=str(self.bold_space)
            )
            msg = f"No BOLD file found for features at {loc}"
            if len(orphan_bold_keys) > 1:
                msg += f" ({len(orphan_bold_keys) - 1} other coverage gaps exist)"
            raise FileNotFoundError(msg)

        resolved_feature_paths: dict[FeatureKey, Path] = {}
        for feature_key, path in feature_paths.items():
            cell_key = feature_key.cell
            bold_key = BoldKey(cell_key.get("ses"), cell_key.get("run"))
            bold_meta = bold_metas[bold_key]

            if not bold_meta.segments:
                run_cell_keys = cell_keys_by_bold_key[bold_key]
                if len(run_cell_keys) > 1:
                    loc = _format_loc(ses=bold_key.ses, run=bold_key.run)
                    raise ValueError(
                        f"Run is unsegmented but has {len(run_cell_keys)} feature "
                        f"cells at {loc} — provide an events.tsv with BIDS key-value "
                        f"entities to segment the run"
                    )
                illegal_keys = cell_key.keys() - {"ses", "run"}
                if illegal_keys:
                    raise ValueError(
                        f"Feature cell {cell_key!r} carries entities "
                        f"{sorted(illegal_keys)} that do not identify the run. "
                        f"For an unsegmented run, only ses and run are valid on "
                        f"feature filenames — move descriptive metadata to events.json"
                    )
                resolved_feature_paths[feature_key] = path
                continue

            segment_entity = bold_meta.segments[0].entity
            segment_values = {segment.value for segment in bold_meta.segments}

            # Validate the cell carries a known segment value for this run
            segment_value = cell_key.get(segment_entity)
            if segment_value is None:
                loc = _format_loc(ses=bold_key.ses, run=bold_key.run)
                raise ValueError(
                    f"Feature cell {cell_key!r} at {loc} is missing segment "
                    f"entity {segment_entity!r} declared in events"
                )
            if segment_value not in segment_values:
                loc = _format_loc(ses=bold_key.ses, run=bold_key.run)
                raise ValueError(
                    f"Segment value {segment_entity}-{segment_value} at {loc} not "
                    f"found in events — valid values: {sorted(segment_values)}"
                )

            segment = next(
                seg for seg in bold_meta.segments if seg.value == segment_value
            )

            # Reject filename entities that are not run-locating, segment, or metadata
            legal_keys = {"ses", "run"} | {segment_entity} | segment.metadata.keys()
            illegal_keys = cell_key.keys() - legal_keys
            if illegal_keys:
                raise ValueError(
                    f"Feature cell {cell_key!r} carries entities "
                    f"{sorted(illegal_keys)} that are absent from events.json "
                    f"metadata. Descriptive attributes must live in events.json, "
                    f"not feature filenames"
                )

            # Merge metadata: filename value takes precedence only if it agrees
            extra_metadata: dict[str, str] = {}
            for entity, value in segment.metadata.items():
                if cell_key.get(entity) is not None and cell_key[entity] != value:
                    raise ValueError(
                        f"Feature filename and events.json disagree on {entity!r} "
                        f"for {segment_entity}-{segment_value}: "
                        f"filename has {cell_key[entity]!r}, sidecar has {value!r}"
                    )
                if cell_key.get(entity) is None:
                    extra_metadata[entity] = value
            if extra_metadata:
                resolved_cell_key = CellKey(
                    **{**dict(cell_key.items()), **extra_metadata}
                )
                resolved_feature_paths[
                    FeatureKey(cell=resolved_cell_key, feature=feature_key.feature)
                ] = path
            else:
                resolved_feature_paths[feature_key] = path

        return resolved_feature_paths

    def _apply_filters(
        self,
        feature_paths: dict[FeatureKey, Path],
        bold_metas: dict[BoldKey, BoldMeta],
    ) -> tuple[dict[FeatureKey, Path], dict[BoldKey, BoldMeta]]:
        """Apply bids_filters to feature cells and BOLD runs.

        Filters are applied against CellKey entities for features, and against
        filename entities for BOLD. Same-entity filter values OR-match within a
        group; different entities AND-match across groups. A filter key absent from
        one side is skipped on that side rather than rejecting all rows. A filter
        key absent from both sides raises ValueError (typo diagnostic) before any
        empty-result condition surfaces as a coverage error.
        """
        if not self.bids_filters:
            return feature_paths, bold_metas

        # Group filter values by entity for matching later
        allowed_values_by_entity: dict[str, list[str]] = {}
        for bids_filter in self.bids_filters:
            entity_key, entity_value = bids_filter.split("-", 1)
            allowed_values_by_entity.setdefault(entity_key, []).append(entity_value)

        # Collect entity key schema from both sides for typo detection
        cell_entity_keys = frozenset(
            entity_key
            for feature_key in feature_paths
            for entity_key in feature_key.cell.keys()
        )
        bold_entity_keys = frozenset(
            entity_key
            for meta in bold_metas.values()
            for entity_key in meta.bids.entities
        )
        known_entity_keys = cell_entity_keys | bold_entity_keys

        for entity_key in allowed_values_by_entity:
            if entity_key not in known_entity_keys:
                raise ValueError(
                    f"bids_filters entity {entity_key!r} not found on any "
                    f"feature cell or BOLD file for this subject — check for a typo"
                )

        def _cell_matches(cell: CellKey) -> bool:
            return all(
                cell.get(entity_key) in entity_values
                for entity_key, entity_values in allowed_values_by_entity.items()
                if entity_key in cell_entity_keys
            )

        def _bold_matches(bids: BIDSPath) -> bool:
            return all(
                bids.entities.get(entity_key) in entity_values
                for entity_key, entity_values in allowed_values_by_entity.items()
                if entity_key in bold_entity_keys
            )

        filtered_features = {
            feature_key: path
            for feature_key, path in feature_paths.items()
            if _cell_matches(feature_key.cell)
        }
        filtered_bold = {
            bold_key: meta
            for bold_key, meta in bold_metas.items()
            if _bold_matches(meta.bids)
        }

        return filtered_features, filtered_bold

    def _validate_coverage(
        self,
        feature_paths: dict[FeatureKey, Path],
        bold_metas: dict[BoldKey, BoldMeta],
    ) -> None:
        """Validate bidirectional ses/run coverage between filtered features and BOLD.

        Also checks sub/task invariance across all feature and BOLD files.
        Raises if either side is empty — indicates filters selected nothing.
        """
        if not feature_paths:
            raise FileNotFoundError("No feature files match the given filters")
        if not bold_metas:
            raise FileNotFoundError("No BOLD files match the given filters")

        feature_bids = [BIDSPath(path) for path in feature_paths.values()]
        bold_bids = [meta.bids for meta in bold_metas.values()]
        sub_id = (feature_bids or bold_bids)[0].entities.get("sub")
        try:
            validate_entity_invariance(feature_bids + bold_bids, ("sub", "task"))
        except ValueError as e:
            raise ValueError(f"{e} (subject {sub_id})") from e

        cell_keys = {feature_key.cell for feature_key in feature_paths}
        covered_bold_keys = {
            BoldKey(key.get("ses"), key.get("run")) for key in cell_keys
        }

        bold_without_features = bold_metas.keys() - covered_bold_keys
        if bold_without_features:
            bold_key = next(iter(bold_without_features))
            loc = _format_loc(sub=sub_id, ses=bold_key.ses, run=bold_key.run)
            msg = f"No feature files found for BOLD at {loc}"
            if len(bold_without_features) > 1:
                msg += f" ({len(bold_without_features) - 1} other coverage gaps exist)"
            raise FileNotFoundError(msg)

        features_without_bold = covered_bold_keys - bold_metas.keys()
        if features_without_bold:
            bold_key = next(iter(features_without_bold))
            loc = _format_loc(sub=sub_id, ses=bold_key.ses, run=bold_key.run)
            msg = f"No BOLD file found for features at {loc}"
            if len(features_without_bold) > 1:
                msg += f" ({len(features_without_bold) - 1} other coverage gaps exist)"
            raise FileNotFoundError(msg)

    def _build_xy(
        self,
        feature_paths: dict[FeatureKey, Path],
        bold_metas: dict[BoldKey, BoldMeta],
    ) -> TrainingData:
        """Assemble X and Y matrices for regression from feature files and BOLD arrays.

        Cells are sorted deterministically so row positions in X and Y are stable
        across runs. Column layout is derived from the first cell and assumed
        invariant — all cells must yield the same feature dimensionality.

        Segment slice coverage is validated against actual BOLD array length before
        any data is assembled; a mismatch raises early rather than producing a
        silently truncated Y.
        """
        bold_arrays: dict[BoldKey, np.ndarray] = {
            key: _load_bold_array(meta.bids.path) for key, meta in bold_metas.items()
        }

        for bold_key, bold_meta in bold_metas.items():
            if not bold_meta.segments:
                continue
            expected = max(seg.slice.stop for seg in bold_meta.segments)
            actual = bold_arrays[bold_key].shape[0]
            if expected > actual:
                first_bold = next(iter(bold_metas.values()))
                sub_id = first_bold.bids.entities.get("sub")
                loc = _format_loc(sub=sub_id, ses=bold_key.ses, run=bold_key.run)
                raise ValueError(
                    f"Segment slices extend to TR {expected} "
                    f"but BOLD has only {actual} TRs for {loc}"
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
            if not bold_meta.segments:
                onset_tr, n_trs = 0, bold_data.shape[0]
            else:
                segment_entity = bold_meta.segments[0].entity
                segment_value = cell_key[segment_entity]
                seg = next(s for s in bold_meta.segments if s.value == segment_value)
                onset_tr, n_trs = seg.slice.start, seg.slice.stop - seg.slice.start
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
