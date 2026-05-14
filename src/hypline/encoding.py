import reprlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, NamedTuple

import numpy as np
from pydantic import BaseModel

from hypline.bids import (
    BIDSPath,
    normalize_bids_filters,
    validate_entity_invariance,
)
from hypline.bold import (
    BOLD_EXTENSIONS,
    BoldMeta,
    load_bold_meta,
    parse_bold_space,
)
from hypline.enums import Device
from hypline.features._utils import (
    Downsample,
    read_feature,
    read_feature_metadata,
    resample_feature,
)
from hypline.layout import BIDSLayout


class BoldKey(NamedTuple):
    ses: str | None
    run: str | None


class CellKey:
    """Open-schema key identifying a single feature time window.

    EXCLUDE defines which entities must never appear on a cell key:
    - sub, task, acq, ce, rec, dir: invariant across a training call
    - desc, res, den, echo: image-variant entities (BOLD derivatives only)
    - space, feat: orthogonal axes — handled by dedicated arguments

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
            "feat",
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


def _diff_meta(reference: dict, compare: dict) -> list[str]:
    """Return `key: ref_val != cmp_val` lines for differing keys.

    Values are truncated via `reprlib`; missing keys render as `<missing>`.
    """
    missing = object()
    lines = []
    for key in sorted(reference.keys() | compare.keys()):
        rv, cv = reference.get(key, missing), compare.get(key, missing)
        if rv != cv:
            rs = "<missing>" if rv is missing else reprlib.repr(rv)
            cs = "<missing>" if cv is missing else reprlib.repr(cv)
            lines.append(f"{key}: {rs} != {cs}")
    return lines


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
        layout: BIDSLayout,
        features: list[str],
        bold_space: str,
        downsample: str | Downsample = "mean",
        bids_filters: list[str] | None = None,
    ):
        import torch

        if config.device is Device.CUDA and not torch.cuda.is_available():
            raise RuntimeError("CUDA is requested but not available")
        self.config = config
        self._layout = layout

        if not features:
            raise ValueError("features must be a non-empty list")
        if len(features) != len(set(features)):
            dupes = sorted({f for f in features if features.count(f) > 1})
            raise ValueError(f"Duplicate entries in features: {dupes}")
        self.features = features

        self.bold_space = parse_bold_space(bold_space)
        self.downsample = Downsample(downsample)

        self.bids_filters = normalize_bids_filters(
            bids_filters, reserved={"sub", "space", "feat"}
        )

    def train(self, sub_id: str):
        feature_bids = self._discover_features(sub_id)
        bold_metas = self._discover_bold(sub_id)
        feature_bids = self._resolve_cell_keys(sub_id, feature_bids, bold_metas)
        feature_bids, bold_metas = self._apply_filters(sub_id, feature_bids, bold_metas)
        self._validate_coverage(sub_id, feature_bids, bold_metas)
        data = self._build_xy(sub_id, feature_bids, bold_metas)

        # TODO: modeling (banded ridge regression) goes here
        data

    def _discover_features(self, sub_id: str) -> dict[FeatureKey, BIDSPath]:
        """Discover and validate feature file paths for a subject.

        Scans the features directory by BIDS filename alone — no feature data is read.
        Only sub and feature are used as structural filters; bids_filters are applied
        post-enrichment in _apply_filters. Duplicate files for the same (cell, feature)
        pair raise immediately.

        Returns a flat dict mapping each (cell, feature) pair to its BIDSPath.
        Every cell is guaranteed to have all requested features — a missing feature
        at any cell raises rather than silently producing an incomplete matrix.
        All files are guaranteed to share the same cell schema (entity key set).
        """
        feature_bids: dict[FeatureKey, BIDSPath] = {}
        for feature_name in self.features:
            feature_files = self._layout.find.features(sub=sub_id, kind=feature_name)
            if not feature_files:
                raise FileNotFoundError(
                    f"No matching feature files found for sub={sub_id}, "
                    f"feat={feature_name}"
                )

            for bids in feature_files:
                cell_key = CellKey(
                    **{
                        key: val
                        for key, val in bids.entities.items()
                        if key not in CellKey.EXCLUDE
                    }
                )
                feature_key = FeatureKey(cell=cell_key, feature=feature_name)
                if feature_key in feature_bids:
                    loc = _format_loc(sub=sub_id, **dict(cell_key.items()))
                    raise ValueError(
                        f"Multiple feature files for feat={feature_name}, {loc}:\n"
                        f"  {feature_bids[feature_key].path}\n  {bids.path}"
                    )
                feature_bids[feature_key] = bids

        # Validate: all files share the same entity key set
        schema: frozenset[str] | None = None
        schema_path: Path | None = None
        for feature_key, bids in feature_bids.items():
            file_schema = feature_key.cell.keys()
            if schema is None:
                schema, schema_path = file_schema, bids.path
            elif file_schema != schema:
                raise ValueError(
                    f"Inconsistent feature file schemas:\n"
                    f"  {schema_path}\n  {bids.path}"
                )

        # Validate: metadata is identical across files for each feature
        # (keys prefixed with '_' are exempt — reserved for per-file metadata)
        per_feature_meta: dict[str, tuple[dict, Path]] = {}
        for feature_key, bids in feature_bids.items():
            meta = {
                key: val
                for key, val in read_feature_metadata(bids.path).items()
                if not key.startswith("_")
            }
            feature_name = feature_key.feature
            if feature_name not in per_feature_meta:
                per_feature_meta[feature_name] = (meta, bids.path)
            elif per_feature_meta[feature_name][0] != meta:
                ref_meta, ref_path = per_feature_meta[feature_name]
                diff = "\n".join(f"    {line}" for line in _diff_meta(ref_meta, meta))
                raise ValueError(
                    f"Inconsistent metadata for feat={feature_name}:\n"
                    f"  {ref_path}\n  {bids.path}\n  differing keys:\n{diff}"
                )

        # Validate: task entity is invariant across all files
        try:
            validate_entity_invariance(list(feature_bids.values()), ("task",))
        except ValueError as e:
            raise ValueError(f"{e} (subject {sub_id})") from e

        # Validate: all features present at every cell
        expected = {
            FeatureKey(cell_key, feature_name)
            for cell_key in {feature_key.cell for feature_key in feature_bids}
            for feature_name in self.features
        }
        missing = expected - feature_bids.keys()
        if missing:
            feature_key = next(iter(missing))
            loc = _format_loc(sub=sub_id, **dict(feature_key.cell.items()))
            msg = f"Missing feat={feature_key.feature} at {loc}"
            if len(missing) > 1:
                msg += f" ({len(missing) - 1} other coverage gaps exist)"
            raise FileNotFoundError(msg)

        return feature_bids

    def _discover_bold(self, sub_id: str) -> dict[BoldKey, BoldMeta]:
        """Discover BOLD files and load their metadata for a subject.

        Scans the BOLD directory by filename without loading image arrays. TR is
        read from the sidecar JSON, falling back to the image header. Segmentation
        is inferred from the colocated events TSV when present. All runs are guaranteed
        to share the same TR, BOLD-level entity invariants, and segment entity (or all
        unsegmented).
        """
        bold_ext = BOLD_EXTENSIONS[type(self.bold_space)]
        bold_files = self._layout.find.fmriprep(
            sub=sub_id,
            suffix="bold",
            ext=bold_ext,
            bids_filters=[
                f"space-{self.bold_space}",
                "desc-clean",  # hardcoded until parameterization is needed
            ],
        )
        if not bold_files:
            raise FileNotFoundError(
                f"No BOLD files found for sub={sub_id}, space={self.bold_space}, "
                f"desc=clean"
            )

        bold_metas: dict[BoldKey, BoldMeta] = {}
        for bids in bold_files:
            bold_key = BoldKey(
                ses=bids.entities.get("ses"),
                run=bids.entities.get("run"),
            )
            if bold_key in bold_metas:
                loc = _format_loc(sub=sub_id, ses=bold_key.ses, run=bold_key.run)
                raise ValueError(
                    f"Duplicate BOLD file at {loc}:\n"
                    f"  {bold_metas[bold_key].bids.path}\n  {bids.path}"
                )
            try:
                bold_metas[bold_key] = load_bold_meta(bids.path)
            except ValueError as e:
                loc = _format_loc(sub=sub_id, ses=bold_key.ses, run=bold_key.run)
                raise ValueError(f"Failed to load BOLD at {loc}: {e}") from e

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
        sub_id: str,
        feature_bids: dict[FeatureKey, BIDSPath],
        bold_metas: dict[BoldKey, BoldMeta],
    ) -> dict[FeatureKey, BIDSPath]:
        """Validate and resolve feature CellKeys against BOLD segment metadata.

        For each feature cell, locates the matching BOLD run and segment, then
        merges segment.metadata into the CellKey. Filename entities beyond ses, run,
        and the segment entity are rejected unless they echo a metadata key from
        events.json — descriptive metadata must live in events.json, not filenames.

        Invariant: _discover_bold guarantees all segments share the same metadata
        schema across runs, so resolved cells always end up with a uniform key set.
        """
        cell_keys_by_bold_key: dict[BoldKey, set[CellKey]] = {}
        for feature_key in feature_bids:
            bold_key = BoldKey(feature_key.cell.get("ses"), feature_key.cell.get("run"))
            cell_keys_by_bold_key.setdefault(bold_key, set()).add(feature_key.cell)

        orphan_bold_keys = cell_keys_by_bold_key.keys() - bold_metas.keys()
        if orphan_bold_keys:
            bold_key = next(iter(orphan_bold_keys))
            loc = _format_loc(
                sub=sub_id,
                ses=bold_key.ses,
                run=bold_key.run,
                space=self.bold_space,
            )
            msg = f"No BOLD file found for features at {loc}"
            if len(orphan_bold_keys) > 1:
                msg += f" ({len(orphan_bold_keys) - 1} other coverage gaps exist)"
            raise FileNotFoundError(msg)

        resolved_feature_bids: dict[FeatureKey, BIDSPath] = {}
        for feature_key, bids in feature_bids.items():
            cell_key = feature_key.cell
            bold_key = BoldKey(cell_key.get("ses"), cell_key.get("run"))
            bold_meta = bold_metas[bold_key]

            if not bold_meta.segments:
                run_cell_keys = cell_keys_by_bold_key[bold_key]
                if len(run_cell_keys) > 1:
                    loc = _format_loc(
                        sub=sub_id,
                        ses=bold_key.ses,
                        run=bold_key.run,
                        space=self.bold_space,
                    )
                    raise ValueError(
                        f"Run is unsegmented but has {len(run_cell_keys)} feature "
                        f"files at {loc} — provide an events.tsv with BIDS key-value "
                        f"entities to segment the run"
                    )
                illegal_keys = cell_key.keys() - {"ses", "run"}
                if illegal_keys:
                    loc = _format_loc(
                        sub=sub_id,
                        ses=bold_key.ses,
                        run=bold_key.run,
                        space=self.bold_space,
                    )
                    raise ValueError(
                        f"Unsegmented run at {loc} has feature filename with "
                        f"unexpected entities {sorted(illegal_keys)} — only ses "
                        f"and run are valid on feature filenames for unsegmented "
                        f"runs. To attach metadata, declare a segment row in "
                        f"events.tsv and add descriptive attributes to "
                        f"events.json Levels."
                    )
                resolved_feature_bids[feature_key] = bids
                continue

            segment_entity = bold_meta.segments[0].entity
            segment_values = {segment.value for segment in bold_meta.segments}

            # Validate the cell carries a known segment value for this run
            segment_value = cell_key.get(segment_entity)
            if segment_value is None:
                loc = _format_loc(
                    sub=sub_id,
                    ses=bold_key.ses,
                    run=bold_key.run,
                    space=self.bold_space,
                )
                raise ValueError(
                    f"Feature filename at {loc} is missing segment entity "
                    f"{segment_entity!r} declared in events"
                )
            if segment_value not in segment_values:
                loc = _format_loc(
                    sub=sub_id,
                    ses=bold_key.ses,
                    run=bold_key.run,
                    space=self.bold_space,
                )
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
                loc = _format_loc(
                    sub=sub_id,
                    ses=bold_key.ses,
                    run=bold_key.run,
                    space=self.bold_space,
                )
                raise ValueError(
                    f"Feature filename at {loc} carries entities "
                    f"{sorted(illegal_keys)} absent from events.json "
                    f"events.json Levels metadata — descriptive attributes "
                    f"must live in events.json, not feature filenames"
                )

            # Merge metadata: filename value takes precedence only if it agrees
            extra_metadata: dict[str, str] = {}
            for entity, value in segment.metadata.items():
                if cell_key.get(entity) is not None and cell_key[entity] != value:
                    loc = _format_loc(
                        sub=sub_id,
                        ses=bold_key.ses,
                        run=bold_key.run,
                        space=self.bold_space,
                    )
                    raise ValueError(
                        f"Feature filename and events.json disagree on {entity!r} "
                        f"at {loc} for {segment_entity}-{segment_value}: "
                        f"filename has {cell_key[entity]!r}, sidecar has {value!r}"
                    )
                if cell_key.get(entity) is None:
                    extra_metadata[entity] = value
            if extra_metadata:
                resolved_cell_key = CellKey(
                    **{**dict(cell_key.items()), **extra_metadata}
                )
                resolved_feature_bids[
                    FeatureKey(cell=resolved_cell_key, feature=feature_key.feature)
                ] = bids
            else:
                resolved_feature_bids[feature_key] = bids

        return resolved_feature_bids

    def _apply_filters(
        self,
        sub_id: str,
        feature_bids: dict[FeatureKey, BIDSPath],
        bold_metas: dict[BoldKey, BoldMeta],
    ) -> tuple[dict[FeatureKey, BIDSPath], dict[BoldKey, BoldMeta]]:
        """Apply bids_filters to feature cells and BOLD runs.

        Filters are applied against CellKey entities for features, and against
        filename entities for BOLD. Same-entity filter values OR-match within a
        group; different entities AND-match across groups. A filter key absent from
        one side is skipped on that side rather than rejecting all rows. A filter
        key absent from both sides raises ValueError (typo diagnostic) before any
        empty-result condition surfaces as a coverage error.
        """
        if not self.bids_filters:
            return feature_bids, bold_metas

        # Group filter values by entity for matching later
        allowed_values_by_entity: dict[str, list[str]] = {}
        for bids_filter in self.bids_filters:
            entity_key, entity_value = bids_filter.split("-", 1)
            allowed_values_by_entity.setdefault(entity_key, []).append(entity_value)

        # Collect entity key schema from both sides for typo detection
        cell_entity_keys = frozenset(
            entity_key
            for feature_key in feature_bids
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
                    f"feature cell or BOLD file for sub={sub_id} — check for a typo"
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
            feature_key: bids
            for feature_key, bids in feature_bids.items()
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
        sub_id: str,
        feature_bids: dict[FeatureKey, BIDSPath],
        bold_metas: dict[BoldKey, BoldMeta],
    ) -> None:
        """Validate bidirectional ses/run coverage between filtered features and BOLD.

        Also checks sub/task invariance across all feature and BOLD files.
        Raises if either side is empty — indicates filters selected nothing.
        """
        if not feature_bids:
            raise FileNotFoundError("No feature files match the given filters")
        if not bold_metas:
            raise FileNotFoundError("No BOLD files match the given filters")

        bold_bids = [meta.bids for meta in bold_metas.values()]
        try:
            validate_entity_invariance(
                list(feature_bids.values()) + bold_bids, ("sub", "task")
            )
        except ValueError as e:
            raise ValueError(f"{e} (subject {sub_id})") from e

        cell_keys = {feature_key.cell for feature_key in feature_bids}
        covered_bold_keys = {
            BoldKey(key.get("ses"), key.get("run")) for key in cell_keys
        }

        bold_without_features = bold_metas.keys() - covered_bold_keys
        if bold_without_features:
            bold_key = next(iter(bold_without_features))
            loc = _format_loc(
                sub=sub_id,
                ses=bold_key.ses,
                run=bold_key.run,
                space=self.bold_space,
            )
            msg = f"No feature files found for BOLD at {loc}"
            if len(bold_without_features) > 1:
                msg += f" ({len(bold_without_features) - 1} other coverage gaps exist)"
            raise FileNotFoundError(msg)

        features_without_bold = covered_bold_keys - bold_metas.keys()
        if features_without_bold:
            bold_key = next(iter(features_without_bold))
            loc = _format_loc(
                sub=sub_id,
                ses=bold_key.ses,
                run=bold_key.run,
                space=self.bold_space,
            )
            msg = f"No BOLD file found for features at {loc}"
            if len(features_without_bold) > 1:
                msg += f" ({len(features_without_bold) - 1} other coverage gaps exist)"
            raise FileNotFoundError(msg)

    def _build_xy(
        self,
        sub_id: str,
        feature_bids: dict[FeatureKey, BIDSPath],
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
                loc = _format_loc(
                    sub=sub_id,
                    ses=bold_key.ses,
                    run=bold_key.run,
                    space=self.bold_space,
                )
                raise ValueError(
                    f"events.tsv declares segments extending to TR {expected} "
                    f"but BOLD at {loc} has only {actual} TRs"
                )

        # None sorts before any value; empty string is a stable tiebreaker for ses/run
        def _sort_key(k: CellKey) -> tuple:
            ses = k.get("ses")
            run = k.get("run")
            rest = sorted(val for key, val in k.items() if key not in ("ses", "run"))
            return (ses is not None, ses or "", run is not None, run or "", *rest)

        cell_keys = sorted(
            {feature_key.cell for feature_key in feature_bids}, key=_sort_key
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
                df = read_feature(feature_bids[FeatureKey(cell_key, feature_name)].path)
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
