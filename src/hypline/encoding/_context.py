from __future__ import annotations

import reprlib
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from sklearn.model_selection import BaseCrossValidator
    from sklearn.pipeline import Pipeline

from hypline.bids import BIDSPath
from hypline.bold import BOLD_EXTENSIONS, BoldMeta, load_bold_meta
from hypline.downsample import downsample
from hypline.events import merge_filename_and_sidecar, segment_tr_slice
from hypline.io import (
    read_confound,
    read_confound_metadata,
    read_feature,
    read_feature_metadata,
    stack_array_column,
)
from hypline.layout import BIDSLayout

from ._artifact import XRecipe
from ._schema import (
    BoldKey,
    CellDelayer,
    CellKey,
    RegressorKey,
    RegressorMeta,
    XData,
)

_SOLVER_N_ITER = 100
_SOLVER_DIAGONALIZE_METHOD = "svd"

# Reserved col_slices key for the single confound band. Feature/confound entries
# match BIDS_ENTITY_VALUE_RE (alphanumeric only), so the surrounding underscores
# make this key uncollidable with any feature name.
_CONFOUND_BAND = "__confounds__"


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


def _require_uniform_schema(regressor_bids: dict[RegressorKey, BIDSPath]) -> None:
    """Assert every discovered regressor file shares one cell-entity schema.

    A non-uniform schema means cells can't line up on X's row axis. The two
    conflicting paths name the offending files, so no role label is needed.
    """
    schema: frozenset[str] | None = None
    schema_path: Path | None = None
    for regressor_key, bids in regressor_bids.items():
        file_schema = regressor_key.cell.keys()
        if schema is None:
            schema, schema_path = file_schema, bids.path
        elif file_schema != schema:
            raise ValueError(
                f"Inconsistent schemas across regressor files:"
                f"\n  {schema_path}\n  {bids.path}"
            )


def _require_consistent_metadata(
    regressor_bids: dict[RegressorKey, BIDSPath],
    read_meta: Callable[[Path], dict],
    label: str,
) -> None:
    """Assert all files of one entry carry identical generator metadata.

    Keys prefixed with '_' are exempt — reserved for per-file metadata. `read_meta`
    parameterizes the data source (`read_feature_metadata`/`read_confound_metadata`),
    both of which return a plain dict, so the comparison stays source-agnostic.
    """
    per_entry_meta: dict[str, tuple[dict, Path]] = {}
    for regressor_key, bids in regressor_bids.items():
        meta = {
            key: val
            for key, val in read_meta(bids.path).items()
            if not key.startswith("_")
        }
        entry = regressor_key.entry
        if entry not in per_entry_meta:
            per_entry_meta[entry] = (meta, bids.path)
        elif per_entry_meta[entry][0] != meta:
            ref_meta, ref_path = per_entry_meta[entry]
            diff = "\n".join(f"    {line}" for line in _diff_meta(ref_meta, meta))
            raise ValueError(
                f"Inconsistent metadata for {label}={entry}:\n"
                f"  {ref_path}\n  {bids.path}\n  differing keys:\n{diff}"
            )


def _require_full_coverage(
    regressor_bids: dict[RegressorKey, BIDSPath],
    entries: dict[str, tuple[str, str | None]],
    sub_id: str,
    label: str,
) -> None:
    """Assert every entry is present at every discovered cell.

    A missing entry at any cell would silently produce an incomplete X, so it
    raises instead.
    """
    expected = {
        RegressorKey(cell_key, entry)
        for cell_key in {key.cell for key in regressor_bids}
        for entry in entries
    }
    missing = expected - regressor_bids.keys()
    if missing:
        regressor_key = next(iter(missing))
        loc = _format_loc(sub=sub_id, **dict(regressor_key.cell.items()))
        msg = f"Missing {label}={regressor_key.entry} at {loc}"
        if len(missing) > 1:
            msg += f" ({len(missing) - 1} other coverage gaps exist)"
        raise FileNotFoundError(msg)


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


def _align_y(
    bold_metas: dict[BoldKey, BoldMeta], row_slices: dict[CellKey, slice]
) -> np.ndarray:
    """Slice each cell's BOLD response onto X's row geometry.

    For each cell in `row_slices`, recompute `onset_tr` from *this*
    `bold_metas`' TR and segment, then take `arr[onset : onset + n_trs]` where
    `n_trs` is the row-slice length. Onset is recomputed (not reused from X's
    build) so a cross-subject Y, whose TR may differ from X's source, slices the
    correct rows: a segment at 40 s is onset_tr 20 at TR 2.0 but 40 at TR 1.0.

    Raises if a cell's run is absent from `bold_metas`, or if the recomputed
    `n_trs` disagrees with the row-slice length (TR/duration drift between the
    subject that built X and this one).
    """
    Y_parts: list[np.ndarray] = []
    arrays: dict[BoldKey, np.ndarray] = {}  # cache per run; segments share a file
    for cell_key, sl in row_slices.items():
        bold_key = cell_key.to_bold_key()
        if bold_key not in bold_metas:
            raise ValueError(f"No BOLD run for cell {cell_key} (bold_key {bold_key})")
        bold_meta = bold_metas[bold_key]
        if bold_key not in arrays:
            arrays[bold_key] = _load_bold_array(bold_meta.bids.path)
        bold_data = arrays[bold_key]

        if not bold_meta.segments:
            onset_tr, n_trs = 0, bold_data.shape[0]
        else:
            segment_value = cell_key[bold_meta.segments[0].entity]
            seg = next(s for s in bold_meta.segments if s.value == segment_value)
            tr_slice = segment_tr_slice(seg, bold_meta.repetition_time)
            onset_tr, n_trs = tr_slice.start, tr_slice.stop - tr_slice.start

        # drift guard: fires cross-subject when this run's TR/duration makes the
        # recomputed span diverge from X's row geometry; inert in same-subject train
        expected = sl.stop - sl.start
        if n_trs != expected:
            raise ValueError(
                f"Cell {cell_key} spans {expected} row(s) in X but BOLD here "
                f"yields {n_trs} TR(s) — TR or segment-duration drift between "
                f"the subject that built X and this one"
            )

        # bounds check: a too-short array would silently truncate Y (numpy slice)
        if onset_tr + n_trs > bold_data.shape[0]:
            raise ValueError(
                f"Segment for cell {cell_key} extends to TR {onset_tr + n_trs} "
                f"but BOLD at {bold_meta.bids.path.name} has only "
                f"{bold_data.shape[0]} TRs"
            )

        Y_parts.append(bold_data[onset_tr : onset_tr + n_trs])

    return np.concatenate(Y_parts)


def _build_pipeline(
    *,
    col_slices: dict[str, slice],
    cell_lengths: list[int],
    delays: list[int],
    alphas: list[float],
    cv: BaseCrossValidator,
) -> Pipeline:
    """Assemble an unfitted banded-ridge pipeline, one kernel band per regressor band.

    Each band in `col_slices` (feature names plus the reserved confound band) gets
    its own sub-pipeline:
    `StandardScaler -> CellDelayer -> Kernelizer(linear)`. Bands are bundled via
    `ColumnKernelizer` (one precomputed kernel each) and scored by
    `MultipleKernelRidgeCV`. The caller must have set the himalaya backend before
    calling this — estimators bind the backend at construction.

    A single feature still goes through `MultipleKernelRidgeCV` over one band,
    not plain `KernelRidge`, to pin the shape later phases extend.
    """
    from himalaya.kernel_ridge import (
        ColumnKernelizer,
        Kernelizer,
        MultipleKernelRidgeCV,
    )
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.preprocessing import StandardScaler

    transformers = [
        (
            band_name,
            make_pipeline(
                StandardScaler(),
                CellDelayer(delays=delays, cell_lengths=cell_lengths),
                Kernelizer(kernel="linear"),
            ),
            col_slice,
        )
        for band_name, col_slice in col_slices.items()
    ]
    column_kernelizer = ColumnKernelizer(transformers)

    model = MultipleKernelRidgeCV(
        kernels="precomputed",
        solver_params=dict(
            alphas=np.asarray(alphas),
            n_iter=_SOLVER_N_ITER,
            diagonalize_method=_SOLVER_DIAGONALIZE_METHOD,
            progress_bar=False,
        ),
        cv=cv,
    )

    return Pipeline([("kernelizer", column_kernelizer), ("model", model)])


class _EncodingContext:
    """Shared recipe-derived discovery/build path for train and predict.

    Not instantiated directly. Both `EncodingTrainer` and `EncodingPredictor`
    inherit these methods; they differ only in how the recipe attrs are
    populated (validated constructor args vs. a loaded artifact).
    """

    # Attribute contract: both subclasses set `self._recipe` (trainer builds it
    # from validated args, predictor takes `artifact.recipe`), and the shared
    # discovery/build path reads everything off it. `_layout` is caller-supplied
    # (bids_root), deliberately not part of X identity.
    _layout: BIDSLayout
    _recipe: XRecipe

    def _discover_features(self, sub_id: str) -> dict[RegressorKey, BIDSPath]:
        """Discover and validate feature file paths for a subject.

        Scans the features directory by BIDS filename alone — no feature data is read.
        Features are dyad-keyed, so `sub_id` is resolved to its dyad via `dyad_of`;
        only dyad and feature are used as structural filters; bids_filters are applied
        post-enrichment in _apply_filters. Duplicate files for the same (cell, feature)
        pair raise immediately.

        Returns a flat dict mapping each (cell, feature) pair to its BIDSPath.
        Every cell is guaranteed to have all requested features — a missing feature
        at any cell raises rather than silently producing an incomplete matrix.
        All files are guaranteed to share the same cell schema (entity key set).
        """
        # Features are dyad-keyed; resolve this subject's dyad via participants.tsv
        dyad_id = self._layout.dyad_of(sub_id)
        feature_bids: dict[RegressorKey, BIDSPath] = {}
        for feature_name, (kind, desc) in self._recipe.features.items():
            feature_files = self._layout.find.features(
                dyad=dyad_id,
                kind=kind,
                desc=desc,
                bids_filters=self._recipe.task_filters,
            )

            for bids in feature_files:
                cell_key = CellKey(
                    **{
                        key: val
                        for key, val in bids.entities.items()
                        if key not in CellKey.EXCLUDE
                    }
                )
                feature_key = RegressorKey(cell=cell_key, entry=feature_name)
                if feature_key in feature_bids:
                    loc = _format_loc(sub=sub_id, **dict(cell_key.items()))
                    raise ValueError(
                        f"Multiple feature files for feat={feature_name}, {loc}:\n"
                        f"  {feature_bids[feature_key].path}\n  {bids.path}"
                    )
                feature_bids[feature_key] = bids

        _require_uniform_schema(feature_bids)
        _require_consistent_metadata(feature_bids, read_feature_metadata, "feat")
        _require_full_coverage(feature_bids, self._recipe.features, sub_id, "feat")
        return feature_bids

    def _discover_confounds(self, sub_id: str) -> dict[RegressorKey, BIDSPath]:
        """Discover and validate confound file paths for a subject.

        Mirrors `_discover_features` so confound cells resolve to the same cell
        schema and full per-cell coverage — they must line up on X's row axis with
        the feature cells. The differences from features: files are read via
        `find.confounds`/`read_confound_metadata`, and `RegressorKey.entry` holds
        the confound name (the recipe's `confounds` key). The per-entry key is
        incidental — confounds all collapse into one ridge band in `_build_x`, so it
        only labels coverage and dedup here.

        Returns an empty dict when no confounds are configured.
        """
        if not self._recipe.confounds:
            return {}

        # Confounds are dyad-keyed; resolve this subject's dyad via participants.tsv
        dyad_id = self._layout.dyad_of(sub_id)
        confound_bids: dict[RegressorKey, BIDSPath] = {}
        for entry, (kind, desc) in self._recipe.confounds.items():
            confound_files = self._layout.find.confounds(
                dyad=dyad_id,
                kind=kind,
                desc=desc,
                bids_filters=self._recipe.task_filters,
            )

            for bids in confound_files:
                cell_key = CellKey(
                    **{
                        key: val
                        for key, val in bids.entities.items()
                        if key not in CellKey.EXCLUDE
                    }
                )
                confound_key = RegressorKey(cell=cell_key, entry=entry)
                if confound_key in confound_bids:
                    loc = _format_loc(sub=sub_id, **dict(cell_key.items()))
                    raise ValueError(
                        f"Multiple confound files for conf={entry}, {loc}:\n"
                        f"  {confound_bids[confound_key].path}\n  {bids.path}"
                    )
                confound_bids[confound_key] = bids

        _require_uniform_schema(confound_bids)
        _require_consistent_metadata(confound_bids, read_confound_metadata, "conf")
        _require_full_coverage(confound_bids, self._recipe.confounds, sub_id, "conf")
        return confound_bids

    def _discover_bold(self, sub_id: str) -> dict[BoldKey, BoldMeta]:
        """Discover BOLD files and load their metadata for a subject.

        Scans the BOLD directory by filename without loading image arrays. TR is
        read from the sidecar JSON, falling back to the image header. Segmentation
        is inferred from the colocated events TSV when present. All runs are guaranteed
        to share the same TR, BOLD-level entity invariants, and segment entity (or all
        unsegmented).
        """
        bold_ext = BOLD_EXTENSIONS[type(self._recipe.bold_space)]
        # Encoding consumes hypline postprocessing outputs, not fmriprep's raw preproc
        bold_files = self._layout.find.hypline(
            sub=sub_id,
            suffix="bold",
            ext=bold_ext,
            bids_filters=[
                f"space-{self._recipe.bold_space}",
                f"desc-{self._recipe.bold_desc}",
                *self._recipe.task_filters,
            ],
        )

        bold_metas: dict[BoldKey, BoldMeta] = {}
        for bids in bold_files:
            bold_key = BoldKey(
                ses=bids.entities.get("ses"),
                task=bids.entities["task"],
                run=bids.entities.get("run"),
            )
            if bold_key in bold_metas:
                loc = _format_loc(
                    sub=sub_id,
                    ses=bold_key.ses,
                    task=bold_key.task,
                    run=bold_key.run,
                )
                raise ValueError(
                    f"Duplicate BOLD file at {loc}:\n"
                    f"  {bold_metas[bold_key].bids.path}\n  {bids.path}"
                )
            try:
                bold_metas[bold_key] = load_bold_meta(self._layout, bids)
            except ValueError as e:
                loc = _format_loc(
                    sub=sub_id,
                    ses=bold_key.ses,
                    task=bold_key.task,
                    run=bold_key.run,
                )
                raise ValueError(f"Failed to load BOLD at {loc}: {e}") from e

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
        regressor_bids: dict[RegressorKey, BIDSPath],
        bold_metas: dict[BoldKey, BoldMeta],
    ) -> dict[RegressorKey, BIDSPath]:
        """Validate and resolve regressor CellKeys against BOLD segment metadata.

        For each cell, locates the matching BOLD run and segment, then merges
        segment.metadata into the CellKey. Filename entities beyond ses, run, and the
        segment entity are rejected unless they echo a metadata key from events.json
        — descriptive metadata must live in events.json, not filenames.

        Invariant: _discover_bold guarantees all segments share the same metadata
        schema across runs, so resolved cells always end up with a uniform key set.
        """

        def _loc(bold_key: BoldKey) -> str:
            return _format_loc(
                sub=sub_id,
                ses=bold_key.ses,
                task=bold_key.task,
                run=bold_key.run,
                space=self._recipe.bold_space,
            )

        cell_keys_by_bold_key: dict[BoldKey, set[CellKey]] = {}
        for regressor_key in regressor_bids:
            bold_key = regressor_key.cell.to_bold_key()
            cell_keys_by_bold_key.setdefault(bold_key, set()).add(regressor_key.cell)

        orphan_bold_keys = cell_keys_by_bold_key.keys() - bold_metas.keys()
        if orphan_bold_keys:
            bold_key = next(iter(orphan_bold_keys))
            msg = f"No BOLD file found for regressor at {_loc(bold_key)}"
            if len(orphan_bold_keys) > 1:
                msg += f" ({len(orphan_bold_keys) - 1} other coverage gaps exist)"
            raise FileNotFoundError(msg)

        resolved_regressor_bids: dict[RegressorKey, BIDSPath] = {}
        for regressor_key, bids in regressor_bids.items():
            cell_key = regressor_key.cell
            bold_key = cell_key.to_bold_key()
            bold_meta = bold_metas[bold_key]

            if not bold_meta.segments:
                run_cell_keys = cell_keys_by_bold_key[bold_key]
                if len(run_cell_keys) > 1:
                    raise ValueError(
                        f"Run is unsegmented but has {len(run_cell_keys)} regressor "
                        f"files at {_loc(bold_key)} — provide an events.tsv with "
                        f"BIDS key-value entities to segment the run"
                    )
                illegal_keys = cell_key.keys() - {"ses", "task", "run"}
                if illegal_keys:
                    raise ValueError(
                        f"Unsegmented run at {_loc(bold_key)} has a filename "
                        f"with unexpected entities {sorted(illegal_keys)} — only "
                        f"ses, task, and run are valid on filenames for "
                        f"unsegmented runs. To attach metadata, declare a segment "
                        f"row in events.tsv and add descriptive attributes to "
                        f"events.json Levels."
                    )
                resolved_regressor_bids[regressor_key] = bids
                continue

            segment_entity = bold_meta.segments[0].entity
            segment_values = {segment.value for segment in bold_meta.segments}

            # Validate the cell carries a known segment value for this run
            segment_value = cell_key.get(segment_entity)
            if segment_value is None:
                raise ValueError(
                    f"Filename at {_loc(bold_key)} is missing segment "
                    f"entity {segment_entity!r} declared in events"
                )
            if segment_value not in segment_values:
                raise ValueError(
                    f"Segment value {segment_entity}-{segment_value} at "
                    f"{_loc(bold_key)} not found in events — valid values: "
                    f"{sorted(segment_values)}"
                )

            segment = next(
                seg for seg in bold_meta.segments if seg.value == segment_value
            )

            filename_entities = dict(cell_key.items())
            try:
                merged = merge_filename_and_sidecar(
                    filename_entities=filename_entities,
                    sidecar_metadata=segment.metadata,
                    structural_keys=frozenset({"ses", "task", "run", segment_entity}),
                )
            except ValueError as err:
                raise ValueError(f"{err} (at {_loc(bold_key)})") from None
            if merged == filename_entities:
                resolved_regressor_bids[regressor_key] = bids
            else:
                resolved_regressor_bids[
                    RegressorKey(cell=CellKey(**merged), entry=regressor_key.entry)
                ] = bids

        return resolved_regressor_bids

    def _enrich_regressor_metas(
        self,
        regressor_bids: dict[RegressorKey, BIDSPath],
        bold_metas: dict[BoldKey, BoldMeta],
    ) -> dict[RegressorKey, RegressorMeta]:
        """Enrich each regressor path into a `RegressorMeta` with its TR-grid placement.

        Role-neutral: features and confounds both flow through here, since placement
        depends only on the cell's BOLD timeline, not the regressor's role. The
        crossover between cells and the BOLD timeline happens here, once: derive
        `(onset_tr, n_trs)` from the segment TR-slice for segmented runs, or from the
        run's header `n_trs` for unsegmented runs, and stamp `repetition_time`.
        `_build_x` then reads placement off the metas and never touches `bold_metas`.

        No BOLD voxel data is read — placement comes from `bold_metas` alone.
        The header `n_trs` carried by each meta is trusted; `_align_y` reconciles
        it against the actual array.
        """
        cells_by_bold_key: dict[BoldKey, set[CellKey]] = {}
        for regressor_key in regressor_bids:
            bold_key = regressor_key.cell.to_bold_key()
            cells_by_bold_key.setdefault(bold_key, set()).add(regressor_key.cell)

        placement_by_cell: dict[CellKey, tuple[int, int, float]] = {}
        for bold_key, cells in cells_by_bold_key.items():
            bold_meta = bold_metas[bold_key]
            if not bold_meta.segments:
                # Unsegmented: the whole run is one cell spanning the full timeline
                for cell_key in cells:
                    placement_by_cell[cell_key] = (
                        0,
                        bold_meta.n_trs,
                        bold_meta.repetition_time,
                    )
            else:
                for cell_key in cells:
                    segment = next(
                        seg
                        for seg in bold_meta.segments
                        if seg.value == cell_key[seg.entity]
                    )
                    tr_slice = segment_tr_slice(segment, bold_meta.repetition_time)
                    placement_by_cell[cell_key] = (
                        tr_slice.start,  # onset_tr
                        tr_slice.stop - tr_slice.start,  # n_trs
                        bold_meta.repetition_time,
                    )

        regressor_metas: dict[RegressorKey, RegressorMeta] = {}
        for regressor_key, bids in regressor_bids.items():
            onset_tr, n_trs, repetition_time = placement_by_cell[regressor_key.cell]
            regressor_metas[regressor_key] = RegressorMeta(
                bids=bids,
                n_trs=n_trs,
                onset_tr=onset_tr,
                repetition_time=repetition_time,
            )
        return regressor_metas

    def _read_feature_array(self, meta: RegressorMeta, n_trs: int) -> np.ndarray:
        """Read one feature file and downsample it onto the cell's TR grid."""
        df = read_feature(meta.bids.path)
        # Drop untimed rows — downsample needs TR alignment
        df = df.filter(df.get_column("start_time").is_not_null())
        return downsample(
            stack_array_column(df.get_column("feature")),
            start_times=df.get_column("start_time").to_numpy(),
            n_trs=n_trs,
            repetition_time=meta.repetition_time,
            method=self._recipe.downsample,
        )

    def _read_confound_array(
        self, meta: RegressorMeta, n_trs: int, entry: str, cell_key: CellKey
    ) -> np.ndarray:
        """Read one confound file, already at TR level, asserting its row count.

        Unlike features there is no downsample step: confounds are written TR-level
        and per-segment, so the file must already span the cell's `n_trs`. The check
        is explicit rather than left to `downsample`'s pass-through, which would
        silently bin instead of raise on a mismatched grid.
        """
        arr = stack_array_column(read_confound(meta.bids.path).get_column("confound"))
        if arr.shape[0] != n_trs:
            raise ValueError(
                f"Confound conf={entry} at cell {cell_key} has {arr.shape[0]} "
                f"TR row(s) but the cell spans {n_trs} — confounds must be saved "
                f"at the BOLD TR grid for this segment"
            )
        return arr

    def _build_x(self, regressor_metas: dict[RegressorKey, RegressorMeta]) -> XData:
        """Assemble the X regressor matrix and its row/column geometry, no target.

        Reads `n_trs`/`repetition_time` off each cell's `RegressorMeta` — the
        crossover with the BOLD timeline already happened in
        `_enrich_regressor_metas`, so X no longer touches `bold_metas` and no BOLD
        data enters X.

        `regressor_metas` is the merged feature+confound dict; the recipe's
        `features`/`confounds` maps decide each entry's role. Cells are sorted
        deterministically so row positions are stable across runs. Column layout is
        derived from the first cell and assumed invariant — all cells must yield the
        same regressor dimensionality.

        Features each get their own ridge band; all confounds collapse into a single
        trailing band keyed `_CONFOUND_BAND`, regardless of how many were configured.
        """

        # None sorts before any value; empty string is a stable tiebreaker for ses/run
        def _sort_key(k: CellKey) -> tuple:
            ses = k.get("ses")
            task = k["task"]
            run = k.get("run")
            rest = sorted(
                val for key, val in k.items() if key not in ("ses", "run", "task")
            )
            return (ses is not None, ses or "", task, run is not None, run or "", *rest)

        cell_keys = sorted(
            {regressor_key.cell for regressor_key in regressor_metas}, key=_sort_key
        )

        X_parts: list[np.ndarray] = []
        row_slices: dict[CellKey, slice] = {}
        col_slices: dict[str, slice] = {}
        row_offset = 0
        col_offset = 0
        col_slices_initialized = False

        for cell_key in cell_keys:
            # Geometry is per-cell, shared across this cell's regressor metas
            cell_meta = regressor_metas[
                RegressorKey(cell_key, next(iter(self._recipe.features)))
            ]
            n_trs = cell_meta.n_trs
            row_slices[cell_key] = slice(row_offset, row_offset + n_trs)
            row_offset += n_trs

            # Construct X for the given cell
            feature_arrays = [
                self._read_feature_array(
                    regressor_metas[RegressorKey(cell_key, name)], n_trs
                )
                for name in self._recipe.features
            ]
            # One band for all confounds: hstack each entry's already-TR-level vector
            confound_arrays = [
                self._read_confound_array(
                    regressor_metas[RegressorKey(cell_key, entry)],
                    n_trs,
                    entry,
                    cell_key,
                )
                for entry in self._recipe.confounds
            ]

            if not col_slices_initialized:
                for feature_name, arr in zip(self._recipe.features, feature_arrays):
                    n_cols = arr.shape[1]
                    col_slices[feature_name] = slice(col_offset, col_offset + n_cols)
                    col_offset += n_cols
                if confound_arrays:
                    n_cols = sum(arr.shape[1] for arr in confound_arrays)
                    col_slices[_CONFOUND_BAND] = slice(col_offset, col_offset + n_cols)
                    col_offset += n_cols
                col_slices_initialized = True  # col slices are invariant across cells
            X_parts.append(np.hstack([*feature_arrays, *confound_arrays]))

        X = np.concatenate(X_parts, axis=0)
        return XData(X=X, row_slices=row_slices, col_slices=col_slices)
