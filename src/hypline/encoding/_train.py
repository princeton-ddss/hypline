from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal, get_args

import numpy as np

if TYPE_CHECKING:
    from sklearn.model_selection import BaseCrossValidator

from hypline.bids import (
    BIDS_ENTITY_VALUE_RE,
    BIDSPath,
    normalize_bids_filters,
    parse_kind_desc,
)
from hypline.bold import BoldMeta, parse_bold_space
from hypline.enums import Device
from hypline.layout import BIDSLayout

from ._artifact import EncodingArtifact, FittedModel, FoldSpec, XRecipe
from ._context import _align_y, _build_pipeline, _EncodingContext, _format_loc
from ._schema import (
    BoldKey,
    CellKey,
    EncodingConfig,
    FeatureDownsampleMethod,
    FeatureKey,
    FeatureMeta,
    TrainingData,
)


def _group_cells_by(cells: set[CellKey], entity: str) -> dict[str, set[CellKey]]:
    """Group cells by their value for `entity`.

    Raises if `entity` is absent from any cell. Because `_resolve_cell_keys`
    yields a uniform schema (a key is on every cell or none), uniform absence
    means the dataset has no such axis — a `fold_by` config error, not a
    data-shape edge case.
    """
    groups: dict[str, set[CellKey]] = {}
    for cell in cells:
        if entity not in cell.keys():
            raise ValueError(
                f"entity {entity!r} not present on cell {cell} — this dataset "
                f"has no {entity!r} axis to group on"
            )
        groups.setdefault(cell[entity], set()).add(cell)
    return groups


def _partition_groups(
    groups: dict[str, set[CellKey]], n_folds: int | Literal["loo"]
) -> list[set[CellKey]]:
    """Map a fold count to per-fold held-out cell sets over whole groups.

    Sorts the group values, then splits them into K held-out buckets — `int`
    folds into K contiguous chunks, `"loo"` into one group per bucket. Cells are
    never split across folds (the split unit is the group).

    The sort makes bucketing reproducible within a run (predict never recomputes
    folds, so no seed is needed).
    """
    n_groups = len(groups)
    ordered = sorted(groups)

    if n_folds == "loo":
        if n_groups < 2:
            raise ValueError(f"n_folds='loo' needs >= 2 groups to fold; got {n_groups}")
        return [groups[value] for value in ordered]

    if n_folds > n_groups:
        raise ValueError(
            f"n_folds={n_folds} exceeds the {n_groups} group(s) found for "
            f"the fold_by entity"
        )
    # Contiguous chunks; earlier buckets absorb the remainder
    base, extra = divmod(n_groups, n_folds)
    buckets: list[set[CellKey]] = []
    start = 0
    for i in range(n_folds):
        size = base + (1 if i < extra else 0)
        bucket: set[CellKey] = set()
        for value in ordered[start : start + size]:
            bucket |= groups[value]
        buckets.append(bucket)
        start += size
    return buckets


def _select_rows(
    data: TrainingData, cells: set[CellKey]
) -> tuple[np.ndarray, np.ndarray, list[CellKey]]:
    """Slice `data` down to `cells`, preserving `_build_x` row order.

    Iterates `data.row_slices` directly — a dict already built in `_sort_key`
    order by `_build_x`, so surviving cells keep that order with no re-sort.
    The ordered surviving cells are returned so the inner-CV selector and
    `CellDelayer` can recover per-cell row counts from `data.row_slices` in the
    same row order.
    """
    X_parts: list[np.ndarray] = []
    Y_parts: list[np.ndarray] = []
    ordered_cells: list[CellKey] = []
    for cell, sl in data.row_slices.items():
        if cell not in cells:
            continue
        X_parts.append(data.X[sl])
        Y_parts.append(data.Y[sl])
        ordered_cells.append(cell)
    return np.concatenate(X_parts), np.concatenate(Y_parts), ordered_cells


def _inner_cv(
    *,
    ordered_cells: list[CellKey],
    cell_lengths: list[int],
    segment_entity: str | None,
    fold: FoldSpec | None,
) -> BaseCrossValidator:
    """Build the hyperparameter-search splitter confined to one model's train set.

    Applies the 3-step inner-unit rule over this model's train cells:

    1. `fold.by` itself, if the train set holds >=2 distinct values — symmetric
       with outer folding, valid for structural and descriptive `fold_by` alike.
    2. Descend the structural chain `[ses, task, run, segment_entity]`
       coarsest-first, skipping only `fold.by` (already failed step 1); first
       entity with >=2 distinct train values wins. The full chain is walked —
       entities coarser than a structural `fold_by` are NOT skipped, because BIDS
       labels are not strictly nested (a `run` value recurs across sessions), so a
       coarser entity can legitimately vary while `fold.by` is constant.
       Descriptive keys (e.g. `cond`) are excluded here: a descriptive value can
       straddle a run boundary and leak. They re-enter only via step 1. When only
       `segment_entity` varies, this yields leave-one-trial-out, which is leaky
       under FIR delays — accepted as the honest expression of "this fold only
       varies at the trial level"; the leak is a property of the data (no coarser
       structure), not something the CV strategy should mask.
    3. No eligible entity (train set collapsed to a single cell) → contiguous
       `KFold(n_splits=2, shuffle=False)`. No seed — contiguous halving keeps
       FIR-correlated adjacent TRs on the same side except at one boundary.

    Steps 1–2 return a `PredefinedSplit` over whole cells: each cell's chosen
    entity value (mapped to an int) is repeated by the cell's row count, so a
    group id is constant within a cell and leave-one-value-out never splits a
    cell. `segment_entity` is `None` when runs are unsegmented (it drops out of
    the chain naturally). `fold` is `None` for unfolded models — step 1 is then
    skipped and the rule starts at step 2.
    """
    from sklearn.model_selection import KFold, PredefinedSplit

    def _split_on(entity: str):
        # absent entity → None; needs >=2 distinct non-None values to split
        values = [cell.get(entity) for cell in ordered_cells]
        if len({v for v in values if v is not None}) < 2:
            return None
        value_to_id = {v: i for i, v in enumerate(dict.fromkeys(values))}
        per_row_group_ids = np.repeat([value_to_id[v] for v in values], cell_lengths)
        return PredefinedSplit(per_row_group_ids)

    fold_by = fold.by if fold is not None else None

    # step 1: fold_by as the inner unit
    if fold_by is not None:
        cv = _split_on(fold_by)
        if cv is not None:
            return cv

    # step 2: descend the structural chain, skipping only fold_by
    chain = ["ses", "task", "run"]
    if segment_entity is not None:
        chain.append(segment_entity)
    for entity in chain:
        if entity == fold_by:
            continue
        cv = _split_on(entity)
        if cv is not None:
            return cv

    # step 3: single-cell train fold → contiguous 2-way halves
    if sum(cell_lengths) < 2:
        raise ValueError(
            "Inner CV cannot form a 2-way split: a train fold collapsed to a "
            f"single cell with {sum(cell_lengths)} row(s). Cells: "
            f"{ordered_cells}"
        )
    return KFold(n_splits=2, shuffle=False)


class EncodingTrainer(_EncodingContext):
    """Recipe -> fit -> artifact. Owns validation, filtering, and Y-build."""

    def __init__(
        self,
        *,
        config: EncodingConfig,
        bids_root: str | Path,
        features: list[str],
        tasks: list[str],
        bold_space: str,
        bold_desc: str = "denoised",
        downsample: FeatureDownsampleMethod = "mean",
        bids_filters: list[str] | None = None,
        fold_by: str | None = None,
        n_folds: int | Literal["loo"] | None = None,
    ):
        import torch

        if config.device is Device.CUDA and not torch.cuda.is_available():
            raise RuntimeError("CUDA is requested but not available")

        if not features:
            raise ValueError("features must be a non-empty list")
        parsed = [parse_kind_desc(entry) for entry in features]
        kinds = [kind for kind, _ in parsed]
        if len(kinds) != len(set(kinds)):
            dupes = sorted({k for k in kinds if kinds.count(k) > 1})
            raise ValueError(
                f"Duplicate feature kind(s) {dupes} in features"
                " — each kind may appear once"
            )
        # Iteration order fixes X column layout
        feature_map = dict(zip(features, parsed))

        if not tasks:
            raise ValueError("tasks must be a non-empty list")
        if len(tasks) != len(set(tasks)):
            dupes = sorted({t for t in tasks if tasks.count(t) > 1})
            raise ValueError(f"Duplicate entries in tasks: {dupes}")

        parsed_bold_space = parse_bold_space(bold_space)

        if not BIDS_ENTITY_VALUE_RE.match(bold_desc):
            raise ValueError(f"Invalid bold_desc: {bold_desc!r}")

        if downsample not in get_args(FeatureDownsampleMethod):
            raise ValueError(
                f"downsample must be one of {get_args(FeatureDownsampleMethod)};"
                f" got {downsample!r}"
            )

        normalized_bids_filters = normalize_bids_filters(
            bids_filters, reserved={"sub", "task", "space", "feat", "desc"}
        )

        # fold_by/n_folds are paired: both define one fold group or neither does.
        # Only subject-independent checks run here; group-count and entity-presence
        # validation is data-dependent and deferred to train / _partition_groups.
        if (fold_by is None) != (n_folds is None):
            raise ValueError(
                "fold_by and n_folds must be set together or both left unset; "
                f"got fold_by={fold_by!r}, n_folds={n_folds!r}"
            )
        fold: FoldSpec | None = None
        if fold_by is not None:
            assert n_folds is not None
            if fold_by in CellKey.EXCLUDE:
                raise ValueError(
                    f"fold_by={fold_by!r} is not a cell axis (it never appears on "
                    f"a cell); valid axes exclude {sorted(CellKey.EXCLUDE)}"
                )
            if n_folds != "loo" and n_folds < 2:
                raise ValueError(
                    f"n_folds must be >= 2 or 'loo'; got {n_folds!r}. A single fold "
                    "is no split — pass n_folds=None for a single model"
                )
            fold = FoldSpec(by=fold_by, n=n_folds)

        self._config = config
        self._layout = BIDSLayout(bids_root)
        self._fold = fold
        # col_slices is filled by train from the assembled TrainingData
        self._recipe = XRecipe(
            features=feature_map,
            tasks=tasks,
            bold_space=parsed_bold_space,
            bold_desc=bold_desc,
            downsample=downsample,
            bids_filters=normalized_bids_filters,
            delays=config.delays,
            alphas=config.alphas,
        )

    def train(self, sub_id: str) -> EncodingArtifact:
        """Fit the encoding model for a subject and return the in-memory artifact.

        Pure compute: discovers, builds X/Y, fits, and returns the
        `EncodingArtifact`. Persistence is the caller's concern — pass the result
        to `write_artifact` to store it, and gate the fit on `skip_existing`
        before calling if a check-before-compute cache is wanted.
        """
        feature_bids = self._discover_features(sub_id)
        bold_metas = self._discover_bold(sub_id)
        feature_bids = self._resolve_cell_keys(sub_id, feature_bids, bold_metas)
        feature_bids, bold_metas = self._apply_filters(sub_id, feature_bids, bold_metas)
        self._validate_coverage(sub_id, feature_bids, bold_metas)
        feature_metas = self._enrich_feature_metas(feature_bids, bold_metas)
        data = self._build_training_data(feature_metas, bold_metas)

        # himalaya binds the backend at estimator construction, not at fit — set it
        # before building the pipeline or fitting silently falls back to CPU
        from himalaya.backend import set_backend

        set_backend("torch_cuda" if self._config.device is Device.CUDA else "torch")

        # segment entity is invariant across bold_metas (validated in _discover_bold);
        # None when runs are unsegmented
        segment_metas = [meta for meta in (bold_metas or {}).values() if meta.segments]
        segment_entity = segment_metas[0].segments[0].entity if segment_metas else None

        def _fit_model(
            X: np.ndarray,
            Y: np.ndarray,
            ordered_cells: list[CellKey],
        ):
            # Each fold needs a fresh pipeline: cell_lengths differ per fold and
            # CellDelayer's boundary masks are built from them at construction.
            # Inner CV is rebuilt per model so its hyperparameter search stays
            # confined to that model's own train cells.
            cell_lengths = data.cell_lengths(ordered_cells)
            cv = _inner_cv(
                ordered_cells=ordered_cells,
                cell_lengths=cell_lengths,
                segment_entity=segment_entity,
                fold=self._fold,
            )
            pipeline = _build_pipeline(
                col_slices=data.col_slices,
                cell_lengths=cell_lengths,
                delays=self._config.delays,
                alphas=self._config.alphas,
                cv=cv,
            )
            # torch backends want float32; float64 doubles memory and can error on CUDA
            pipeline.fit(X.astype(np.float32), Y.astype(np.float32))
            return pipeline

        recipe = replace(self._recipe, col_slices=data.col_slices)

        if self._fold is None:
            # cell order tracks data.row_slices (= _build_x / _sort_key order)
            ordered_cells = list(data.row_slices)
            pipeline = _fit_model(data.X, data.Y, ordered_cells)
            artifact = EncodingArtifact(
                recipe=recipe,
                fold=None,
                models=[
                    FittedModel(pipeline=pipeline, train_cells=set(data.row_slices))
                ],
                universe=None,
            )
        else:
            all_cells = set(data.row_slices)
            groups = _group_cells_by(all_cells, self._fold.by)
            held_out = _partition_groups(groups, self._fold.n)
            models = []
            for held in held_out:
                train_cells = all_cells - held
                X_sub, Y_sub, ordered_cells = _select_rows(data, train_cells)
                pipeline = _fit_model(X_sub, Y_sub, ordered_cells)
                models.append(FittedModel(pipeline=pipeline, train_cells=train_cells))
            artifact = EncodingArtifact(
                recipe=recipe, fold=self._fold, models=models, universe=all_cells
            )

        return artifact

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
        if not self._recipe.bids_filters:
            return feature_bids, bold_metas

        # Group filter values by entity for matching later
        allowed_values_by_entity: dict[str, list[str]] = {}
        for bids_filter in self._recipe.bids_filters:
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
        """Validate bidirectional (ses, task, run) coverage between features and BOLD.

        Raises if either side is empty — indicates filters selected nothing.
        """

        def _loc(bold_key: BoldKey) -> str:
            return _format_loc(
                sub=sub_id,
                ses=bold_key.ses,
                task=bold_key.task,
                run=bold_key.run,
                space=self._recipe.bold_space,
            )

        if not feature_bids:
            raise FileNotFoundError("No feature files match the given filters")
        if not bold_metas:
            raise FileNotFoundError("No BOLD files match the given filters")

        cell_keys = {feature_key.cell for feature_key in feature_bids}
        covered_bold_keys = {key.to_bold_key() for key in cell_keys}

        bold_without_features = bold_metas.keys() - covered_bold_keys
        if bold_without_features:
            bold_key = next(iter(bold_without_features))
            msg = f"No feature files found for BOLD at {_loc(bold_key)}"
            if len(bold_without_features) > 1:
                msg += f" ({len(bold_without_features) - 1} other coverage gaps exist)"
            raise FileNotFoundError(msg)

        features_without_bold = covered_bold_keys - bold_metas.keys()
        if features_without_bold:
            bold_key = next(iter(features_without_bold))
            msg = f"No BOLD file found for features at {_loc(bold_key)}"
            if len(features_without_bold) > 1:
                msg += f" ({len(features_without_bold) - 1} other coverage gaps exist)"
            raise FileNotFoundError(msg)

    def _build_training_data(
        self,
        feature_metas: dict[FeatureKey, FeatureMeta],
        bold_metas: dict[BoldKey, BoldMeta],
    ) -> TrainingData:
        """Bundle X (from `_build_x`) with the aligned BOLD target Y onto its rows.

        The X+Y carrier is train-only — predict stops at X (`_build_x`) and never
        builds Y. Segment coverage vs. actual BOLD array length is checked in
        `_align_y`; a mismatch raises rather than producing a silently truncated Y.
        """
        x_data = self._build_x(feature_metas)
        Y = _align_y(bold_metas, x_data.row_slices)
        return TrainingData(
            X=x_data.X,
            row_slices=x_data.row_slices,
            col_slices=x_data.col_slices,
            Y=Y,
        )
