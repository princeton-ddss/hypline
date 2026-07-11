from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    import xarray as xr
    from sklearn.pipeline import Pipeline

from hypline import __version__
from hypline.bids import parse_filter_groups, validate_bids_entities
from hypline.layout import BIDSLayout

from ._artifact import EncodingArtifact, FittedModel, load_artifact
from ._context import _align_y, _EncodingContext
from ._eval import (
    ROLES,
    _role_masks,
    _score_roles,
)
from ._schema import CellDelayer, CellKey, Prediction, RegressorKey, RegressorMeta


def _assert_same_voxels(*, Y_hat: np.ndarray, Y_true: np.ndarray) -> None:
    """Guard the shared voxel axis that pairing `Y_hat` with `Y_true` requires.

    `Y_hat`'s voxels are the model subject's trained targets; `Y_true`'s are the
    target subject's BOLD. Same-study + same-space puts both on one standard grid
    (MNI volume or fsaverageN surface), so the axes match — a mismatch means a
    cross-space or cross-study pairing. Raise a clear error at the pairing seam
    rather than let downstream scoring fail with an opaque shape error.
    """
    if Y_hat.shape[-1] != Y_true.shape[-1]:
        raise ValueError(
            f"Voxel-axis mismatch: Y_hat has {Y_hat.shape[-1]} voxels but Y_true "
            f"has {Y_true.shape[-1]} — model and target must share a standard grid "
            f"(same study, same bold_space)"
        )


def _rebind_cell_lengths(pipeline: Pipeline, cell_lengths: list[int]) -> None:
    """Overwrite each band's `CellDelayer.cell_lengths` to the predict geometry.

    `CellDelayer` froze the *train* subject's per-cell row counts at fit; its
    `transform` builds FIR boundary masks from them and raises if X's row count
    disagrees. Predict's cells differ, so the frozen lengths are wrong — rebind
    them before predict. Safe because `cell_lengths` only drives boundary masking
    and the row-count check (both functions of the current X), never the fitted
    weights.

    Targets `transformers_` — the fitted transformer clones sklearn made at fit
    (the same tree `_numpyfy_fitted` walks) — not the unfitted `.transformers`
    spec, which predict never touches. `cell_lengths` order must match X's
    `row_slices` order.
    """
    for _, transformer, _ in pipeline.named_steps["kernelizer"].transformers_:
        for _, step in transformer.steps:
            if isinstance(step, CellDelayer):
                step.cell_lengths = cell_lengths


def _select_cells(
    available: set[CellKey],
    artifact: EncodingArtifact,
    model: FittedModel,
    *,
    test_on: list[str] | None = None,
) -> set[CellKey]:
    """Choose which cells `model` predicts on — out-of-sample by default, or `test_on`.

    `available` is the source subject's *full* discovered cell set. Recipe filters
    are deliberately not replayed: the stored `train`/`universe` sets are trusted,
    and `test_on` is allowed to name cells outside the train corpus, so narrowing
    `available` up front would wrongly exclude valid targets.

    `test_on` is a list of `<entity>-<value>` refs (the codebase-wide filter shape,
    like `bids_filters`): same-entity values OR-match within a group, different
    entities AND-match across groups (via `parse_filter_groups`). A named entity
    absent from the cell schema is a typo — raise, mirroring `_apply_filters`.

    The mode (read off the code below) is the easy part; the subtle bits:

    - `test_on` overrides everything and only *warns* on overlap with `train_cells`
      — predicting on trained-on cells is usually a leak but occasionally deliberate.
    - K-fold (`universe` set) bounds OOS to the universe so cells the train fold
      never saw (e.g. a later run discovered only at predict time) aren't selected.
    - The bounded-OOS presence check exists because K-fold OOS cells come from the
      *train* subject's universe: one absent on this source would be silently dropped
      downstream by the regressor-subset filter — fewer predictions, no error. Raise.

    Raises on an empty selection.
    """
    # If source and train cells have different entity keys, they never hash/compare
    # equal, so `available - train_cells` below subtracts nothing — wrongly keeping
    # trained-on cells in the OOS set. Catch the mismatch here instead.
    schemas = {c.keys() for c in available} | {c.keys() for c in model.train_cells}
    if len(schemas) > 1:
        raise ValueError(
            f"Cell-schema mismatch between source and train cells: "
            f"{sorted(sorted(s) for s in schemas)}"
        )

    if test_on:
        validate_bids_entities(*test_on)
        groups = parse_filter_groups(test_on)
        # `available` shares one schema here — the mismatch guard above collapsed
        # it to a single frozenset, so any element gives the cell entity keys.
        cell_entity_keys = next(iter(schemas))
        unknown = set(groups) - cell_entity_keys
        if unknown:
            raise ValueError(
                f"test_on entities {sorted(unknown)} not found on any available "
                f"cell (schema: {sorted(cell_entity_keys)}) — check for a typo"
            )
        selected = {
            cell
            for cell in available
            if all(cell.get(entity) in values for entity, values in groups.items())
        }
        if selected & model.train_cells:
            logger.warning(
                "test_on selects {} cell(s) the model was trained on",
                len(selected & model.train_cells),
            )
    elif artifact.universe is not None:
        selected = artifact.universe - model.train_cells
    else:
        selected = available - model.train_cells

    if not selected:
        if test_on:
            raise ValueError(f"test_on matched no available cells: {test_on}")
        raise ValueError("empty out-of-sample set — pass test_on to name cells")

    if artifact.universe is not None:
        missing = selected - available
        if missing:
            raise ValueError(
                f"bounded OOS cells absent on source subject: "
                f"{sorted(map(repr, missing))}"
            )
    return selected


class EncodingPredictor(_EncodingContext):
    """Loaded artifact -> inference. Rebuilds X via the shared path."""

    def __init__(self, *, bids_root: str | Path, artifact: EncodingArtifact) -> None:
        """Wrap a loaded artifact to drive predict on a given `bids_root`.

        Stores `artifact.recipe` as `self._recipe` (the shared discovery/build path
        reads everything off it; validated at train, so no re-validation here) and
        stashes the whole `artifact`: the `predict` loop reads
        `artifact.models`/`artifact.universe` to select cells per model,
        `_predict_model` reads `self._recipe.col_slices` for the rebuild guard, and
        `analyze` reads `artifact.sub_id` for provenance. All come off the one
        loaded artifact.

        No `config`/`device`: predict runs on the numpy backend (CPU always) and
        the discovery/build path reads no `self._config`. `bids_root` is a caller
        argument, deliberately not part of X identity; the loaded recipe carries
        `col_slices` (train-filled) for the rebuild guard.
        """
        self._layout = BIDSLayout(bids_root)
        self._recipe = artifact.recipe
        self._artifact = artifact

    @classmethod
    def load(
        cls,
        *,
        bids_root: str | Path,
        sub_id: str,
        desc: str,
    ) -> EncodingPredictor:
        """Load a persisted artifact by `(sub_id, desc)` and wrap it for predict.

        `(sub_id, kind="encodingModel", desc)` fully determines the file. `sub_id`
        is the *model* subject (whose trained weights); the source subject is passed to
        `predict`, and may differ.
        """
        layout = BIDSLayout(bids_root)
        out = layout.path.result(sub=sub_id, kind="encodingModel", desc=desc)
        return cls(bids_root=bids_root, artifact=load_artifact(out.path))

    def predict(
        self,
        *,
        source_sub_id: str,
        test_on: list[str] | None = None,
    ) -> list[Prediction]:
        """Predict each model's `Y_hat` from a source subject's regressors.

        Discovers and enriches the regressors for `source_sub_id` once (the per-model
        loop reuses one filesystem scan), then for each model in the artifact selects
        its cells (OOS by default, or `test_on`), subsets the enriched metas, and
        runs `_predict_model`. Returns one `Prediction` per model, in
        `artifact.models` order — a single-model artifact yields a length-1 list.

        Predict-only: no target Y. `source_sub_id` provides regressors (X); the
        model's weights are baked in at load (`EncodingPredictor.__init__`).

        Uses full discovery, not the recipe's `bids_filters`: `_select_cells`
        trusts the stored `train`/`universe` sets and may name cells (`test_on`)
        outside the train corpus, so the source's available cells must not be
        pre-narrowed by replaying train-time filters. `_apply_filters` and
        `_validate_coverage` (a train coverage invariant) are deliberately skipped.
        """
        feature_bids = self._discover_features(source_sub_id)
        confound_bids = self._discover_confounds(source_sub_id)
        bold_metas = self._discover_bold(source_sub_id)
        feature_bids = self._resolve_cell_keys(source_sub_id, feature_bids, bold_metas)
        confound_bids = self._resolve_cell_keys(
            source_sub_id, confound_bids, bold_metas
        )
        # Merge into one regressor dict; `_build_x` rebuilds X from features+confounds
        # the same way train did. Cross-stream coverage is not re-checked here (predict
        # trusts the stored cell sets, like _validate_coverage); a source cell missing
        # a confound surfaces as a KeyError in `_build_x`, same as for features.
        regressor_bids = {**feature_bids, **confound_bids}
        regressor_metas = self._enrich_regressor_metas(regressor_bids, bold_metas)
        available = {key.cell for key in feature_bids}

        results: list[Prediction] = []
        for model in self._artifact.models:
            cells = _select_cells(available, self._artifact, model, test_on=test_on)
            sub_metas = {
                key: meta for key, meta in regressor_metas.items() if key.cell in cells
            }
            results.append(self._predict_model(source_sub_id, model, sub_metas))
        return results

    def analyze(
        self,
        *,
        source_sub_id: str,
        target_sub_id: str,
        test_on: list[str] | None = None,
    ) -> xr.Dataset:
        """Score a source-driven prediction against a target subject's actual BOLD.

        Wraps `predict(source_sub_id, test_on)`, then per fold aligns the target's Y,
        builds target-turn role masks, and scores per band/role. Returns the pure
        in-memory `xr.Dataset` (`corr (fold, band, role, voxel)`); the CLI owns
        persistence and the `desc` path tag, so neither is a parameter here.

        The three subject roles are independent: `source_sub_id` drives X, the loaded
        model drives weights, `target_sub_id` drives actual Y and the prod/comp turns.
        Cross-dyad triples run mechanically (a scramble/null control) but only warn —
        `_align_y`'s length-drift guard is the hard net if runs mismatch.
        """
        if self._layout.dyad_of(source_sub_id) != self._layout.dyad_of(target_sub_id):
            logger.warning(
                "source ({}) and target ({}) are different dyads — correlations are "
                "a cross-conversation control, not a fit",
                source_sub_id,
                target_sub_id,
            )
        source_preds = self.predict(source_sub_id=source_sub_id, test_on=test_on)
        target_bold = self._discover_bold(target_sub_id)

        # One enriched regressor meta per cell for the target's dyad, so `_role_masks`
        # can place the target's prod/comp turns (which live in the target's own
        # conversation, not the source's). `_turn_masks` reads only
        # `bids`/`n_trs`/`repetition_time` off the meta, so collapsing each cell to its
        # first feature meta suffices. Built once — all folds share the conversation.
        feature_bids = self._discover_features(target_sub_id)
        feature_bids = self._resolve_cell_keys(target_sub_id, feature_bids, target_bold)
        regressor_metas = self._enrich_regressor_metas(feature_bids, target_bold)
        target_cell_metas: dict[CellKey, RegressorMeta] = {}
        for key, meta in regressor_metas.items():
            target_cell_metas.setdefault(key.cell, meta)

        corr_folds: list[np.ndarray] = []
        fold_cells: list[list[CellKey]] = []
        for pred in source_preds:
            Y_true = _align_y(target_bold, pred.row_slices)
            _assert_same_voxels(Y_hat=pred.Y_hat, Y_true=Y_true)
            # Restrict to this fold's cells, in `pred.row_slices` order — that order
            # is what lines the masks up with Y_true. The lookup raises if the target
            # subject never discovered a cell the source predicted on.
            fold_metas = {cell: target_cell_metas[cell] for cell in pred.row_slices}
            masks = _role_masks(
                layout=self._layout,
                target_sub_id=target_sub_id,
                cell_metas=fold_metas,
                delays=self._recipe.delays,
            )
            fold_scores = _score_roles(Y_true=Y_true, Y_hat=pred.Y_hat, masks=masks)
            corr_folds.append(fold_scores)
            fold_cells.append(list(pred.row_slices.keys()))

        # xarray is optional at module load, so import it lazily here (mirrors the
        # `TYPE_CHECKING`-only import at the top)
        import xarray as xr

        # Axis labels are coords so they index the cube (`.sel(band=…, role=…)`) and
        # travel with the array; `voxel` gets no coord — a bare integer index is honest
        # without real voxel IDs. `fold_cells` stays a live `list[list[CellKey]]` on
        # `attrs`; JSON-encoding it is a `save_eval` serialization detail, never seen by
        # a programmatic caller.
        return xr.Dataset(
            {"corr": (("fold", "band", "role", "voxel"), np.stack(corr_folds))},
            coords={
                "fold": list(range(len(source_preds))),
                "band": source_preds[0].band_names,
                "role": list(ROLES),
            },
            attrs={
                "model_sub": self._artifact.sub_id,
                "source_sub": source_sub_id,
                "target_sub": target_sub_id,
                # JSON, not a Python repr, so archival consumers (R/MATLAB/ncdump)
                # parse it. Encoded here, not at the save_eval seam, because it is a
                # scalar string, not a live nested value like fold_cells.
                "test_on": "OOS" if test_on is None else json.dumps(test_on),
                "delays": np.asarray(self._recipe.delays, dtype=int),
                "bold_space": str(self._recipe.bold_space),
                "fold_cells": fold_cells,
                "hypline_version": __version__,
            },
        )

    def _predict_model(
        self,
        source_sub_id: str,
        model: FittedModel,
        regressor_metas: dict[RegressorKey, RegressorMeta],
    ) -> Prediction:
        """Run one model over a pre-selected cell set, returning its `Y_hat` (no Y).

        Consumes pre-discovered, pre-enriched `regressor_metas` (features +
        confounds) already subset to the selected cells (the analyze loop calls this
        K times; re-discovering per call means K+1 filesystem scans). Rebuilds X via
        the same `_build_x` path as train, asserts `col_slices` matches the recipe
        (rebuild guard), rebinds the loaded pipeline's frozen train cell-lengths to
        this X's geometry, and predicts on the numpy backend.

        Returns a `Prediction` — per-model, no actual Y. The caller does
        `_align_y(target_bold, prediction.row_slices)` for Y to compare against.

        `source_sub_id` is the subject whose regressors these are; `_build_x`
        resolves the prod/comp split mask against it, so the rebuilt X carries the
        same subject-relative column meaning the model was trained under. The
        `col_slices` drift guard below rejects any mismatch.
        """
        from himalaya.backend import set_backend

        data = self._build_x(source_sub_id, regressor_metas)
        if data.col_slices != self._recipe.col_slices:
            raise ValueError(
                f"col_slices drift: {data.col_slices} != {self._recipe.col_slices}"
            )
        set_backend("numpy")
        _rebind_cell_lengths(model.pipeline, data.cell_lengths())
        # split=True returns per-band Y_hat (band, row, voxel); analyze scores each
        # band, and summing it (Y_hat.sum(0)) recovers the combined prediction.
        # himalaya orders bands by ColumnKernelizer order == col_slices build order,
        # so band_names below labels them straight from col_slices.keys().
        Y_hat = model.pipeline.predict(data.X.astype(np.float32), split=True)
        return Prediction(
            Y_hat=Y_hat,
            row_slices=data.row_slices,
            band_names=list(data.col_slices.keys()),
        )
