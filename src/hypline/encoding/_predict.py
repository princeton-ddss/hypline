from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline

from hypline.layout import BIDSLayout

from ._artifact import EncodingArtifact, FittedModel, load_artifact
from ._context import _EncodingContext
from ._schema import CellDelayer, CellKey, Prediction, RegressorKey, RegressorMeta


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
    test_on: dict[str, str] | None = None,
) -> set[CellKey]:
    """Choose which cells `model` predicts on — out-of-sample by default, or `test_on`.

    `available` is the source subject's *full* discovered cell set. Recipe filters
    are deliberately not replayed: the stored `train`/`universe` sets are trusted,
    and `test_on` is allowed to name cells outside the train corpus, so narrowing
    `available` up front would wrongly exclude valid targets.

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
        selected = {
            cell
            for cell in available
            if all(cell.get(entity) == value for entity, value in test_on.items())
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
        `artifact.models`/`artifact.universe` to select cells per model, and
        `_predict_model` reads `self._recipe.col_slices` for the rebuild guard. Both
        come off the one loaded artifact.

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

        `(sub_id, kind="encoding", desc)` fully determines the file. `sub_id` is the
        *model* subject (whose trained weights); the source subject is passed to
        `predict`, and may differ.
        """
        layout = BIDSLayout(bids_root)
        out = layout.path.result(sub=sub_id, kind="encoding", desc=desc)
        return cls(bids_root=bids_root, artifact=load_artifact(out.path))

    def predict(
        self,
        source_sub_id: str,
        test_on: dict[str, str] | None = None,
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
            results.append(self._predict_model(model, sub_metas))
        return results

    def _predict_model(
        self,
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
        """
        from himalaya.backend import set_backend

        data = self._build_x(regressor_metas)
        if data.col_slices != self._recipe.col_slices:
            raise ValueError(
                f"col_slices drift: {data.col_slices} != {self._recipe.col_slices}"
            )
        set_backend("numpy")
        _rebind_cell_lengths(model.pipeline, data.cell_lengths())
        Y_hat = model.pipeline.predict(data.X.astype(np.float32))
        return Prediction(row_slices=data.row_slices, Y_hat=Y_hat)
