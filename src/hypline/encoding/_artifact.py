from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from loguru import logger
from sklearn.base import BaseEstimator

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline

from hypline._version import __version__
from hypline.downsample import FeatureDownsampleMethod
from hypline.enums import SurfaceSpace, VolumeSpace

from ._schema import CellKey


@dataclass(frozen=True)
class XRecipe:
    """Everything needed to rebuild X identically on another subject.

    One source of truth for X identity so the train-time writer and the
    predict-time rebuilder cannot drift. `device` is excluded — it is a
    runtime/hardware choice, not part of X identity, and stays on
    `EncodingConfig`.

    Attributes
    ----------
    features
        Feature ref name -> its `(kind, desc)` selector. Iteration order fixes
        X's feature-band column layout, so it is load-bearing, not incidental.
    confounds
        Confound ref name -> its `(kind, desc)` selector. All confounds collapse
        into one shared ridge band, so within-band order is irrelevant (a single
        alpha covers them).
    bold_space
        Target BOLD space; surface (e.g. `fsaverage6`) or volume.
    bold_desc
        `desc` entity of the target BOLD (e.g. `denoised`) — selects which
        derivative the model is fit against.
    downsample
        Method mapping feature time series onto the BOLD TR grid.
    bids_filters
        Extra BIDS filters constraining the cell set — the sole record of
        which tasks/runs/conditions shaped X. `task` is an ordinary corpus
        filter here; only `sub`/`space`/`feat`/`desc` are reserved and cannot
        appear.
    delays
        FIR delays (in TRs) stacked per regressor, giving the model its HRF
        lag basis.
    alphas
        Ridge penalties searched per band during inner-CV.
    split
        Whether himalaya keeps per-band prediction shares (`split=True`)
        rather than only the summed prediction.
    col_slices
        Band key -> its column block in X. Empty at construction; `train`
        fills it from the assembled data. Build order is screens, features,
        confounds — the band axis labels downstream must match this order.
    """

    features: dict[str, tuple[str, str | None]]
    confounds: dict[str, tuple[str, str | None]]
    bold_space: SurfaceSpace | VolumeSpace
    bold_desc: str
    downsample: FeatureDownsampleMethod
    bids_filters: list[str]
    delays: list[int]
    alphas: list[float]
    split: bool = True
    col_slices: dict[str, slice] = field(default_factory=dict)


@dataclass(frozen=True)
class FittedModel:
    """One fitted model and the cell set it was fit on.

    Attributes
    ----------
    pipeline
        The fitted banded-ridge pipeline: a `ColumnKernelizer` that delays and
        linear-kernelizes each band (`StandardScaler -> CellDelayer ->
        Kernelizer`, screens skipping the scaler), scored by
        `MultipleKernelRidgeCV`.
    train_cells
        Post-filter `row_slices` key set — the cells actually fit on.
    """

    pipeline: Pipeline
    train_cells: set[CellKey]


@dataclass(frozen=True)
class FoldSpec:
    """One fold configuration: the cell axis to fold over and the fold count.

    Internal carrier for the paired `fold_by`/`n_folds` constructor args; the
    public surface stays flat.

    Attributes
    ----------
    by
        Cell axis to fold over (e.g. `run`).
    n
        Fold count, or the `"loo"` sentinel for leave-one-out.
    """

    by: str
    n: int | Literal["loo"]


@dataclass(frozen=True)
class EncodingArtifact:
    """On-disk encoding result: a shared `recipe` plus one-or-more fitted `models`.

    Filters that shaped the cell set live on `recipe.bids_filters` (X identity),
    not here.

    Attributes
    ----------
    recipe
        Shared X identity; every model here was built from it.
    sub_id
        Subject the models were fit on.
    fold
        Fold configuration these models were produced under, or `None` when
        unfolded (a single model).
    models
        The fitted models: one when unfolded, one per fold otherwise.
    universe
        Bounds the OOS cell set for fold groups; `None` for a single model,
        whose OOS is the target subject's available cells minus its train set.
    """

    recipe: XRecipe
    sub_id: str
    fold: FoldSpec | None
    models: list[FittedModel]
    universe: set[CellKey] | None


def _numpyfy_fitted(obj: object, _seen: set[int] | None = None) -> None:
    """In-place convert any torch-tensor fitted attr on `obj` to numpy.

    himalaya binds its backend at estimator construction and fitted weights
    (`MultipleKernelRidgeCV.dual_coef_`/`deltas_`, each band's `Kernelizer.X_fit_`,
    scaler stats) stay backend-bound — torch tensors when fit on a torch backend.
    `set_backend("numpy")` does not retroactively convert them, so the blob would
    carry torch arrays and fail to load without torch/CUDA. This walks the fitted
    estimator tree and rewrites tensors as numpy arrays, making the artifact
    portable and loadable on the numpy backend.

    Convert via `.detach().cpu().numpy()` directly, not the active backend's
    `to_numpy`: the numpy backend's `to_numpy` is a no-op (`return array`), so a
    CUDA tensor would never leave the GPU and `np.asarray` would raise — the
    device transfer lives only in the torch backend. Going through torch's own
    tensor API is device-safe regardless of which backend is active.
    """
    import torch

    seen = _seen if _seen is not None else set()
    if id(obj) in seen:
        return
    seen.add(id(obj))

    def _convert(value: object) -> object:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        if isinstance(value, BaseEstimator):
            _numpyfy_fitted(value, seen)
            return value
        if isinstance(value, list):
            return [_convert(v) for v in value]
        if isinstance(value, tuple):
            return tuple(_convert(v) for v in value)
        if isinstance(value, dict):
            return {k: _convert(v) for k, v in value.items()}
        return value

    for attr, value in vars(obj).items():
        setattr(obj, attr, _convert(value))


def save_artifact(artifact: EncodingArtifact, path: str | Path) -> None:
    """Dump `artifact` to `path` (.joblib) plus a non-pipeline JSON sidecar.

    Forces fitted weights to numpy before the joblib dump (see `_numpyfy_fitted`)
    so the blob loads without torch. The sidecar mirrors everything except the
    pipeline — recipe, per-model `train` cell sets, `fold`, `universe`, and
    `hypline_version` — making provenance greppable without unpickling.
    """
    import joblib
    from himalaya.backend import set_backend

    path = Path(path)

    # Switch to numpy so the in-memory pipeline this artifact still references
    # predicts on the same backend as a reloaded copy (predict reads the active
    # backend at call time). The weight conversion below is backend-independent.
    set_backend("numpy")
    for model in artifact.models:
        _numpyfy_fitted(model.pipeline)

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    path.with_suffix(".json").write_text(json.dumps(_sidecar(artifact), indent=2))


def _sidecar(artifact: EncodingArtifact) -> dict:
    """Build the JSON-serializable mirror of `artifact`, minus the pipeline."""
    recipe = artifact.recipe
    return {
        "hypline_version": __version__,
        "sub_id": artifact.sub_id,
        "recipe": {
            "features": {name: list(spec) for name, spec in recipe.features.items()},
            "confounds": {name: list(spec) for name, spec in recipe.confounds.items()},
            "bold_space": str(recipe.bold_space),
            "bold_desc": recipe.bold_desc,
            "downsample": recipe.downsample,
            "bids_filters": recipe.bids_filters,
            "delays": recipe.delays,
            "alphas": recipe.alphas,
            "split": recipe.split,
            "col_slices": {
                name: [s.start, s.stop] for name, s in recipe.col_slices.items()
            },
        },
        "fold": (
            None
            if artifact.fold is None
            else {"fold_by": artifact.fold.by, "n_folds": artifact.fold.n}
        ),
        "models": [
            {"train_cells": [dict(cell.items()) for cell in model.train_cells]}
            for model in artifact.models
        ],
        "universe": (
            None
            if artifact.universe is None
            else [dict(cell.items()) for cell in artifact.universe]
        ),
    }


def load_artifact(path: str | Path) -> EncodingArtifact:
    """Load an encoding artifact from its `.joblib` blob.

    Logs a warning (does not fail) on a `hypline_version` mismatch read from the
    sidecar — a version skew is a provenance signal, not a hard incompatibility here.
    A missing sidecar silently skips the version check.

    Parameters
    ----------
    path
        Path to the artifact `.joblib` blob.

    Returns
    -------
    EncodingArtifact
        The artifact with fitted weights as numpy arrays (converted at save
        time), so it loads and predicts without torch or CUDA.
    """
    import joblib

    path = Path(path)

    sidecar_path = path.with_suffix(".json")
    if sidecar_path.exists():
        stamped = json.loads(sidecar_path.read_text()).get("hypline_version")
        if stamped is not None and stamped != __version__:
            logger.warning(
                "Artifact {} was written by hypline {}, loading under {}",
                path.name,
                stamped,
                __version__,
            )

    return joblib.load(path)
