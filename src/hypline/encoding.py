import json
import reprlib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Literal, NamedTuple, get_args

import numpy as np
from loguru import logger
from pydantic import BaseModel
from sklearn.base import BaseEstimator, TransformerMixin

if TYPE_CHECKING:
    from sklearn.model_selection import BaseCrossValidator
    from sklearn.pipeline import Pipeline

from hypline._version import __version__
from hypline.bids import (
    BIDS_ENTITY_VALUE_RE,
    BIDSPath,
    normalize_bids_filters,
    parse_kind_desc,
)
from hypline.bold import (
    BOLD_EXTENSIONS,
    BoldMeta,
    load_bold_meta,
    parse_bold_space,
)
from hypline.downsample import DownsampleMethod, downsample
from hypline.enums import Device, SurfaceSpace, VolumeSpace
from hypline.events import merge_filename_and_sidecar, segment_tr_slice
from hypline.io import (
    read_feature,
    read_feature_metadata,
    stack_array_column,
)
from hypline.layout import BIDSLayout

FeatureDownsampleMethod = Literal["mean", "sum"]

_SOLVER_N_ITER = 100
_SOLVER_DIAGONALIZE_METHOD = "svd"

# Public Encoding-facing methods must be a subset of all methods
if not set(get_args(FeatureDownsampleMethod)) <= set(get_args(DownsampleMethod)):
    raise RuntimeError("FeatureDownsampleMethod must be a subset of DownsampleMethod")


class BoldKey(NamedTuple):
    ses: str | None
    task: str
    run: str | None


class CellKey:
    """Open-schema key identifying a single feature time window.

    EXCLUDE defines which entities must never appear on a cell key:
    - sub, dyad: invariant identities across a training call (features are
      dyad-keyed, BOLD is sub-keyed; neither is a cell axis)
    - desc, res, den: image-variant entities (BOLD derivatives only)
    - space, feat: orthogonal axes — handled by dedicated arguments

    `task` flows through as a cell axis: a training call may declare multiple
    tasks (`tasks=["A", "B"]`), in which case cells from different tasks become
    distinct rows in X/Y. Single-task calls leave `task` constant on every cell.

    Equality and hashing are order-independent.
    """

    EXCLUDE: frozenset[str] = frozenset(
        (
            "sub",
            "dyad",
            "desc",
            "res",
            "den",
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

    def to_bold_key(self) -> BoldKey:
        return BoldKey(self.get("ses"), self["task"], self.get("run"))


class FeatureKey(NamedTuple):
    cell: CellKey
    feature: str


class EncodingConfig(BaseModel):
    device: Device = Device.CPU
    delays: list[int] = [0, 1, 2, 3, 4, 5]
    alphas: list[float] = np.logspace(0, 12, 13).tolist()


@dataclass(frozen=True)
class FeatureMeta:
    """One feature cell's TR-grid placement.

    Resolved once by `_enrich_feature_metas` so `_build_x` never reads BOLD.
    `onset_tr` is currently unconsumed — X rows start at 0 per cell, and `_align_y`
    recomputes its own onset from the target subject's BOLD.
    """

    bids: BIDSPath
    n_trs: int
    onset_tr: int
    repetition_time: float


@dataclass(frozen=True)
class XData:
    """Assembled feature matrix and its row/column geometry, no target.

    `row_slices` maps each cell to its contiguous block of rows in X (in
    `_sort_key` order); `col_slices` maps each feature name to its contiguous
    block of columns. This is what predict needs — X alone — and what train
    extends with Y (see `TrainingData`).
    """

    X: np.ndarray
    row_slices: dict[CellKey, slice]
    col_slices: dict[str, slice]

    def cell_lengths(self, ordered_cells: list[CellKey] | None = None) -> list[int]:
        """Per-cell row counts — the geometry `CellDelayer` and `_inner_cv` consume.

        Order matters and is positional: these counts must line up with X's row
        blocks, since `CellDelayer`'s FIR mask keys off cumulative offsets. Pass
        `ordered_cells` for a cell subset in a specific order (e.g. a fold's train
        cells); omit it to take all cells in `row_slices` order.
        """
        cells = self.row_slices.keys() if ordered_cells is None else ordered_cells
        return [
            self.row_slices[cell].stop - self.row_slices[cell].start for cell in cells
        ]


@dataclass(frozen=True)
class TrainingData(XData):
    """`XData` plus the aligned BOLD target Y, on X's same row axis.

    Subclassing appends `Y` as the last field — construction is kwargs-only at
    every site, so the positional reorder is inert. Do not switch any
    `TrainingData(...)` call to positional args without accounting for this.
    """

    Y: np.ndarray


@dataclass(frozen=True)
class Prediction:
    """One model's predicted BOLD and the row geometry it sits on.

    `row_slices` maps each predicted cell to its contiguous block of rows in
    `Y_hat` (`_build_x` order). Predict-only: there is no actual Y — `analyze`
    recovers it from a target subject via `_align_y` on this same geometry.
    """

    row_slices: dict[CellKey, slice]
    Y_hat: np.ndarray


@dataclass(frozen=True)
class XRecipe:
    """Everything needed to rebuild X identically on another subject.

    One source of truth for X identity so the train-time writer and the
    predict-time rebuilder cannot drift. `device` is excluded — it is a
    runtime/hardware choice, not part of X identity, and stays on
    `EncodingConfig`.
    """

    features: dict[str, tuple[str, str | None]]
    tasks: list[str]
    bold_space: SurfaceSpace | VolumeSpace
    bold_desc: str
    downsample: FeatureDownsampleMethod
    bids_filters: list[str]
    delays: list[int]
    alphas: list[float]
    col_slices: dict[str, slice] = field(default_factory=dict)

    @property
    def task_filters(self) -> list[str]:
        return [f"task-{task}" for task in self.tasks]


@dataclass(frozen=True)
class FittedModel:
    """One fitted model and the cell set it was fit on.

    `train_cells` is the post-filter `row_slices` key set — the cells actually
    fit on.
    """

    pipeline: "Pipeline"
    train_cells: set[CellKey]


@dataclass(frozen=True)
class FoldSpec:
    """One fold configuration: the cell axis to fold over and the fold count.

    Internal carrier for the paired `fold_by`/`n_folds` constructor args; the
    public surface stays flat. `n` is a count or the `"loo"` sentinel.
    """

    by: str
    n: int | Literal["loo"]


@dataclass(frozen=True)
class EncodingArtifact:
    """On-disk encoding result: a shared `recipe` plus one-or-more fitted `models`.

    `fold` records the fold configuration these models were produced under, or
    `None` when unfolded (a single model). `universe` bounds the OOS cell set for
    fold groups; it is `None` for a single model, whose OOS is just the target
    subject's available cells minus its train set.

    Filters that shaped the cell set live on `recipe.bids_filters` (X identity),
    not here.
    """

    recipe: XRecipe
    fold: FoldSpec | None
    models: list[FittedModel]
    universe: set[CellKey] | None


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
    fold: "FoldSpec | None",
) -> "BaseCrossValidator":
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


class CellDelayer(BaseEstimator, TransformerMixin):
    """Stack finite-impulse-response delays of X, one column block per delay.

    A row's delayed source `row - d` is zeroed when it falls before the start of
    that row's cell, so a cell never sees feature values from the cell above it.
    `cell_lengths` gives the per-cell row counts in the same cell order as
    `TrainingData.row_slices`; it is set per `train`/`predict` call rather than
    frozen, since cell lengths are per-subject.

    Assumes `delays >= 0`. Negative delays would need the mirror-image mask
    (zeroing rows near a cell's *end*) and are out of scope.
    """

    def __init__(self, delays: list[int], cell_lengths: list[int]):
        self.delays = delays
        self.cell_lengths = cell_lengths

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if any(d < 0 for d in self.delays):
            raise ValueError(
                f"CellDelayer supports delays >= 0 only; got {self.delays}"
            )
        X = np.asarray(X)
        if X.shape[0] != sum(self.cell_lengths):
            raise ValueError(
                f"X has {X.shape[0]} rows but cell_lengths sum to "
                f"{sum(self.cell_lengths)}"
            )

        cell_starts = np.repeat(
            np.concatenate(([0], np.cumsum(self.cell_lengths)[:-1])),
            self.cell_lengths,
        )
        rows = np.arange(X.shape[0])

        delayed_blocks = []
        for d in self.delays:
            block = np.zeros_like(X)
            # row i takes source i-d, valid only when i-d stays within i's cell
            valid = rows - d >= cell_starts
            block[valid] = X[rows[valid] - d]
            delayed_blocks.append(block)

        return np.hstack(delayed_blocks)


def _build_pipeline(
    *,
    col_slices: dict[str, slice],
    cell_lengths: list[int],
    delays: list[int],
    alphas: list[float],
    cv: "BaseCrossValidator",
) -> "Pipeline":
    """Assemble an unfitted banded-ridge pipeline, one kernel band per feature.

    Each feature in `col_slices` gets its own band:
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
            feature_name,
            make_pipeline(
                StandardScaler(),
                CellDelayer(delays=delays, cell_lengths=cell_lengths),
                Kernelizer(kernel="linear"),
            ),
            col_slice,
        )
        for feature_name, col_slice in col_slices.items()
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


def _rebind_cell_lengths(pipeline: "Pipeline", cell_lengths: list[int]) -> None:
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
    artifact: "EncodingArtifact",
    model: "FittedModel",
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
      downstream by the feature-subset filter — fewer predictions, no error. Raise.

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


def write_artifact(artifact: EncodingArtifact, path: Path) -> None:
    """Dump `artifact` to `path` (.joblib) plus a non-pipeline JSON sidecar.

    Forces fitted weights to numpy before the joblib dump (see `_numpyfy_fitted`)
    so the blob loads without torch. The sidecar mirrors everything except the
    pipeline — recipe, per-model `train` cell sets, `fold`, `universe`, and
    `hypline_version` — making provenance greppable without unpickling.
    """
    import joblib
    from himalaya.backend import set_backend

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
        "recipe": {
            "features": {name: list(spec) for name, spec in recipe.features.items()},
            "tasks": recipe.tasks,
            "bold_space": str(recipe.bold_space),
            "bold_desc": recipe.bold_desc,
            "downsample": recipe.downsample,
            "bids_filters": recipe.bids_filters,
            "delays": recipe.delays,
            "alphas": recipe.alphas,
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


def load_artifact(path: Path) -> EncodingArtifact:
    """Load an encoding artifact from its `.joblib` blob.

    Logs a warning (does not fail) on a `hypline_version` mismatch read from the
    sidecar — a version skew is a provenance signal, not a hard incompatibility here.
    """
    import joblib

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

    def _discover_features(self, sub_id: str) -> dict[FeatureKey, BIDSPath]:
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
        feature_bids: dict[FeatureKey, BIDSPath] = {}
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

        # Validate: all features present at every cell
        expected = {
            FeatureKey(cell_key, feature_name)
            for cell_key in {feature_key.cell for feature_key in feature_bids}
            for feature_name in self._recipe.features
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

        def _loc(bold_key: BoldKey) -> str:
            return _format_loc(
                sub=sub_id,
                ses=bold_key.ses,
                task=bold_key.task,
                run=bold_key.run,
                space=self._recipe.bold_space,
            )

        cell_keys_by_bold_key: dict[BoldKey, set[CellKey]] = {}
        for feature_key in feature_bids:
            bold_key = feature_key.cell.to_bold_key()
            cell_keys_by_bold_key.setdefault(bold_key, set()).add(feature_key.cell)

        orphan_bold_keys = cell_keys_by_bold_key.keys() - bold_metas.keys()
        if orphan_bold_keys:
            bold_key = next(iter(orphan_bold_keys))
            msg = f"No BOLD file found for features at {_loc(bold_key)}"
            if len(orphan_bold_keys) > 1:
                msg += f" ({len(orphan_bold_keys) - 1} other coverage gaps exist)"
            raise FileNotFoundError(msg)

        resolved_feature_bids: dict[FeatureKey, BIDSPath] = {}
        for feature_key, bids in feature_bids.items():
            cell_key = feature_key.cell
            bold_key = cell_key.to_bold_key()
            bold_meta = bold_metas[bold_key]

            if not bold_meta.segments:
                run_cell_keys = cell_keys_by_bold_key[bold_key]
                if len(run_cell_keys) > 1:
                    raise ValueError(
                        f"Run is unsegmented but has {len(run_cell_keys)} feature "
                        f"files at {_loc(bold_key)} — provide an events.tsv with "
                        f"BIDS key-value entities to segment the run"
                    )
                illegal_keys = cell_key.keys() - {"ses", "task", "run"}
                if illegal_keys:
                    raise ValueError(
                        f"Unsegmented run at {_loc(bold_key)} has feature filename "
                        f"with unexpected entities {sorted(illegal_keys)} — only "
                        f"ses, task, and run are valid on feature filenames for "
                        f"unsegmented runs. To attach metadata, declare a segment "
                        f"row in events.tsv and add descriptive attributes to "
                        f"events.json Levels."
                    )
                resolved_feature_bids[feature_key] = bids
                continue

            segment_entity = bold_meta.segments[0].entity
            segment_values = {segment.value for segment in bold_meta.segments}

            # Validate the cell carries a known segment value for this run
            segment_value = cell_key.get(segment_entity)
            if segment_value is None:
                raise ValueError(
                    f"Feature filename at {_loc(bold_key)} is missing segment "
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
                resolved_feature_bids[feature_key] = bids
            else:
                resolved_feature_bids[
                    FeatureKey(cell=CellKey(**merged), feature=feature_key.feature)
                ] = bids

        return resolved_feature_bids

    def _enrich_feature_metas(
        self,
        feature_bids: dict[FeatureKey, BIDSPath],
        bold_metas: dict[BoldKey, BoldMeta],
    ) -> dict[FeatureKey, FeatureMeta]:
        """Enrich each feature path into a `FeatureMeta` with its TR-grid placement.

        The crossover between feature cells and the BOLD timeline happens here,
        once: derive `(onset_tr, n_trs)` from the segment TR-slice for segmented
        runs, or from the run's header `n_trs` for unsegmented runs, and stamp
        `repetition_time`. `_build_x` then reads placement off the metas and never
        touches `bold_metas`.

        No BOLD voxel data is read — placement comes from `bold_metas` alone.
        The header `n_trs` carried by each meta is trusted; `_align_y` reconciles
        it against the actual array.
        """
        cells_by_bold_key: dict[BoldKey, set[CellKey]] = {}
        for feature_key in feature_bids:
            bold_key = feature_key.cell.to_bold_key()
            cells_by_bold_key.setdefault(bold_key, set()).add(feature_key.cell)

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

        feature_metas: dict[FeatureKey, FeatureMeta] = {}
        for feature_key, bids in feature_bids.items():
            onset_tr, n_trs, repetition_time = placement_by_cell[feature_key.cell]
            feature_metas[feature_key] = FeatureMeta(
                bids=bids,
                n_trs=n_trs,
                onset_tr=onset_tr,
                repetition_time=repetition_time,
            )
        return feature_metas

    def _build_x(self, feature_metas: dict[FeatureKey, FeatureMeta]) -> XData:
        """Assemble the X feature matrix and its row/column geometry, no target.

        Reads `n_trs`/`repetition_time` off each cell's `FeatureMeta` — the
        crossover with the BOLD timeline already happened in `_enrich_feature_metas`,
        so X no longer touches `bold_metas` and no BOLD data enters X.

        Cells are sorted deterministically so row positions are stable across
        runs. Column layout is derived from the first cell and assumed invariant —
        all cells must yield the same feature dimensionality.
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
            {feature_key.cell for feature_key in feature_metas}, key=_sort_key
        )

        X_parts: list[np.ndarray] = []
        row_slices: dict[CellKey, slice] = {}
        col_slices: dict[str, slice] = {}
        row_offset = 0
        col_offset = 0
        col_slices_initialized = False

        for cell_key in cell_keys:
            # Geometry is per-cell, shared across this cell's feature metas
            cell_meta = feature_metas[
                FeatureKey(cell_key, next(iter(self._recipe.features)))
            ]
            n_trs = cell_meta.n_trs
            row_slices[cell_key] = slice(row_offset, row_offset + n_trs)
            row_offset += n_trs

            # Construct X for the given cell
            feature_arrays: list[np.ndarray] = []
            for feature_name in self._recipe.features:
                meta = feature_metas[FeatureKey(cell_key, feature_name)]
                df = read_feature(meta.bids.path)
                # Drop untimed rows — downsample needs TR alignment
                df = df.filter(df.get_column("start_time").is_not_null())
                arr = downsample(
                    stack_array_column(df.get_column("feature")),
                    start_times=df.get_column("start_time").to_numpy(),
                    n_trs=n_trs,
                    repetition_time=meta.repetition_time,
                    method=self._recipe.downsample,
                )
                feature_arrays.append(arr)
            if not col_slices_initialized:
                for feature_name, arr in zip(self._recipe.features, feature_arrays):
                    n_cols = arr.shape[1]
                    col_slices[feature_name] = slice(col_offset, col_offset + n_cols)
                    col_offset += n_cols
                col_slices_initialized = True  # col slices are invariant across cells
            X_parts.append(np.hstack(feature_arrays))

        X = np.concatenate(X_parts, axis=0)
        return XData(X=X, row_slices=row_slices, col_slices=col_slices)


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
    ) -> "EncodingPredictor":
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
        """Predict each model's `Y_hat` from a source subject's features.

        Discovers and enriches the features for `source_sub_id` once (the per-model
        loop reuses one filesystem scan), then for each model in the artifact selects
        its cells (OOS by default, or `test_on`), subsets the enriched metas, and
        runs `_predict_model`. Returns one `Prediction` per model, in
        `artifact.models` order — a single-model artifact yields a length-1 list.

        Predict-only: no target Y. `source_sub_id` provides features (X); the
        model's weights are baked in at load (`EncodingPredictor.__init__`).

        Uses full discovery, not the recipe's `bids_filters`: `_select_cells`
        trusts the stored `train`/`universe` sets and may name cells (`test_on`)
        outside the train corpus, so the source's available cells must not be
        pre-narrowed by replaying train-time filters. `_apply_filters` and
        `_validate_coverage` (a train coverage invariant) are deliberately skipped.
        """
        feature_bids = self._discover_features(source_sub_id)
        bold_metas = self._discover_bold(source_sub_id)
        feature_bids = self._resolve_cell_keys(source_sub_id, feature_bids, bold_metas)
        feature_metas = self._enrich_feature_metas(feature_bids, bold_metas)
        available = {feature_key.cell for feature_key in feature_metas}

        results: list[Prediction] = []
        for model in self._artifact.models:
            cells = _select_cells(available, self._artifact, model, test_on=test_on)
            sub_metas = {
                feature_key: meta
                for feature_key, meta in feature_metas.items()
                if feature_key.cell in cells
            }
            results.append(self._predict_model(model, sub_metas))
        return results

    def _predict_model(
        self,
        model: FittedModel,
        feature_metas: dict[FeatureKey, FeatureMeta],
    ) -> Prediction:
        """Run one model over a pre-selected cell set, returning its `Y_hat` (no Y).

        Consumes pre-discovered, pre-enriched `feature_metas` already subset to the
        selected cells (the analyze loop calls this K times; re-discovering per call
        means K+1 filesystem scans). Rebuilds X via the same `_build_x` path as
        train, asserts `col_slices` matches the recipe (rebuild guard), rebinds the
        loaded pipeline's frozen train cell-lengths to this X's geometry, and
        predicts on the numpy backend.

        Returns a `Prediction` — per-model, no actual Y. The caller does
        `_align_y(target_bold, prediction.row_slices)` for Y to compare against.
        """
        from himalaya.backend import set_backend

        data = self._build_x(feature_metas)
        if data.col_slices != self._recipe.col_slices:
            raise ValueError(
                f"col_slices drift: {data.col_slices} != {self._recipe.col_slices}"
            )
        set_backend("numpy")
        _rebind_cell_lengths(model.pipeline, data.cell_lengths())
        Y_hat = model.pipeline.predict(data.X.astype(np.float32))
        return Prediction(row_slices=data.row_slices, Y_hat=Y_hat)
