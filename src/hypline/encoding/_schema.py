from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, NamedTuple

import numpy as np
from pydantic import BaseModel
from sklearn.base import BaseEstimator, TransformerMixin

from hypline.bids import BIDSPath
from hypline.enums import Device


class BoldKey(NamedTuple):
    ses: str | None
    task: str
    run: str | None


class CellKey:
    """Open-schema key identifying a single regressor time window.

    EXCLUDE defines which entities must never appear on a cell key:
    - sub, dyad: invariant identities across a training call (features are
      dyad-keyed, BOLD is sub-keyed; neither is a cell axis)
    - desc, res, den: image-variant entities (BOLD derivatives only)
    - space, feat, conf: orthogonal axes — handled by dedicated arguments
      (feat/conf are the category entities naming the regressor kind, not a
      cell axis; excluding both lets feature and confound cells compare equal)

    `task` flows through as a cell axis, filtered like any other corpus entity via
    `bids_filters` (e.g. `task-A,task-B`). An unfiltered call pools every task, in
    which case cells from different tasks become distinct rows in X/Y; narrowing to
    one task leaves `task` constant on every cell.

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
            "conf",
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


class RegressorKey(NamedTuple):
    cell: CellKey
    name: str


class EncodingConfig(BaseModel):
    device: Device = Device.CPU
    delays: list[int] = [0, 1, 2, 3, 4, 5]
    alphas: list[float] = np.logspace(0, 12, 13).tolist()


@dataclass(frozen=True)
class RegressorMeta:
    """One regressor cell's TR-grid placement.

    Resolved once by `_enrich_regressor_metas` so `_build_x` never reads BOLD.
    `onset_tr` is currently unconsumed — X rows start at 0 per cell, and `_align_y`
    recomputes its own onset from the target subject's BOLD.
    """

    bids: BIDSPath
    n_trs: int
    onset_tr: int
    repetition_time: float


@dataclass(frozen=True)
class XData:
    """Assembled regressor matrix and its row/column geometry, no target.

    `row_slices` maps each cell to its contiguous block of rows in X (in
    `_sort_key` order); `col_slices` maps each band key (a feature name, the
    reserved confound-band key, or the reserved screen-band key) to its contiguous
    block of columns. This is what predict needs — X alone — and what train extends
    with Y (see `TrainingData`).
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

    `segment_entity` is the within-run BIDS entity (e.g. 'block') appended as the
    finest level of `_inner_cv`'s structural fold chain (ses -> task -> run ->
    segment); None when unsegmented. It determines the split only when no coarser
    level varies in the train set.
    """

    Y: np.ndarray
    segment_entity: str | None = None


@dataclass(frozen=True)
class Prediction:
    """One model's predicted BOLD and the row/band geometry it sits on.

    `Y_hat` is 3-D `(band, row, voxel)` from himalaya's `split=True`: each band is
    that kernel's *share* of the joint-model prediction, and the shares sum to the
    combined prediction (`Y_hat.sum(0)`).

    `row_slices` maps each predicted cell to its contiguous block of rows in
    `Y_hat` (`_build_x` order). Predict-only: there is no actual Y — `analyze`
    recovers it from a target subject via `_align_y` on this same geometry.

    `band_names` labels the band axis, in `col_slices` build order (screens,
    features, confounds), so band `i` is `band_names[i]`.
    """

    Y_hat: np.ndarray
    row_slices: dict[CellKey, slice]
    band_names: list[str]


class CellDelayer(BaseEstimator, TransformerMixin):
    """Stack finite-impulse-response delays of X, one column block per delay.

    A row's delayed source `row - d` is zeroed when it falls before the start of
    that row's cell, so a cell never sees regressor values from the cell above it.
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
