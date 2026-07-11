"""The eval-Dataset lifecycle: build target-turn masks -> score -> serialize.

Mirrors `_artifact.py`'s model lifecycle for the analyze output.
`EncodingPredictor.analyze` composes these free functions; the storage seam
(`save_eval`/`load_eval`) lives here too so the `xr.Dataset` shape and its
on-disk encoding cannot drift.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from hypline.events import Turn

if TYPE_CHECKING:
    import xarray as xr

    from hypline.layout import BIDSLayout

from ._schema import BoldKey, CellKey, RegressorMeta
from ._turns import _turn_masks

# The role axis, in array/coord order. `_score_roles` builds the cube's role axis
# from this and `analyze` labels it from the same tuple, so the correlation rows and
# their `.sel(role=…)` labels cannot drift apart.
ROLES = ("prod", "comp", "both")


def _smear_mask(mask: np.ndarray, delays: list[int]) -> np.ndarray:
    """FIR-smear a per-cell boolean mask: OR together its `delays`-shifted copies.

    Row `i` is True if the raw mask was True at any of `i, i-d1, i-d2, …`; the
    first `d` rows precede the cell's start, so delay `d` never sets them. `mask`
    must be a single cell's rows — the FIR window must not bleed across cell
    boundaries (the same rule `CellDelayer` enforces for X via `cell_lengths`).
    """
    smeared = np.zeros_like(mask)
    for d in delays:
        if d == 0:
            smeared |= mask
        else:
            smeared[d:] |= mask[:-d]
    return smeared


def _role_masks(
    *,
    layout: BIDSLayout,
    target_sub_id: str,
    cell_metas: dict[CellKey, RegressorMeta],
    delays: list[int],
) -> dict[str, np.ndarray]:
    """Per-row prod/comp/both masks over X's stacked rows, from target turns.

    For each cell, derive the target's raw per-TR prod/comp masks (`_turn_masks`),
    FIR-smear each with `delays` per cell, then:

        prod = prod_sm & ~comp_sm   # exclusive pure-production rows
        comp = comp_sm & ~prod_sm   # exclusive pure-comprehension rows
        both = prod_sm |  comp_sm   # union — any speech-active row, prod or comp

    Exclusive is not a no-op only because of smearing: raw prod/comp are already
    disjoint (one floor-holder per TR), so the AND-NOT drops the delay-contaminated
    boundary rows the smear introduced. `both` re-includes those rows on purpose.

    Cells are processed in `cell_metas` iteration order, so the caller must pass the
    fold's cells in X's row order (i.e. keyed off `pred.row_slices`) for the masks to
    line up row-for-row with `_align_y`'s Y_true. `_turn_masks` sizes each cell's grid
    from `meta.n_trs`; that equals the row span because `analyze` runs `_align_y`
    first, whose per-cell drift guard raises on any target-vs-source mismatch.
    """
    prod_parts: list[np.ndarray] = []
    comp_parts: list[np.ndarray] = []
    turns_cache: dict[BoldKey, list[Turn]] = {}
    for meta in cell_metas.values():
        prod_raw, comp_raw = _turn_masks(
            layout,
            sub_id=target_sub_id,
            meta=meta,
            turns_cache=turns_cache,
        )
        prod_parts.append(_smear_mask(prod_raw, delays))
        comp_parts.append(_smear_mask(comp_raw, delays))

    prod_sm = np.concatenate(prod_parts)
    comp_sm = np.concatenate(comp_parts)
    return {
        "prod": prod_sm & ~comp_sm,
        "comp": comp_sm & ~prod_sm,
        "both": prod_sm | comp_sm,
    }


def _score_roles(
    *, Y_true: np.ndarray, Y_hat: np.ndarray, masks: dict[str, np.ndarray]
) -> np.ndarray:
    """Per-role split correlations, pooled across a fold's rows; empty role yields NaN.

    Returns `(band, role, voxel)` with roles in `ROLES` order. Each role selects its
    rows from the pooled `Y_true`/`Y_hat` and scores them at once — one correlation
    per role over all its masked rows, not a per-cell average. A zero-row mask fills
    NaN — a correlation over zero varying pairs is undefined, and NaN is skipped by
    `nanmean` across folds instead of masquerading as a real low correlation.
    """
    from himalaya.scoring import correlation_score_split

    n_bands, _, n_voxels = Y_hat.shape
    out = np.full((n_bands, len(ROLES), n_voxels), np.nan, dtype=np.float32)
    for m, role in enumerate(ROLES):
        mask = masks[role]
        if mask.sum() == 0:
            continue
        scores = correlation_score_split(Y_true[mask], Y_hat[:, mask, :])
        out[:, m, :] = np.asarray(scores)
    return out


def save_eval(ds: xr.Dataset, path: str | Path) -> None:
    """Write an analyze `Dataset` to self-contained netCDF-4 at `path`.

    The single storage seam (with `load_eval`): serializes the one nested value
    `fold_cells` to a JSON string (netCDF attrs hold only scalars/strings/numeric
    arrays), leaving the rest of `ds` — which `analyze` already built with
    attr-safe values — untouched. Does not mutate the caller's `ds`.

    Each `CellKey` serializes via `dict(cell.items())`; `load_eval` inverts this
    with `CellKey(**cell)`, so the round-trip is lossless and both sides stay live
    `CellKey`s (entity values are uniformly str — filename entities by parse,
    sidecar metadata by `_validate_segment_records` — so JSON preserves them
    exactly).
    """
    to_write = ds.copy()
    to_write.attrs = dict(ds.attrs)
    to_write.attrs["fold_cells"] = json.dumps(
        [[dict(cell.items()) for cell in fold] for fold in ds.attrs["fold_cells"]]
    )
    to_write.to_netcdf(path, engine="h5netcdf")


def load_eval(path: str | Path) -> xr.Dataset:
    """Open a saved eval netCDF and return the subsettable `xr.Dataset`.

    The controlled read path (mirrors `save_eval`): pins the engine and parses
    `fold_cells` back from its JSON string to `list[list[CellKey]]` — reconstructing
    each cell via `CellKey(**cell)` so loaded keys hash and compare against live
    ones (a plain dict never would, since `CellKey.__eq__` rejects non-`CellKey`).
    `corr` is subsettable by `.sel(band=…, role=…)`. Kept thin — no analysis logic —
    so the storage boundary stays in exactly these two functions.
    """
    import xarray as xr

    ds = xr.open_dataset(path, engine="h5netcdf")
    ds.attrs["fold_cells"] = [
        [CellKey(**cell) for cell in fold]
        for fold in json.loads(ds.attrs["fold_cells"])
    ]
    return ds
