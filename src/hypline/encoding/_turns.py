from __future__ import annotations

import numpy as np
import polars as pl

from hypline.events import (
    Turn,
    frame_onset,
    load_turns,
    stamp_turns,
)
from hypline.layout import BIDSLayout

from ._schema import BoldKey, RegressorMeta


def _turn_masks(
    layout: BIDSLayout,
    *,
    sub_id: str,
    meta: RegressorMeta,
    turns_cache: dict[BoldKey, list[Turn]],
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-TR (prod, comp) boolean masks for `sub_id` on one cell.

    prod TRs are those whose onset falls in `sub_id`'s speaking window; comp
    TRs are those held by any other named speaker; silence TRs (no floor-holder)
    are False in both — the implicit baseline. A TR straddling a turn boundary is
    assigned by its onset (`start_time = tr_index * repetition_time`), matching
    the source study's per-TR boxcar.

    The per-word `turn_sub` stamped into feature files is dropped at downsample,
    so the per-TR boxcar is re-derived here from the same `load_turns`/
    `stamp_turns` primitives against a synthetic TR-cadence grid. The grid spans
    `meta.n_trs` — the cell's TR count as placed on the BOLD timeline — so the
    masks always line up with the cell's features and confounds.

    Turns are per-run, so they are cached by (ses, task, run) across a run's
    segments. `meta.bids` is dyad-keyed (features/confounds are discovered by
    dyad), so it serves directly as the `load_turns`/`frame_onset` source.

    Raises if the dyad is not exactly two subjects — "the partner" (and thus
    comp) would be ambiguous — or if `sub_id` is not in the dyad, since
    `subjects_of` (participants.tsv) and file discovery are separate sources
    that can drift.
    """
    dyad_id = meta.bids.entities["dyad"]
    subjects = layout.subjects_of(dyad_id)
    if len(subjects) != 2:
        raise ValueError(
            f"prod/comp split requires a 2-subject dyad; dyad-{dyad_id} has "
            f"{len(subjects)}: {subjects}"
        )
    if sub_id not in subjects:
        raise ValueError(
            f"sub-{sub_id} is not in dyad-{dyad_id} {subjects}; "
            f"prod/comp masks would be undefined"
        )

    cache_key = BoldKey(  # turns are per-run; cache across a run's segments
        ses=meta.bids.entities.get("ses"),
        task=meta.bids.entities["task"],
        run=meta.bids.entities.get("run"),
    )
    if cache_key not in turns_cache:
        turns_cache[cache_key] = load_turns(layout, meta.bids)
    turns = turns_cache[cache_key]

    onset = frame_onset(layout, meta.bids)
    tr_starts = np.arange(meta.n_trs) * meta.repetition_time
    tr_grid = pl.DataFrame({"start_time": tr_starts})
    stamped, _ = stamp_turns(tr_grid, turns, frame_onset=onset)
    turn_sub = stamped.get_column("turn_sub").to_list()

    prod = np.array([t == sub_id for t in turn_sub], dtype=bool)
    comp = np.array([t is not None and t != sub_id for t in turn_sub], dtype=bool)
    return prod, comp
