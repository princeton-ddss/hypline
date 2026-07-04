from typing import Literal

from hypline.encoding import EncodingConfig, EncodingTrainer

from ..conftest import DEFAULT_BOLD_N_TRS, BIDSTree

SUB = "001"
PARTNER = "002"
DYAD = "101"
TASK = "conv"
SPACE = "MNI152NLin6Asym"


def _add_dyad_turns(
    tree: BIDSTree,
    *,
    task: str = TASK,
    run: str | None = "1",
    n_trs: int = DEFAULT_BOLD_N_TRS,
    tr: float = 2.0,
) -> None:
    """Register `SUB`+`PARTNER` on `DYAD` and split the run into two turn halves.

    Screens are always assembled from turn_speaker windows (see `_build_x`), so
    every X-assembling fixture must carry a 2-subject dyad with non-degenerate
    turns. `SUB` holds the floor over the first half of the run, `PARTNER` the
    second, so `SUB`'s prod and comp masks are both non-empty. Turn onsets are
    run-relative; `_turn_masks` lifts each segment's TRs by `frame_onset`.

    Sets up the bare pair only; tests that also need a *segmented* run write their
    own events (see `_add_block_segmented_run`).
    """
    tree.add_participants({SUB: DYAD, PARTNER: DYAD})
    half = (n_trs // 2) * tr
    sub_turn = {"trial_type": "turn_speaker", "onset": 0.0, "duration": half}
    partner_turn = {
        "trial_type": "turn_speaker",
        "onset": half,
        "duration": n_trs * tr - half,
    }
    tree.add_events(sub=SUB, task=task, run=run, rows=[sub_turn])
    tree.add_events(sub=PARTNER, task=task, run=run, rows=[partner_turn])


def _make_encoding(
    tree: BIDSTree,
    features: list[str],
    *,
    confounds: list[str] | None = None,
    tasks: list[str] | None = None,
    bold_space: str = SPACE,
    bold_desc: str = "denoised",
    bids_filters: list[str] | None = None,
    fold_by: str | None = None,
    n_folds: int | Literal["loo"] | None = None,
    split: bool = False,
) -> EncodingTrainer:
    # Default split=False: the split path needs a 2-subject dyad with turn_speaker
    # events, which most fixtures do not set up; split-specific tests opt in.
    return EncodingTrainer(
        config=EncodingConfig(),
        bids_root=tree.root,
        features=features,
        confounds=confounds,
        tasks=tasks if tasks is not None else [TASK],
        bold_space=bold_space,
        bold_desc=bold_desc,
        bids_filters=bids_filters,
        fold_by=fold_by,
        n_folds=n_folds,
        split=split,
    )
