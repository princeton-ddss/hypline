from typing import Literal

import polars as pl

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


def _two_run_dyad(tree: BIDSTree) -> None:
    """A 2-subject dyad with two conv runs — enough for a length-2 OOS fold set.

    Both SUB and PARTNER get BOLD (features are dyad-keyed, so shared), so the
    same dyad can be analyzed with either as target — the cross-subject case.

    Lays down only the feature layer every caller needs. Tests that also exercise
    the feature+confound merge add the optional confound layer via
    `_add_phonemic_confound`. Feature values are filler: callers assert on X/band
    structure, never on the numbers.
    """
    feature_df = pl.DataFrame(
        {
            "start_time": [2.0 * i for i in range(DEFAULT_BOLD_N_TRS)],
            "feature": [[float(i + 1)] for i in range(DEFAULT_BOLD_N_TRS)],
        },
        schema={"start_time": pl.Float64, "feature": pl.Array(pl.Float64, 1)},
    )
    for run in ("1", "2"):
        _add_dyad_turns(tree, run=run)
        for sub in (SUB, PARTNER):
            tree.add_bold(
                sub=sub, task=TASK, space=SPACE, run=run, tr=2.0, desc="denoised"
            )
        tree.add_feature(dyad=DYAD, task=TASK, kind="mfcc", run=run, df=feature_df)


def _add_phonemic_confound(tree: BIDSTree) -> None:
    """Lay down a `phonemic-onset` confound on both conv runs of `DYAD`.

    Peer to `_two_run_dyad`; call it after to exercise the feature+confound merge
    path. Pairs with `confounds=["phonemic-onset"]` in `_make_encoding`.
    """
    # Confounds are TR-level: one row per TR (spaced by the 2.0 TR), no downsample.
    confound_df = pl.DataFrame(
        {
            "start_time": [2.0 * i for i in range(DEFAULT_BOLD_N_TRS)],
            "confound": [[float(i)] for i in range(DEFAULT_BOLD_N_TRS)],
        },
        schema={"start_time": pl.Float64, "confound": pl.Array(pl.Float64, 1)},
    )
    for run in ("1", "2"):
        tree.add_confound(
            dyad=DYAD,
            task=TASK,
            kind="phonemic",
            run=run,
            desc="onset",
            df=confound_df,
        )
