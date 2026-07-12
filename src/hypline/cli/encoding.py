from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import typer
from loguru import logger

from hypline.downsample import FeatureDownsampleMethod
from hypline.enums import BoldSpace, Device

from ._utils import run_per_id, split_csv

if TYPE_CHECKING:
    from hypline.layout import BIDSLayout

app = typer.Typer()


def _parse_fold_by(value: str) -> str | None:
    # fold_by is required, so "none" is the explicit opt-out from folding
    return None if value == "none" else value


def _parse_n_folds(value: str | None) -> int | Literal["loo"] | None:
    if value is None:
        return None
    if value == "loo":
        return "loo"
    try:
        return int(value)
    except ValueError:
        raise typer.BadParameter(
            "must be an integer or 'loo'", param_hint="--n-folds"
        ) from None


def _parse_int_csv(value: str | None, *, param_hint: str) -> list[int] | None:
    items = split_csv(value, param_hint=param_hint)
    if items is None:
        return None
    try:
        return [int(v) for v in items]
    except ValueError:
        raise typer.BadParameter("must be integers", param_hint=param_hint) from None


def _parse_float_csv(value: str | None, *, param_hint: str) -> list[float] | None:
    items = split_csv(value, param_hint=param_hint)
    if items is None:
        return None
    try:
        return [float(v) for v in items]
    except ValueError:
        raise typer.BadParameter("must be numbers", param_hint=param_hint) from None


def _parse_test_on(value: str | None) -> list[str] | None:
    """Parse `--test-on` into a bids-filter list.

    Same grammar as `--data-filters` (same-entity OR, different-entity AND), but
    no entities are reserved — analyze has no dedicated selector flags, so
    `--test-on task-opinion` is a legitimate cell selector.
    """
    from hypline.bids import normalize_bids_filters

    items = split_csv(value, param_hint="--test-on")
    if items is None:
        return None
    try:
        return normalize_bids_filters(items)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--test-on") from None


def _resolve_sub(value: str, *, target_sub_id: str, layout: BIDSLayout) -> str:
    """Resolve a `self`/`partner` keyword (relative to `target_sub_id`) or pass through.

    `partner` goes via `layout.partner_of`, which owns the clean-pair invariant.
    """
    if value == "self":
        return target_sub_id
    if value == "partner":
        try:
            return layout.partner_of(target_sub_id)
        except (KeyError, ValueError) as exc:
            raise typer.BadParameter(str(exc)) from None
    return value


@app.command(name="train")
def train(
    bids_root: Annotated[
        Path,
        typer.Argument(
            help="BIDS dataset root (contains derivatives/, features/, results/)",
            show_default=False,
        ),
    ],
    tasks: Annotated[
        str,
        typer.Option(
            "--tasks",
            help="Comma-separated task labels to train on (e.g., neutral,opinion)",
            show_default=False,
        ),
    ],
    features: Annotated[
        str,
        typer.Option(
            "--features",
            help="""
            Comma-separated feature refs as <kind>[-<desc>] (e.g.,
            semantic-gpt3,phonemic-noArt); each ref becomes its own ridge band
            """,
            show_default=False,
        ),
    ],
    desc: Annotated[
        str,
        typer.Option(
            "--desc",
            help="""
            Variant label for this model (alphanumeric); output lands under
            results/sub-XX/encodingModel-<desc>/ (e.g., --desc v1)
            """,
            show_default=False,
        ),
    ],
    fold_by: Annotated[
        str,
        typer.Option(
            "--fold-by",
            help="""
            Cross-validation grouping axis: a BIDS entity to fold on (e.g.,
            run), or 'none' to fit a single model without folding
            """,
            show_default=False,
        ),
    ],
    n_folds: Annotated[
        str | None,
        typer.Option(
            "--n-folds",
            help="""
            Number of cross-validation folds: an integer (>=2) or 'loo'.
            Required with --fold-by <entity>; omit when --fold-by none
            """,
            show_default=False,
        ),
    ] = None,
    confounds: Annotated[
        str | None,
        typer.Option(
            "--confounds",
            help="""
            Comma-separated confound refs as <kind>[-<desc>] (e.g.,
            phonemic-onset,phonemic-rate); all share one ridge band. Must be
            saved at TR level, one row per TR per segment
            """,
            show_default=False,
        ),
    ] = None,
    bold_space: Annotated[
        BoldSpace,
        typer.Option(
            "--bold-space",
            help="BOLD data space to train on",
        ),
    ] = BoldSpace.MNI_152_NLIN_2009_C_ASYM,
    bold_desc: Annotated[
        str,
        typer.Option(
            "--bold-desc",
            help="BOLD desc entity to select the input runs",
        ),
    ] = "denoised",
    downsample: Annotated[
        FeatureDownsampleMethod,
        typer.Option(
            "--downsample",
            help="Feature-to-TR downsampling method",
        ),
    ] = "mean",
    delays: Annotated[
        str | None,
        typer.Option(
            "--delays",
            help="""
            Comma-separated FIR delays in TRs; omit for the default 0,1,2,3,4,5
            """,
            show_default=False,
        ),
    ] = None,
    alphas: Annotated[
        str | None,
        typer.Option(
            "--alphas",
            help="""
            Comma-separated ridge alphas to search (e.g., 1,10,100); omit for the
            default 1-1e12 log grid (13 points)
            """,
            show_default=False,
        ),
    ] = None,
    device: Annotated[
        Device,
        typer.Option(
            "--device",
            help="Compute device for the fit",
        ),
    ] = Device.CPU,
    sub_ids: Annotated[
        str | None,
        typer.Option(
            "--sub-ids",
            help="Comma-separated subject IDs to train (e.g., 01,02); omit for all",
            show_default=False,
        ),
    ] = None,
    bids_filters: Annotated[
        str | None,
        typer.Option(
            "--data-filters",
            help="""
            Comma-separated BIDS entity filters; same-entity values OR'd, different
            entities AND'd (e.g., run-2,run-4,cond-G → (run=2 OR run=4) AND cond=G)
            """,
            show_default=False,
        ),
    ] = None,
    no_split: Annotated[
        bool,
        typer.Option(
            "--no-split",
            help="Fit one model over all screens instead of separate prod/comp models",
        ),
    ] = False,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Overwrite existing outputs (default skips them)",
        ),
    ] = False,
):
    """Fit a voxelwise ridge encoding model per subject, writing to results/."""
    from hypline.encoding import EncodingConfig, EncodingTrainer, write_artifact
    from hypline.layout import BIDSLayout

    _fold_by = _parse_fold_by(fold_by)
    _n_folds = _parse_n_folds(n_folds)
    # fold_by and n_folds are paired both-or-neither. --fold-by is always given
    # (required), so anchor the check on whether an entity was named vs "none".
    if (_fold_by is None) != (_n_folds is None):
        raise typer.BadParameter(
            "give --n-folds with --fold-by <entity>, and omit it with --fold-by none",
            param_hint="--fold-by/--n-folds",
        )

    _tasks = split_csv(tasks, param_hint="--tasks")
    _features = split_csv(features, param_hint="--features")
    # Required options: split_csv returns None only on a None input, which typer
    # rules out. Assert to narrow for the trainer's list[str] params.
    assert _tasks is not None and _features is not None

    _confounds = split_csv(confounds, param_hint="--confounds")
    _bids_filters = split_csv(bids_filters, param_hint="--data-filters")
    _delays = _parse_int_csv(delays, param_hint="--delays")
    _alphas = _parse_float_csv(alphas, param_hint="--alphas")

    _sub_ids = split_csv(sub_ids, param_hint="--sub-ids")
    _sub_ids = _sub_ids or BIDSLayout(bids_root).list.subjects(area="fmriprep")
    if not _sub_ids:
        logger.warning("No subjects found — nothing to train")
        return

    # Omit unpassed flags so they keep their EncodingConfig defaults (delays, alphas)
    config_overrides: dict = {"device": device}
    if _delays is not None:
        config_overrides["delays"] = _delays
    if _alphas is not None:
        config_overrides["alphas"] = _alphas
    config = EncodingConfig(**config_overrides)

    trainer = EncodingTrainer(
        config=config,
        bids_root=bids_root,
        features=_features,
        tasks=_tasks,
        confounds=_confounds,
        bold_space=bold_space.value,
        bold_desc=bold_desc,
        downsample=downsample,
        bids_filters=_bids_filters,
        fold_by=_fold_by,
        n_folds=_n_folds,
        split=not no_split,
    )

    layout = BIDSLayout(bids_root)

    def _train_sub(sub_id: str) -> None:
        out = layout.path.result(sub=sub_id, kind="encodingModel", desc=desc)
        if not force and out.path.exists():
            logger.info("sub-{} result exists — skipping", sub_id)
            return
        artifact = trainer.train(sub_id)
        write_artifact(artifact, out.path)

    run_per_id(
        bids_root,
        "encoding",
        "train",
        id_key="sub",
        id_values=_sub_ids,
        task=_train_sub,
    )


@app.command(name="analyze")
def analyze(
    bids_root: Annotated[
        Path,
        typer.Argument(
            help="BIDS dataset root (contains derivatives/, features/, results/)",
            show_default=False,
        ),
    ],
    target_sub: Annotated[
        str,
        typer.Option(
            "--target-sub",
            help="Subject whose actual BOLD and prod/comp turns are scored against",
            show_default=False,
        ),
    ],
    model_sub: Annotated[
        str,
        typer.Option(
            "--model-sub",
            help="""
            Subject whose trained model is loaded: an ID, or 'self'/'partner'
            (relative to --target-sub)
            """,
            show_default=False,
        ),
    ],
    model_desc: Annotated[
        str,
        typer.Option(
            "--model-desc",
            help="The --desc passed to `encoding train` (its encodingModel-<desc> tag)",
            show_default=False,
        ),
    ],
    desc: Annotated[
        str,
        typer.Option(
            "--desc",
            help="""
            Variant label for this eval (alphanumeric); output lands under
            results/sub-<target>/encodingEval-<desc>/ (e.g., --desc v1)
            """,
            show_default=False,
        ),
    ],
    source_sub: Annotated[
        str,
        typer.Option(
            "--source-sub",
            help="""
            Subject whose regressors drive the prediction: an ID, or 'self'/'partner'
            (relative to --target-sub)
            """,
        ),
    ] = "self",
    test_on: Annotated[
        str | None,
        typer.Option(
            "--test-on",
            help="""
            Comma-separated BIDS entity filters naming which cells to score (e.g.,
            run-6, or run-6,run-8); same-entity values OR'd, different entities
            AND'd. Omit to score each model's out-of-sample cells
            """,
            show_default=False,
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Overwrite existing outputs (default skips them)",
        ),
    ] = False,
):
    """Score a model's cross-subject predictions against a target's BOLD, per role."""
    from hypline.encoding import EncodingPredictor, save_eval
    from hypline.layout import BIDSLayout

    _test_on = _parse_test_on(test_on)

    layout = BIDSLayout(bids_root)

    def _analyze_target(sub_id: str) -> None:
        out = layout.path.result(sub=sub_id, kind="encodingEval", desc=desc, ext=".nc")
        if not force and out.path.exists():
            logger.info("sub-{} result exists — skipping", sub_id)
            return
        # `self`/`partner` are relative to this target, so resolve inside the loop
        model_sub_id = _resolve_sub(model_sub, target_sub_id=sub_id, layout=layout)
        source_sub_id = _resolve_sub(source_sub, target_sub_id=sub_id, layout=layout)
        predictor = EncodingPredictor.load(
            bids_root=bids_root, sub_id=model_sub_id, desc=model_desc
        )
        ds = predictor.analyze(
            source_sub_id=source_sub_id, target_sub_id=sub_id, test_on=_test_on
        )
        # save_eval writes straight to `path` (unlike write_artifact, it does not
        # create parents), so make the encodingEval-<desc>/ dir here
        out.path.parent.mkdir(parents=True, exist_ok=True)
        save_eval(ds, out.path)

    run_per_id(
        bids_root,
        "encoding",
        "analyze",
        id_key="sub",
        id_values=[target_sub],
        task=_analyze_target,
    )
