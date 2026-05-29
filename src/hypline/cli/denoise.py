from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

from hypline.enums import BoldSpace

from ._utils import run_per_subject, split_csv

app = typer.Typer()


@app.command(name="denoise")
def denoise(
    bids_root: Annotated[
        Path,
        typer.Argument(
            help="BIDS dataset root (contains stimuli/, features/, derivatives/)",
            show_default=False,
        ),
    ],
    confounds: Annotated[
        str,
        typer.Option(
            "--confounds",
            help="""
            Comma-separated confound refs <kind>[-<desc>] to regress out
            (e.g. fmriprep-minimal,phonemic)
            """,
            show_default=False,
        ),
    ],
    space: Annotated[
        BoldSpace,
        typer.Option(
            "--space",
            help="BOLD data space to clean",
        ),
    ] = BoldSpace.MNI_152_NLIN_2009_C_ASYM,
    sub_ids: Annotated[
        str | None,
        typer.Option(
            help="Comma-separated subject IDs to process (e.g., 01,02); omit for all",
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
):
    """Regress out confounds from preprocessed BOLD, writing desc-clean in place."""
    from hypline.denoise import Denoiser

    _confounds = split_csv(confounds, param_hint="--confounds")
    if not _confounds:
        raise typer.BadParameter(
            "at least one confound ref required", param_hint="--confounds"
        )

    _sub_ids = split_csv(sub_ids, param_hint="--sub-ids")
    _bids_filters = split_csv(bids_filters, param_hint="--data-filters")

    denoiser = Denoiser(
        bids_root=bids_root,
        space=space.value,
        confounds=_confounds,
        bids_filters=_bids_filters,
    )

    _sub_ids = _sub_ids or denoiser._layout.list.subjects(area="fmriprep")

    if not _sub_ids:
        logger.warning("No subjects found — nothing to denoise")
        return

    run_per_subject(bids_root, "denoise", sub_ids=_sub_ids, task=denoiser.denoise)
