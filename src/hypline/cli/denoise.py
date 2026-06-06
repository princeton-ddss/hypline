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
    columns: Annotated[
        str | None,
        typer.Option(
            "--columns",
            help="""
            Comma-separated fmriprep confound columns: exact tsv names (e.g.,
            trans_x,rot_x) plus group prefixes that expand to all matches
            (cosine, motion_outlier)
            """,
            show_default=False,
        ),
    ] = None,
    compcor: Annotated[
        str | None,
        typer.Option(
            "--compcor",
            help="""
            Comma-separated CompCor selectors type:mask:n (e.g., a:CSF:5 = top-5
            aCompCor in CSF; t::10 = top-10 tCompCor). type a=anatomical (mask
            required), t=temporal (no mask); n = top-N int or variance fraction (0-1)
            """,
            show_default=False,
        ),
    ] = None,
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
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Overwrite existing outputs (default skips them)",
        ),
    ] = False,
):
    """Regress fmriprep confounds out of preprocessed BOLD, writing desc-clean."""
    from hypline.denoise import Denoiser

    _columns = split_csv(columns, param_hint="--columns") or []
    _compcor = split_csv(compcor, param_hint="--compcor") or []
    if not _columns and not _compcor:
        raise typer.BadParameter(
            "at least one of --columns or --compcor must be given",
            param_hint="--columns/--compcor",
        )

    _sub_ids = split_csv(sub_ids, param_hint="--sub-ids")
    _bids_filters = split_csv(bids_filters, param_hint="--data-filters")

    denoiser = Denoiser(
        bids_root=bids_root,
        space=space.value,
        columns=_columns,
        compcor=_compcor,
        bids_filters=_bids_filters,
        force=force,
    )

    _sub_ids = _sub_ids or denoiser._layout.list.subjects(area="fmriprep")

    if not _sub_ids:
        logger.warning("No subjects found — nothing to denoise")
        return

    run_per_subject(bids_root, "denoise", sub_ids=_sub_ids, task=denoiser.denoise)
