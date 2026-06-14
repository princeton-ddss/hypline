from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

from ._utils import run_per_id, split_csv

app = typer.Typer()


@app.command(name="phonemic")
def generate_phonemic_confound(
    bids_root: Annotated[
        Path,
        typer.Argument(
            help="BIDS dataset root (contains stimuli/, features/, derivatives/)",
            show_default=False,
        ),
    ],
    dyad_ids: Annotated[
        str | None,
        typer.Option(
            help="Comma-separated dyad IDs to process (e.g., 01,02); omit for all",
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
    """Generate phonemic confounds (onset, rate) from phonemic feature files."""
    from hypline.confounds.phonemic import PhonemicConfound
    from hypline.layout import BIDSLayout

    _dyad_ids = split_csv(dyad_ids, param_hint="--dyad-ids")
    _bids_filters = split_csv(bids_filters, param_hint="--data-filters")

    _dyad_ids = _dyad_ids or BIDSLayout(bids_root).list.dyads(area="features")
    if not _dyad_ids:
        logger.warning("No dyads found — nothing to generate")
        return

    confound = PhonemicConfound(
        bids_root=bids_root,
        bids_filters=_bids_filters,
        force=force,
    )

    run_per_id(
        bids_root,
        "confoundgen",
        "phonemic",
        id_key="dyad",
        id_values=_dyad_ids,
        task=confound.generate,
    )


@app.command(name="semantic")
def generate_semantic_confound(
    bids_root: Annotated[
        Path,
        typer.Argument(
            help="BIDS dataset root (contains stimuli/, features/, derivatives/)",
            show_default=False,
        ),
    ],
    dyad_ids: Annotated[
        str | None,
        typer.Option(
            help="Comma-separated dyad IDs to process (e.g., 01,02); omit for all",
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
    """Generate semantic confounds (onset, rate) from semantic feature files."""
    from hypline.confounds.semantic import SemanticConfound
    from hypline.layout import BIDSLayout

    _dyad_ids = split_csv(dyad_ids, param_hint="--dyad-ids")
    _bids_filters = split_csv(bids_filters, param_hint="--data-filters")

    _dyad_ids = _dyad_ids or BIDSLayout(bids_root).list.dyads(area="features")
    if not _dyad_ids:
        logger.warning("No dyads found — nothing to generate")
        return

    confound = SemanticConfound(
        bids_root=bids_root,
        bids_filters=_bids_filters,
        force=force,
    )

    run_per_id(
        bids_root,
        "confoundgen",
        "semantic",
        id_key="dyad",
        id_values=_dyad_ids,
        task=confound.generate,
    )
