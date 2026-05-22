from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

from ._utils import split_csv, subject_log

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
    """Generate phonemic confounds (onset, rate) from phonemic feature files."""
    from hypline.confounds.phonemic import PhonemicConfound

    _sub_ids = split_csv(sub_ids, param_hint="--sub-ids")
    _bids_filters = split_csv(bids_filters, param_hint="--data-filters")

    confound = PhonemicConfound(
        bids_root=bids_root,
        bids_filters=_bids_filters,
    )

    _sub_ids = _sub_ids or confound._layout.list.subjects(area="stimuli")

    if not _sub_ids:
        logger.warning("No subjects found — nothing to generate")
        return

    for sub_id in _sub_ids:
        with subject_log(bids_root, "confoundgen", "phonemic", sub_id=sub_id):
            confound.generate(sub_id)
