from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

from ._utils import split_csv, subject_log

app = typer.Typer()


@app.command(name="phonemic")
def generate_phonemic_feature(
    bids_root: Annotated[
        Path,
        typer.Argument(
            help="BIDS dataset root (contains stimuli/, features/, derivatives/)",
            show_default=False,
        ),
    ],
    no_articulatory: Annotated[
        bool,
        typer.Option(
            "--no-articulatory",
            help="Do not use articulatory features",
        ),
    ] = False,
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
    desc: Annotated[
        str | None,
        typer.Option(
            "--desc",
            help="""
            Label to tag outputs (alphanumeric), e.g. --desc v2;
            appears as desc-<label> in filenames
            """,
            show_default=False,
        ),
    ] = None,
):
    """Generate phonemic feature from word-level transcripts."""
    from hypline.features.phonemic import PhonemicFeature
    from hypline.layout import BIDSLayout

    resolved_sub_ids = split_csv(sub_ids, param_hint="--sub-ids")
    resolved_bids_filters = split_csv(bids_filters, param_hint="--data-filters")

    layout = BIDSLayout(bids_root)

    feature = PhonemicFeature(
        layout=layout,
        use_articulatory=not no_articulatory,
        bids_filters=resolved_bids_filters,
        desc=desc,
    )

    resolved_sub_ids = resolved_sub_ids or layout.list.subjects(area="stimuli")

    if not resolved_sub_ids:
        logger.warning("No subjects found — nothing to generate")
        return

    for sub_id in resolved_sub_ids:
        with subject_log(bids_root, "featuregen", "phonemic", sub_id=sub_id):
            feature.generate(sub_id)
