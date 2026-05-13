from pathlib import Path
from typing import Annotated

import typer

from ._utils import split_csv

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
            help="Comma-separated BIDS entity filters (e.g., run-2,run-4,cond-G)",
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
    )

    # TODO: warn when no subjects found (pending logging setup)
    resolved_sub_ids = resolved_sub_ids or layout.list.subjects(area="stimuli")

    for sub_id in resolved_sub_ids:
        feature.generate(sub_id)
