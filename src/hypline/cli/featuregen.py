from pathlib import Path
from typing import Annotated

import typer

from ._utils import split_csv

app = typer.Typer()


@app.command(name="phonemic")
def generate_phonemic_feature(
    input_dir: Annotated[
        Path,
        typer.Argument(
            help="Directory containing word-level transcripts (CSV files)",
            show_default=False,
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Argument(
            help="Directory to store phonemic feature data (Parquet files)",
            show_default=False,
        ),
    ],
    sub_ids: Annotated[
        str | None,
        typer.Option(
            help="""
            Comma-separated subject IDs to process (e.g., 01,02); omit to process all
            """,
            show_default=False,
        ),
    ] = None,
    bids_filters: Annotated[
        str | None,
        typer.Option(
            help="Comma-separated BIDS entity filters (e.g., run-2,run-4,cond-G)",
            show_default=False,
        ),
    ] = None,
    no_articulatory: Annotated[
        bool,
        typer.Option(
            "--no-articulatory",
            help="Do not use articulatory features",
        ),
    ] = False,
):
    """Generate phonemic feature from word-level transcripts."""
    from hypline.bids import BIDSPath
    from hypline.features.phonemic import PhonemicFeature

    resolved_sub_ids = split_csv(sub_ids, param_hint="--sub-ids")
    resolved_bids_filters = split_csv(bids_filters, param_hint="--bids-filters")

    feature = PhonemicFeature(
        input_dir,
        output_dir,
        use_articulatory=not no_articulatory,
        bids_filters=resolved_bids_filters,
    )

    resolved_sub_ids = resolved_sub_ids or list(
        {
            BIDSPath(f).entities["sub"]
            for f in input_dir.iterdir()
            if f.is_file() and "sub" in BIDSPath(f).entities
        }
    )

    for sub_id in resolved_sub_ids:
        feature.generate(sub_id)
