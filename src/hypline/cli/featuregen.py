from pathlib import Path
from typing import Annotated

import typer

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
    no_articulatory: Annotated[
        bool,
        typer.Option(
            "--no-articulatory",
            help="Do not use articulatory features",
        ),
    ] = False,
    bids_filters: Annotated[
        list[str] | None,
        typer.Option(
            "--bids-filter",
            help="""
            [Repeatable] Filter input files by BIDS entity
            (e.g., run-5) present in the filenames
            """,
            show_default=False,
        ),
    ] = None,
):
    """Generate phonemic feature from word-level transcripts."""
    from hypline.features.phonemic import PhonemicFeature

    feature = PhonemicFeature()

    feature.generate(
        input_dir,
        output_dir,
        use_articulatory=not no_articulatory,
        bids_filters=bids_filters,
    )
