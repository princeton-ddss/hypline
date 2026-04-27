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
    sub_ids: Annotated[
        list[str] | None,
        typer.Option(
            "--sub",
            help="[Repeatable] Subject ID to process (e.g., 01); omit to process all",
            show_default=False,
        ),
    ] = None,
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

    feature = PhonemicFeature(
        input_dir,
        output_dir,
        use_articulatory=not no_articulatory,
        bids_filters=bids_filters,
    )

    resolved_sub_ids = sub_ids or list(
        {
            BIDSPath(f).entities["sub"]
            for f in input_dir.iterdir()
            if f.is_file() and "sub" in BIDSPath(f).entities
        }
    )

    for sub_id in resolved_sub_ids:
        feature.generate(sub_id)
