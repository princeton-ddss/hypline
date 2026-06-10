from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

from ._utils import run_per_id, split_csv

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
    desc: Annotated[
        str | None,
        typer.Option(
            "--desc",
            help="""
            Label to tag outputs (alphanumeric), e.g., --desc v2;
            appears as desc-<label> in filenames
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
    skip_confoundgen: Annotated[
        bool,
        typer.Option(
            "--skip-confoundgen",
            help="Do not also generate phonemic confounds from the new features",
        ),
    ] = False,
):
    """Generate phonemic feature from word-level transcripts."""
    from hypline.confounds.phonemic import PhonemicConfound
    from hypline.features.phonemic import PhonemicFeature

    _dyad_ids = split_csv(dyad_ids, param_hint="--dyad-ids")
    _bids_filters = split_csv(bids_filters, param_hint="--data-filters")

    feature = PhonemicFeature(
        bids_root=bids_root,
        use_articulatory=not no_articulatory,
        bids_filters=_bids_filters,
        desc=desc,
        force=force,
    )

    _dyad_ids = _dyad_ids or feature._layout.list.dyads(area="stimuli")

    if not _dyad_ids:
        logger.warning("No dyads found — nothing to generate")
        return

    if skip_confoundgen:
        task = feature.generate
    else:
        confound = PhonemicConfound(
            bids_root=bids_root,
            bids_filters=_bids_filters,
            force=force,
        )

        def task(dyad_id: str):
            feature.generate(dyad_id)
            confound.generate(dyad_id)

    run_per_id(
        bids_root,
        "featuregen",
        "phonemic",
        id_key="dyad",
        id_values=_dyad_ids,
        task=task,
    )
