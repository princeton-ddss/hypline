from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

from hypline.enums import Device

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
    from hypline.layout import BIDSLayout

    _dyad_ids = split_csv(dyad_ids, param_hint="--dyad-ids")
    _bids_filters = split_csv(bids_filters, param_hint="--data-filters")

    _dyad_ids = _dyad_ids or BIDSLayout(bids_root).list.dyads(area="stimuli")
    if not _dyad_ids:
        logger.warning("No dyads found — nothing to generate")
        return

    feature = PhonemicFeature(
        bids_root=bids_root,
        use_articulatory=not no_articulatory,
        bids_filters=_bids_filters,
        desc=desc,
        force=force,
    )

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


@app.command(name="semantic")
def generate_semantic_feature(
    bids_root: Annotated[
        Path,
        typer.Argument(
            help="BIDS dataset root (contains stimuli/, features/, derivatives/)",
            show_default=False,
        ),
    ],
    model: Annotated[
        str,
        typer.Option(
            "--model",
            help="Hugging Face causal-LM id (e.g., gpt2-xl, meta-llama/Llama-3.2-1B)",
            show_default=False,
        ),
    ],
    model_dir: Annotated[
        Path | None,
        typer.Option(
            "--model-dir",
            help="Cache dir for downloaded weights",
            show_default="~/.cache/hypline/huggingface",
        ),
    ] = None,
    device: Annotated[
        Device,
        typer.Option(
            help="Hardware target for running the model",
        ),
    ] = Device.CPU,
    layer: Annotated[
        int | None,
        typer.Option(
            "--layer",
            help="Hidden layer index in 0..num_hidden_layers; omit for middle layer",
            show_default=False,
        ),
    ] = None,
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
            help="Do not also generate semantic confounds from the new features",
        ),
    ] = False,
):
    """Generate contextual word embeddings from any Hugging Face causal LM."""
    from hypline.confounds.semantic import SemanticConfound
    from hypline.features.semantic import HFModelConfig, SemanticFeature
    from hypline.layout import BIDSLayout

    _dyad_ids = split_csv(dyad_ids, param_hint="--dyad-ids")
    _bids_filters = split_csv(bids_filters, param_hint="--data-filters")

    # Discover first: SemanticFeature.__init__ downloads the model,
    # so skip constructing it when there is nothing to generate
    _dyad_ids = _dyad_ids or BIDSLayout(bids_root).list.dyads(area="stimuli")
    if not _dyad_ids:
        logger.warning("No dyads found — nothing to generate")
        return

    config = HFModelConfig(
        name=model,
        model_dir=model_dir,
        device=device,
        layer=layer,
    )

    feature = SemanticFeature(
        config,
        bids_root=bids_root,
        bids_filters=_bids_filters,
        desc=desc,
        force=force,
    )

    if skip_confoundgen:
        task = feature.generate
    else:
        confound = SemanticConfound(
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
        "semantic",
        id_key="dyad",
        id_values=_dyad_ids,
        task=task,
    )
