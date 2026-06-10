from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

from hypline.enums import Device, WhisperModel

from ._utils import run_per_id, split_csv


def transcribe(
    bids_root: Annotated[
        Path,
        typer.Argument(
            help="BIDS dataset root (contains stimuli/, features/, derivatives/)",
            show_default=False,
        ),
    ],
    audio_ext: Annotated[
        str,
        typer.Option(
            help="Extension of the audio files (e.g., .wav)",
            show_default=False,
        ),
    ],
    model: Annotated[
        WhisperModel,
        typer.Option(
            help="Whisper automatic speech recognition (ASR) model to use",
        ),
    ] = WhisperModel.LARGE_V2,
    model_dir: Annotated[
        Path | None,
        typer.Option(
            help="""
            Directory to find the model weights;
            if not found, downloads them here
            """,
            show_default="system cache directory",
        ),
    ] = None,
    device: Annotated[
        Device,
        typer.Option(
            help="Hardware target for running the model",
        ),
    ] = Device.CPU,
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
    """
    Transcribe audio files using a Whisper ASR model.
    """
    from hypline.transcribe import Transcriber, WhisperConfig

    _dyad_ids = split_csv(dyad_ids, param_hint="--dyad-ids")
    _bids_filters = split_csv(bids_filters, param_hint="--data-filters")

    config = WhisperConfig(
        model=model,
        model_dir=model_dir,
        device=device,
    )

    transcriber = Transcriber(
        config,
        bids_root=bids_root,
        audio_ext=audio_ext,
        bids_filters=_bids_filters,
        force=force,
    )

    _dyad_ids = _dyad_ids or transcriber._layout.list.dyads(area="stimuli")

    if not _dyad_ids:
        logger.warning("No dyads found — nothing to transcribe")
        return

    run_per_id(
        bids_root,
        "transcribe",
        id_key="dyad",
        id_values=_dyad_ids,
        task=transcriber.transcribe,
    )
