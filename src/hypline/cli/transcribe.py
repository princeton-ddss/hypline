from pathlib import Path
from typing import Annotated

import typer

from hypline.enums import Device
from hypline.layout import BIDSLayout
from hypline.transcriber import Transcriber, WhisperConfig, WhisperModel

from ._utils import split_csv


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
            help="Extension of the audio files",
        ),
    ] = ".wav",
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
            help="Comma-separated BIDS entity filters (e.g., run-2,run-4,cond-G)",
            show_default=False,
        ),
    ] = None,
):
    """
    Transcribe audio files using a Whisper ASR model.
    """
    resolved_sub_ids = split_csv(sub_ids, param_hint="--sub-ids")
    resolved_bids_filters = split_csv(bids_filters, param_hint="--bids-filters")

    layout = BIDSLayout(bids_root)

    config = WhisperConfig(
        model=model,
        model_dir=model_dir,
        device=device,
    )

    transcriber = Transcriber(
        config,
        layout=layout,
        audio_ext=audio_ext,
        bids_filters=resolved_bids_filters,
    )

    # TODO: warn when no subjects found (pending logging setup)
    resolved_sub_ids = resolved_sub_ids or layout.list.subjects(area="stimuli")

    for sub_id in resolved_sub_ids:
        transcriber.transcribe(sub_id)
