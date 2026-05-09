from pathlib import Path
from typing import Annotated

import typer

from hypline.enums import Device
from hypline.transcriber import Transcriber, WhisperConfig, WhisperModel

from ._utils import split_csv


def transcribe(
    input_dir: Annotated[
        Path,
        typer.Argument(
            help="Directory containing audio files",
            show_default=False,
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Argument(
            help="Directory to store word-level transcripts (CSV files)",
            show_default=False,
        ),
    ],
    file_ext: Annotated[
        str,
        typer.Argument(
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
):
    """
    Transcribe audio files using a Whisper ASR model.
    """
    from hypline.bids import BIDSPath

    resolved_sub_ids = split_csv(sub_ids, param_hint="--sub-ids")
    resolved_bids_filters = split_csv(bids_filters, param_hint="--bids-filters")

    config = WhisperConfig(
        model=model,
        model_dir=model_dir,
        device=device,
    )

    transcriber = Transcriber(
        config,
        input_dir=input_dir,
        output_dir=output_dir,
        audio_ext=file_ext,
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
        transcriber.transcribe(sub_id)
