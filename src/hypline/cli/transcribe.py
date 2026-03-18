from pathlib import Path
from typing import Annotated

import typer

from hypline.transcriber import Device, Transcriber, TranscriberConfig, WhisperModel


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
            help="Directory to store word-level transcripts",
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
    bids_filter: Annotated[
        list[str] | None,
        typer.Option(
            help="BIDS entity tag to filter (e.g., run-5; repeatable)",
            show_default=False,
        ),
    ] = None,
):
    """
    Transcribe audio files using a Whisper ASR model.
    """
    config = TranscriberConfig(
        model=model,
        model_dir=model_dir,
        device=device,
    )

    transcriber = Transcriber(config)

    transcriber.transcribe(
        input_dir=input_dir,
        output_dir=output_dir,
        audio_ext=file_ext,
        bids_filters=bids_filter,
    )
