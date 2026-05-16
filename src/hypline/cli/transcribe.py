from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

from hypline.enums import Device, WhisperModel

from ._utils import split_csv, subject_log


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
            help="""
            Comma-separated BIDS entity filters; same-entity values OR'd, different
            entities AND'd (e.g., run-2,run-4,cond-G → (run=2 OR run=4) AND cond=G)
            """,
            show_default=False,
        ),
    ] = None,
):
    """
    Transcribe audio files using a Whisper ASR model.
    """
    from hypline.layout import BIDSLayout
    from hypline.transcriber import Transcriber, WhisperConfig

    resolved_sub_ids = split_csv(sub_ids, param_hint="--sub-ids")
    resolved_bids_filters = split_csv(bids_filters, param_hint="--data-filters")

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

    resolved_sub_ids = resolved_sub_ids or layout.list.subjects(area="stimuli")

    if not resolved_sub_ids:
        logger.warning("No subjects found — nothing to transcribe")
        return

    for sub_id in resolved_sub_ids:
        with subject_log(bids_root, "transcribe", sub_id=sub_id):
            transcriber.transcribe(sub_id)
