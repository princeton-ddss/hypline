import shutil
import tempfile
from enum import Enum
from pathlib import Path

import polars as pl
from pydantic import BaseModel, field_validator

from hypline.utils import find_files, validate_dirs


class WhisperModel(str, Enum):
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE_V2 = "large-v2"
    LARGE_V3 = "large-v3"


class Device(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"


class TranscriberConfig(BaseModel):
    model: WhisperModel = WhisperModel.LARGE_V2
    model_dir: Path | None = None
    device: Device = Device.CPU
    batch_size: int | None = None

    @field_validator("model_dir", mode="after")
    @classmethod
    def _default_model_dir(cls, v: Path | None) -> Path:
        if v is None:
            v = Path(tempfile.gettempdir()) / "hypline" / "whisperx"
            v.mkdir(parents=True, exist_ok=True)
            return v
        elif not v.is_dir():
            raise ValueError(f"model_dir does not exist: {v}")
        return v


class Transcriber:
    def __init__(self, config: TranscriberConfig):
        import torch
        import whisperx

        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "ffmpeg is required but not found on PATH. "
                "Install it with: brew install ffmpeg (macOS) "
                "or apt install ffmpeg (Ubuntu/Debian)"
            )

        if config.device is Device.CUDA and not torch.cuda.is_available():
            raise RuntimeError("CUDA is requested but not available")

        assert config.model_dir is not None  # For type inference

        self.config = config

        self.model = whisperx.load_model(
            whisper_arch=config.model.value,
            device=config.device.value,
            compute_type="float16" if config.device is Device.CUDA else "int8",
            download_root=str(config.model_dir),
            vad_method="silero",  # Good enough as we need no diarization
            language="en",
        )

        self.align_model, self.align_metadata = whisperx.load_align_model(
            language_code="en",
            device=config.device.value,
            model_dir=str(config.model_dir),
        )

    def transcribe(
        self,
        input_dir: Path,
        output_dir: Path,
        audio_ext: str,
        *,
        bids_filters: list[str] | None = None,
    ):
        import whisperx

        validate_dirs(input_dir, output_dir)

        audio_files = find_files(input_dir, audio_ext, bids_filters=bids_filters)

        batch_size = self.config.batch_size
        if batch_size is None:
            batch_size = 16 if self.config.device is Device.CUDA else 1

        for audio_file in audio_files:
            audio = whisperx.load_audio(str(audio_file))

            result = self.model.transcribe(audio, batch_size=batch_size)
            result = whisperx.align(
                transcript=result["segments"],
                model=self.align_model,
                align_model_metadata=self.align_metadata,
                audio=audio,
                device=self.config.device.value,
                return_char_alignments=False,
            )

            df = pl.DataFrame(result["word_segments"]).rename(
                {
                    "start": "start_time",
                    "end": "end_time",
                    "score": "confidence_score",
                }
            )
            output_file = output_dir / f"{audio_file.stem}.csv"
            df.write_csv(output_file)
