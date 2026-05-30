import shutil
import tempfile
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, field_validator

from hypline.bids import normalize_bids_filters, validate_extension
from hypline.enums import Device, WhisperModel
from hypline.io import skip_existing
from hypline.layout import BIDSLayout


class WhisperConfig(BaseModel):
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
    def __init__(
        self,
        config: WhisperConfig,
        *,
        bids_root: str | Path,
        audio_ext: str,
        bids_filters: list[str] | None = None,
        force: bool = False,
    ):
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
        self._layout = BIDSLayout(bids_root)

        validate_extension(audio_ext)
        self._audio_ext = audio_ext

        self._bids_filters = normalize_bids_filters(
            bids_filters, reserved={"sub", "stim"}
        )

        self._force = force

        self._model = whisperx.load_model(
            whisper_arch=config.model,
            device=config.device,
            compute_type="float16" if config.device is Device.CUDA else "int8",
            download_root=str(config.model_dir),
            vad_method="silero",  # Good enough as we need no diarization
            language="en",
        )

        self._align_model, self._align_metadata = whisperx.load_align_model(
            language_code="en",
            device=config.device,
            model_dir=str(config.model_dir),
        )

    def transcribe(self, sub_id: str):
        import polars as pl
        import whisperx

        audio_files = self._layout.find.stimuli(
            sub=sub_id,
            kind="audio",
            ext=self._audio_ext,
            bids_filters=self._bids_filters,
        )

        batch_size = self.config.batch_size
        if batch_size is None:
            batch_size = 16 if self.config.device is Device.CUDA else 1

        for audio_file in audio_files:
            out = self._layout.path.stimulus(
                source=audio_file,
                kind="transcript",
                ext=".csv",
            )
            if skip_existing(out.path, force=self._force):
                continue

            logger.info("Transcribing {}", audio_file.path.name)
            audio = whisperx.load_audio(str(audio_file.path))

            result = self._model.transcribe(audio, batch_size=batch_size)
            result = whisperx.align(
                transcript=result["segments"],
                model=self._align_model,
                align_model_metadata=self._align_metadata,
                audio=audio,
                device=self.config.device,
                return_char_alignments=False,
            )

            df = pl.DataFrame(result["word_segments"]).rename(
                {
                    "start": "start_time",
                    "end": "end_time",
                    "score": "confidence_score",
                }
            )
            out.path.parent.mkdir(parents=True, exist_ok=True)
            df.write_csv(out.path)
            logger.debug("Wrote transcript to {}", out.path)
