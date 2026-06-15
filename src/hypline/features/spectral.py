from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from loguru import logger
from pydantic import BaseModel, field_validator

from hypline.bids import BIDS_ENTITY_VALUE_RE, BIDSPath, normalize_bids_filters
from hypline.bold import load_bold_meta, resolve_bold_image
from hypline.cache import hypline_cache_dir
from hypline.confounds._utils import segment_n_trs
from hypline.downsample import downsample
from hypline.enums import WhisperModel
from hypline.io import skip_existing, write_feature
from hypline.layout import BIDSLayout

if TYPE_CHECKING:
    from transformers import WhisperFeatureExtractor

# Pre-aligning to the run's BOLD TR grid is a producer's prerogative when the
# feature is dense, regular, and event-free: every consumer would bin it
# identically, so deferring (the default) buys nothing. Spectral qualifies.
_DOWNSAMPLE_METHOD = "mean"


class WhisperFeatureExtractorConfig(BaseModel):
    """Loader config for a Whisper `WhisperFeatureExtractor` (log-Mel front-end).

    `model` is a `WhisperModel` because the 30-s chunk geometry and extractor
    attributes are Whisper's contract; the enum is the guard against non-Whisper
    checkpoints. No `device` field — `AutoFeatureExtractor` is a CPU Mel
    transform with no forward pass. SR / n_mels / hop / chunk are extractor-
    derived (not config fields) to avoid drift on a model swap.

    Shares the `huggingface` cache subdir with semantic: both load transformers
    artifacts via `from_pretrained`, so transformers dedupes them.
    """

    model: WhisperModel = WhisperModel.TINY
    model_dir: Path | None = None

    @field_validator("model_dir", mode="after")
    @classmethod
    def _default_model_dir(cls, v: Path | None) -> Path:
        if v is None:
            return hypline_cache_dir("huggingface")
        if not v.is_dir():
            raise ValueError(f"model_dir does not exist: {v}")
        return v


class SpectralFeature:
    def __init__(
        self,
        config: WhisperFeatureExtractorConfig,
        *,
        bids_root: str | Path,
        audio_ext: str,
        bids_filters: list[str] | None = None,
        desc: str | None = None,
        force: bool = False,
    ):
        if desc is not None and not BIDS_ENTITY_VALUE_RE.match(desc):
            raise ValueError(f"Invalid desc: {desc!r}")

        self._config = config
        self._layout = BIDSLayout(bids_root)
        self._audio_ext = audio_ext
        self._bids_filters = normalize_bids_filters(bids_filters, reserved={"dyad"})
        self._desc = desc
        self._force = force

        from transformers import AutoFeatureExtractor

        self._extractor: WhisperFeatureExtractor = AutoFeatureExtractor.from_pretrained(
            f"openai/whisper-{config.model.value}",
            cache_dir=str(config.model_dir),
        )

    def generate(self, dyad_id: str):
        audio_files = self._layout.find.stimuli(
            dyad=dyad_id,
            kind="audio",
            ext=self._audio_ext,
            bids_filters=self._bids_filters,
        )

        # A dyad feature carries no `sub`, but downsampling needs n_trs/TR from a
        # real BOLD. Partners share one simultaneous scan sequence, so their TR
        # and n_trs are identical by construction — any partner stands in (per
        # the phonemic confound). Raises KeyError if the dyad is unmapped.
        first_sub = self._layout.subjects_of(dyad_id)[0]

        for audio_file in audio_files:
            out = self._layout.path.feature(
                source=audio_file, kind="spectral", desc=self._desc
            )
            if skip_existing(out.path, force=self._force):
                continue
            logger.info("Generating spectral features for {}", audio_file.path.name)
            self._generate_one(audio_file, first_sub, out.path)

    def _generate_one(self, source: BIDSPath, first_sub: str, out_path: Path) -> None:
        import whisperx

        extractor = self._extractor
        sampling_rate = extractor.sampling_rate
        hop_s = extractor.hop_length / sampling_rate

        audio = whisperx.load_audio(str(source.path), sr=sampling_rate)

        # Slice by fixed `n_samples` so only the LAST chunk is short and padded.
        # Splitting into N equal pieces makes every chunk a non-30-s length the
        # extractor pads internally, so silence lands between real frames and
        # frame_times (which assume contiguous 30-s chunks) drift after chunk 1.
        chunks = [
            audio[i : i + extractor.n_samples]
            for i in range(0, len(audio), extractor.n_samples)
        ]
        features = extractor(chunks, sampling_rate=sampling_rate, return_tensors="np")
        # `input_features` is (n_chunks, n_mels, nb_max_frames); concatenate
        # along time, then transpose to (n_frames, n_mels).
        mels = np.concatenate(list(features["input_features"]), axis=1).T

        # Trim frames past the true audio duration so the final chunk's padding
        # cannot leak into a bin.
        frame_times = np.arange(mels.shape[0]) * hop_s
        keep = frame_times < len(audio) / sampling_rate
        mels = mels[keep]
        frame_times = frame_times[keep]

        sub_source = source.with_identity("sub", first_sub)
        bold = resolve_bold_image(self._layout, sub_source)
        bold_meta = load_bold_meta(self._layout, bold)
        n_trs = segment_n_trs(source, bold_meta)
        tr = bold_meta.repetition_time

        spectrogram = downsample(
            mels,
            start_times=frame_times,
            n_trs=n_trs,
            repetition_time=tr,
            method=_DOWNSAMPLE_METHOD,
        )

        n_mels = extractor.feature_size
        out_df = pl.DataFrame(
            {
                "start_time": np.arange(n_trs) * tr,
                "feature": spectrogram.tolist(),
            },
            schema={
                "start_time": pl.Float64,
                "feature": pl.Array(pl.Float64, n_mels),
            },
        )
        metadata = {
            "model": self._config.model.value,
            "sampling_rate": sampling_rate,
            "hop_length": extractor.hop_length,
            "n_mels": n_mels,
            "chunk_length": extractor.chunk_length,
            "repetition_time": tr,
            "downsample_method": _DOWNSAMPLE_METHOD,
        }
        write_feature(out_df, out_path, metadata=metadata)
        logger.debug("Wrote spectral feature to {}", out_path)
