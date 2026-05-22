import warnings

# pyannote.audio warns loudly at import time when torchcodec fails to load
# its native lib. We don't hit pyannote's file-decode path — whisperx's
# load_audio shells out to FFmpeg directly and feeds in-memory tensors to
# pyannote's Silero VAD — so the warning is noise.
warnings.filterwarnings(
    "ignore",
    message=r"(?s).*torchcodec is not installed correctly.*",
    category=UserWarning,
)

from ._version import __version__  # noqa: E402,F401
from .cli import app  # noqa: E402
from .io import (  # noqa: E402
    read_confound,
    read_confound_metadata,
    read_feature,
    read_feature_metadata,
    save_confound,
    save_feature,
)

__all__ = [
    "read_confound",
    "read_confound_metadata",
    "read_feature",
    "read_feature_metadata",
    "save_confound",
    "save_feature",
]


def main():
    app()
