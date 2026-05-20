import warnings
from importlib.metadata import version

# pyannote.audio warns loudly at import time when torchcodec fails to load
# its native lib. We don't hit pyannote's file-decode path — whisperx's
# load_audio shells out to FFmpeg directly and feeds in-memory tensors to
# pyannote's Silero VAD — so the warning is noise.
warnings.filterwarnings(
    "ignore",
    message=r"(?s).*torchcodec is not installed correctly.*",
    category=UserWarning,
)

from .cli import app  # noqa: E402

__version__ = version("hypline")


def main():
    app()
