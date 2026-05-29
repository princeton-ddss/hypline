from enum import StrEnum


class Device(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"


class WhisperModel(StrEnum):
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE_V2 = "large-v2"
    LARGE_V3 = "large-v3"


class SurfaceSpace(StrEnum):
    FS_AVERAGE_5 = "fsaverage5"
    FS_AVERAGE_6 = "fsaverage6"


class VolumeSpace(StrEnum):
    MNI_152_NLIN_6_ASYM = "MNI152NLin6Asym"
    MNI_152_NLIN_2009_C_ASYM = "MNI152NLin2009cAsym"


# Flat union of all BOLD spaces for CLI type hints; typer rejects union types and
# a bare str gives no --help choices. Defined explicitly (not built functionally)
# so type checkers see its members; the assert guards against drift from the two
# source enums at import.
class BoldSpace(StrEnum):
    FS_AVERAGE_5 = "fsaverage5"
    FS_AVERAGE_6 = "fsaverage6"
    MNI_152_NLIN_6_ASYM = "MNI152NLin6Asym"
    MNI_152_NLIN_2009_C_ASYM = "MNI152NLin2009cAsym"


assert {m.value for m in BoldSpace} == {
    m.value for e in (SurfaceSpace, VolumeSpace) for m in e
}, "BoldSpace must equal the union of SurfaceSpace and VolumeSpace"
