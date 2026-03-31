from enum import StrEnum


class Device(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"


class SurfaceSpace(StrEnum):
    FS_AVERAGE_5 = "fsaverage5"
    FS_AVERAGE_6 = "fsaverage6"


class VolumeSpace(StrEnum):
    MNI_152_NLIN_6_ASYM = "MNI152NLin6Asym"
    MNI_152_NLIN_2009_C_ASYM = "MNI152NLin2009cAsym"
