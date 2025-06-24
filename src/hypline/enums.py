from enum import Enum


class CompCorMethod(Enum):
    ANATOMICAL = "aCompCor"
    TEMPORAL = "tCompCor"
    MEAN = "Mean"


class CompCorMask(Enum):
    CSF = "CSF"
    WM = "WM"
    COMBINED = "combined"


class SurfaceSpace(Enum):
    FS_AVERAGE_5 = "fsaverage5"
    FS_AVERAGE_6 = "fsaverage6"


class VolumeSpace(Enum):
    MNI_152_NLIN_6_ASYM = "MNI152NLin6Asym"
    MNI_152_NLIN_2009_C_ASYM = "MNI152NLin2009cAsym"
