import pytest

from hypline.enums import SurfaceSpace, VolumeSpace
from hypline.regression import ConfoundRegression


@pytest.mark.parametrize(
    "subject_id, session_name, task_name, data_space, expected_output",
    [
        (
            "001",
            "1",
            "Black",
            VolumeSpace.MNI_152_NLIN_2009_C_ASYM,
            "sub-001/**/sub-001*ses-1*task-Black*space-MNI152NLin2009cAsym*bold.nii.gz",
        ),
        (
            "001",
            "1",
            "Black",
            VolumeSpace.MNI_152_NLIN_6_ASYM,
            "sub-001/**/sub-001*ses-1*task-Black*space-MNI152NLin6Asym*bold.nii.gz",
        ),
        (
            "001",
            "1",
            "Black",
            SurfaceSpace.FS_AVERAGE_5,
            "sub-001/**/sub-001*ses-1*task-Black*space-fsaverage5*bold.func.gii",
        ),
        (
            "001",
            "1",
            "Black",
            SurfaceSpace.FS_AVERAGE_6,
            "sub-001/**/sub-001*ses-1*task-Black*space-fsaverage6*bold.func.gii",
        ),
        (
            "001",
            "*",
            "Black",
            SurfaceSpace.FS_AVERAGE_6,
            "sub-001/**/sub-001*task-Black*space-fsaverage6*bold.func.gii",
        ),
        (
            "001",
            "*",
            "*",
            SurfaceSpace.FS_AVERAGE_6,
            "sub-001/**/sub-001*space-fsaverage6*bold.func.gii",
        ),
    ],
)
def test_compose_glob_pattern_for_bold(
    subject_id: str,
    session_name: str,
    task_name: str,
    data_space: VolumeSpace | SurfaceSpace,
    expected_output: str,
):
    output = ConfoundRegression._compose_glob_pattern_for_bold(
        subject_id, session_name, task_name, data_space
    )
    assert output == expected_output
