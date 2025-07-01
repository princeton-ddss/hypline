import pytest

from hypline.regression import ConfoundRegression


@pytest.mark.parametrize(
    "file, new_file",
    [
        (
            "sub-004_ses-1_task-Black_space-fsaverage6_hemi-L_bold.func.gii",
            "sub-004_ses-1_task-Black_space-fsaverage6_hemi-L_desc-clean_bold.func.gii",
        ),
        (
            "sub-005_ses-2_task-Conv_space-fsaverage6_hemi-R_bold.func.gii",
            "sub-005_ses-2_task-Conv_space-fsaverage6_hemi-R_desc-clean_bold.func.gii",
        ),
        (
            "sub-006_ses-3_task-Conv_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
            "sub-006_ses-3_task-Conv_space-MNI152NLin2009cAsym_desc-clean_bold.nii.gz",
        ),
    ],
)
def test_make_new_filepath(
    # Fixture(s)
    confound_regression: ConfoundRegression,
    # Parameter(s)
    file: str,
    new_file: str,
):
    filepath = confound_regression.fmriprep_dir / "some_intermediate_dir" / file
    new_filepath = confound_regression.output_dir / "some_intermediate_dir" / new_file

    assert confound_regression._make_new_filepath(filepath) == new_filepath
