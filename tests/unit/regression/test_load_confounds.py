from pathlib import Path

import polars as pl
import pytest
from pytest_mock import MockerFixture

from hypline.regression import ConfoundRegression
from hypline.schemas import ModelSpec


@pytest.fixture(scope="function")
def confound_regression(mocker: MockerFixture, tmp_path: Path):
    """
    An instance of `ConfoundRegression` with mock initiation.
    """
    mocker.patch("hypline.regression.logging")
    mocker.patch("hypline.regression.yaml")
    mocker.patch("hypline.regression.Config")
    mocker.patch("hypline.regression.TypeAdapter")
    mocker.patch(
        "hypline.regression.ConfoundRegression._extract_confounds",
        side_effect=lambda *args: args[0],  # Return input data as is
    )

    config_filepath = tmp_path / "config.yaml"
    config_filepath.write_text("")

    config_file = str(config_filepath)
    fmriprep_dir = str(tmp_path)
    confound_regression = ConfoundRegression(config_file, fmriprep_dir)

    return confound_regression


@pytest.mark.parametrize(
    "bold_file, confounds_file",
    [
        (
            "sub-004_ses-1_task-A_run-1_space-fsaverage6_hemi-L_bold.func.gii",
            "sub-004_ses-1_task-A_run-1_desc-confounds_timeseries.tsv",
        ),
        (
            "sub-005_ses-2_task-B_run-3_space-fsaverage6_hemi-R_bold.func.gii",
            "sub-005_ses-2_task-B_run-3_desc-confounds_timeseries.tsv",
        ),
        (
            "sub-006_task-C_run-2_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
            "sub-006_task-C_run-2_desc-confounds_timeseries.tsv",
        ),
    ],
)
def test_load_standard_confounds(
    # Fixture(s)
    confound_regression: ConfoundRegression,
    tmp_path: Path,
    # Parameter(s)
    bold_file: str,
    confounds_file: str,
):
    bold_filepath = tmp_path / bold_file
    bold_filepath.write_text("")

    confounds_filepath = tmp_path / confounds_file
    confounds_filepath.with_suffix(".tsv").write_text("X\tY\tZ\n1\t2\t3")
    confounds_filepath.with_suffix(".json").write_text("")

    model_spec = ModelSpec(confounds=["X"])  # Ignored due to mocking

    confounds_df = confound_regression._load_confounds(bold_filepath, model_spec)

    assert confounds_df.equals(pl.DataFrame({"X": [1], "Y": [2], "Z": [3]}))


@pytest.mark.parametrize(
    "bold_file",
    [
        "sub-004_ses-1_task-Black_space-fsaverage6_hemi-L_bold.func.gii",
        "sub-005_ses-2_task-Conv_space-fsaverage6_hemi-R_bold.func.gii",
        "sub-006_ses-3_task-Conv_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
    ],
)
def test_bold_file_with_no_run_number(
    # Fixture(s)
    confound_regression: ConfoundRegression,
    tmp_path: Path,
    # Parameter(s)
    bold_file: str,
):
    bold_filepath = tmp_path / bold_file

    model_spec = ModelSpec(confounds=["X"])  # Ignored due to mocking

    with pytest.raises(ValueError, match=r"Run number is missing"):
        confound_regression._load_confounds(bold_filepath, model_spec)


@pytest.mark.parametrize(
    "bold_file",
    [
        "sub-004_ses-1_task-Black_run-1_space-fsaverage6_hemi-L_bold.func.gii",
        "sub-005_ses-2_task-Conv_run-3_space-fsaverage6_hemi-R_bold.func.gii",
        "sub-006_ses-3_task-Conv_run-2_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
    ],
)
def test_standard_confounds_file_missing(
    # Fixture(s)
    confound_regression: ConfoundRegression,
    tmp_path: Path,
    # Parameter(s)
    bold_file: str,
):
    bold_filepath = tmp_path / bold_file
    bold_filepath.write_text("")

    model_spec = ModelSpec(confounds=["X"])  # Ignored due to mocking

    with pytest.raises(FileNotFoundError, match=r"Confounds not found"):
        confound_regression._load_confounds(bold_filepath, model_spec)


def test_load_custom_confounds(
    # Fixture(s)
    confound_regression: ConfoundRegression,
    tmp_path: Path,
):
    bold_file = "sub-005_task-A_run-3_space-fsaverage6_hemi-R_bold.func.gii"
    bold_filepath = tmp_path / bold_file
    bold_filepath.write_text("")

    confounds_file = "sub-005_task-A_run-3_desc-confounds_timeseries.tsv"
    confounds_filepath = tmp_path / confounds_file
    confounds_filepath.write_text("X\tY\tZ\n1\t2\t3")
    confounds_filepath.with_suffix(".json").write_text("")

    custom_confounds_file = "sub-005_task-A_run-3_desc-customConfounds_timeseries.tsv"
    custom_confounds_filepath = tmp_path / custom_confounds_file
    custom_confounds_filepath.write_text("A\tB\tC\n10\t20\t30")

    model_spec = ModelSpec(
        confounds=["X"],  # Ignored due to mocking
        custom_confounds=["A"],
    )

    confound_regression._custom_confounds_dir = tmp_path

    confound_df = confound_regression._load_confounds(bold_filepath, model_spec)

    assert confound_df.equals(pl.DataFrame({"X": [1], "Y": [2], "Z": [3], "A": [10]}))


def test_custom_confounds_dir_missing(
    # Fixture(s)
    confound_regression: ConfoundRegression,
    tmp_path: Path,
):
    bold_file = "sub-005_task-A_run-3_space-fsaverage6_hemi-R_bold.func.gii"
    bold_filepath = tmp_path / bold_file
    bold_filepath.write_text("")

    confounds_file = "sub-005_task-A_run-3_desc-confounds_timeseries.tsv"
    confounds_filepath = tmp_path / confounds_file
    confounds_filepath.write_text("X\tY\tZ\n1\t2\t3")
    confounds_filepath.with_suffix(".json").write_text("")

    custom_confounds_file = "sub-005_task-A_run-3_desc-customConfounds_timeseries.tsv"
    custom_confounds_filepath = tmp_path / custom_confounds_file
    custom_confounds_filepath.write_text("A\tB\tC\n10\t20\t30")

    model_spec = ModelSpec(
        confounds=["X"],  # Ignored due to mocking
        custom_confounds=["A"],
    )

    with pytest.raises(AssertionError, match=r"Missing directory path"):
        confound_regression._load_confounds(bold_filepath, model_spec)


def test_custom_confounds_file_missing(
    # Fixture(s)
    confound_regression: ConfoundRegression,
    tmp_path: Path,
):
    bold_file = "sub-005_task-A_run-3_space-fsaverage6_hemi-R_bold.func.gii"
    bold_filepath = tmp_path / bold_file
    bold_filepath.write_text("")

    confounds_file = "sub-005_task-A_run-3_desc-confounds_timeseries.tsv"
    confounds_filepath = tmp_path / confounds_file
    confounds_filepath.write_text("X\tY\tZ\n1\t2\t3")
    confounds_filepath.with_suffix(".json").write_text("")

    model_spec = ModelSpec(
        confounds=["X"],  # Ignored due to mocking
        custom_confounds=["A"],
    )

    confound_regression._custom_confounds_dir = tmp_path

    with pytest.raises(FileNotFoundError, match=r"Custom confounds not found"):
        confound_regression._load_confounds(bold_filepath, model_spec)


def test_custom_confounds_with_missing_values(
    # Fixture(s)
    confound_regression: ConfoundRegression,
    tmp_path: Path,
):
    bold_file = "sub-005_task-A_run-3_space-fsaverage6_hemi-R_bold.func.gii"
    bold_filepath = tmp_path / bold_file
    bold_filepath.write_text("")

    confounds_file = "sub-005_task-A_run-3_desc-confounds_timeseries.tsv"
    confounds_filepath = tmp_path / confounds_file
    confounds_filepath.write_text("X\tY\tZ\n1\t2\t3")
    confounds_filepath.with_suffix(".json").write_text("")

    custom_confounds_file = "sub-005_task-A_run-3_desc-customConfounds_timeseries.tsv"
    custom_confounds_filepath = tmp_path / custom_confounds_file
    custom_confounds_filepath.write_text("A\tB\tC\n\t20\t30")

    model_spec = ModelSpec(
        confounds=["X"],  # Ignored due to mocking
        custom_confounds=["A"],
    )

    confound_regression._custom_confounds_dir = tmp_path

    with pytest.raises(ValueError, match=r"Missing / NaN values"):
        confound_regression._load_confounds(bold_filepath, model_spec)


def test_custom_confounds_of_incompatible_shape(
    # Fixture(s)
    confound_regression: ConfoundRegression,
    tmp_path: Path,
):
    bold_file = "sub-005_task-A_run-3_space-fsaverage6_hemi-R_bold.func.gii"
    bold_filepath = tmp_path / bold_file
    bold_filepath.write_text("")

    confounds_file = "sub-005_task-A_run-3_desc-confounds_timeseries.tsv"
    confounds_filepath = tmp_path / confounds_file
    confounds_filepath.write_text("X\tY\tZ\n1\t2\t3")
    confounds_filepath.with_suffix(".json").write_text("")

    custom_confounds_file = "sub-005_task-A_run-3_desc-customConfounds_timeseries.tsv"
    custom_confounds_filepath = tmp_path / custom_confounds_file
    custom_confounds_filepath.write_text("A\tB\tC\n10\t20\t30\n100\t200\t300")

    model_spec = ModelSpec(
        confounds=["X"],  # Ignored due to mocking
        custom_confounds=["A"],
    )

    confound_regression._custom_confounds_dir = tmp_path

    with pytest.raises(ValueError, match=r"Unequal number of rows"):
        confound_regression._load_confounds(bold_filepath, model_spec)
