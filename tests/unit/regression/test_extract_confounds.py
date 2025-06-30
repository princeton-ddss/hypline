import polars as pl
import pytest

from hypline.regression import ConfoundRegression
from hypline.schemas import ModelSpec


@pytest.fixture(scope="module")
def confounds_df():
    confounds_df = pl.DataFrame(
        {
            "global_signal": [1, 2],
            "csf": [3, 4],
            "white_matter": [5, 6],
            "cosine00": [7, 8],
            "cosine01": [9, 10],
            "cosine02": [11, 12],
            "motion_outlier00": [13, 14],
            "motion_outlier01": [15, 16],
            "motion_outlier02": [17, 18],
        }
    )

    return confounds_df


@pytest.mark.parametrize(
    "model_confounds, expected_output_confounds",
    [
        (["global_signal"], ["global_signal"]),
        (["global_signal", "csf"], ["global_signal", "csf"]),
        (
            ["global_signal", "csf", "white_matter"],
            ["global_signal", "csf", "white_matter"],
        ),
        (
            ["global_signal", "csf", "cosine"],
            ["global_signal", "csf", "cosine00", "cosine01", "cosine02"],
        ),
        (
            ["global_signal", "csf", "motion_outlier"],
            [
                "global_signal",
                "csf",
                "motion_outlier00",
                "motion_outlier01",
                "motion_outlier02",
            ],
        ),
    ],
)
def test_extract_confounds(
    # Fixture(s)
    confound_regression: ConfoundRegression,
    confounds_df: pl.DataFrame,
    # Parameter(s)
    model_confounds: list[str],
    expected_output_confounds: list[str],
):
    confounds_meta = {}
    model_spec = ModelSpec(confounds=model_confounds)
    extracted_df = confound_regression._extract_confounds(
        confounds_df, confounds_meta, model_spec
    )
    assert extracted_df.equals(confounds_df[expected_output_confounds])


def test_extract_nonexisting_confounds(
    # Fixture(s)
    confound_regression: ConfoundRegression,
    confounds_df: pl.DataFrame,
):
    confounds_meta = {}
    model_spec = ModelSpec(confounds=["a"])
    with pytest.raises(ValueError, match="Model confounds missing from confound data"):
        _ = confound_regression._extract_confounds(
            confounds_df, confounds_meta, model_spec
        )
