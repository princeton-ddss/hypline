import pytest

from hypline.enums import CompCorMask, CompCorMethod
from hypline.regression import ConfoundRegression
from hypline.schemas import ConfoundMetadata


@pytest.mark.parametrize(
    "method, n_comps, mask, expected_output",
    [
        # aCompCor with CSF mask
        (
            CompCorMethod.ANATOMICAL,
            1,
            CompCorMask.CSF,
            ["a_comp_cor_00"],
        ),
        (
            CompCorMethod.ANATOMICAL,
            3,
            CompCorMask.CSF,
            ["a_comp_cor_00", "a_comp_cor_01", "a_comp_cor_02"],
        ),
        (
            CompCorMethod.ANATOMICAL,
            10,
            CompCorMask.CSF,
            [
                "a_comp_cor_00",
                "a_comp_cor_01",
                "a_comp_cor_02",
                "a_comp_cor_03",
                "a_comp_cor_04",
                "a_comp_cor_05",
                "a_comp_cor_06",
                "a_comp_cor_07",
                "a_comp_cor_08",
                "a_comp_cor_09",
            ],
        ),
        (
            CompCorMethod.ANATOMICAL,
            0.3,
            CompCorMask.CSF,
            [
                "a_comp_cor_00",
                "a_comp_cor_01",
                "a_comp_cor_02",
                "a_comp_cor_03",
                "a_comp_cor_04",
            ],
        ),
        # aCompCor with WM mask
        (
            CompCorMethod.ANATOMICAL,
            3,
            CompCorMask.WM,
            ["a_comp_cor_12", "a_comp_cor_13", "a_comp_cor_14"],
        ),
        (
            CompCorMethod.ANATOMICAL,
            0.1,
            CompCorMask.WM,
            ["a_comp_cor_12", "a_comp_cor_13"],
        ),
        # aCompCor with combined mask
        (
            CompCorMethod.ANATOMICAL,
            3,
            CompCorMask.COMBINED,
            ["a_comp_cor_100", "a_comp_cor_101", "a_comp_cor_102"],
        ),
        (
            CompCorMethod.ANATOMICAL,
            0.1,
            CompCorMask.COMBINED,
            ["a_comp_cor_100", "a_comp_cor_101"],
        ),
        # tCompCor
        (
            CompCorMethod.TEMPORAL,
            1,
            None,
            ["t_comp_cor_00"],
        ),
        (
            CompCorMethod.TEMPORAL,
            3,
            None,
            ["t_comp_cor_00", "t_comp_cor_01", "t_comp_cor_02"],
        ),
        (
            CompCorMethod.TEMPORAL,
            10,
            None,
            ["t_comp_cor_00", "t_comp_cor_01", "t_comp_cor_02"],
        ),
        (
            CompCorMethod.TEMPORAL,
            0.4,
            None,
            ["t_comp_cor_00", "t_comp_cor_01"],
        ),
        (
            CompCorMethod.TEMPORAL,
            0.4,
            CompCorMask.CSF,  # Expected to be ignored
            ["t_comp_cor_00", "t_comp_cor_01"],
        ),
    ],
)
def test_select_comps(
    # Fixture(s)
    confound_regression: ConfoundRegression,
    confounds_meta: ConfoundMetadata,
    # Parameter(s)
    method: CompCorMethod,
    n_comps: int | float,
    mask: CompCorMask | None,
    expected_output: list[str],
):
    output = confound_regression._select_comps(
        confounds_meta=confounds_meta,
        method=method,
        n_comps=n_comps,
        mask=mask,
    )
    assert output == expected_output


def test_invalid_n_comps(
    # Fixture(s)
    confound_regression: ConfoundRegression,
    confounds_meta: ConfoundMetadata,
):
    with pytest.raises(AssertionError, match="`n_comps` must be positive"):
        confound_regression._select_comps(
            confounds_meta=confounds_meta,
            method=CompCorMethod.TEMPORAL,
            n_comps=-1,
            mask=None,
        )


def test_missing_mask_for_acompcor(
    # Fixture(s)
    confound_regression: ConfoundRegression,
    confounds_meta: ConfoundMetadata,
):
    with pytest.raises(AssertionError, match="Mask must be specified for aCompCor"):
        confound_regression._select_comps(
            confounds_meta=confounds_meta,
            method=CompCorMethod.ANATOMICAL,
            n_comps=1,
            mask=None,
        )


def test_unsupported_method(
    # Fixture(s)
    confound_regression: ConfoundRegression,
    confounds_meta: ConfoundMetadata,
):
    with pytest.raises(
        ValueError, match=f"Unsupported CompCor method: {CompCorMethod.MEAN}"
    ):
        confound_regression._select_comps(
            confounds_meta=confounds_meta,
            method=CompCorMethod.MEAN,
            n_comps=1,
            mask=None,
        )
