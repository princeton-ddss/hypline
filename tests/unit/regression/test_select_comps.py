import pytest

from hypline.enums import CompCorMethod, CompCorTissue
from hypline.regression import ConfoundRegression
from hypline.schemas import ConfoundMetadata


@pytest.mark.parametrize(
    "method, n_comps, tissue, expected_output",
    [
        # aCompCor with CSF tissue
        (
            CompCorMethod.ANATOMICAL,
            1,
            CompCorTissue.CSF,
            ["a_comp_cor_00"],
        ),
        (
            CompCorMethod.ANATOMICAL,
            3,
            CompCorTissue.CSF,
            ["a_comp_cor_00", "a_comp_cor_01", "a_comp_cor_02"],
        ),
        (
            CompCorMethod.ANATOMICAL,
            10,
            CompCorTissue.CSF,
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
            CompCorTissue.CSF,
            [
                "a_comp_cor_00",
                "a_comp_cor_01",
                "a_comp_cor_02",
                "a_comp_cor_03",
                "a_comp_cor_04",
            ],
        ),
        # aCompCor with WM tissue
        (
            CompCorMethod.ANATOMICAL,
            3,
            CompCorTissue.WM,
            ["a_comp_cor_12", "a_comp_cor_13", "a_comp_cor_14"],
        ),
        (
            CompCorMethod.ANATOMICAL,
            0.1,
            CompCorTissue.WM,
            ["a_comp_cor_12", "a_comp_cor_13"],
        ),
        # aCompCor with combined tissue
        (
            CompCorMethod.ANATOMICAL,
            3,
            CompCorTissue.COMBINED,
            ["a_comp_cor_100", "a_comp_cor_101", "a_comp_cor_102"],
        ),
        (
            CompCorMethod.ANATOMICAL,
            0.1,
            CompCorTissue.COMBINED,
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
            CompCorTissue.CSF,  # Expected to be ignored
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
    tissue: CompCorTissue | None,
    expected_output: list[str],
):
    output = confound_regression._select_comps(
        confounds_meta=confounds_meta,
        method=method,
        n_comps=n_comps,
        tissue=tissue,
    )
    assert output == expected_output
