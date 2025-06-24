from pathlib import Path

import pytest
from pydantic import TypeAdapter
from pytest_mock import MockerFixture

from hypline.regression import ConfoundRegression
from hypline.schemas import ConfoundMetadata


@pytest.fixture(scope="function")
def confound_regression(mocker: MockerFixture):
    """
    An instance of `ConfoundRegression` with mock initiation.
    """
    mocker.patch("hypline.regression.Path")
    mocker.patch("hypline.regression.Config")
    mocker.patch("hypline.regression.yaml")
    mocker.patch("hypline.regression.logging")
    config_file = ""
    fmriprep_dir = ""
    confound_regression = ConfoundRegression(config_file, fmriprep_dir)

    return confound_regression


@pytest.fixture(scope="function")
def confounds_meta():
    path = Path(__file__).parents[2] / "data" / "confounds_timeseries.json"
    text = path.read_text()
    meta = TypeAdapter(dict[str, ConfoundMetadata]).validate_json(text)

    return meta
