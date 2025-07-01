from pathlib import Path

import pytest
from pydantic import TypeAdapter
from pytest_mock import MockerFixture

from hypline.regression import ConfoundRegression
from hypline.schemas import ConfoundMetadata


@pytest.fixture(scope="function")
def confound_regression(mocker: MockerFixture, tmp_path: Path):
    """
    An instance of `ConfoundRegression` with mock initiation.
    """
    mocker.patch("hypline.regression.logging")
    mocker.patch("hypline.regression.yaml")
    mocker.patch("hypline.regression.Config")

    config_filepath = tmp_path / "config.yaml"
    config_filepath.write_text("")

    config_file = str(config_filepath)
    fmriprep_dir = str(tmp_path)
    confound_regression = ConfoundRegression(config_file, fmriprep_dir)

    return confound_regression


@pytest.fixture(scope="session")
def confounds_meta():
    path = Path(__file__).parents[2] / "data" / "confounds_timeseries.json"
    text = path.read_text()
    meta = TypeAdapter(dict[str, ConfoundMetadata]).validate_json(text)

    return meta
