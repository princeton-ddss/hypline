import json
import re
from enum import Enum
from pathlib import Path
from types import MappingProxyType

import nibabel as nib
import numpy as np
import polars as pl

from .config import Config, ModelSpec


class VolumeSpace(Enum):
    MNI_152_NLIN_6_ASYM = "MNI152NLin6Asym"
    MNI_152_NLIN_2009_C_ASYM = "MNI152NLin2009cAsym"


class SurfaceSpace(Enum):
    FS_AVERAGE_5 = "fsaverage5"
    FS_AVERAGE_6 = "fsaverage6"


# Read-only mapping between a data space name and its enum variant
DATA_SPACES = MappingProxyType(
    {space.value: space for space in list(VolumeSpace) + list(SurfaceSpace)}
)


class ConfoundRegression:
    def __init__(
        self,
        config: str,
        fmriprep_dir: str,
        output_dir: str | None = None,
        custom_confounds_dir: str | None = None,
    ):
        # TODO: Parse and validate config file
        self._config: Config

        # Set the directory path to fMRIPrep data
        self._fmriprep_dir = Path(fmriprep_dir)
        if self._fmriprep_dir.exists() is False:
            raise FileNotFoundError(f"Path does not exist: {fmriprep_dir}")

        # Set the directory path to store cleaned outputs
        self._output_dir = (
            Path(output_dir)
            if output_dir
            else self._fmriprep_dir.with_name(self._fmriprep_dir.name + "_cleaned")
        )
        if self._output_dir.exists() is False:
            self._output_dir.mkdir()

        # Set the directory path to custom confounds
        self._custom_confounds_dir = None
        if custom_confounds_dir:
            self._custom_confounds_dir = Path(custom_confounds_dir)
            if self._custom_confounds_dir.exists() is False:
                raise FileNotFoundError(f"Path does not exist: {custom_confounds_dir}")

    def clean_bold_for_all_subjects(
        self,
        model_name: str,
        data_space_name: str = VolumeSpace.MNI_152_NLIN_2009_C_ASYM.value,
    ):
        # TODO: Use multiprocessing
        pass

    def clean_bold(
        self,
        subject_id: int,
        model_name: str,
        data_space_name: str = VolumeSpace.MNI_152_NLIN_2009_C_ASYM.value,
    ):
        model_spec = self._config.model_specs.get(model_name)
        if model_spec is None:
            raise ValueError(f"Undefined model: {model_name}")

        data_space = DATA_SPACES.get(data_space_name)
        if isinstance(data_space, VolumeSpace):
            self._clean_bold_in_volume_space(subject_id, model_spec, data_space)
        elif isinstance(data_space, SurfaceSpace):
            self._clean_bold_in_surface_space(subject_id, model_spec, data_space)
        else:
            raise ValueError(f"Unsupported data space: {data_space_name}")

    def _clean_bold_in_volume_space(
        self, subject_id: int, model_spec: ModelSpec, data_space: VolumeSpace
    ):
        pass

    def _clean_bold_in_surface_space(
        self, subject_id: int, model_spec: ModelSpec, data_space: SurfaceSpace
    ):
        files = self._fmriprep_dir.glob(
            f"sub-{subject_id}/**/*space-{data_space.value}*bold.func.gii"
        )

        for f in files:
            # Read raw BOLD data
            img = nib.load(f)
            assert isinstance(img, nib.GiftiImage)
            bold = img.agg_data()
            assert isinstance(bold, np.ndarray)
            bold = bold.T  # Shape of (TRs, voxels)

            # Load confounds required for the model

            # Perform confound regression (use different functions for volume vs. surface space)

            # Store cleaned BOLD data (different for volume vs. surface space)

    def _load_confounds(self, bold_filepath: Path, model_spec: ModelSpec) -> np.ndarray:
        # Extract file name up to the run number segment
        match = re.search(r"^(.*?run-\d+)", bold_filepath.name)
        if match is None:
            raise ValueError(f"Run number is missing: {bold_filepath.name}")
        identifier = match.group(1)  # Includes subject/session/task/run info

        # Load standard confounds required for the model
        files = bold_filepath.parent.glob(f"{identifier}*desc-confounds*timeseries.*")
        confounds_filepath = next(files, None)
        if confounds_filepath is None:
            raise FileNotFoundError(f"Confounds data not found for: {identifier}")
        confounds_df = (
            pl.read_csv(confounds_filepath.with_suffix(".tsv"), separator="\t")
            .fill_nan(None)  # For interpolation
            .fill_null(strategy="backward")  # Assume missing data in the beginning only
        )
        with open(confounds_filepath.with_suffix(".json")) as f:
            confounds_meta: dict[str, dict] = json.load(f)
        confounds_df = self._extract_confounds(confounds_df, confounds_meta, model_spec)

        # Load custom confounds required for the model (if necessary)

        # Gather all confounds

    @staticmethod
    def _extract_confounds(
        confounds_df: pl.DataFrame,
        confounds_meta: dict[str, dict],
        model_spec: ModelSpec,
    ) -> pl.DataFrame:
        pass
