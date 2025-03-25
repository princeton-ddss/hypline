import json
import re
from enum import Enum
from pathlib import Path
from types import MappingProxyType

import nibabel as nib
import numpy as np
import polars as pl
from natsort import natsorted

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

    @classmethod
    def _extract_confounds(
        cls,
        confounds_df: pl.DataFrame,
        confounds_meta: dict[str, dict],
        model_spec: ModelSpec,
    ) -> pl.DataFrame:
        """
        Extract confounds (including CompCor ones).

        Notes
        -----
        Adapted from https://github.com/snastase/narratives/blob/master/code/extract_confounds.py.
        """
        # Pop out confound groups of variable number
        groups = set(model_spec.confounds).intersection({"cosine", "motion_outlier"})

        # Grab the requested (non-group) confounds
        confounds = confounds_df[[c for c in model_spec.confounds if c not in groups]]

        # Grab confound groups if requested
        if groups:
            group_cols = [
                col
                for col in confounds_df.columns
                if any(group in col for group in groups)
            ]
            confounds = pl.concat(
                [confounds, confounds_df[group_cols]], how="horizontal"
            )

        # Grab CompCor confounds if requested
        compcors = model_spec.model_fields_set.intersection(["aCompCor", "tCompCor"])
        if compcors:
            compcor_dfs = [
                cls._extract_compcor(
                    confounds_df,
                    confounds_meta,
                    method=compcor,
                    **kwargs,
                )
                for compcor in compcors
                for kwargs in getattr(model_spec, compcor)
            ]
            confounds = pl.concat([confounds] + compcor_dfs, how="horizontal")

        return confounds

    @staticmethod
    def _extract_compcor(
        confounds_df: pl.DataFrame,
        confounds_meta: dict[str, dict],
        method: str = "tCompCor",
        n_comps: int = 5,
        tissue: str | None = None,
    ) -> pl.DataFrame:
        """
        Extract CompCor confounds.

        Notes
        -----
        Adapted from https://github.com/snastase/narratives/blob/master/code/extract_confounds.py.
        """
        # Check that we sensible number of components
        assert n_comps > 0

        # Check that method is specified correctly
        assert method in ["aCompCor", "tCompCor"]

        # Check that tissue is specified for aCompCor
        if method == "aCompCor" and tissue not in ["combined", "CSF", "WM"]:
            raise AssertionError(
                "Must specify a tissue type (combined, CSF, or WM) for aCompCor"
            )

        # Ignore tissue if specified for tCompCor
        if method == "tCompCor" and tissue:
            print(
                "Warning: tCompCor is not restricted to a tissue "
                f"mask - ignoring tissue specification ({tissue})"
            )
            tissue = None

        # Get CompCor metadata for relevant method
        compcor_meta = {
            c: confounds_meta[c]
            for c in confounds_meta
            if confounds_meta[c]["Method"] == method and confounds_meta[c]["Retained"]
        }

        # If aCompCor, filter metadata for tissue mask
        if method == "aCompCor":
            compcor_meta = {
                c: compcor_meta[c]
                for c in compcor_meta
                if compcor_meta[c]["Mask"] == tissue
            }

        # Make sure metadata components are sorted properly
        comp_sorted = natsorted(compcor_meta)
        for i, comp in enumerate(comp_sorted):
            if comp != comp_sorted[-1]:
                comp_next = comp_sorted[i + 1]
                assert (
                    compcor_meta[comp]["SingularValue"]
                    > compcor_meta[comp_next]["SingularValue"]
                )

        # Either get top n components
        if n_comps >= 1.0:
            n_comps = int(n_comps)
            if len(comp_sorted) >= n_comps:
                comp_selector = comp_sorted[:n_comps]
            else:
                comp_selector = comp_sorted
                print(
                    f"Warning: Only {len(comp_sorted)} {method} "
                    f"components available ({n_comps} requested)"
                )

        # Or components necessary to capture n proportion of variance
        else:
            comp_selector = []
            for comp in comp_sorted:
                comp_selector.append(comp)
                if compcor_meta[comp]["CumulativeVarianceExplained"] > n_comps:
                    break

        # Check we didn't end up with degenerate 0 components
        assert len(comp_selector) > 0

        # Grab the actual component time series
        confounds_compcor = confounds_df[comp_selector]

        return confounds_compcor
