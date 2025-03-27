import re
from pathlib import Path
from types import MappingProxyType

import nibabel as nib
import numpy as np
import polars as pl
from natsort import natsorted
from nilearn import signal
from pydantic import PositiveFloat, PositiveInt, TypeAdapter

from .enums import CompCor, CompCorTissue, SurfaceSpace, VolumeSpace
from .schemas import CompCorOptions, Config, ConfoundMetadata, ModelSpec

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

        for filepath in files:
            # Read raw BOLD data
            img = nib.load(filepath)
            assert isinstance(img, nib.GiftiImage)
            bold = img.agg_data()
            assert isinstance(bold, np.ndarray)
            bold = bold.T  # Shape of (TRs, voxels)

            # Extract TR value (assumed constant in a given run)
            repetition_time = img.darrays[0].meta.get("TimeStep")  # In milliseconds
            if repetition_time is None:
                raise ValueError(f"TR metadata is missing: {filepath.name}")
            TR = float(repetition_time) / 1000  # Convert to seconds

            # Load confounds for the requested model
            confounds_df = self._load_confounds(filepath, model_spec)
            confounds = confounds_df.to_numpy()  # Shape of (TRs, confounds)
            if confounds.shape[0] != bold.shape[0]:
                raise ValueError(
                    f"Unequal number of rows (TRs) between BOLD and confounds data: {filepath.name}"
                )

            # Perform confound regression
            cleaned_bold = signal.clean(
                bold,
                confounds=confounds,
                detrend=True,
                t_r=TR,
                ensure_finite=True,
                standardize="zscore_sample",
                standardize_confounds=True,
            )

            # Store cleaned BOLD data (different for volume vs. surface space)

    def _load_confounds(
        self, bold_filepath: Path, model_spec: ModelSpec
    ) -> pl.DataFrame:
        # Extract file name up to the run number segment
        match = re.search(r"^(.*?run-\d+)", bold_filepath.name)
        if match is None:
            raise ValueError(f"Run number is missing: {bold_filepath.name}")
        identifier = match.group(1)  # Includes subject/session/task/run info

        # Load standard confounds for the requested model
        files = bold_filepath.parent.glob(f"{identifier}*desc-confounds*timeseries.*")
        confounds_filepath = next(files, None)
        if confounds_filepath is None:
            raise FileNotFoundError(f"Confounds not found for: {identifier}")
        confounds_df = (
            pl.read_csv(confounds_filepath.with_suffix(".tsv"), separator="\t")
            .fill_nan(None)  # For interpolation
            .fill_null(strategy="backward")  # Assume missing data in the beginning only
        )
        confounds_meta = TypeAdapter(dict[str, ConfoundMetadata]).validate_json(
            confounds_filepath.with_suffix(".json").read_text()
        )
        confounds_df = self._extract_confounds(confounds_df, confounds_meta, model_spec)

        # Load custom confounds for the requested model
        if model_spec.custom_confounds:
            files = self._custom_confounds_dir.glob(
                f"**/{identifier}*desc-customConfounds*timeseries.tsv"
            )
            custom_confounds_filepath = next(files, None)
            if custom_confounds_filepath is None:
                raise FileNotFoundError(f"Custom confounds not found for: {identifier}")
            custom_confounds_df = pl.read_csv(
                custom_confounds_filepath,
                separator="\t",
                columns=model_spec.custom_confounds,
            )
            if custom_confounds_df.fill_nan(None).null_count().pipe(sum).item() > 0:
                raise ValueError(
                    f"Missing / NaN values in custom confounds data: {identifier}"
                )
            if custom_confounds_df.height != confounds_df.height:
                raise ValueError(
                    f"Unequal number of rows (TRs) between standard and custom confounds data: {identifier}"
                )
            confounds_df = pl.concat(
                [confounds_df, custom_confounds_df], how="horizontal"
            )

        return confounds_df

    @classmethod
    def _extract_confounds(
        cls,
        confounds_df: pl.DataFrame,
        confounds_meta: dict[str, ConfoundMetadata],
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
        compcors = [c for c in CompCor if c.value in model_spec.model_fields_set]
        if compcors:
            comps_selected: list[str] = []
            for compcor in compcors:
                for options in getattr(model_spec, compcor.value):
                    assert isinstance(options, CompCorOptions)
                    comps_selected.extend(
                        cls._select_comps(
                            confounds_meta,
                            compcor,
                            n_comps=options.n_comps,
                            tissue=options.tissue,
                        )
                    )
            confounds = pl.concat(
                [confounds, confounds_df[comps_selected]], how="horizontal"
            )

        return confounds

    @staticmethod
    def _select_comps(
        confounds_meta: dict[str, ConfoundMetadata],
        method: CompCor,
        n_comps: PositiveInt | PositiveFloat = 5,
        tissue: CompCorTissue | None = None,
    ) -> list[str]:
        """
        Select relevant CompCor components.

        Notes
        -----
        Adapted from https://github.com/snastase/narratives/blob/master/code/extract_confounds.py.
        """
        # Ignore tissue if specified for tCompCor
        if method == CompCor.TEMPORAL and tissue:
            print(
                "Warning: tCompCor is not restricted to a tissue "
                f"mask - ignoring tissue specification ({tissue})"
            )
            tissue = None

        # Get CompCor metadata for relevant method
        compcor_meta = {
            k: v
            for k, v in confounds_meta.items()
            if v.Method == method and v.Retained is True
        }

        # If aCompCor, filter metadata for tissue mask
        if method == CompCor.ANATOMICAL:
            compcor_meta = {k: v for k, v in confounds_meta.items() if v.Mask == tissue}

        # Make sure metadata components are sorted properly
        comps_sorted = natsorted(compcor_meta)
        for i, comp in enumerate(comps_sorted):
            if comp != comps_sorted[-1]:
                comp_next = comps_sorted[i + 1]
                assert (
                    compcor_meta[comp].SingularValue
                    > compcor_meta[comp_next].SingularValue
                )

        # Either get top n components
        if n_comps >= 1.0:
            n_comps = int(n_comps)
            if len(comps_sorted) >= n_comps:
                comps_selected = comps_sorted[:n_comps]
            else:
                comps_selected = comps_sorted
                print(
                    f"Warning: Only {len(comps_sorted)} {method} "
                    f"components available ({n_comps} requested)"
                )

        # Or components necessary to capture n proportion of variance
        else:
            comps_selected = []
            for comp in comps_sorted:
                comps_selected.append(comp)
                if compcor_meta[comp].CumulativeVarianceExplained > n_comps:
                    break

        # Check we didn't end up with degenerate 0 components
        assert len(comps_selected) > 0

        return comps_selected
