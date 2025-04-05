import json
import logging
import re
from copy import deepcopy
from pathlib import Path
from types import MappingProxyType

import nibabel as nib
import numpy as np
import polars as pl
import yaml
from natsort import natsorted
from nibabel.gifti import GiftiDataArray, GiftiImage
from nibabel.nifti1 import Nifti1Image
from nilearn import image as nimg
from nilearn import signal
from pydantic import TypeAdapter
from rich import print
from rich.progress import track

from .enums import CompCorMethod, CompCorTissue, SurfaceSpace, VolumeSpace
from .schemas import CompCorOptions, Config, ConfoundMetadata, ModelSpec


class ConfoundRegression:
    # Read-only mapping between a data space name and its enum variant
    DATA_SPACES = MappingProxyType(
        {space.value: space for space in list(VolumeSpace) + list(SurfaceSpace)}
    )

    def __init__(
        self,
        config_file: str,
        fmriprep_dir: str,
        output_dir: str | None = None,
        custom_confounds_dir: str | None = None,
    ):
        # Parse and validate config data
        config_filepath = Path(config_file)
        if config_filepath.exists() is False:
            raise FileNotFoundError(f"Path does not exist: {config_file}")
        self._config = Config.model_validate(
            yaml.safe_load(config_filepath.read_text())
        )

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
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Set the directory path to custom confounds
        self._custom_confounds_dir = None
        if custom_confounds_dir:
            self._custom_confounds_dir = Path(custom_confounds_dir)
            if self._custom_confounds_dir.exists() is False:
                raise FileNotFoundError(f"Path does not exist: {custom_confounds_dir}")

        # Create logging-related attributes
        self._log_dir = self._output_dir / "logs"
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)
        self._formatter = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    @property
    def config(self) -> Config:
        return deepcopy(self._config)

    @property
    def fmriprep_dir(self) -> Path:
        return deepcopy(self._fmriprep_dir)

    @property
    def output_dir(self) -> Path:
        return deepcopy(self._output_dir)

    @property
    def custom_confounds_dir(self) -> Path | None:
        return deepcopy(self._custom_confounds_dir)

    def clean_bold(
        self,
        model_name: str,
        subject_ids: list[str],
        session_name: str = "*",
        task_name: str = "*",
        data_space_name: str = "MNI152NLin2009cAsym",
    ):
        # Mapping between a data space type and the corresponding method
        CLEAN_BOLD = {
            VolumeSpace: self._clean_bold_in_volume_space,
            SurfaceSpace: self._clean_bold_in_surface_space,
        }

        model_spec = self._config.model_specs.get(model_name)
        if model_spec is None:
            raise ValueError(f"Undefined model: {model_name}")

        data_space = self.DATA_SPACES.get(data_space_name)
        data_space_type = type(data_space)
        if data_space_type not in CLEAN_BOLD:
            raise ValueError(f"Unsupported data space: {data_space_name}")

        for sub_id in track(subject_ids, description="Processing..."):
            file_handler = logging.FileHandler(self._log_dir / f"sub-{sub_id}.log")
            file_handler.setFormatter(self._formatter)
            self._logger.addHandler(file_handler)

            bold_pattern = self._compose_glob_pattern_for_bold(
                subject_id=sub_id,
                session_name=session_name,
                task_name=task_name,
                data_space=data_space,
            )
            bold_filepaths = self._fmriprep_dir.glob(bold_pattern)

            for filepath in bold_filepaths:
                self._logger.info("Cleaning starting: %s", filepath.name)
                try:
                    CLEAN_BOLD[data_space_type](filepath, model_spec)
                    self._logger.info("Cleaning complete: %s", filepath.name)
                except Exception as e:
                    self._logger.error(e)
                    print("[red]Processing failed:[/red]", filepath.name)

            self._logger.removeHandler(file_handler)  # Reset for next iteration

    def _clean_bold_in_volume_space(self, filepath: Path, model_spec: ModelSpec):
        # Read raw BOLD data
        bold = nimg.load_img(filepath)  # Shape of (x, y, z, TRs)
        assert isinstance(bold, Nifti1Image)

        # Extract TR value (assumed constant in a given run)
        p = filepath.parent / (filepath.name.split(".")[0] + ".json")
        with open(p, "r") as f:
            data = json.load(f)
        assert isinstance(data, dict)
        repetition_time = data.get("RepetitionTime")  # In seconds
        if repetition_time is None:
            raise ValueError(f"TR metadata is missing: {p.name}")
        TR = float(repetition_time)

        # Load confounds for the requested model
        confounds_df = self._load_confounds(filepath, model_spec)
        confounds = confounds_df.to_numpy()  # Shape of (TRs, confounds)
        if confounds.shape[0] != bold.shape[-1]:
            raise ValueError(
                f"Unequal number of TRs between BOLD and confounds data: {filepath.name}"
            )

        # Perform confound regression
        cleaned_bold = nimg.clean_img(
            bold,
            confounds=confounds,
            detrend=True,
            t_r=TR,
            ensure_finite=True,
            standardize="zscore_sample",
            standardize_confounds=True,
        )
        assert isinstance(cleaned_bold, Nifti1Image)

        # Store cleaned BOLD data
        entities = filepath.name.split("_")
        if entities[-2].startswith("desc-"):
            entities[-2] = "desc-clean"
        else:
            entities.insert(-1, "desc-clean")
        new_filename = "_".join(entities)
        intermediate_dir = filepath.relative_to(self._fmriprep_dir).parent
        new_filepath = self._output_dir / intermediate_dir / new_filename
        new_filepath.parent.mkdir(parents=True, exist_ok=True)
        nib.save(cleaned_bold, new_filepath)

    def _clean_bold_in_surface_space(self, filepath: Path, model_spec: ModelSpec):
        # Read raw BOLD data
        img = nib.load(filepath)
        assert isinstance(img, GiftiImage)
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
                f"Unequal number of TRs between BOLD and confounds data: {filepath.name}"
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

        # Store cleaned BOLD data
        new_img = GiftiImage(
            darrays=[
                GiftiDataArray(data=row, intent="NIFTI_INTENT_TIME_SERIES")
                for row in cleaned_bold
            ],
            header=img.header,
            extra=img.extra,
        )
        entities = filepath.name.split("_")
        if entities[-2].startswith("desc-"):
            entities[-2] = "desc-clean"
        else:
            entities.insert(-1, "desc-clean")
        new_filename = "_".join(entities)
        intermediate_dir = filepath.relative_to(self._fmriprep_dir).parent
        new_filepath = self._output_dir / intermediate_dir / new_filename
        new_filepath.parent.mkdir(parents=True, exist_ok=True)
        nib.save(new_img, new_filepath)

    def _load_confounds(
        self, bold_filepath: Path, model_spec: ModelSpec
    ) -> pl.DataFrame:
        # Extract file name up to the run number segment
        match = re.search(r"^(.*?run-\d+)", bold_filepath.name)
        if match is None:
            raise ValueError(f"Run number is missing: {bold_filepath.name}")
        identifier = match.group(1)  # Includes subject/session/task/run info

        # Load standard confounds for the requested model
        files = bold_filepath.parent.glob(f"{identifier}*desc-confounds*timeseries.tsv")
        confounds_filepath = next(files, None)
        if confounds_filepath is None:
            raise FileNotFoundError(f"Confounds not found for: {identifier}")
        confounds_df = (
            pl.read_csv(confounds_filepath, separator="\t")
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

    def _extract_confounds(
        self,
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
        compcors = [c for c in CompCorMethod if c.value in model_spec.model_fields_set]
        if compcors:
            comps_selected: list[str] = []
            for compcor in compcors:
                for options in getattr(model_spec, compcor.value):
                    assert isinstance(options, CompCorOptions)
                    comps_selected.extend(
                        self._select_comps(
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

    def _select_comps(
        self,
        confounds_meta: dict[str, ConfoundMetadata],
        method: CompCorMethod,
        n_comps: int | float,
        tissue: CompCorTissue | None,
    ) -> list[str]:
        """
        Select relevant CompCor components.

        Notes
        -----
        Adapted from https://github.com/snastase/narratives/blob/master/code/extract_confounds.py.
        """
        # Check that we sensible number of components
        assert n_comps > 0

        # Ignore tissue if specified for tCompCor
        if method == CompCorMethod.TEMPORAL and tissue:
            self._logger.warning(
                "tCompCor is not restricted to a tissue mask - ignoring tissue specification (%s)",
                tissue.value,
            )
            tissue = None

        # Get CompCor metadata for relevant method
        compcor_meta = {
            k: v
            for k, v in confounds_meta.items()
            if v.Method == method and v.Retained is True
        }

        # If aCompCor, filter metadata for tissue mask
        if method == CompCorMethod.ANATOMICAL:
            compcor_meta = {k: v for k, v in compcor_meta.items() if v.Mask == tissue}

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
                self._logger.warning(
                    "Only %d %s components available (%d requested)",
                    len(comps_sorted),
                    method.value,
                    n_comps,
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

    @staticmethod
    def _compose_glob_pattern_for_bold(
        subject_id: str,
        session_name: str,
        task_name: str,
        data_space: VolumeSpace | SurfaceSpace,
    ) -> str:
        SUFFIX_MAP = {VolumeSpace: "bold.nii.gz", SurfaceSpace: "bold.func.gii"}

        subject = f"sub-{subject_id}"
        session = "" if session_name == "*" else f"ses-{session_name}"
        task = "" if task_name == "*" else f"task-{task_name}"
        space = f"space-{data_space.value}"
        suffix = SUFFIX_MAP[type(data_space)]

        filepath_pattern = "*".join(
            filter(None, [subject, session, task, space, suffix])
        )

        return f"{subject}/**/{filepath_pattern}"
