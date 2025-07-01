import json
import logging
import re
from copy import deepcopy
from pathlib import Path
from types import MappingProxyType

import nibabel as nib
import numpy as np
import polars as pl
import yaml  # type: ignore
from nibabel.gifti import GiftiDataArray, GiftiImage
from nibabel.nifti1 import Nifti1Image
from nilearn import image as nimg
from nilearn import signal
from pydantic import TypeAdapter
from rich import print
from rich.progress import track

from .enums import CompCorMask, CompCorMethod, SurfaceSpace, VolumeSpace
from .schemas import CompCorOptions, Config, ConfoundMetadata, ModelSpec


class ConfoundRegression:
    """
    Performs confound regression to remove noise from BOLD fMRI data.

    Parameters
    ----------
    config_file : str
        Configuration file containing model specs, etc.
    fmriprep_dir : str
        Directory containing fMRIPrep outputs.
    output_dir : str, optional
        Directory to store cleaned BOLD data.
        Defaults to `<fmriprep_dir>_cleaned` if unspecified.
    custom_confounds_dir: str, optional
        Directory containing custom confounds.
    """

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
        """
        Perform confound regression to clean BOLD data.

        Parameters
        ----------
        model_name : str
            Confound regression model to run (defined in configuration).
        subject_ids : list of str
            Target subject IDs.
        session_name : str, optional
            Target session name. Defaults to all if unspecified.
        task_name : str, optional
            Target task name. Defaults to all if unspecified.
        data_space_name : str, optional
            Target BOLD data space. Defaults to `MNI152NLin2009cAsym` if unspecified.

        Returns
        -------
        None
        """
        # Mapping between a data space type and the corresponding method
        CLEAN_BOLD = {
            VolumeSpace: self._clean_bold_in_volume_space,
            SurfaceSpace: self._clean_bold_in_surface_space,
        }

        model_spec = self._config.model_specs.get(model_name)
        if model_spec is None:
            raise ValueError(f"Undefined model: {model_name}")

        data_space = self.DATA_SPACES.get(data_space_name)
        if data_space is None:
            raise ValueError(f"Unsupported data space: {data_space_name}")
        data_space_type = type(data_space)

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
        """
        Clean BOLD data stored in volumetric space.

        Parameters
        ----------
        filepath : Path
            Path to BOLD data to be cleaned.
        model_spec : ModelSpec
            Model specification for confound regression.

        Returns
        -------
        None
        """
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
                "Unequal number of TRs between BOLD and "
                f"confounds data: {filepath.name}"
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
        new_filepath = self._make_new_filepath(filepath)
        nib.save(cleaned_bold, new_filepath)

    def _clean_bold_in_surface_space(self, filepath: Path, model_spec: ModelSpec):
        """
        Clean BOLD data stored in surface space.

        Parameters
        ----------
        filepath : Path
            Path to BOLD data to be cleaned.
        model_spec : ModelSpec
            Model specification for confound regression.

        Returns
        -------
        None
        """
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
                "Unequal number of TRs between BOLD and "
                f"confounds data: {filepath.name}"
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
        new_filepath = self._make_new_filepath(filepath)
        nib.save(new_img, new_filepath)

    def _make_new_filepath(self, filepath: Path) -> Path:
        """
        Make a new file path to store cleaned data.

        Parameters
        ----------
        filepath : Path
            Path to the data to be cleaned.

        Returns
        -------
        Path
            Path to save the cleaned data.
        """
        entities = filepath.name.split("_")
        if entities[-2].startswith("desc-"):
            entities[-2] = "desc-clean"
        else:
            entities.insert(-1, "desc-clean")
        new_filename = "_".join(entities)
        intermediate_dir = filepath.relative_to(self._fmriprep_dir).parent
        new_filepath = self._output_dir / intermediate_dir / new_filename
        new_filepath.parent.mkdir(parents=True, exist_ok=True)

        return new_filepath

    def _load_confounds(
        self, bold_filepath: Path, model_spec: ModelSpec
    ) -> pl.DataFrame:
        """
        Load all confounds required for the given model.

        Parameters
        ----------
        bold_filepath : Path
            Path to the BOLD data to be cleaned.
        model_spec : ModelSpec
            Model specification for confound regression.

        Returns
        -------
        pl.DataFrame
            Table containing all confounds
            necessary for confound regression.
        """
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
            # JSON assumed present with TSV
            confounds_filepath.with_suffix(".json").read_text()
        )
        confounds_df = self._extract_confounds(confounds_df, confounds_meta, model_spec)

        # Load custom confounds for the requested model
        if model_spec.custom_confounds:
            assert self._custom_confounds_dir is not None, (
                "Missing directory path for custom confounds"
            )
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
            if sum(custom_confounds_df.fill_nan(None).null_count().row(0)) > 0:
                raise ValueError(
                    f"Missing / NaN values in custom confounds data: {identifier}"
                )
            if custom_confounds_df.height != confounds_df.height:
                raise ValueError(
                    "Unequal number of rows (TRs) between standard and "
                    f"custom confounds data: {identifier}"
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
        Extract standard confounds (including CompCor ones).

        Parameters
        ----------
        confounds_df : pl.DataFrame
            Table containing all standard confounds (columns)
            and their values across TRs (rows).
        confounds_meta : dict of str to ConfoundMetadata
            Mapping between component name and data.
        model_spec : ModelSpec
            Model specification for confound regression.

        Returns
        -------
        pl.DataFrame
            Table containing standards confounds
            necessary for confound regression.

        Notes
        -----
        Adapted from https://github.com/snastase/narratives/blob/master/code/extract_confounds.py.
        """
        # Pop out confound groups of variable number
        groups = set(model_spec.confounds).intersection({"cosine", "motion_outlier"})

        # Grab the requested (non-group) confounds
        confounds = [c for c in model_spec.confounds if c not in groups]

        # Grab confound groups if requested
        if groups:
            group_cols = [
                col
                for col in confounds_df.columns
                if any(group in col for group in groups)
            ]
            confounds.extend(group_cols)

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
                            mask=options.mask,
                        )
                    )
            confounds.extend(comps_selected)

        if not set(confounds).issubset(confounds_df.columns):
            raise ValueError("Model confounds missing from confound data")

        return confounds_df[confounds]

    def _select_comps(
        self,
        confounds_meta: dict[str, ConfoundMetadata],
        method: CompCorMethod,
        n_comps: int | float,
        mask: CompCorMask | None,
    ) -> list[str]:
        """
        Select relevant CompCor components.

        Parameters
        ----------
        confounds_meta : dict of str to ConfoundMetadata
            Mapping between component name and data.
        method : CompCorMethod
            Either anatomical or temporal CompCor.
        n_comps : int or float
            If integer, the number of top components to select.
            If float, the proportion of cumulative variance to capture.
        mask: CompCorMask or None
            ROI where the decomposition that generated the component was performed.
            Applicable for anatomical CompCor only.

        Returns
        -------
        list of str
            Selected component names.

        Notes
        -----
        Adapted from https://github.com/snastase/narratives/blob/master/code/extract_confounds.py.
        """
        # Ensure a sensible number of components is requested
        assert n_comps > 0, "`n_comps` must be positive"

        # Get CompCor metadata for relevant method
        compcor_meta = {
            k: v
            for k, v in confounds_meta.items()
            if v.Method == method and v.Retained is True
        }

        # Apply method-specific processing
        if method == CompCorMethod.ANATOMICAL:
            assert mask is not None, "Mask must be specified for aCompCor"
            compcor_meta = {k: v for k, v in compcor_meta.items() if v.Mask == mask}
        elif method == CompCorMethod.TEMPORAL:
            if mask:
                self._logger.warning(
                    "tCompCor is not restricted to a mask "
                    "- ignoring mask specification (%s)",
                    mask.value,
                )
                mask = None  # Ignore (not applicable)
        else:
            raise ValueError(f"Unsupported CompCor method: {method}")

        # Sort metadata components
        comps_sorted = sorted(
            compcor_meta,
            key=lambda k: (compcor_meta[k].SingularValue or 0.0),
            reverse=True,
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
                if (compcor_meta[comp].CumulativeVarianceExplained or 1.0) > n_comps:
                    break

        # Check we didn't end up with degenerate 0 components
        assert len(comps_selected) > 0, "Zero components selected"

        return comps_selected

    @staticmethod
    def _compose_glob_pattern_for_bold(
        subject_id: str,
        session_name: str,
        task_name: str,
        data_space: VolumeSpace | SurfaceSpace,
    ) -> str:
        """
        Compose a file name pattern to match.

        Parameters
        ----------
        subject_id : str
            ID of the study participant (should correspond to fMRIPrep output).
        session_name : str
            Name of the hyperscanning session (should correspond to fMRIPrep output).
        task_name : str
            Name of the hyperscanning task (should correspond to fMRIPrep output).
        data_space : VolumeSpace or SurfaceSpace
            A spatial representation of the dataset.
            Use `VolumeSpace` for volumetric data (e.g., MNI152NLin2009cAsym),
            and `SurfaceSpace` for surface-based data (e.g., fsaverage6).

        Returns
        -------
        str
            Corresponding file name pattern.
        """
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
