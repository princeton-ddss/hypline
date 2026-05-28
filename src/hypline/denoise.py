from enum import StrEnum
from pathlib import Path

import nibabel as nib
import polars as pl
from loguru import logger
from nibabel.gifti import GiftiDataArray, GiftiImage
from pydantic import BaseModel, Field, PositiveFloat, PositiveInt, TypeAdapter

from hypline.bids import (
    BOLD_IDENTITY_ENTITIES,
    BIDSPath,
    normalize_bids_filters,
    parse_kind_desc,
)
from hypline.bold import BOLD_EXTENSIONS, get_n_trs, get_repetition_time
from hypline.enums import SurfaceSpace, VolumeSpace
from hypline.io import read_confound, stack_array_column
from hypline.layout import BIDSLayout


class CompCorMethod(StrEnum):
    ANATOMICAL = "aCompCor"
    TEMPORAL = "tCompCor"
    MEAN = "Mean"


class CompCorMask(StrEnum):
    CSF = "CSF"
    WM = "WM"
    COMBINED = "combined"


class CompCorOptions(BaseModel):
    n_comps: PositiveInt | PositiveFloat = 5
    mask: CompCorMask | None = None


class ConfoundMetadata(BaseModel):
    Method: CompCorMethod
    Retained: bool | None = None
    Mask: CompCorMask | None = None
    SingularValue: float | None = None
    VarianceExplained: float | None = None
    CumulativeVarianceExplained: float | None = None


class ModelSpec(BaseModel):
    confounds: list[str] = Field(min_length=1)
    custom_confounds: list[str] | None = None  # <kind> or <kind>-<desc>
    aCompCor: list[CompCorOptions] | None = None
    tCompCor: list[CompCorOptions] | None = None


class Config(BaseModel):
    model_specs: dict[str, ModelSpec]


def _identity_filters(bold: BIDSPath) -> list[str]:
    """Build `entity-value` filters from a BOLD run's identity entities.

    Includes `sub`; finders take `sub` as a dedicated argument, so callers
    strip it before passing the rest as `bids_filters`.
    """
    return [f"{k}-{v}" for k, v in bold.entities.items() if k in BOLD_IDENTITY_ENTITIES]


class Denoiser:
    """Confound regression to remove noise from preprocessed BOLD fMRI data.

    Finds `desc-preproc` BOLD in the fmriprep tree, regresses out the confounds
    named by `model_spec`, and writes the cleaned run in place as `desc-clean`.
    The input descriptor is fixed as `desc-preproc` so the denoiser never
    re-cleans its own output. Volume and surface BOLD dispatch on `type(space)`;
    surface runs are per-hemisphere and cleaned independently.
    """

    def __init__(
        self,
        model_spec: ModelSpec,
        *,
        bids_root: str | Path,
        space: SurfaceSpace | VolumeSpace,
        bids_filters: list[str] | None = None,
    ):
        self._model_spec = model_spec
        self._layout = BIDSLayout(bids_root)
        self._space = space
        self._custom_confounds = [
            parse_kind_desc(entry) for entry in (model_spec.custom_confounds or [])
        ]
        self._bids_filters = normalize_bids_filters(
            bids_filters, reserved={"sub", "desc", "space"}
        )

    def denoise(self, sub_id: str) -> None:
        clean = {
            VolumeSpace: self._clean_volume,
            SurfaceSpace: self._clean_surface,
        }[type(self._space)]

        bolds = self._layout.find.fmriprep(
            sub=sub_id,
            suffix="bold",
            ext=BOLD_EXTENSIONS[type(self._space)],
            bids_filters=[
                "desc-preproc",
                f"space-{self._space.value}",
                *self._bids_filters,
            ],
        )

        for bold in bolds:
            logger.info("Cleaning starting: {}", bold.path.name)
            clean(bold)
            logger.info("Cleaning complete: {}", bold.path.name)

    def _clean_volume(self, bold: BIDSPath) -> None:
        from nilearn import image as nimg

        img = nimg.load_img(bold.path)  # Shape of (x, y, z, TRs)
        TR = get_repetition_time(self._layout, bold)

        confounds = self._load_confounds(bold).to_numpy()  # (TRs, confounds)
        if confounds.shape[0] != get_n_trs(bold):
            raise ValueError(
                f"Unequal number of TRs between BOLD and confounds: {bold.path.name}"
            )

        cleaned = nimg.clean_img(
            img,
            confounds=confounds,
            detrend=True,
            t_r=TR,
            ensure_finite=True,
            standardize="zscore_sample",
            standardize_confounds=True,
        )
        nib.save(cleaned, bold.with_entity("desc", "clean").path)

    def _clean_surface(self, bold: BIDSPath) -> None:
        from nilearn import signal

        img = nib.load(bold.path)
        assert isinstance(img, GiftiImage)
        data = img.agg_data().T  # Shape of (TRs, voxels)
        TR = get_repetition_time(self._layout, bold)

        confounds = self._load_confounds(bold).to_numpy()  # (TRs, confounds)
        if confounds.shape[0] != data.shape[0]:
            raise ValueError(
                f"Unequal number of TRs between BOLD and confounds: {bold.path.name}"
            )

        cleaned = signal.clean(
            data,
            confounds=confounds,
            detrend=True,
            t_r=TR,
            ensure_finite=True,
            standardize="zscore_sample",
            standardize_confounds=True,
        )
        new_img = GiftiImage(
            darrays=[
                GiftiDataArray(data=row, intent="NIFTI_INTENT_TIME_SERIES")
                for row in cleaned
            ],
            header=img.header,
            extra=img.extra,
        )
        nib.save(new_img, bold.with_entity("desc", "clean").path)

    def _load_confounds(self, bold: BIDSPath) -> pl.DataFrame:
        """Load all confounds (standard + custom) for the given BOLD run."""
        confounds_bids = self._layout.find.fmriprep(
            sub=bold.entities["sub"],
            suffix="timeseries",
            ext=".tsv",
            bids_filters=[
                "desc-confounds",
                *(f for f in _identity_filters(bold) if not f.startswith("sub-")),
            ],
        )
        if len(confounds_bids) != 1:
            raise ValueError(
                f"Expected one confounds file for {bold.path.name}, "
                f"found {len(confounds_bids)}"
            )
        confounds_path = confounds_bids[0].path

        confounds_df = (
            pl.read_csv(confounds_path, separator="\t")
            .fill_nan(None)  # For interpolation
            .fill_null(strategy="backward")  # Assume missing data at the beginning only
        )
        confounds_meta = TypeAdapter(dict[str, ConfoundMetadata]).validate_json(
            confounds_path.with_suffix(".json").read_text()  # JSON assumed present
        )
        confounds_df = self._extract_confounds(confounds_df, confounds_meta)

        if self._custom_confounds:
            custom_df = self._load_custom_confounds(bold)
            if custom_df.height != confounds_df.height:
                raise ValueError(
                    "Unequal number of rows (TRs) between standard and "
                    f"custom confounds data: {bold.path.name}"
                )
            confounds_df = pl.concat([confounds_df, custom_df], how="horizontal")

        return confounds_df

    def _load_custom_confounds(self, bold: BIDSPath) -> pl.DataFrame:
        """Read requested custom confounds from the `confounds/` parquet area.

        Each spec entry resolves to one `conf-<kind>[-<desc>]` file for the run.
        A confound's `confound` column is a fixed-width array; it expands to one
        scalar column per element so the concatenated frame yields a flat
        `(TRs, regressors)` matrix for nilearn. Distinct entries that resolve to
        the same file are loaded once.
        """
        run_filters = [f for f in _identity_filters(bold) if not f.startswith("sub-")]

        columns: dict[str, list[float]] = {}
        seen: set[Path] = set()
        for kind, desc in self._custom_confounds:
            matches = self._layout.find.confounds(
                sub=bold.entities["sub"],
                kind=kind,
                desc=desc,
                bids_filters=run_filters,
            )
            if len(matches) != 1:
                label = f"{kind}-{desc}" if desc else kind
                raise ValueError(
                    f"Expected one {label!r} confound for {bold.path.name}, "
                    f"found {len(matches)}"
                )
            path = matches[0].path
            if path in seen:
                continue
            seen.add(path)

            block = stack_array_column(read_confound(path).get_column("confound"))
            name = f"{kind}-{desc}" if desc else kind
            for i in range(block.shape[1]):
                columns[f"{name}_{i}"] = block[:, i].tolist()

        return pl.DataFrame(columns)

    def _extract_confounds(
        self,
        confounds_df: pl.DataFrame,
        confounds_meta: dict[str, ConfoundMetadata],
    ) -> pl.DataFrame:
        """Extract standard confounds (including CompCor ones).

        Notes
        -----
        Adapted from https://github.com/snastase/narratives/blob/master/code/extract_confounds.py.
        """
        model_spec = self._model_spec

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
        compcors = [c for c in CompCorMethod if c in model_spec.model_fields_set]
        if compcors:
            comps_selected: list[str] = []
            for compcor in compcors:
                for options in getattr(model_spec, compcor):
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
        """Select relevant CompCor components.

        Parameters
        ----------
        n_comps : int or float
            If integer, the number of top components to select.
            If float, the proportion of cumulative variance to capture.
        mask: CompCorMask or None
            ROI where the decomposition that generated the component was performed.
            Applicable for anatomical CompCor only.

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
                logger.warning(
                    "tCompCor is not restricted to a mask "
                    "- ignoring mask specification ({})",
                    mask,
                )
                mask = None  # Ignore (not applicable)
        else:
            raise ValueError(f"Unsupported CompCor method: {method}")

        # Sort metadata components
        comps_sorted = sorted(
            compcor_meta,
            key=lambda k: compcor_meta[k].SingularValue or 0.0,
            reverse=True,
        )

        # Either get top n components
        if n_comps >= 1.0:
            n_comps = int(n_comps)
            if len(comps_sorted) >= n_comps:
                comps_selected = comps_sorted[:n_comps]
            else:
                comps_selected = comps_sorted
                logger.warning(
                    "Only {} {} components available ({} requested)",
                    len(comps_sorted),
                    method,
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
