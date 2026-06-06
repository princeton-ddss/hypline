from pathlib import Path

import nibabel as nib
import numpy as np
from loguru import logger
from nibabel.gifti import GiftiDataArray, GiftiImage

from hypline.bids import (
    BOLD_IDENTITY_ENTITIES,
    BIDSPath,
    normalize_bids_filters,
)
from hypline.bold import (
    BOLD_EXTENSIONS,
    get_repetition_time,
    parse_bold_space,
)
from hypline.enums import SurfaceSpace, VolumeSpace
from hypline.fmriprep import (
    parse_compcor,
    read_fmriprep_confounds,
    select_fmriprep_columns,
)
from hypline.io import skip_existing
from hypline.layout import BIDSLayout


class Denoiser:
    """Confound regression to remove noise from preprocessed BOLD fMRI data.

    Finds `desc-preproc` BOLD in the fmriprep tree, regresses out the fmriprep
    confound columns selected by `columns`/`compcor`, and writes the cleaned run
    in place as `desc-clean`. The input descriptor is fixed as `desc-preproc` so
    the denoiser never re-cleans its own output. Volume and surface BOLD
    dispatch on `type(space)`; surface runs are per-hemisphere and cleaned
    independently.
    """

    def __init__(
        self,
        *,
        bids_root: str | Path,
        space: str,
        columns: list[str],
        compcor: list[str],
        bids_filters: list[str] | None = None,
        force: bool = False,
    ):
        if not columns and not compcor:
            raise ValueError("at least one of columns or compcor must be given")
        self._layout = BIDSLayout(bids_root)
        self._space = parse_bold_space(space)
        self._columns = columns
        self._compcor = parse_compcor(compcor)
        self._bids_filters = normalize_bids_filters(
            bids_filters, reserved={"sub", "desc", "space"}
        )
        self._force = force

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
            if skip_existing(bold.with_entity("desc", "clean").path, force=self._force):
                continue
            logger.info("Cleaning starting: {}", bold.path.name)
            clean(bold)
            logger.info("Cleaning complete: {}", bold.path.name)

    def _clean_volume(self, bold: BIDSPath) -> None:
        from nilearn import image as nimg

        img = nimg.load_img(bold.path)  # Shape of (x, y, z, TRs)
        TR = get_repetition_time(self._layout, bold)

        nuisance = self._load_nuisance(bold)  # (TRs, regressors)
        if nuisance.shape[0] != img.shape[3]:
            raise ValueError(
                f"Unequal number of TRs between BOLD and nuisance: {bold.path.name}"
            )

        cleaned = nimg.clean_img(
            img,
            confounds=nuisance,
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

        nuisance = self._load_nuisance(bold)  # (TRs, regressors)
        if nuisance.shape[0] != data.shape[0]:
            raise ValueError(
                f"Unequal number of TRs between BOLD and nuisance: {bold.path.name}"
            )

        cleaned = signal.clean(
            data,
            confounds=nuisance,
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

    def _load_nuisance(self, bold: BIDSPath) -> np.ndarray:
        """Build the `(TRs, regressors)` matrix from the run's fmriprep tsv.

        Resolves the run's `desc-confounds_timeseries.tsv` (one match or raise)
        in the fmriprep tree, reads it via `read_fmriprep_confounds`, and selects
        `columns`/`compcor` into a single flat block for nilearn.
        """
        run_filters = [
            f"{k}-{bold.entities[k]}"
            for k in BOLD_IDENTITY_ENTITIES - {"sub"}
            if k in bold.entities
        ]

        matches = self._layout.find.fmriprep(
            sub=bold.entities["sub"],
            suffix="timeseries",
            ext=".tsv",
            bids_filters=["desc-confounds", *run_filters],
        )
        if len(matches) != 1:
            raise ValueError(
                f"Expected one confounds tsv for {bold.path.name}, found {len(matches)}"
            )

        df, meta = read_fmriprep_confounds(matches[0].path)
        names = select_fmriprep_columns(
            df, meta, columns=self._columns, compcor=self._compcor
        )
        return df.select(names).to_numpy()  # (TRs, regressors)
