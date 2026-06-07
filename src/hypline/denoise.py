from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
import polars as pl
from loguru import logger
from nibabel.gifti import GiftiDataArray, GiftiImage

from hypline.bids import (
    BIDSPath,
    normalize_bids_filters,
    parse_kind_desc,
)
from hypline.bold import (
    BOLD_EXTENSIONS,
    get_repetition_time,
    parse_bold_space,
    run_identity_filters,
)
from hypline.enums import SurfaceSpace, VolumeSpace
from hypline.fmriprep import (
    parse_compcor,
    read_fmriprep_confounds,
    select_fmriprep_columns,
)
from hypline.io import read_nuisance, skip_existing
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
        custom_sources: list[str],
        custom_columns: list[str],
        bids_filters: list[str] | None = None,
        force: bool = False,
    ):
        if not columns and not compcor and not custom_sources:
            raise ValueError(
                "at least one of columns, compcor, or custom_sources must be given"
            )
        if bool(custom_sources) != bool(custom_columns):
            raise ValueError("custom_sources and custom_columns must be given together")
        self._layout = BIDSLayout(bids_root)
        self._space = parse_bold_space(space)
        self._columns = columns
        self._compcor = parse_compcor(compcor)
        self._custom_sources = [parse_kind_desc(ref) for ref in custom_sources]
        self._custom_columns = custom_columns
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
        """Build the `(rows, regressors)` matrix from all nuisance channels.

        Concatenates two channels into one block: (1) fmriprep confound columns
        selected by `columns`/`compcor` from the run's native
        `desc-confounds_timeseries.tsv`, and (2) custom nuisance columns selected
        by `custom_columns` from the `nuisance/` sources named in
        `custom_sources`. Each fmriprep/custom source is resolved one-match-or-
        raise against the run.

        Validates only *internal* consistency: every channel shares the same row
        count, and the final selected column names are unique (across custom
        sources and against fmriprep). The row count is *not* checked against the
        BOLD here — the caller compares the returned matrix against the run it is
        cleaning.
        """
        frames: list[pl.DataFrame] = []
        if self._columns or self._compcor:
            frames.append(self._load_fmriprep_block(bold))
        if self._custom_sources:
            frames.append(self._load_custom_block(bold))

        heights = {f.height for f in frames}
        if len(heights) > 1:
            raise ValueError(
                f"Nuisance channels disagree on row count for {bold.path.name}: "
                f"{sorted(heights)}"
            )

        dupes = sorted(
            n for n, c in Counter(c for f in frames for c in f.columns).items() if c > 1
        )
        if dupes:
            raise ValueError(f"Nuisance column name collision across channels: {dupes}")

        combined = pl.concat(frames, how="horizontal")
        return combined.to_numpy()  # (rows, regressors)

    def _load_fmriprep_block(self, bold: BIDSPath) -> pl.DataFrame:
        """Select `columns`/`compcor` from the run's fmriprep confounds tsv."""
        matches = self._layout.find.fmriprep(
            sub=bold.entities["sub"],
            suffix="timeseries",
            ext=".tsv",
            bids_filters=["desc-confounds", *run_identity_filters(bold)],
        )
        if len(matches) != 1:
            raise ValueError(
                f"Expected one confounds tsv for {bold.path.name}, found {len(matches)}"
            )

        df, meta = read_fmriprep_confounds(matches[0].path)
        names = select_fmriprep_columns(
            df, meta, columns=self._columns, compcor=self._compcor
        )
        return df.select(names)

    def _load_custom_block(self, bold: BIDSPath) -> pl.DataFrame:
        """Concat custom nuisance sources and select `custom_columns`.

        Each `(kind, desc)` ref resolves one-match-or-raise against the run. The
        h-concat namespace must be collision-free (selection is post-concat, so
        any duplicate column name across sources is ambiguous), and the selected
        names are validated to exist.
        """
        sources: list[pl.DataFrame] = []
        height: int | None = None
        for kind, desc in self._custom_sources:
            matches = self._layout.find.nuisance(
                sub=bold.entities["sub"],
                kind=kind,
                desc=desc,
                bids_filters=run_identity_filters(bold),
            )
            if len(matches) != 1:
                ref = kind if desc is None else f"{kind}-{desc}"
                raise ValueError(
                    f"Expected one nuisance file for {bold.path.name} ({ref}), "
                    f"found {len(matches)}"
                )
            df = read_nuisance(matches[0].path)
            # h-concat silently null-pads unequal heights, defeating read_nuisance's
            # finiteness guarantee; check per-file to fail loud first
            if height is not None and df.height != height:
                raise ValueError(
                    f"Custom nuisance row count mismatch for {bold.path.name}: "
                    f"{matches[0].path.name} has {df.height}, expected {height}"
                )
            height = df.height
            sources.append(df)

        dupes = sorted(
            n
            for n, c in Counter(c for f in sources for c in f.columns).items()
            if c > 1
        )
        if dupes:
            raise ValueError(f"Duplicate custom nuisance column(s): {dupes}")

        concat = pl.concat(sources, how="horizontal")
        missing = [c for c in self._custom_columns if c not in concat.columns]
        if missing:
            raise ValueError(f"Custom nuisance columns missing from sources: {missing}")
        return concat.select(self._custom_columns)
