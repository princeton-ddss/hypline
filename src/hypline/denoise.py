from pathlib import Path

import nibabel as nib
import polars as pl
from loguru import logger
from nibabel.gifti import GiftiDataArray, GiftiImage

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


def _identity_filters(bold: BIDSPath) -> list[str]:
    """Build `entity-value` filters from a BOLD run's identity entities.

    Includes `sub`; finders take `sub` as a dedicated argument, so callers
    strip it before passing the rest as `bids_filters`.
    """
    return [f"{k}-{v}" for k, v in bold.entities.items() if k in BOLD_IDENTITY_ENTITIES]


class Denoiser:
    """Confound regression to remove noise from preprocessed BOLD fMRI data.

    Finds `desc-preproc` BOLD in the fmriprep tree, regresses out the confounds
    named by `confounds`, and writes the cleaned run in place as `desc-clean`.
    The input descriptor is fixed as `desc-preproc` so the denoiser never
    re-cleans its own output. Volume and surface BOLD dispatch on `type(space)`;
    surface runs are per-hemisphere and cleaned independently.
    """

    def __init__(
        self,
        *,
        bids_root: str | Path,
        space: SurfaceSpace | VolumeSpace,
        confounds: list[str],
        bids_filters: list[str] | None = None,
    ):
        # Each entry is a `<kind>-<desc>` (or bare `<kind>`) ref into `confounds/`;
        # fmriprep tsv columns arrive as `conf-fmriprep_desc-*` bundles too.
        if not confounds:
            raise ValueError("confounds must be non-empty")
        self._layout = BIDSLayout(bids_root)
        self._space = space
        self._confounds = [parse_kind_desc(entry) for entry in confounds]
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
        """Load all confounds for the given BOLD run from `confounds/`.

        Each `confounds` ref resolves to one `conf-<kind>[-<desc>]` parquet file
        for the run. A file's `confound` column is a fixed-width array; it
        expands to one scalar column per element so the concatenated frame
        yields a flat `(TRs, regressors)` matrix for nilearn. Distinct refs that
        resolve to the same file are loaded once. All bundles must share the
        same row count (TRs); a mismatch fails fast.
        """
        run_filters = [f for f in _identity_filters(bold) if not f.startswith("sub-")]

        columns: dict[str, list[float]] = {}
        seen: set[Path] = set()
        n_trs: int | None = None
        for kind, desc in self._confounds:
            matches = self._layout.find.confounds(
                sub=bold.entities["sub"],
                kind=kind,
                desc=desc,
                bids_filters=run_filters,
            )
            label = f"{kind}-{desc}" if desc else kind
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one {label!r} confound for {bold.path.name}, "
                    f"found {len(matches)}"
                )
            path = matches[0].path
            if path in seen:
                continue
            seen.add(path)

            block = stack_array_column(read_confound(path).get_column("confound"))
            if n_trs is None:
                n_trs = block.shape[0]
            elif block.shape[0] != n_trs:
                raise ValueError(
                    "Unequal number of rows (TRs) between confound bundles: "
                    f"{label!r} has {block.shape[0]}, expected {n_trs} "
                    f"({bold.path.name})"
                )
            for i in range(block.shape[1]):
                columns[f"{label}_{i}"] = block[:, i].tolist()

        return pl.DataFrame(columns)
