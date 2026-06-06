from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger

from hypline.bids import BIDS_ENTITY_VALUE_RE, BIDSPath, normalize_bids_filters
from hypline.bold import get_n_trs, get_repetition_time
from hypline.fmriprep import (
    CompCorOptions,
    read_fmriprep_confounds,
    select_fmriprep_columns,
    validate_compcor,
)
from hypline.io import skip_existing, write_confound
from hypline.layout import BIDSLayout

# fmriprep writes confound columns already aligned to the BOLD TR grid; no
# resampling happens here, so all `conf-fmriprep` bundles share this marker.
_TR_METHOD = "native"


class FmriprepConfound:
    """Extract a chosen group of fmriprep confound columns into one bundle.

    Reads `desc-confounds_timeseries.tsv` (+ JSON sidecar) per run and selects
    columns two ways, additively: `columns` by name (literals plus the
    `cosine`/`motion_outlier` group prefixes) and `compcor` by metadata criteria
    resolved via the sidecar. The selected columns stack into one
    `(n_trs, N)` array written as `conf-fmriprep_desc-<desc>`. Inputs are
    trusted typed values — the CLI owns parsing and validation.
    """

    def __init__(
        self,
        *,
        bids_root: str | Path,
        desc: str,
        columns: list[str],
        compcor: list[CompCorOptions],
        bids_filters: list[str] | None = None,
        force: bool = False,
    ):
        if not BIDS_ENTITY_VALUE_RE.match(desc):
            raise ValueError(f"desc must be alphanumeric, got {desc!r}")
        for options in compcor:
            validate_compcor(options)

        self._layout = BIDSLayout(bids_root)
        self._desc = desc
        self._columns = columns
        self._compcor = compcor
        self._bids_filters = normalize_bids_filters(bids_filters, reserved={"sub"})
        self._force = force

    def generate(self, sub_id: str):
        tsv_files = self._layout.find.fmriprep(
            sub=sub_id,
            suffix="timeseries",
            ext=".tsv",
            bids_filters=["desc-confounds", *(self._bids_filters or [])],
        )

        for tsv in tsv_files:
            out = self._layout.path.confound(
                source=tsv,
                kind="fmriprep",
                desc=self._desc,
            )
            if skip_existing(out.path, force=self._force):
                continue

            logger.info("Generating fmriprep confounds for {}", tsv.path.name)
            df, meta = read_fmriprep_confounds(tsv.path)
            names = select_fmriprep_columns(
                df, meta, columns=self._columns, compcor=self._compcor
            )
            block = df.select(names).to_numpy()  # (n_trs, N)

            self._assert_n_trs(tsv, n_trs=block.shape[0])
            repetition_time = get_repetition_time(self._layout, tsv)
            out_df = pl.DataFrame(
                {
                    "start_time": np.arange(block.shape[0]) * repetition_time,
                    "confound": block.tolist(),
                },
                schema={
                    "start_time": pl.Float64,
                    "confound": pl.Array(pl.Float64, block.shape[1]),
                },
            )
            write_confound(
                out_df,
                out.path,
                repetition_time=repetition_time,
                tr_method=_TR_METHOD,
                # `_`-prefixed: per-file metadata, exempt from cross-file
                # equality — label set varies per run (motion_outlier count,
                # CompCor component names), which is expected for fmriprep.
                metadata={"_confound_dim_labels": names},
            )
            logger.debug("Wrote fmriprep confound to {}", out.path)

    def _assert_n_trs(self, tsv: BIDSPath, *, n_trs: int) -> None:
        """Assert the tsv row count matches the run's raw BOLD volume count.

        TR-aligned confound rows must equal BOLD volumes; the raw volumetric
        BOLD is the canonical anchor (surface derivatives inherit its count).
        Skipped when the raw BOLD is absent — TR resolution already requires a
        raw sidecar or image, so a missing image surfaces there.
        """
        raw_bold = self._layout.path.raw(source=tsv, suffix="bold", ext=".nii.gz")
        if not raw_bold.path.exists():
            return
        bold_n_trs = get_n_trs(raw_bold)
        if bold_n_trs != n_trs:
            raise ValueError(
                f"tsv has {n_trs} rows but raw BOLD has {bold_n_trs} volumes: "
                f"{tsv.path.name}"
            )
