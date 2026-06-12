from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger

from hypline.bids import normalize_bids_filters
from hypline.bold import load_bold_meta, resolve_bold_image
from hypline.downsample import DownsampleMethod, downsample
from hypline.io import read_feature, skip_existing, write_confound
from hypline.layout import BIDSLayout

from ._utils import pick_timing_source, segment_n_trs

_VARIANTS: tuple[tuple[str, DownsampleMethod], ...] = (
    ("onset", "any"),
    ("rate", "count"),
)


class PhonemicConfound:
    def __init__(
        self,
        *,
        bids_root: str | Path,
        bids_filters: list[str] | None = None,
        force: bool = False,
    ):
        self._layout = BIDSLayout(bids_root)
        self._bids_filters = normalize_bids_filters(
            bids_filters, reserved={"dyad", "feat", "desc"}
        )
        self._force = force

    def generate(self, dyad_id: str):
        feature_files = self._layout.find.features(
            dyad=dyad_id,
            kind="phonemic",
            desc="*",
            bids_filters=self._bids_filters,
        )
        feature_files = pick_timing_source(feature_files)

        # A dyad feature carries no `sub`, but downsampling needs n_trs/TR from a
        # real BOLD. Partners share one simultaneous scan sequence, so their TR
        # and n_trs are identical by construction — any partner stands in. Raises
        # KeyError if the dyad is unmapped.
        first_sub = self._layout.subjects_of(dyad_id)[0]

        for feat_file in feature_files:
            pending = []
            for desc, method in _VARIANTS:
                out = self._layout.path.confound(
                    source=feat_file,
                    kind="phonemic",
                    desc=desc,
                )
                if not skip_existing(out.path, force=self._force):
                    pending.append((out, method))
            if not pending:
                continue

            logger.info("Generating phonemic confounds for {}", feat_file.path.name)
            df = read_feature(feat_file.path)
            # Drop untimed rows — downsample needs TR alignment
            df = df.filter(pl.col("start_time").is_not_null())
            start_times = df.get_column("start_time").to_numpy()

            sub_source = feat_file.with_identity("sub", first_sub)
            bold = resolve_bold_image(self._layout, sub_source)
            bold_meta = load_bold_meta(self._layout, bold)
            n_trs = segment_n_trs(feat_file, bold_meta)

            for out, method in pending:
                series = downsample(
                    np.zeros(len(start_times)),
                    start_times=start_times,
                    n_trs=n_trs,
                    repetition_time=bold_meta.repetition_time,
                    method=method,
                )
                out_df = pl.DataFrame(
                    {
                        "start_time": np.arange(n_trs) * bold_meta.repetition_time,
                        "confound": series.reshape(-1, 1).tolist(),
                    },
                    schema={
                        "start_time": pl.Float64,
                        "confound": pl.Array(pl.Float64, 1),
                    },
                )
                write_confound(
                    out_df,
                    out.path,
                    repetition_time=bold_meta.repetition_time,
                    tr_method=method,
                )
                logger.debug("Wrote phonemic confound to {}", out.path)
