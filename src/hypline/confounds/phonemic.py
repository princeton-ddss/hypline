from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger

from hypline.bold import BOLD_EXTENSIONS, load_bold_meta
from hypline.downsample import DownsampleMethod, downsample
from hypline.enums import VolumeSpace
from hypline.io import read_feature, write_confound
from hypline.layout import BIDSLayout

from ._utils import collapse_desc_variants, segment_n_trs

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
    ):
        self._layout = BIDSLayout(bids_root)
        self._bids_filters = bids_filters

    def generate(self, sub_id: str):
        feature_files = self._layout.find.features(
            sub=sub_id,
            kind="phonemic",
            desc="*",
            bids_filters=self._bids_filters,
        )
        feature_files = collapse_desc_variants(feature_files)

        for feat_file in feature_files:
            logger.info("Generating phonemic confounds for {}", feat_file.path.name)
            df = read_feature(feat_file.path)
            start_times = df.get_column("start_time").to_numpy()

            raw_bold = self._layout.path.raw(
                source=feat_file,
                suffix="bold",
                ext=BOLD_EXTENSIONS[VolumeSpace],
            )
            bold_meta = load_bold_meta(self._layout, raw_bold)
            n_trs = segment_n_trs(feat_file, bold_meta)

            for desc, method in _VARIANTS:
                series = downsample(
                    np.zeros(len(start_times)),
                    start_times=start_times,
                    n_trs=n_trs,
                    repetition_time=bold_meta.repetition_time,
                    method=method,
                )
                out = self._layout.path.confound(
                    source=feat_file,
                    kind="phonemic",
                    desc=desc,
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
