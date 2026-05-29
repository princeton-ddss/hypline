from enum import StrEnum
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger
from pydantic import BaseModel, PositiveFloat, PositiveInt, TypeAdapter

from hypline.bids import BIDS_ENTITY_VALUE_RE, BIDSPath, normalize_bids_filters
from hypline.bold import get_n_trs, get_repetition_time
from hypline.io import write_confound
from hypline.layout import BIDSLayout

# fmriprep writes confound columns already aligned to the BOLD TR grid; no
# resampling happens here, so all `conf-fmriprep` bundles share this marker.
_TR_METHOD = "native"

# Confound names that denote a variable-size group rather than a single column;
# each expands to every tsv column containing the token.
_GROUP_PREFIXES = ("cosine", "motion_outlier")


class CompCorMethod(StrEnum):
    ANATOMICAL = "aCompCor"
    TEMPORAL = "tCompCor"
    MEAN = "Mean"  # appears in sidecar metadata only; never a selectable method


class CompCorMask(StrEnum):
    CSF = "CSF"
    WM = "WM"
    COMBINED = "combined"


class CompCorOptions(BaseModel):
    # Optional default only; `_validate_compcor` rejects None.
    method: CompCorMethod | None = None
    n_comps: PositiveInt | PositiveFloat = 5
    mask: CompCorMask | None = None


class ConfoundMetadata(BaseModel):
    Method: CompCorMethod
    Retained: bool | None = None
    Mask: CompCorMask | None = None
    SingularValue: float | None = None
    VarianceExplained: float | None = None
    CumulativeVarianceExplained: float | None = None


def _validate_compcor(options: CompCorOptions) -> None:
    """Enforce the mask-iff-aCompCor invariant on a single selector.

    The shared `CompCorOptions` model permits `method=None` and any mask (the
    denoiser still relies on that), so the rule is checked here rather than on
    the model. aCompCor requires a mask; tCompCor must not carry one.
    """
    if options.method is None:
        raise ValueError("CompCor method must be set (aCompCor or tCompCor)")
    if options.method == CompCorMethod.ANATOMICAL and options.mask is None:
        raise ValueError("aCompCor requires a mask")
    if options.method == CompCorMethod.TEMPORAL and options.mask is not None:
        raise ValueError(f"tCompCor must not carry a mask, got {options.mask}")


def _select_comps(
    confounds_meta: dict[str, ConfoundMetadata],
    method: CompCorMethod,
    *,
    n_comps: int | float,
    mask: CompCorMask | None,
) -> list[str]:
    """Select relevant CompCor components from sidecar metadata.

    `n_comps` selects either the top N components (integer) or the fewest
    components capturing that proportion of cumulative variance (float < 1).
    `mask` restricts anatomical components to the ROI their decomposition ran
    in; it is ignored for temporal CompCor.

    Notes
    -----
    Adapted from https://github.com/snastase/narratives/blob/master/code/extract_confounds.py.
    """
    assert n_comps > 0, "`n_comps` must be positive"

    compcor_meta = {
        k: v
        for k, v in confounds_meta.items()
        if v.Method == method and v.Retained is True
    }

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

    assert len(comps_selected) > 0, "Zero components selected"

    return comps_selected


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
    ):
        if not BIDS_ENTITY_VALUE_RE.match(desc):
            raise ValueError(f"desc must be alphanumeric, got {desc!r}")
        for options in compcor:
            _validate_compcor(options)

        self._layout = BIDSLayout(bids_root)
        self._desc = desc
        self._columns = columns
        self._compcor = compcor
        self._bids_filters = normalize_bids_filters(bids_filters, reserved={"sub"})

    def generate(self, sub_id: str):
        tsv_files = self._layout.find.fmriprep(
            sub=sub_id,
            suffix="timeseries",
            ext=".tsv",
            bids_filters=["desc-confounds", *(self._bids_filters or [])],
        )

        for tsv in tsv_files:
            logger.info("Generating fmriprep confounds for {}", tsv.path.name)
            df = (
                pl.read_csv(tsv.path, separator="\t")
                .fill_nan(None)  # For interpolation
                .fill_null(strategy="backward")  # Assume missing data at the start only
            )
            meta = TypeAdapter(dict[str, ConfoundMetadata]).validate_json(
                tsv.path.with_suffix(".json").read_text()  # JSON assumed present
            )

            names = self._select_columns(df, meta)
            block = df.select(names).to_numpy()  # (n_trs, N)

            self._assert_n_trs(tsv, n_trs=block.shape[0])
            repetition_time = get_repetition_time(self._layout, tsv)
            out = self._layout.path.confound(
                source=tsv,
                kind="fmriprep",
                desc=self._desc,
            )
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

    def _select_columns(
        self,
        df: pl.DataFrame,
        meta: dict[str, ConfoundMetadata],
    ) -> list[str]:
        """Resolve all selected confound column names, in deterministic order.

        Name-based `columns` come first (literals, then group-prefix matches in
        column order), then compcor groups in the order given. Raises if any
        named literal is absent from the tsv.
        """
        groups = set(self._columns).intersection(_GROUP_PREFIXES)
        names = [c for c in self._columns if c not in groups]

        if groups:
            names.extend(col for col in df.columns if any(g in col for g in groups))

        for options in self._compcor:
            assert options.method is not None  # guaranteed by _validate_compcor
            names.extend(
                _select_comps(
                    meta, options.method, n_comps=options.n_comps, mask=options.mask
                )
            )

        if not set(names).issubset(df.columns):
            missing = sorted(set(names) - set(df.columns))
            raise ValueError(f"Confound columns missing from tsv: {missing}")

        return names
