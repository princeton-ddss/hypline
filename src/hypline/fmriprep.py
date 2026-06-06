"""Reading and selecting from fmriprep's `desc-confounds_timeseries.tsv`.

Format knowledge for fmriprep confounds: the CompCor selector types, the
sidecar metadata schema, and the column-selection logic shared by denoise (which
reads these natively) and the `conf-fmriprep` generator. Parsing/validation of
user input lives at the CLI; the helpers here take trusted typed values.
"""

from enum import StrEnum
from pathlib import Path

import polars as pl
from loguru import logger
from pydantic import BaseModel, PositiveFloat, PositiveInt, TypeAdapter


class CompCorMethod(StrEnum):
    ANATOMICAL = "aCompCor"
    TEMPORAL = "tCompCor"
    MEAN = "Mean"  # appears in sidecar metadata only; never a selectable method


class CompCorMask(StrEnum):
    CSF = "CSF"
    WM = "WM"
    COMBINED = "combined"


class CompCorOptions(BaseModel):
    # Optional default only; `validate_compcor` rejects None.
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


def validate_compcor(options: CompCorOptions) -> None:
    """Enforce the mask-iff-aCompCor invariant on a single selector.

    The rule lives here rather than on `CompCorOptions` because that model
    intentionally permits `method=None` and any mask (the denoiser relies on
    that permissiveness).
    """
    if options.method is None:
        raise ValueError("CompCor method must be set (aCompCor or tCompCor)")
    if options.method == CompCorMethod.ANATOMICAL and options.mask is None:
        raise ValueError("aCompCor requires a mask")
    if options.method == CompCorMethod.TEMPORAL and options.mask is not None:
        raise ValueError(f"tCompCor must not carry a mask, got {options.mask}")


def _parse_n_comps(n_str: str, *, token: str) -> int | float:
    """Parse the `n` field: a positive int (top-N) or positive float (variance)."""
    if not n_str:
        raise ValueError(f"'{token}' missing n (e.g., a:CSF:5)")
    try:
        n_comps: int | float = int(n_str)
    except ValueError:
        try:
            n_comps = float(n_str)
        except ValueError:
            raise ValueError(f"'{token}' n must be a number, got '{n_str}'") from None
    if n_comps <= 0:
        raise ValueError(f"'{token}' n must be positive")
    return n_comps


_COMPCOR_METHODS = {"a": CompCorMethod.ANATOMICAL, "t": CompCorMethod.TEMPORAL}


def parse_compcor(tokens: list[str]) -> list[CompCorOptions]:
    """Parse `[a|t]:[mask]:[n]` selector tokens into validated CompCorOptions.

    Each token has fixed 3-field arity. `a`=aCompCor (mask required), `t`=tCompCor
    (mask slot must be empty — tCompCor is not mask-restricted). `n` is a positive
    int (top-N) or float (variance fraction). Owns the full compcor grammar
    including the mask-iff-method rule; raises `ValueError` on any malformed token.
    """
    options: list[CompCorOptions] = []
    for token in tokens:
        fields = token.split(":")
        if len(fields) != 3:
            raise ValueError(
                f"'{token}' must have 3 colon-separated fields ([a|t]:[mask]:[n])"
            )
        method_str, mask_str, n_str = fields

        if method_str not in _COMPCOR_METHODS:
            raise ValueError(f"'{token}' method must be 'a' or 't', got '{method_str}'")
        method = _COMPCOR_METHODS[method_str]

        mask: CompCorMask | None = None
        if method is CompCorMethod.ANATOMICAL:
            if not mask_str:
                raise ValueError(f"'{token}' aCompCor requires a mask (e.g., a:CSF:5)")
            try:
                mask = CompCorMask(mask_str)
            except ValueError:
                allowed = ", ".join(m.value for m in CompCorMask)
                raise ValueError(f"'{token}' mask must be one of: {allowed}") from None
        elif mask_str:
            raise ValueError(
                f"'{token}' tCompCor is not mask-restricted — leave mask empty "
                "(e.g., t::10)"
            )

        n_comps = _parse_n_comps(n_str, token=token)
        options.append(CompCorOptions(method=method, n_comps=n_comps, mask=mask))

    return options


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


def read_fmriprep_confounds(
    path: str | Path,
) -> tuple[pl.DataFrame, dict[str, ConfoundMetadata]]:
    """Read a fmriprep `desc-confounds_timeseries.tsv` and its JSON sidecar.

    fmriprep writes leading `n/a` cells by design (`*_derivative1`,
    `framewise_displacement`, `dvars`, `std_dvars` are NaN at row 0). `n/a` must
    be parsed as null (polars does not do this by default — an otherwise-numeric
    column would infer as string), then backfilled so a selected derivative
    column does not become a string or NaN regressor. The `.json` sidecar
    (assumed present) carries the CompCor component metadata.
    """
    path = Path(path)
    df = (
        pl.read_csv(path, separator="\t", null_values=["n/a"])
        .fill_nan(None)  # For interpolation
        .fill_null(strategy="backward")  # Assume missing data at the start only
    )
    meta = TypeAdapter(dict[str, ConfoundMetadata]).validate_json(
        path.with_suffix(".json").read_text()
    )
    return df, meta


# Confound names that denote a variable-size group rather than a single column;
# each expands to every tsv column containing the token.
_GROUP_PREFIXES = ("cosine", "motion_outlier")


def select_fmriprep_columns(
    df: pl.DataFrame,
    meta: dict[str, ConfoundMetadata],
    *,
    columns: list[str],
    compcor: list[CompCorOptions],
) -> list[str]:
    """Resolve all selected confound column names, in deterministic order.

    Name-based `columns` come first (literals, then group-prefix matches in
    column order), then compcor groups in the order given. Raises if any named
    literal is absent from the tsv.
    """
    groups = set(columns).intersection(_GROUP_PREFIXES)
    names = [c for c in columns if c not in groups]

    if groups:
        names.extend(col for col in df.columns if any(g in col for g in groups))

    for options in compcor:
        # method guaranteed set by parse_compcor / validate_compcor
        assert options.method is not None
        names.extend(
            _select_comps(
                meta, options.method, n_comps=options.n_comps, mask=options.mask
            )
        )

    if not set(names).issubset(df.columns):
        missing = sorted(set(names) - set(df.columns))
        raise ValueError(f"Confound columns missing from tsv: {missing}")

    return names
