from typing import Literal, get_args

import numpy as np

DownsampleMethod = Literal["mean", "sum", "max", "min", "any", "count"]

FeatureDownsampleMethod = Literal["mean", "sum"]

# Public Encoding-facing methods must be a subset of all methods
if not set(get_args(FeatureDownsampleMethod)) <= set(get_args(DownsampleMethod)):
    raise RuntimeError("FeatureDownsampleMethod must be a subset of DownsampleMethod")


def downsample(
    values: np.ndarray,
    *,
    start_times: np.ndarray,
    n_trs: int,
    repetition_time: float,
    method: DownsampleMethod,
) -> np.ndarray:
    """Downsample an event-level array to TR resolution.

    If `start_times` already form a TR-cadence grid (count equals `n_trs`
    and uniform spacing equals `repetition_time`), `values` is returned
    as a copy unchanged. Otherwise each row is binned by
    `floor(start_time / repetition_time)` and aggregated.

    Parameters
    ----------
    values
        Shape `(n_events,)` or `(n_events, dim)`. Ignored when
        `method` is `"any"` or `"count"`, but a correctly shaped array
        is still required.
    start_times
        Shape `(n_events,)`, seconds from the start of the source file.
    n_trs
        Number of TR bins in the output.
    repetition_time
        TR duration in seconds.
    method
        Aggregation method: `"mean"`, `"sum"`, `"max"`, `"min"`,
        `"any"` (1 if any event falls in the bin, else 0), or
        `"count"` (number of events per bin; `values` is ignored).

    Raises
    ------
    ValueError
        If `n_trs` is not positive, `method` is unrecognized, or
        `start_times` contains NaN (null timestamps must be cleaned
        upstream, not silently binned).

    Returns
    -------
    np.ndarray
        Shape `(n_trs,)` or `(n_trs, dim)`, matching the input
        dimensionality. `method="any"` and `method="count"` always
        return shape `(n_trs,)`. Empty bins are `0.0` for every method,
        including `max`/`min` — callers cannot distinguish "empty" from
        "true zero."

    Notes
    -----
    Assumes each event's duration ≤ TR. Events spanning multiple TRs are
    misassigned by start-time binning.
    """
    if n_trs <= 0:
        raise ValueError(f"n_trs must be positive, got {n_trs}")
    if method not in get_args(DownsampleMethod):
        raise ValueError(f"Unrecognized method: {method!r}")
    if np.isnan(start_times).any():
        raise ValueError("start_times contains NaN; clean null timestamps upstream")

    squeeze = values.ndim == 1
    if squeeze:
        values = values[:, np.newaxis]
    dim = values.shape[1]

    # Pass through if already at TR level
    if len(start_times) == n_trs:
        intervals = np.diff(start_times)
        if len(intervals) > 0 and np.allclose(intervals, repetition_time):
            out = values.astype(np.float64, copy=True)
            return out.squeeze(axis=1) if squeeze else out

    bins = np.floor(start_times / repetition_time).astype(int)
    mask = (bins >= 0) & (bins < n_trs)
    valid_bins = bins[mask]
    valid_values = values[mask]

    # `any` and `count` are bin-level; ignore `dim` and return 1-D
    if method == "any":
        result_1d = np.zeros(n_trs, dtype=np.float64)
        result_1d[np.unique(valid_bins)] = 1.0
        return result_1d
    if method == "count":
        counts = np.zeros(n_trs, dtype=np.intp)
        np.add.at(counts, valid_bins, 1)
        return counts.astype(np.float64)

    if method == "mean":
        result = np.zeros((n_trs, dim), dtype=np.float64)
        counts = np.zeros(n_trs, dtype=np.intp)
        np.add.at(result, valid_bins, valid_values)
        np.add.at(counts, valid_bins, 1)
        nonzero = counts > 0
        result[nonzero] /= counts[nonzero, np.newaxis]
    elif method == "sum":
        result = np.zeros((n_trs, dim), dtype=np.float64)
        np.add.at(result, valid_bins, valid_values)
    elif method == "max":
        result = np.full((n_trs, dim), -np.inf, dtype=np.float64)
        np.maximum.at(result, valid_bins, valid_values)
        result[result == -np.inf] = 0.0
    elif method == "min":
        result = np.full((n_trs, dim), np.inf, dtype=np.float64)
        np.minimum.at(result, valid_bins, valid_values)
        result[result == np.inf] = 0.0
    else:
        raise NotImplementedError(f"Unhandled method: {method}")

    return result.squeeze(axis=1) if squeeze else result
