import numpy as np
import pytest

from hypline.downsample import downsample


def _features(rows: list[list[float]]) -> np.ndarray:
    return np.array(rows, dtype=np.float64)


def _times(seconds: list[float]) -> np.ndarray:
    return np.array(seconds, dtype=np.float64)


class TestDownsample:
    # Fast path

    def test_pass_through_when_already_tr_aligned(self):
        result = downsample(
            _features([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            start_times=_times([0.0, 2.0, 4.0]),
            n_trs=3,
            repetition_time=2.0,
            method="mean",
        )
        assert np.allclose(result, [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    def test_pass_through_for_phase_shifted_tr_aligned_data(self):
        result = downsample(
            _features([[1.0, 2.0], [3.0, 4.0]]),
            start_times=_times([0.5, 2.5]),
            n_trs=2,
            repetition_time=2.0,
            method="mean",
        )
        assert np.allclose(result, [[1.0, 2.0], [3.0, 4.0]])

    def test_single_tr_falls_through_to_binning(self):
        result = downsample(
            _features([[1.0, 2.0]]),
            start_times=_times([0.0]),
            n_trs=1,
            repetition_time=2.0,
            method="mean",
        )
        assert np.allclose(result[0], [1.0, 2.0])

    # Per-method binning

    def test_mean_multiple_events_per_tr(self):
        result = downsample(
            _features([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            start_times=_times([0.0, 0.5, 2.0]),
            n_trs=2,
            repetition_time=2.0,
            method="mean",
        )
        assert np.allclose(result[0], [2.0, 3.0])
        assert np.allclose(result[1], [5.0, 6.0])

    def test_sum_multiple_events_per_tr(self):
        result = downsample(
            _features([[1.0], [3.0], [5.0]]),
            start_times=_times([0.0, 0.5, 2.0]),
            n_trs=2,
            repetition_time=2.0,
            method="sum",
        )
        assert np.allclose(result[0], [4.0])
        assert np.allclose(result[1], [5.0])

    def test_max_multiple_events_per_tr(self):
        result = downsample(
            _features([[1.0], [3.0], [5.0]]),
            start_times=_times([0.0, 0.5, 2.0]),
            n_trs=2,
            repetition_time=2.0,
            method="max",
        )
        assert np.allclose(result[0], [3.0])
        assert np.allclose(result[1], [5.0])

    def test_max_empty_bins_are_zero(self):
        result = downsample(
            _features([[1.0]]),
            start_times=_times([0.0]),
            n_trs=2,
            repetition_time=2.0,
            method="max",
        )
        assert np.allclose(result[1], [0.0])

    def test_max_preserves_negative_values(self):
        result = downsample(
            _features([[-3.0], [-1.0]]),
            start_times=_times([0.0, 0.5]),
            n_trs=1,
            repetition_time=2.0,
            method="max",
        )
        assert result[0] == -1.0

    def test_min_multiple_events_per_tr(self):
        result = downsample(
            _features([[1.0], [3.0], [5.0]]),
            start_times=_times([0.0, 0.5, 2.0]),
            n_trs=2,
            repetition_time=2.0,
            method="min",
        )
        assert np.allclose(result[0], [1.0])
        assert np.allclose(result[1], [5.0])

    def test_min_empty_bins_are_zero(self):
        result = downsample(
            _features([[1.0]]),
            start_times=_times([0.0]),
            n_trs=2,
            repetition_time=2.0,
            method="min",
        )
        assert np.allclose(result[1], [0.0])

    def test_min_preserves_positive_values(self):
        result = downsample(
            _features([[3.0], [1.0]]),
            start_times=_times([0.0, 0.5]),
            n_trs=1,
            repetition_time=2.0,
            method="min",
        )
        assert result[0] == 1.0

    def test_any_returns_1d_bin_level_indicator(self):
        # `any` ignores `values` and per-column structure; result is bin-level
        result = downsample(
            _features([[1.0, 0.0], [0.0, 0.0]]),
            start_times=_times([0.0, 2.0]),
            n_trs=3,
            repetition_time=2.0,
            method="any",
        )
        assert result.shape == (3,)
        assert np.allclose(result, [1.0, 1.0, 0.0])

    def test_count_returns_events_per_bin(self):
        result = downsample(
            _features([[0.0], [0.0], [0.0]]),
            start_times=_times([0.0, 0.5, 2.0]),
            n_trs=3,
            repetition_time=2.0,
            method="count",
        )
        assert result.shape == (3,)
        assert np.allclose(result, [2.0, 1.0, 0.0])

    # Bin boundaries and event range

    def test_event_at_onset_zero_goes_to_tr_zero(self):
        result = downsample(
            _features([[1.0, 2.0]]),
            start_times=_times([0.0]),
            n_trs=2,
            repetition_time=2.0,
            method="mean",
        )
        assert np.allclose(result[0], [1.0, 2.0])

    def test_bin_boundary_event_at_exact_tr_goes_to_next_bin(self):
        result = downsample(
            _features([[1.0, 2.0]]),
            start_times=_times([2.0]),
            n_trs=2,
            repetition_time=2.0,
            method="mean",
        )
        assert np.allclose(result[0], [0.0, 0.0])
        assert np.allclose(result[1], [1.0, 2.0])

    def test_empty_tr_bins_are_zero(self):
        result = downsample(
            _features([[1.0, 2.0]]),
            start_times=_times([0.0]),
            n_trs=3,
            repetition_time=2.0,
            method="mean",
        )
        assert np.allclose(result[1], [0.0, 0.0])
        assert np.allclose(result[2], [0.0, 0.0])

    def test_events_outside_tr_range_are_ignored(self):
        result = downsample(
            _features([[1.0, 2.0], [9.0, 9.0]]),
            start_times=_times([0.0, 10.0]),
            n_trs=2,
            repetition_time=2.0,
            method="mean",
        )
        assert np.allclose(result[0], [1.0, 2.0])
        assert np.allclose(result[1], [0.0, 0.0])

    # Output shape and dtype

    def test_output_shape_2d(self):
        result = downsample(
            _features([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            start_times=_times([0.0, 0.5, 2.0]),
            n_trs=2,
            repetition_time=2.0,
            method="mean",
        )
        assert result.shape == (2, 2)

    def test_1d_input_returns_1d_output(self):
        result = downsample(
            np.array([1.0, 2.0, 3.0]),
            start_times=_times([0.0, 0.5, 2.0]),
            n_trs=2,
            repetition_time=2.0,
            method="sum",
        )
        assert result.shape == (2,)
        assert np.allclose(result, [3.0, 3.0])

    def test_output_dtype_is_float64(self):
        result = downsample(
            _features([[1.0, 2.0], [3.0, 4.0]]),
            start_times=_times([0.0, 2.0]),
            n_trs=2,
            repetition_time=2.0,
            method="mean",
        )
        assert result.dtype == np.float64

    # Error and edge inputs

    def test_n_trs_zero_raises(self):
        with pytest.raises(ValueError, match="n_trs must be positive"):
            downsample(
                _features([[1.0], [2.0]]),
                start_times=_times([0.0, 2.0]),
                n_trs=0,
                repetition_time=2.0,
                method="mean",
            )

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            downsample(
                _features([[1.0]]),
                start_times=_times([0.0]),
                n_trs=1,
                repetition_time=2.0,
                method="invalid",  # type: ignore[arg-type]
            )

    def test_nan_start_time_raises(self):
        with pytest.raises(ValueError, match="NaN"):
            downsample(
                _features([[1.0], [2.0]]),
                start_times=_times([0.0, np.nan]),
                n_trs=2,
                repetition_time=2.0,
                method="count",
            )

    def test_empty_input_returns_zero_filled_output(self):
        result = downsample(
            np.empty((0, 2), dtype=np.float64),
            start_times=np.array([]),
            n_trs=3,
            repetition_time=2.0,
            method="mean",
        )
        assert result.shape == (3, 2)
        assert np.allclose(result, np.zeros((3, 2)))
