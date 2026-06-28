import numpy as np
import pytest

from hypline.encoding._schema import CellDelayer, CellKey


class TestCellDelayer:
    def test_cell_delayer_resets_at_boundaries(self):
        # Two stacked cells of 4 rows; values encode (cell, row) so bleed is visible
        cell_lengths = [4, 4]
        X = np.arange(1, 9, dtype=float).reshape(-1, 1)
        delays = [0, 1, 2]
        out = CellDelayer(delays=delays, cell_lengths=cell_lengths).transform(X)

        # Output columns are [delay0, delay1, delay2]; cell 2 spans rows 4..7
        # The first max(delays)=2 rows of cell 2 must not pull from cell 1
        col_d1, col_d2 = out[:, 1], out[:, 2]
        assert col_d1[4] == 0  # row 4, delay 1 would source row 3 (cell 1)
        assert col_d2[4] == 0  # row 4, delay 2 would source row 2 (cell 1)
        assert col_d2[5] == 0  # row 5, delay 2 would source row 3 (cell 1)
        # Within-cell delays still work
        assert col_d1[5] == X[4, 0]  # row 5, delay 1 sources row 4 (same cell)
        assert col_d2[6] == X[4, 0]  # row 6, delay 2 sources row 4 (same cell)

    def test_cell_delayer_single_cell_matches_plain_delay(self):
        cell_lengths = [6]
        X = np.arange(1, 7, dtype=float).reshape(-1, 1)
        delays = [0, 1, 2]
        out = CellDelayer(delays=delays, cell_lengths=cell_lengths).transform(X)

        expected_blocks = []
        for d in delays:
            block = np.zeros_like(X)
            if d == 0:
                block[:] = X
            else:
                block[d:] = X[:-d]
            expected_blocks.append(block)
        np.testing.assert_array_equal(out, np.hstack(expected_blocks))

    def test_cell_delayer_short_cell_all_zero_deep_delay(self):
        # Cell of 2 rows with a delay of 3 → that delay block is all-zero, no error
        cell_lengths = [2]
        X = np.arange(1, 3, dtype=float).reshape(-1, 1)
        out = CellDelayer(delays=[0, 3], cell_lengths=cell_lengths).transform(X)
        assert np.all(out[:, 1] == 0)

    def test_cell_delayer_negative_delay_raises(self):
        with pytest.raises(ValueError, match="delays >= 0"):
            CellDelayer(delays=[-1], cell_lengths=[3]).transform(np.zeros((3, 1)))

    def test_cell_delayer_row_count_mismatch_raises(self):
        with pytest.raises(ValueError, match="cell_lengths sum"):
            CellDelayer(delays=[0], cell_lengths=[3]).transform(np.zeros((4, 1)))


class TestCellKey:
    def test_excluded_entity_raises(self):
        for entity in CellKey.EXCLUDE:
            with pytest.raises(ValueError, match="CellKey does not accept"):
                CellKey(**{entity: "x"})

    def test_equality_is_order_independent(self):
        assert CellKey(ses="1", run="2") == CellKey(run="2", ses="1")

    def test_keys_returns_present_entities(self):
        assert CellKey(ses="1", run="2").keys() == {"ses", "run"}

    def test_getitem_missing_raises(self):
        with pytest.raises(KeyError):
            CellKey(run="1")["ses"]

    def test_get_missing_returns_default(self):
        assert CellKey(run="1").get("ses") is None
        assert CellKey(run="1").get("ses", "fallback") == "fallback"
