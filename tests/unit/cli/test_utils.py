import pytest
import typer

from hypline.cli._utils import split_csv


class TestSplitCsv:
    def test_none_returns_none(self):
        assert split_csv(None) is None

    def test_single_value(self):
        assert split_csv("01") == ["01"]

    def test_multiple_values(self):
        assert split_csv("01,02,03") == ["01", "02", "03"]

    def test_two_values(self):
        assert split_csv("01,02") == ["01", "02"]

    def test_whitespace_space_raises(self):
        with pytest.raises(typer.BadParameter, match="whitespace"):
            split_csv("01, 02")

    def test_whitespace_tab_raises(self):
        with pytest.raises(typer.BadParameter, match="whitespace"):
            split_csv("01\t02")

    def test_whitespace_newline_raises(self):
        with pytest.raises(typer.BadParameter, match="whitespace"):
            split_csv("01\n02")

    def test_trailing_comma_raises(self):
        with pytest.raises(typer.BadParameter, match="empty value"):
            split_csv("01,")

    def test_leading_comma_raises(self):
        with pytest.raises(typer.BadParameter, match="empty value"):
            split_csv(",01")

    def test_empty_between_commas_raises(self):
        with pytest.raises(typer.BadParameter, match="empty value"):
            split_csv("01,,02")

    def test_duplicates_raises(self):
        with pytest.raises(typer.BadParameter, match="duplicate"):
            split_csv("01,01")

    def test_duplicate_not_adjacent_raises(self):
        with pytest.raises(typer.BadParameter, match="duplicate"):
            split_csv("01,02,01")

    def test_param_hint_propagates(self):
        with pytest.raises(typer.BadParameter) as exc_info:
            split_csv("01, 02", param_hint="--dyad-ids")
        assert exc_info.value.param_hint == "--dyad-ids"

    def test_param_hint_none_by_default(self):
        with pytest.raises(typer.BadParameter) as exc_info:
            split_csv("01, 02")
        assert exc_info.value.param_hint is None

    def test_empty_string_raises(self):
        with pytest.raises(typer.BadParameter, match="empty value"):
            split_csv("")
