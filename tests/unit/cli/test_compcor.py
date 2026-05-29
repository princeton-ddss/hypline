import pytest
import typer

from hypline.cli._compcor import parse_compcor
from hypline.confounds.fmriprep import CompCorMask, CompCorMethod


class TestParseCompcor:
    def test_none_yields_empty(self):
        assert parse_compcor(None) == []

    def test_acompcor_with_mask(self):
        (options,) = parse_compcor("a:CSF:5")
        assert options.method is CompCorMethod.ANATOMICAL
        assert options.mask is CompCorMask.CSF
        assert options.n_comps == 5

    def test_tcompcor_empty_mask(self):
        (options,) = parse_compcor("t::10")
        assert options.method is CompCorMethod.TEMPORAL
        assert options.mask is None
        assert options.n_comps == 10

    def test_variance_fraction_n(self):
        (options,) = parse_compcor("t::0.5")
        assert options.n_comps == 0.5

    def test_multiple_tokens_order(self):
        a, t = parse_compcor("a:WM:3,t::2")
        assert a.method is CompCorMethod.ANATOMICAL
        assert t.method is CompCorMethod.TEMPORAL

    @pytest.mark.parametrize(
        "value, match",
        [
            ("a:CSF", "3 colon-separated fields"),
            ("a:CSF:5:6", "3 colon-separated fields"),
            ("x:CSF:5", "method must be 'a' or 't'"),
            ("a::5", "aCompCor requires a mask"),
            ("a:BAD:5", "mask must be one of"),
            ("t:CSF:5", "tCompCor is not mask-restricted"),
            ("a:CSF:", "missing n"),
            ("a:CSF:abc", "n must be a number"),
            ("a:CSF:0", "n must be positive"),
            ("a:CSF:-1", "n must be positive"),
        ],
    )
    def test_invalid_tokens(self, value: str, match: str):
        with pytest.raises(typer.BadParameter, match=match):
            parse_compcor(value)
