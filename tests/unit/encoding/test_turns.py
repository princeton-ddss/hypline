import numpy as np
import polars as pl
import pytest

from hypline.encoding import EncodingTrainer
from hypline.encoding._schema import RegressorMeta
from hypline.encoding._turns import _turn_masks

from ..conftest import DEFAULT_BOLD_N_TRS, BIDSTree
from .conftest import DYAD, PARTNER, SPACE, SUB, TASK, _add_dyad_turns, _make_encoding


class TestTurnMasks:
    """`_turn_masks` returns the per-TR prod and comp boolean masks.

    The fixture `_add_dyad_turns` gives a 10-TR run at TR=2.0 (so TR i covers
    second 2*i) where SUB speaks over seconds [0, 10) and PARTNER over [10, 20).
    SUB speaking is "production", the partner speaking is "comprehension", so
    SUB's prod mask is on for TRs 0..4, its comp mask is on for TRs 5..9. The
    assertions rest on that 5-on/5-off layout, with no silent (unspoken) TRs.
    """

    @staticmethod
    def _feature_df() -> pl.DataFrame:
        # One distinct timed value per TR, so a mis-zeroed split column is visible
        return pl.DataFrame(
            {
                "start_time": [2.0 * i for i in range(DEFAULT_BOLD_N_TRS)],
                "feature": [[float(i + 1)] for i in range(DEFAULT_BOLD_N_TRS)],
            },
            schema={"start_time": pl.Float64, "feature": pl.Array(pl.Float64, 1)},
        )

    def _one_meta(self, tree: BIDSTree) -> tuple[EncodingTrainer, RegressorMeta]:
        # Run the real discover/enrich chain and hand back one dyad-keyed meta,
        # so `_turn_masks` is exercised against genuine BIDSPath entities
        _add_dyad_turns(tree)
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="denoised")
        tree.add_feature(
            dyad=DYAD, task=TASK, kind="mfcc", run="1", df=self._feature_df()
        )
        enc = _make_encoding(tree, ["mfcc"])
        bold_metas = enc._discover_bold(SUB)
        feature_bids = enc._resolve_cell_keys(
            SUB, enc._discover_features(SUB), bold_metas
        )
        metas = enc._enrich_regressor_metas(feature_bids, bold_metas)
        return enc, next(iter(metas.values()))

    def test_turn_masks_are_subject_relative(self, tree: BIDSTree):
        # Masks are subject-relative: SUB's prod is PARTNER's comp. The stakes:
        # predict rebuilt against the wrong source subject would silently invert
        # every split copy.
        enc, meta = self._one_meta(tree)
        sub_prod, sub_comp = _turn_masks(
            enc._layout, sub_id=SUB, meta=meta, turns_cache={}
        )
        partner_prod, partner_comp = _turn_masks(
            enc._layout, sub_id=PARTNER, meta=meta, turns_cache={}
        )
        np.testing.assert_array_equal(sub_prod, partner_comp)
        np.testing.assert_array_equal(sub_comp, partner_prod)
        # And pin SUB's absolute layout: prod on TRs 0..4, comp on TRs 5..9
        assert sub_prod.tolist() == [True] * 5 + [False] * 5
        assert sub_comp.tolist() == [False] * 5 + [True] * 5

    def test_turn_masks_subject_outside_dyad_raises(self, tree: BIDSTree):
        # A sub_id can have discoverable files yet be missing from the dyad roster,
        # and then its prod/comp are undefined. Assembly always discovers files for
        # the sub it is given, so it never hits this path; exercise it directly.
        enc, meta = self._one_meta(tree)
        with pytest.raises(ValueError, match="is not in dyad"):
            _turn_masks(enc._layout, sub_id="999", meta=meta, turns_cache={})
