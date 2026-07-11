from pathlib import Path

import numpy as np
import pytest

from hypline.encoding import EncodingPredictor
from hypline.encoding._eval import (
    _role_masks,
    _score_roles,
    _smear_mask,
)
from hypline.encoding._schema import CellKey

from ..conftest import BIDSTree
from .conftest import SUB, _make_encoding, _two_run_dyad


class TestSmearMask:
    """`_smear_mask` = fconv's `delayer.fit_transform(mask).any(-1)`, verified against
    `hyperscanning/fconv/code/encoding.py` (forward shift, delays [0..5])."""

    def test_forward_smear_fills_following_rows(self):
        # A single True at row 2, delays [0,1,2] → rows 2,3,4 True (row i takes i-d)
        mask = np.zeros(6, dtype=bool)
        mask[2] = True
        out = _smear_mask(mask, [0, 1, 2])
        assert out.tolist() == [False, False, True, True, True, False]

    def test_zero_delay_is_identity(self):
        mask = np.array([True, False, True, False], dtype=bool)
        np.testing.assert_array_equal(_smear_mask(mask, [0]), mask)

    def test_smear_does_not_wrap_past_start(self):
        # True at row 0 with delay 2 shifts to row 2, never wraps to the tail
        mask = np.zeros(5, dtype=bool)
        mask[0] = True
        assert _smear_mask(mask, [2]).tolist() == [False, False, True, False, False]


class TestRoleMasks:
    """Exclusive prod/comp + union `both`, hand-computed on a 6-TR toy cell.

    Raw prod = TRs 0..2, raw comp = TRs 3..5 (disjoint, one floor-holder per TR).
    With delays [0,1] the smears overlap at the boundary (row 3 gets delayed prod),
    so exclusive AND-NOT drops boundary rows and `both` keeps them.
    """

    def _patch_turns(
        self,
        monkeypatch: pytest.MonkeyPatch,
        by_meta: dict[str, tuple[np.ndarray, np.ndarray]],
    ) -> None:
        # `_role_masks` calls the module-level `_turn_masks`; patch that seam to return
        # synthetic raw masks keyed by the cell's meta value. Signature mirrors the real
        # free function (`layout` positional, rest keyword-only); layout is ignored.
        def _fake(layout, *, sub_id, meta, turns_cache):
            return by_meta[meta]

        monkeypatch.setattr("hypline.encoding._eval._turn_masks", _fake)

    def test_exclusive_drops_boundary_union_keeps_it(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        prod_raw = np.array([1, 1, 1, 0, 0, 0], dtype=bool)
        comp_raw = np.array([0, 0, 0, 1, 1, 1], dtype=bool)
        self._patch_turns(monkeypatch, {"m": (prod_raw, comp_raw)})
        masks = _role_masks(
            layout=None,  # ty: ignore[invalid-argument-type]
            target_sub_id=SUB,
            cell_metas={CellKey(task="a", run="1"): "m"},  # ty: ignore[invalid-argument-type]
            delays=[0, 1],
        )
        # prod_sm = 0,1,2,3 ; comp_sm = 3,4,5 → row 3 is the contaminated boundary
        assert masks["prod"].tolist() == [1, 1, 1, 0, 0, 0]
        assert masks["comp"].tolist() == [0, 0, 0, 0, 1, 1]
        assert masks["both"].tolist() == [1, 1, 1, 1, 1, 1]

    def test_does_not_bleed_across_cells(self, monkeypatch: pytest.MonkeyPatch):
        # Two 3-TR cells. Cell 0 has prod on its last row (2); cell 1's boundary row
        # (global 3) is empty. So global row 3 is the discriminator: a per-cell smear
        # leaves it 0, but a bug that smears the concatenated array flips it to prod.
        prod0 = np.array([0, 0, 1], dtype=bool)
        comp0 = np.array([1, 1, 0], dtype=bool)
        prod1 = np.array([0, 0, 1], dtype=bool)
        comp1 = np.array([0, 0, 0], dtype=bool)
        c0, c1 = CellKey(task="a", run="1"), CellKey(task="a", run="2")
        self._patch_turns(monkeypatch, {"c0": (prod0, comp0), "c1": (prod1, comp1)})
        masks = _role_masks(
            layout=None,  # ty: ignore[invalid-argument-type]
            target_sub_id=SUB,
            cell_metas={c0: "c0", c1: "c1"},  # ty: ignore[invalid-argument-type]
            delays=[0, 1],
        )
        # Row 3 stays 0 in prod: cell 0's tail did not bleed across the boundary.
        assert masks["prod"].tolist() == [0, 0, 0, 0, 0, 1]
        assert masks["comp"].tolist() == [1, 1, 0, 0, 0, 0]


class TestScoreRoles:
    def test_empty_role_is_nan_others_scored(self):
        rng = np.random.RandomState(0)
        n_bands, n_rows, n_vox = 2, 8, 3
        Y_true = rng.randn(n_rows, n_vox)
        Y_hat = rng.randn(n_bands, n_rows, n_vox)
        masks = {
            "prod": np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=bool),
            "comp": np.zeros(n_rows, dtype=bool),  # empty → NaN
            "both": np.ones(n_rows, dtype=bool),
        }
        out = _score_roles(Y_true=Y_true, Y_hat=Y_hat, masks=masks)
        assert out.shape == (n_bands, 3, n_vox)
        assert np.isnan(out[:, 1, :]).all()  # comp role all NaN
        assert not np.isnan(out[:, 0, :]).any()  # prod scored
        assert not np.isnan(out[:, 2, :]).any()  # both scored

    def test_scores_only_masked_rows_at_role_position(self):
        # Value oracle for the row-selection/axis contract `_score_roles` owns; the
        # correlation math itself is himalaya's.
        rng = np.random.RandomState(1)
        n_rows, n_vox = 8, 3
        prod = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=bool)
        comp = ~prod
        Y_true = rng.randn(n_rows, n_vox)
        Y_hat = rng.randn(1, n_rows, n_vox)
        Y_hat[0, prod] = Y_true[prod]  # perfect on prod rows, noise elsewhere
        masks = {"prod": prod, "comp": comp, "both": np.ones(n_rows, dtype=bool)}

        # Perfect prediction on prod rows must surface as ≈1.0 at role index 0.
        out = _score_roles(Y_true=Y_true.copy(), Y_hat=Y_hat.copy(), masks=masks)
        np.testing.assert_allclose(out[:, 0, :], 1.0)

        # Perturbing only comp rows must leave prod's score untouched — prod reads
        # only its own masked rows.
        Y_true_perturbed, Y_hat_perturbed = Y_true.copy(), Y_hat.copy()
        Y_true_perturbed[comp] += 99.0
        Y_hat_perturbed[0, comp] += -50.0
        out_perturbed = _score_roles(
            Y_true=Y_true_perturbed, Y_hat=Y_hat_perturbed, masks=masks
        )
        np.testing.assert_allclose(out[:, 0, :], out_perturbed[:, 0, :])


class TestSaveLoadRoundTrip:
    def test_round_trip_preserves_cube_and_attrs(self, tree: BIDSTree, tmp_path: Path):
        from hypline.encoding import load_eval, save_eval

        _two_run_dyad(tree)
        enc = _make_encoding(tree, ["mfcc"], fold_by="run", n_folds="loo")
        artifact = enc.train(SUB)
        predictor = EncodingPredictor(bids_root=tree.root, artifact=artifact)
        ds = predictor.analyze(source_sub_id=SUB, target_sub_id=SUB)

        path = tmp_path / "eval.nc"
        save_eval(ds, path)
        loaded = load_eval(path)

        np.testing.assert_allclose(
            loaded["corr"].values, ds["corr"].values, equal_nan=True
        )
        assert list(loaded["band"].values) == list(ds["band"].values)
        assert list(loaded["role"].values) == list(ds["role"].values)
        # Guard every attr against drift. `delays` is a numpy array (needs
        # assert_array_equal); the rest compare by value. `test_on` is None here,
        # so it serializes to the "OOS" sentinel.
        for key in (
            "model_sub",
            "source_sub",
            "target_sub",
            "bold_space",
            "hypline_version",
        ):
            assert loaded.attrs[key] == ds.attrs[key]
        assert loaded.attrs["test_on"] == ds.attrs["test_on"] == "OOS"
        np.testing.assert_array_equal(loaded.attrs["delays"], ds.attrs["delays"])
        # fold_cells round-trips as live CellKeys, so it compares equal directly
        assert loaded.attrs["fold_cells"] == ds.attrs["fold_cells"]
        assert all(
            isinstance(c, CellKey) for fold in loaded.attrs["fold_cells"] for c in fold
        )
