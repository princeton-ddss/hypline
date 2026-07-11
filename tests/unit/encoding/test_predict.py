import numpy as np
import pytest

from hypline.encoding import EncodingPredictor
from hypline.encoding._artifact import EncodingArtifact, FittedModel
from hypline.encoding._predict import (
    _assert_same_voxels,
    _rebind_cell_lengths,
    _select_cells,
)
from hypline.encoding._schema import CellDelayer, CellKey, TrainingData, XData

from ..conftest import BIDSTree
from .conftest import (
    DYAD,
    PARTNER,
    SUB,
    _add_phonemic_confound,
    _make_encoding,
    _two_run_dyad,
)


class TestAssertSameVoxels:
    def test_mismatch_raises(self):
        Y_hat = np.zeros((2, 5, 4))
        Y_true = np.zeros((5, 3))
        with pytest.raises(ValueError, match="Voxel-axis mismatch"):
            _assert_same_voxels(Y_hat=Y_hat, Y_true=Y_true)

    def test_match_passes(self):
        _assert_same_voxels(Y_hat=np.zeros((2, 5, 4)), Y_true=np.zeros((5, 4)))


class TestSelectCells:
    """`_select_cells` picks a model's prediction targets. It reads only
    `artifact.universe` and `model.train_cells` (never the recipe or pipeline), so
    the fixtures leave every other field as a placeholder — a real fit would only
    slow these branch checks down.

    Cells are named by `run` throughout: `_cell("1")` etc. `available` is the
    source's full discovered set; `train_cells`/`universe` come off the model.
    """

    def _cell(self, run: str) -> CellKey:
        return CellKey(task="a", run=run)

    def _artifact(self, universe: set[CellKey] | None) -> EncodingArtifact:
        return EncodingArtifact(
            recipe=None,  # ty: ignore[invalid-argument-type]
            sub_id=SUB,
            fold=None,
            models=[],
            universe=universe,
        )

    def _model(self, train_cells: set[CellKey]) -> FittedModel:
        return FittedModel(pipeline=None, train_cells=train_cells)  # ty: ignore[invalid-argument-type]

    def test_default_oos_excludes_train_cells(self):
        # No universe: OOS is `available - train_cells`.
        avail = {self._cell("1"), self._cell("2"), self._cell("3")}
        model = self._model({self._cell("1")})
        selected = _select_cells(avail, self._artifact(None), model)
        assert selected == {self._cell("2"), self._cell("3")}

    def test_default_empty_oos_raises(self):
        # Every available cell was trained on, so nothing is left out of sample.
        avail = {self._cell("1")}
        model = self._model({self._cell("1")})
        with pytest.raises(ValueError, match="empty out-of-sample set"):
            _select_cells(avail, self._artifact(None), model)

    def test_bounded_oos_is_universe_minus_train(self):
        # With a universe, OOS is bounded to it — not to `available` — so a source
        # cell outside the universe (run 3) is never selected.
        avail = {self._cell("1"), self._cell("2"), self._cell("3")}
        universe = {self._cell("1"), self._cell("2")}
        model = self._model({self._cell("1")})
        selected = _select_cells(avail, self._artifact(universe), model)
        assert selected == {self._cell("2")}

    def test_bounded_oos_missing_on_source_raises(self):
        # A bounded-OOS cell (run 2) that the source subject never discovered would be
        # silently dropped downstream; `_select_cells` raises instead.
        avail = {self._cell("1")}
        universe = {self._cell("1"), self._cell("2")}
        model = self._model({self._cell("1")})
        with pytest.raises(ValueError, match="absent on source subject"):
            _select_cells(avail, self._artifact(universe), model)

    def test_test_on_selects_matching_cells(self):
        # `test_on` selects by entity match across mixed-entity cells (task="a" spans
        # two runs; the task="b" cell is excluded). Override semantics are covered by
        # test_test_on_overrides_train_and_universe.
        avail = {
            CellKey(task="a", run="1"),
            CellKey(task="b", run="1"),
            CellKey(task="a", run="2"),
        }
        model = self._model(set())
        selected = _select_cells(
            avail, self._artifact(None), model, test_on=["task-a"]
        )
        assert selected == {CellKey(task="a", run="1"), CellKey(task="a", run="2")}

    def test_test_on_ors_within_entity(self):
        # Same-entity values OR-match: `run-1,run-2` selects both runs, the run-3
        # cell is excluded. This is the AND/OR grammar shared with `bids_filters`.
        avail = {self._cell("1"), self._cell("2"), self._cell("3")}
        model = self._model(set())
        selected = _select_cells(
            avail, self._artifact(None), model, test_on=["run-1", "run-2"]
        )
        assert selected == {self._cell("1"), self._cell("2")}

    def test_test_on_unknown_entity_raises(self):
        # A typo'd entity (cells carry task/run, not `rn`) is caught up front rather
        # than silently matching nothing and surfacing as the empty-set error.
        avail = {self._cell("1")}
        model = self._model(set())
        with pytest.raises(ValueError, match="not found on any available cell"):
            _select_cells(avail, self._artifact(None), model, test_on=["rn-1"])

    def test_test_on_overrides_train_and_universe(self):
        # test_on outranks both the universe and the train set: run 2 is selected even
        # though it sits outside the universe (run 1 only), and the disjoint train cell
        # (run 3) is ignored, not warned about. Run 2 is outside the universe but
        # present in `available`, so the bounded-OOS presence check stays quiet.
        avail = {self._cell("1"), self._cell("2"), self._cell("3")}
        universe = {self._cell("1")}
        model = self._model({self._cell("3")})
        selected = _select_cells(
            avail, self._artifact(universe), model, test_on=["run-2"]
        )
        assert selected == {self._cell("2")}

    def test_test_on_no_match_raises(self):
        avail = {self._cell("1")}
        model = self._model(set())
        with pytest.raises(ValueError, match="test_on matched no available cells"):
            _select_cells(avail, self._artifact(None), model, test_on=["run-9"])

    def test_test_on_overlap_with_train_warns(self):
        # Predicting on trained-on cells is usually a leak, so overlap only warns.
        import loguru

        avail = {self._cell("1"), self._cell("2")}
        model = self._model({self._cell("1")})
        messages: list[str] = []
        handler_id = loguru.logger.add(
            lambda m: messages.append(str(m)), level="WARNING"
        )
        try:
            selected = _select_cells(
                avail, self._artifact(None), model, test_on=["run-1"]
            )
        finally:
            loguru.logger.remove(handler_id)
        assert selected == {self._cell("1")}
        assert any("trained on" in m for m in messages)

    def test_schema_mismatch_raises(self):
        # Source cells keyed by (task, run); train cells by (task) only. They never
        # hash/compare equal, so `available - train_cells` would wrongly keep the
        # trained-on cell — the guard catches the mismatch up front.
        avail = {CellKey(task="a", run="1")}
        model = self._model({CellKey(task="a")})
        with pytest.raises(ValueError, match="Cell-schema mismatch"):
            _select_cells(avail, self._artifact(None), model)


def _fit_predictor(
    tree: BIDSTree, monkeypatch: pytest.MonkeyPatch
) -> tuple[EncodingPredictor, XData]:
    """Fit a real two-band pipeline over a two-cell layout and wrap it for predict.

    The same X drives fit and predict (patched `_build_x` returns the train X), so
    `col_slices` match the recipe and the drift guard passes. Fitted `CellDelayer`
    cell_lengths are `[10, 10]` (the two 10-row cells), the geometry the rebind test
    overwrites.
    """
    rng = np.random.RandomState(0)
    col_slices = {"phonemic-gpt3": slice(0, 3), "mfcc": slice(3, 7)}
    row_slices = {
        CellKey(task="a", run="1"): slice(0, 10),
        CellKey(task="a", run="2"): slice(10, 20),
    }
    train = TrainingData(
        X=rng.randn(20, 7).astype(np.float64),
        Y=rng.randn(20, 5).astype(np.float64),
        row_slices=row_slices,
        col_slices=col_slices,
    )

    enc = _make_encoding(tree, ["phonemic-gpt3", "mfcc"])
    monkeypatch.setattr(enc, "_assemble_training_data", lambda *a, **k: train)
    artifact = enc.train(SUB)

    predictor = EncodingPredictor(bids_root=tree.root, artifact=artifact)
    x = XData(X=train.X, row_slices=row_slices, col_slices=col_slices)
    monkeypatch.setattr(predictor, "_build_x", lambda *a, **k: x)

    return predictor, x


class TestRebindCellLengths:
    """`_rebind_cell_lengths` overwrites every band's frozen train cell_lengths with
    the predict geometry. It must reach the *fitted* clones (`transformers_`), not the
    unfitted `.transformers` spec that predict never consumes — a rebind that hit the
    spec would leave predict's transform running on stale train lengths and raise.
    """

    def test_rebinds_fitted_clones_only(
        self, tree: BIDSTree, monkeypatch: pytest.MonkeyPatch
    ):
        predictor, _ = _fit_predictor(tree, monkeypatch)
        pipeline = predictor._artifact.models[0].pipeline

        def _delayer_lengths(attr: str) -> list[list[int]]:
            # Collect each band's CellDelayer.cell_lengths off the named transformers.
            column_kernelizer = pipeline.named_steps["kernelizer"]
            found: list[list[int]] = []
            for _, transformer, _ in getattr(column_kernelizer, attr):
                for _, step in transformer.steps:
                    if isinstance(step, CellDelayer):
                        found.append(step.cell_lengths)
            return found

        # Fit froze [10, 10] on every band's fitted clone.
        assert _delayer_lengths("transformers_") == [[10, 10], [10, 10]]

        _rebind_cell_lengths(pipeline, [7, 5])

        # Every fitted-clone band now carries the predict geometry...
        assert _delayer_lengths("transformers_") == [[7, 5], [7, 5]]
        # ...while the unfitted spec is untouched — proof the rebind targeted
        # the clones, not the spec.
        assert all(lengths == [10, 10] for lengths in _delayer_lengths("transformers"))


class TestPredictModel:
    """Guards the split=True contract `_predict_model` returns (analyze depends on it).

    `_predict_model` must return per-band `Y_hat` (band, row, voxel) whose axis labels
    track `col_slices` order. A silent regression to 2-D or a mislabelled band axis
    would corrupt every downstream per-band correlation.
    """

    def test_predict_returns_per_band_y_hat(
        self, tree: BIDSTree, monkeypatch: pytest.MonkeyPatch
    ):
        # himalaya owns that the bands sum back to the combined prediction, so
        # re-verifying that would only re-test himalaya; the shape is the guard.
        predictor, x = _fit_predictor(tree, monkeypatch)
        # regressor_metas is unused (patched _build_x ignores it), so pass {}
        pred = predictor._predict_model(SUB, predictor._artifact.models[0], {})

        n_bands, n_rows, n_voxels = len(x.col_slices), x.X.shape[0], 5
        assert pred.Y_hat.shape == (n_bands, n_rows, n_voxels)

    def test_band_names_track_col_slices_order(
        self, tree: BIDSTree, monkeypatch: pytest.MonkeyPatch
    ):
        # band axis i must be col_slices key i — sum(0) would pass under any label
        # order, so alignment needs its own assertion.
        predictor, x = _fit_predictor(tree, monkeypatch)
        pred = predictor._predict_model(SUB, predictor._artifact.models[0], {})
        assert pred.band_names == list(x.col_slices.keys())
        assert len(pred.band_names) == pred.Y_hat.shape[0]

    def test_col_slices_drift_raises(
        self, tree: BIDSTree, monkeypatch: pytest.MonkeyPatch
    ):
        # The rebuild guard: a rebuilt X whose bands diverge from the recipe (a renamed
        # band here) would silently mislabel Y_hat's band axis. `_predict_model` raises
        # before predicting instead.
        predictor, x = _fit_predictor(tree, monkeypatch)
        drifted = XData(
            X=x.X,
            row_slices=x.row_slices,
            col_slices={"renamed": slice(0, 3), "mfcc": slice(3, 7)},
        )
        monkeypatch.setattr(predictor, "_build_x", lambda *a, **k: drifted)
        with pytest.raises(ValueError, match="col_slices drift"):
            predictor._predict_model(SUB, predictor._artifact.models[0], {})


class TestPredict:
    """End-to-end `predict` over toy BIDS: discover source regressors, merge
    features+confounds, select cells per model, return one `Prediction` per model.

    Uses a real tree (not `TestPredictModel`'s stubbed `_build_x`) precisely because
    the discover/merge path is what's under test here. Every fixture carries a
    `phonemic-onset` confound, so the `{**feature_bids, **confound_bids}` merge and
    `_discover_confounds` run for real — the one predict-only path no other class
    exercises.
    """

    def _predictor(self, tree: BIDSTree) -> EncodingPredictor:
        _two_run_dyad(tree)
        _add_phonemic_confound(tree)
        enc = _make_encoding(
            tree,
            ["mfcc"],
            confounds=["phonemic-onset"],
            fold_by="run",
            n_folds="loo",
        )
        artifact = enc.train(SUB)
        return EncodingPredictor(bids_root=tree.root, artifact=artifact)

    def test_returns_one_prediction_per_model_with_confound_band(self, tree: BIDSTree):
        predictor = self._predictor(tree)
        preds = predictor.predict(source_sub_id=SUB)

        # One Prediction per fitted model.
        assert len(preds) == len(predictor._artifact.models)
        # The confound collapses into the reserved trailing band, so band_names
        # carries it alongside mfcc — proof the merge fed a confound into X.
        band_names = predictor._artifact.recipe.col_slices.keys()
        assert all(pred.band_names == list(band_names) for pred in preds)
        assert any("confound" in name.lower() for name in band_names)

    def test_test_on_narrows_predicted_cells(self, tree: BIDSTree):
        # The public passthrough of test_on into _select_cells (unit-tested there):
        # naming one run yields exactly that run's cell in each Prediction.
        predictor = self._predictor(tree)
        preds = predictor.predict(source_sub_id=SUB, test_on=["run-1"])

        for pred in preds:
            assert all(cell["run"] == "1" for cell in pred.row_slices)


class TestAnalyze:
    """End-to-end `analyze` over toy BIDS. Starts source == target (isolates the
    wiring from cross-subject cell-key drift), the case that must pass."""

    def _predictor(self, tree: BIDSTree) -> EncodingPredictor:
        _two_run_dyad(tree)
        enc = _make_encoding(tree, ["mfcc"], fold_by="run", n_folds="loo")
        artifact = enc.train(SUB)
        return EncodingPredictor(bids_root=tree.root, artifact=artifact)

    def test_returns_labelled_dataset(self, tree: BIDSTree):
        predictor = self._predictor(tree)
        ds = predictor.analyze(source_sub_id=SUB, target_sub_id=SUB)

        assert set(ds["corr"].dims) == {"fold", "band", "role", "voxel"}
        assert list(ds["role"].values) == ["prod", "comp", "both"]
        assert list(ds["band"].values) == list(
            predictor._artifact.recipe.col_slices.keys()
        )
        # subsettable by name
        assert ds["corr"].sel(role="prod").dims == ("fold", "band", "voxel")
        # provenance
        assert ds.attrs["model_sub"] == SUB
        assert ds.attrs["source_sub"] == SUB
        assert ds.attrs["target_sub"] == SUB
        assert ds.attrs["test_on"] == "OOS"
        # fold_cells is a live list of CellKey lists, one per fold
        assert len(ds.attrs["fold_cells"]) == ds.sizes["fold"]
        assert all(
            isinstance(c, CellKey) for fold in ds.attrs["fold_cells"] for c in fold
        )

    def test_partner_same_dyad_scores(self, tree: BIDSTree):
        # The primary scientific case: source=SUB drives X, target=PARTNER supplies
        # actual Y and turns — different brains, one shared conversation (fconv's
        # self-vs-partner). Exercises the load-bearing cross-subject assumption:
        # PARTNER's discovered cells must key-match SUB's prediction row_slices, or
        # analyze's fold_metas lookup KeyErrors. source==target cannot hit this (keys
        # match trivially).
        predictor = self._predictor(tree)  # trained on SUB

        import loguru

        messages: list[str] = []
        handler_id = loguru.logger.add(
            lambda m: messages.append(str(m)), level="WARNING"
        )
        try:
            ds = predictor.analyze(source_sub_id=SUB, target_sub_id=PARTNER)
        finally:
            loguru.logger.remove(handler_id)

        # same dyad, so no cross-dyad warning
        assert not any("different dyads" in m for m in messages)
        assert ds.attrs["source_sub"] == SUB
        assert ds.attrs["target_sub"] == PARTNER
        # the union `both` role has scored (non-NaN) voxels — masks aligned and ran
        assert not np.isnan(ds["corr"].sel(role="both").values).all()

    def test_cross_dyad_warns(self, tree: BIDSTree):
        # A target in a *different* dyad: correlations become a null control, so
        # analyze must warn. Register other_sub in a valid 2-subject second dyad
        # (so dyad_of resolves for both) but give it no BOLD/features — the warning
        # fires first, then discovery raises. We assert only that the warning fired.
        predictor = self._predictor(tree)
        other_sub, other_partner = "003", "004"
        tree.add_participants(
            {SUB: DYAD, PARTNER: DYAD, other_sub: "202", other_partner: "202"}
        )
        import loguru

        messages: list[str] = []
        handler_id = loguru.logger.add(
            lambda m: messages.append(str(m)), level="WARNING"
        )
        try:
            with pytest.raises(Exception):  # noqa: B017 — downstream raise is expected
                predictor.analyze(source_sub_id=SUB, target_sub_id=other_sub)
        finally:
            loguru.logger.remove(handler_id)
        assert any("different dyads" in m for m in messages)
