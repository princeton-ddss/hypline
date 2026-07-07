import json

import numpy as np
import pytest

from hypline.encoding import (
    EncodingTrainer,
    load_artifact,
    write_artifact,
)
from hypline.encoding._context import _CONFOUND_BAND
from hypline.encoding._schema import CellKey, TrainingData

from ..conftest import BIDSTree
from .conftest import SUB, TASK, _make_encoding


class TestArtifactRoundTrip:
    """Write → load reproduces the recipe, cell set, and predictions exactly."""

    @pytest.fixture
    def train_setup(
        self, tree: BIDSTree, monkeypatch: pytest.MonkeyPatch
    ) -> tuple[EncodingTrainer, TrainingData]:
        n_rows, n_voxels = 20, 5
        rng = np.random.RandomState(0)
        data = TrainingData(
            X=rng.randn(n_rows, 7).astype(np.float64),
            Y=rng.randn(n_rows, n_voxels).astype(np.float64),
            row_slices={
                CellKey(task="a", run="1"): slice(0, 10),
                CellKey(task="a", run="2"): slice(10, 20),
            },
            col_slices={"phonemic-gpt3": slice(0, 3), "mfcc": slice(3, 7)},
        )
        enc = _make_encoding(tree, ["phonemic-gpt3", "mfcc"])
        monkeypatch.setattr(enc, "_assemble_training_data", lambda *a, **k: data)
        return enc, data

    def test_round_trip(self, train_setup: tuple[EncodingTrainer, TrainingData]):
        from himalaya.backend import set_backend

        enc, data = train_setup
        X = data.X.astype(np.float32)

        artifact = enc.train(SUB)
        out = enc._layout.path.result(sub=SUB, kind="encodingModel", desc="v1")
        write_artifact(artifact, out.path)

        # Predictions compare numpy-vs-numpy: write_artifact already forced the
        # in-memory pipeline to the numpy backend during the dump, so a plain
        # predict here is the numpy reference for the reloaded pipeline.
        set_backend("numpy")
        ref = np.asarray(artifact.models[0].pipeline.predict(X))

        loaded = load_artifact(out.path)
        got = np.asarray(loaded.models[0].pipeline.predict(X))
        np.testing.assert_array_equal(got, ref)

        # Persisted weights are numpy, not torch — guards a regressed converter
        # (a no-op converter would still round-trip while torch is installed)
        model = loaded.models[0].pipeline.named_steps["model"]
        assert isinstance(model.dual_coef_, np.ndarray)

        assert loaded.recipe == artifact.recipe
        assert loaded.models[0].train_cells == artifact.models[0].train_cells
        assert loaded.universe is None

    def test_sidecar_mirrors_non_pipeline_fields(
        self, train_setup: tuple[EncodingTrainer, TrainingData]
    ):
        enc, _ = train_setup
        artifact = enc.train(SUB)

        out = enc._layout.path.result(sub=SUB, kind="encodingModel", desc="v1")
        write_artifact(artifact, out.path)
        sidecar = json.loads(out.path.with_suffix(".json").read_text())
        assert sidecar["recipe"]["tasks"] == [TASK]
        assert sidecar["recipe"]["col_slices"] == {
            "phonemic-gpt3": [0, 3],
            "mfcc": [3, 7],
        }
        assert sidecar["recipe"]["confounds"] == {}
        assert sidecar["universe"] is None
        assert sidecar["fold"] is None
        assert {frozenset(c.items()) for c in sidecar["models"][0]["train_cells"]} == {
            frozenset({("task", "a"), ("run", "1")}),
            frozenset({("task", "a"), ("run", "2")}),
        }

    def test_confounds_round_trip_through_sidecar(
        self, tree: BIDSTree, monkeypatch: pytest.MonkeyPatch
    ):
        n_rows, n_voxels = 20, 5
        rng = np.random.RandomState(0)
        data = TrainingData(
            X=rng.randn(n_rows, 5).astype(np.float64),
            Y=rng.randn(n_rows, n_voxels).astype(np.float64),
            row_slices={
                CellKey(task="a", run="1"): slice(0, 10),
                CellKey(task="a", run="2"): slice(10, 20),
            },
            col_slices={"mfcc": slice(0, 3), _CONFOUND_BAND: slice(3, 5)},
        )
        enc = _make_encoding(tree, ["mfcc"], confounds=["phonemic-onset"])
        monkeypatch.setattr(enc, "_assemble_training_data", lambda *a, **k: data)

        artifact = enc.train(SUB)
        out = enc._layout.path.result(sub=SUB, kind="encodingModel", desc="v1")
        write_artifact(artifact, out.path)

        sidecar = json.loads(out.path.with_suffix(".json").read_text())
        assert sidecar["recipe"]["confounds"] == {
            "phonemic-onset": ["phonemic", "onset"]
        }
        assert load_artifact(out.path).recipe.confounds == {
            "phonemic-onset": ("phonemic", "onset")
        }

    def test_split_round_trips_through_sidecar(
        self, tree: BIDSTree, monkeypatch: pytest.MonkeyPatch
    ):
        # split lives on the recipe; a non-default value must survive both the
        # joblib pickle (the load path) and the provenance sidecar
        n_rows, n_voxels = 20, 5
        rng = np.random.RandomState(0)
        data = TrainingData(
            X=rng.randn(n_rows, 5).astype(np.float64),
            Y=rng.randn(n_rows, n_voxels).astype(np.float64),
            row_slices={CellKey(task="a", run="1"): slice(0, n_rows)},
            col_slices={"mfcc": slice(0, 5)},
        )
        enc = _make_encoding(tree, ["mfcc"], split=True)
        monkeypatch.setattr(enc, "_assemble_training_data", lambda *a, **k: data)

        artifact = enc.train(SUB)
        out = enc._layout.path.result(sub=SUB, kind="encodingModel", desc="v1")
        write_artifact(artifact, out.path)

        sidecar = json.loads(out.path.with_suffix(".json").read_text())
        assert sidecar["recipe"]["split"] is True
        assert load_artifact(out.path).recipe.split is True
