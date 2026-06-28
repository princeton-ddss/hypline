import numpy as np
import pytest

from hypline.encoding import (
    EncodingTrainer,
    load_artifact,
    write_artifact,
)
from hypline.encoding._schema import CellKey, TrainingData

from ..conftest import BIDSTree
from .conftest import SUB, TASK, _make_encoding


class TestArtifactRoundTrip:
    """Write → load reproduces the recipe, cell set, and predictions exactly."""

    def _trained(
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
        for step in (
            "_discover_features",
            "_discover_bold",
            "_resolve_cell_keys",
            "_validate_coverage",
        ):
            monkeypatch.setattr(enc, step, lambda *a, **k: None)
        monkeypatch.setattr(enc, "_apply_filters", lambda *a, **k: (None, None))
        monkeypatch.setattr(enc, "_enrich_feature_metas", lambda *a, **k: None)
        monkeypatch.setattr(enc, "_build_training_data", lambda *a, **k: data)
        return enc, data

    def test_round_trip(self, tree: BIDSTree, monkeypatch: pytest.MonkeyPatch):
        from himalaya.backend import set_backend

        enc, data = self._trained(tree, monkeypatch)
        X = data.X.astype(np.float32)

        artifact = enc.train(SUB)
        out = enc._layout.path.result(sub=SUB, kind="encoding", desc="v1")
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
        self, tree: BIDSTree, monkeypatch: pytest.MonkeyPatch
    ):
        import json

        enc, _ = self._trained(tree, monkeypatch)
        artifact = enc.train(SUB)

        out = enc._layout.path.result(sub=SUB, kind="encoding", desc="v1")
        write_artifact(artifact, out.path)
        sidecar = json.loads(out.path.with_suffix(".json").read_text())
        assert sidecar["recipe"]["tasks"] == [TASK]
        assert sidecar["recipe"]["col_slices"] == {
            "phonemic-gpt3": [0, 3],
            "mfcc": [3, 7],
        }
        assert sidecar["universe"] is None
        assert sidecar["fold"] is None
        assert {frozenset(c.items()) for c in sidecar["models"][0]["train_cells"]} == {
            frozenset({("task", "a"), ("run", "1")}),
            frozenset({("task", "a"), ("run", "2")}),
        }
