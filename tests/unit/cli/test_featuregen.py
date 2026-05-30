from unittest.mock import Mock, call

from typer.testing import CliRunner

from hypline.cli import app

runner = CliRunner()


def _patch_generators(monkeypatch):
    """Patch both generators' `generate` and return a manager recording call order."""
    from hypline.confounds.phonemic import PhonemicConfound
    from hypline.features.phonemic import PhonemicFeature

    manager = Mock()
    monkeypatch.setattr(PhonemicFeature, "generate", manager.feature)
    monkeypatch.setattr(PhonemicConfound, "generate", manager.confound)
    return manager


class TestFeaturegenPhonemicChain:
    def test_confounds_generated_per_subject_in_order(self, tree, monkeypatch):
        manager = _patch_generators(monkeypatch)

        result = runner.invoke(
            app,
            ["featuregen", "phonemic", str(tree.root), "--sub-ids", "01,02"],
        )

        assert result.exit_code == 0
        assert manager.mock_calls == [
            call.feature("01"),
            call.confound("01"),
            call.feature("02"),
            call.confound("02"),
        ]  # per-subject chain, NOT both features then both confounds

    def test_skip_confoundgen_runs_feature_only(self, tree, monkeypatch):
        manager = _patch_generators(monkeypatch)

        result = runner.invoke(
            app,
            [
                "featuregen",
                "phonemic",
                str(tree.root),
                "--sub-ids",
                "01,02",
                "--skip-confoundgen",
            ],
        )

        assert result.exit_code == 0
        manager.confound.assert_not_called()
        assert manager.feature.mock_calls == [call("01"), call("02")]

    def test_feature_failure_skips_own_confound_others_proceed(self, tree, monkeypatch):
        manager = _patch_generators(monkeypatch)

        def feature_generate(sub_id):
            if sub_id == "01":
                raise RuntimeError("boom")

        manager.feature.side_effect = feature_generate

        result = runner.invoke(
            app,
            ["featuregen", "phonemic", str(tree.root), "--sub-ids", "01,02"],
        )

        assert result.exit_code == 1
        assert manager.mock_calls == [
            call.feature("01"),
            call.feature("02"),
            call.confound("02"),
        ]  # sub-01 feature raised → its confound skipped; sub-02 ran both
