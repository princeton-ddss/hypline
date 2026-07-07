from unittest.mock import Mock

from typer.testing import CliRunner

from hypline.cli import app

# When color is forced (as in CI), Rich wraps each part of an option name in its
# own ANSI codes, so `--n-folds` is not a literal substring of the error output.
# Disable Rich rendering so error assertions match plain text.
app.rich_markup_mode = None

runner = CliRunner()

# Every invocation needs the three required options; tests add/override as needed
REQUIRED = ["--tasks", "conv", "--features", "semantic-test", "--desc", "v1"]


def _patch_trainer(monkeypatch):
    """Patch train + write_artifact, returning a manager recording calls.

    The real EncodingTrainer still constructs (its __init__ validation runs),
    but the fit and the disk write are stubbed so the CLI wiring is tested
    without data or a solver.
    """
    import hypline.encoding as enc_pkg
    from hypline.encoding import EncodingTrainer

    manager = Mock()
    manager.train.return_value = object()  # stand-in artifact
    monkeypatch.setattr(EncodingTrainer, "train", manager.train)
    # The command does `from hypline.encoding import write_artifact` at call
    # time, so patch the name on the package it is imported from
    monkeypatch.setattr(enc_pkg, "write_artifact", manager.write)
    return manager


class TestEncodingTrainWiring:
    def test_trains_per_subject_and_writes(self, tree, monkeypatch):
        manager = _patch_trainer(monkeypatch)

        result = runner.invoke(
            app,
            [
                "encoding",
                "train",
                str(tree.root),
                *REQUIRED,
                "--fold-by",
                "none",
                "--sub-ids",
                "01,02",
            ],
        )

        assert result.exit_code == 0, result.output
        assert manager.train.call_count == 2
        assert manager.write.call_count == 2
        # Output path must carry the encodingModel kind (predict loads by it)
        written = [c.args[1] for c in manager.write.call_args_list]
        assert all("encodingModel-v1" in str(p) for p in written)
        assert {str(p.parent.parent.name) for p in written} == {"sub-01", "sub-02"}

    def test_skips_when_result_present(self, tree, monkeypatch):
        manager = _patch_trainer(monkeypatch)
        from hypline.layout import BIDSLayout

        out = BIDSLayout(tree.root).path.result(
            sub="01", kind="encodingModel", desc="v1"
        )
        out.path.parent.mkdir(parents=True, exist_ok=True)
        out.path.touch()

        result = runner.invoke(
            app,
            [
                "encoding",
                "train",
                str(tree.root),
                *REQUIRED,
                "--fold-by",
                "none",
                "--sub-ids",
                "01",
            ],
        )

        assert result.exit_code == 0, result.output
        manager.train.assert_not_called()
        manager.write.assert_not_called()

    def test_force_overwrites_existing_result(self, tree, monkeypatch):
        manager = _patch_trainer(monkeypatch)
        from hypline.layout import BIDSLayout

        out = BIDSLayout(tree.root).path.result(
            sub="01", kind="encodingModel", desc="v1"
        )
        out.path.parent.mkdir(parents=True, exist_ok=True)
        out.path.touch()

        result = runner.invoke(
            app,
            [
                "encoding",
                "train",
                str(tree.root),
                *REQUIRED,
                "--fold-by",
                "none",
                "--sub-ids",
                "01",
                "--force",
            ],
        )

        assert result.exit_code == 0, result.output
        manager.train.assert_called_once()
        manager.write.assert_called_once()


class TestEncodingTrainValidation:
    def test_fold_by_entity_requires_n_folds(self, tree, monkeypatch):
        _patch_trainer(monkeypatch)

        result = runner.invoke(
            app,
            [
                "encoding",
                "train",
                str(tree.root),
                *REQUIRED,
                "--fold-by",
                "run",
                "--sub-ids",
                "01",
            ],
        )

        assert result.exit_code != 0
        assert "--n-folds" in result.output

    def test_fold_by_none_rejects_n_folds(self, tree, monkeypatch):
        _patch_trainer(monkeypatch)

        result = runner.invoke(
            app,
            [
                "encoding",
                "train",
                str(tree.root),
                *REQUIRED,
                "--fold-by",
                "none",
                "--n-folds",
                "3",
                "--sub-ids",
                "01",
            ],
        )

        assert result.exit_code != 0
        assert "--n-folds" in result.output

    def test_n_folds_loo_accepted_with_fold_by_entity(self, tree, monkeypatch):
        manager = _patch_trainer(monkeypatch)

        result = runner.invoke(
            app,
            [
                "encoding",
                "train",
                str(tree.root),
                *REQUIRED,
                "--fold-by",
                "run",
                "--n-folds",
                "loo",
                "--sub-ids",
                "01",
            ],
        )

        assert result.exit_code == 0, result.output
        assert manager.train.call_count == 1

    def test_non_integer_n_folds_rejected(self, tree, monkeypatch):
        _patch_trainer(monkeypatch)

        result = runner.invoke(
            app,
            [
                "encoding",
                "train",
                str(tree.root),
                *REQUIRED,
                "--fold-by",
                "run",
                "--n-folds",
                "abc",
                "--sub-ids",
                "01",
            ],
        )

        assert result.exit_code != 0
        assert "--n-folds" in result.output

    def test_fold_by_required(self, tree, monkeypatch):
        _patch_trainer(monkeypatch)

        result = runner.invoke(
            app,
            ["encoding", "train", str(tree.root), *REQUIRED, "--sub-ids", "01"],
        )

        assert result.exit_code != 0

    def test_bad_downsample_rejected(self, tree, monkeypatch):
        _patch_trainer(monkeypatch)

        result = runner.invoke(
            app,
            [
                "encoding",
                "train",
                str(tree.root),
                *REQUIRED,
                "--fold-by",
                "none",
                "--sub-ids",
                "01",
                "--downsample",
                "bogus",
            ],
        )

        assert result.exit_code != 0
        assert "--downsample" in result.output


class TestEncodingTrainConfig:
    """Assert flags map to the trainer/config as intended (not just call counts)."""

    def _capture_init(self, monkeypatch):
        _patch_trainer(monkeypatch)
        from hypline.encoding import EncodingConfig, EncodingTrainer

        captured = {}
        orig = EncodingTrainer.__init__

        def spy(self, **kwargs):
            captured["config"] = kwargs["config"]
            captured["split"] = kwargs["split"]
            orig(self, **kwargs)

        monkeypatch.setattr(EncodingTrainer, "__init__", spy)
        return captured, EncodingConfig

    def test_no_split_flips_split_false(self, tree, monkeypatch):
        captured, _ = self._capture_init(monkeypatch)

        result = runner.invoke(
            app,
            [
                "encoding",
                "train",
                str(tree.root),
                *REQUIRED,
                "--fold-by",
                "none",
                "--sub-ids",
                "01",
                "--no-split",
            ],
        )

        assert result.exit_code == 0, result.output
        assert captured["split"] is False

    def test_split_defaults_true(self, tree, monkeypatch):
        captured, _ = self._capture_init(monkeypatch)

        runner.invoke(
            app,
            [
                "encoding",
                "train",
                str(tree.root),
                *REQUIRED,
                "--fold-by",
                "none",
                "--sub-ids",
                "01",
            ],
        )

        assert captured["split"] is True

    def test_delays_alphas_default_when_omitted(self, tree, monkeypatch):
        captured, EncodingConfig = self._capture_init(monkeypatch)

        runner.invoke(
            app,
            [
                "encoding",
                "train",
                str(tree.root),
                *REQUIRED,
                "--fold-by",
                "none",
                "--sub-ids",
                "01",
            ],
        )

        defaults = EncodingConfig()
        assert captured["config"].delays == defaults.delays
        assert captured["config"].alphas == defaults.alphas

    def test_delays_alphas_override_when_passed(self, tree, monkeypatch):
        captured, _ = self._capture_init(monkeypatch)

        runner.invoke(
            app,
            [
                "encoding",
                "train",
                str(tree.root),
                *REQUIRED,
                "--fold-by",
                "none",
                "--sub-ids",
                "01",
                "--delays",
                "0,1,2",
                "--alphas",
                "1,10",
            ],
        )

        assert captured["config"].delays == [0, 1, 2]
        assert captured["config"].alphas == [1.0, 10.0]
