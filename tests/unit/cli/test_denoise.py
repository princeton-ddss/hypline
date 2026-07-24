from typer.testing import CliRunner

from hypline.cli import app

# When color is forced (as in CI), Rich wraps each part of an option name in its
# own ANSI codes, so `--custom-sources` is not a literal substring of the error
# output. Disable Rich rendering so error assertions match plain text.
app.rich_markup_mode = None

runner = CliRunner()


def _capture_denoiser(monkeypatch):
    """Patch Denoiser.__init__ to record kwargs and skip the per-run denoise.

    The command still resolves subjects and constructs Denoiser (so CLI wiring
    runs), but __init__ is stubbed to record `columns`/`compcor`/etc. and
    `run_per_id` is neutered so nothing reads BOLD or writes output.
    """
    import hypline.cli.denoise as cli_denoise
    from hypline.denoise import Denoiser

    captured = {}

    def _spy_init(self, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(Denoiser, "__init__", _spy_init)
    monkeypatch.setattr(cli_denoise, "run_per_id", lambda *a, **k: None)
    return captured


class TestDenoiseChannelFallback:
    def test_no_channel_defaults_to_speer_columns(self, tree, monkeypatch):
        captured = _capture_denoiser(monkeypatch)
        from hypline.denoise import DEFAULT_CONFOUND_COLUMNS

        result = runner.invoke(app, ["denoise", str(tree.root), "--sub-ids", "01"])

        assert result.exit_code == 0, result.output
        assert captured["columns"] == DEFAULT_CONFOUND_COLUMNS
        assert captured["compcor"] == []
        assert captured["custom_sources"] == []

    def test_explicit_columns_suppress_fallback(self, tree, monkeypatch):
        captured = _capture_denoiser(monkeypatch)

        result = runner.invoke(
            app,
            ["denoise", str(tree.root), "--sub-ids", "01", "--columns", "trans_x"],
        )

        assert result.exit_code == 0, result.output
        assert captured["columns"] == ["trans_x"]

    def test_explicit_compcor_suppresses_column_fallback(self, tree, monkeypatch):
        # A user picking CompCor is choosing their own model; the default column
        # set must not be composed onto it.
        captured = _capture_denoiser(monkeypatch)

        result = runner.invoke(
            app,
            ["denoise", str(tree.root), "--sub-ids", "01", "--compcor", "a"],
        )

        assert result.exit_code == 0, result.output
        assert captured["columns"] == []

    def test_explicit_custom_sources_suppresses_column_fallback(
        self, tree, monkeypatch
    ):
        # Same rationale as CompCor: a custom-regressor model is the user's own choice.
        captured = _capture_denoiser(monkeypatch)

        result = runner.invoke(
            app,
            [
                "denoise",
                str(tree.root),
                "--sub-ids",
                "01",
                "--custom-sources",
                "physio",
                "--custom-columns",
                "v1",
            ],
        )

        assert result.exit_code == 0, result.output
        assert captured["columns"] == []

    def test_custom_sources_without_columns_errors(self, tree, monkeypatch):
        # The pairing check reports both option names; this asserts on the
        # plain-text error, which is why Rich rendering is disabled above.
        _capture_denoiser(monkeypatch)

        result = runner.invoke(
            app,
            [
                "denoise",
                str(tree.root),
                "--sub-ids",
                "01",
                "--custom-sources",
                "physio",
            ],
        )

        assert result.exit_code != 0
        assert "--custom-sources/--custom-columns" in result.output
