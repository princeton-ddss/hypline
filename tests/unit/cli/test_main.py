import pytest

from hypline.cli import main

HINT = "Rerun as `hypline --verbose <command> ...`"


def _run(monkeypatch, *args):
    monkeypatch.setattr("sys.argv", ["hypline", *args])
    main()


def _fail_setup(monkeypatch, exc):
    """Make semantic featuregen raise `exc` during setup, before `run_per_id` runs."""
    from hypline.features.semantic import SemanticFeature

    def _raise(self, *args, **kwargs):
        raise exc

    monkeypatch.setattr(SemanticFeature, "__init__", _raise)


def _semantic_args(tree):
    return [
        "featuregen",
        "semantic",
        str(tree.root),
        "--model",
        "gpt2",
        "--dyad-ids",
        "01",
    ]


class TestMain:
    def test_setup_error_prints_message_and_hint(self, tree, monkeypatch, capsys):
        _fail_setup(monkeypatch, OSError("Gated repo.\nPlease log in."))

        with pytest.raises(SystemExit) as exc_info:
            _run(monkeypatch, *_semantic_args(tree))

        assert exc_info.value.code == 1
        lines = capsys.readouterr().err.splitlines()
        # Original line breaks are kept, and the hint gets a line of its own
        assert lines[0].endswith("| OSError: Gated repo.")
        assert lines[1] == "Please log in."
        assert lines[2].endswith(f"| {HINT} for the full traceback")
        assert len(lines) == 3

    def test_empty_message_prints_type_name_only(self, tree, monkeypatch, capsys):
        _fail_setup(monkeypatch, AssertionError())

        with pytest.raises(SystemExit):
            _run(monkeypatch, *_semantic_args(tree))

        assert capsys.readouterr().err.splitlines()[0].endswith("| AssertionError")

    def test_braces_in_message_are_printed_verbatim(self, tree, monkeypatch, capsys):
        _fail_setup(monkeypatch, ValueError("{missing}"))

        with pytest.raises(SystemExit):
            _run(monkeypatch, *_semantic_args(tree))

        assert "ValueError: {missing}" in capsys.readouterr().err

    def test_verbose_reraises(self, tree, monkeypatch):
        _fail_setup(monkeypatch, OSError("Gated repo."))

        with pytest.raises(OSError, match="Gated repo"):
            _run(monkeypatch, "--verbose", *_semantic_args(tree))

    def test_version_exit_is_not_caught(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc_info:
            _run(monkeypatch, "--version")

        assert exc_info.value.code == 0
        assert HINT not in capsys.readouterr().err
