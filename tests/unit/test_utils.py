from pathlib import Path

import pytest

from hypline.utils import load_events


def _write_events(
    dirpath: Path,
    stem: str,
    *,
    content: str = "trial_type\tonset\tduration\n",
) -> Path:
    events_path = dirpath / (stem + "_events.tsv")
    events_path.write_text(content)
    return events_path


def _write_bold(dirpath: Path, stem: str) -> Path:
    bold_path = dirpath / (stem + "_bold.nii.gz")
    bold_path.touch()
    return bold_path


class TestLoadEvents:
    def test_spec_compliant_events_returned(self, tmp_path: Path):
        bold_path = _write_bold(
            tmp_path,
            "sub-01_task-movie_run-1_space-MNI152NLin2009cAsym",
        )
        _write_events(
            tmp_path,
            "sub-01_task-movie_run-1",
        )
        result = load_events(bold_path)
        assert result is not None
        assert "trial_type" in result.columns

    def test_canonical_events_returned_when_misnamed_sibling_also_present(
        self, tmp_path: Path
    ):
        bold_path = _write_bold(
            tmp_path,
            "sub-01_task-movie_run-1_space-MNI152NLin2009cAsym",
        )
        _write_events(tmp_path, "sub-01_task-movie_run-1")
        _write_events(tmp_path, "sub-01_task-movie_run-1_desc-preproc")
        result = load_events(bold_path)
        assert result is not None
        assert "trial_type" in result.columns

    def test_no_events_returns_none(self, tmp_path: Path):
        bold_path = _write_bold(
            tmp_path,
            "sub-01_task-movie_run-1_space-MNI152NLin2009cAsym",
        )
        assert load_events(bold_path) is None

    def test_misnamed_sibling_with_space_raises(self, tmp_path: Path):
        bold_path = _write_bold(
            tmp_path,
            "sub-01_task-movie_run-1_space-MNI152NLin2009cAsym",
        )
        _write_events(
            tmp_path,
            "sub-01_task-movie_run-1_space-MNI152NLin2009cAsym",
        )
        with pytest.raises(ValueError, match="unexpected events file"):
            load_events(bold_path)

    def test_misnamed_sibling_with_desc_raises(self, tmp_path: Path):
        bold_path = _write_bold(
            tmp_path,
            "sub-01_task-movie_run-1_space-MNI152NLin2009cAsym",
        )
        _write_events(
            tmp_path,
            "sub-01_task-movie_run-1_desc-preproc",
        )
        with pytest.raises(ValueError, match="unexpected events file"):
            load_events(bold_path)

    def test_misnamed_sibling_with_space_and_desc_raises(self, tmp_path: Path):
        bold_path = _write_bold(
            tmp_path,
            "sub-01_task-movie_run-1_space-MNI152NLin2009cAsym",
        )
        events_path = _write_events(
            tmp_path,
            "sub-01_task-movie_run-1_space-MNI152NLin2009cAsym_desc-preproc",
        )
        with pytest.raises(ValueError, match=events_path.name):
            load_events(bold_path)

    def test_misnamed_sibling_with_reordered_entities_raises(self, tmp_path: Path):
        bold_path = _write_bold(
            tmp_path,
            "sub-01_task-movie_run-1_space-MNI152NLin2009cAsym",
        )
        _write_events(
            tmp_path,
            "sub-01_run-1_task-movie",
        )
        with pytest.raises(ValueError, match="unexpected events file"):
            load_events(bold_path)

    def test_multiple_misnamed_siblings_all_listed(self, tmp_path: Path):
        bold_path = _write_bold(
            tmp_path,
            "sub-01_task-movie_run-1_space-MNI152NLin2009cAsym",
        )
        for space in ("MNI152NLin2009cAsym", "T1w"):
            _write_events(
                tmp_path,
                f"sub-01_task-movie_run-1_space-{space}",
            )
        with pytest.raises(ValueError) as exc_info:
            load_events(bold_path)
        msg = str(exc_info.value)
        assert "MNI152NLin2009cAsym" in msg
        assert "T1w" in msg

    def test_unrelated_events_in_same_dir_ignored(self, tmp_path: Path):
        bold_path = _write_bold(
            tmp_path,
            "sub-01_task-movie_run-1_space-MNI152NLin2009cAsym",
        )
        _write_events(
            tmp_path,
            "sub-02_task-movie_run-1",
        )
        assert load_events(bold_path) is None

    def test_different_run_events_in_same_dir_ignored(self, tmp_path: Path):
        bold_path = _write_bold(
            tmp_path,
            "sub-01_task-movie_run-1_space-MNI152NLin2009cAsym",
        )
        _write_events(
            tmp_path,
            "sub-01_task-movie_run-2",
        )
        assert load_events(bold_path) is None

    def test_misnamed_sibling_for_different_run_ignored(self, tmp_path: Path):
        bold_path = _write_bold(
            tmp_path,
            "sub-01_task-movie_run-1_space-MNI152NLin2009cAsym",
        )
        _write_events(
            tmp_path,
            "sub-01_task-movie_run-2_space-MNI152NLin2009cAsym",
        )
        assert load_events(bold_path) is None
