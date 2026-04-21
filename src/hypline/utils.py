import os
from multiprocessing import Process
from pathlib import Path

from hypline.bids import validate_bids_entities


def validate_dirs(*paths: str | os.PathLike[str]) -> None:
    for path in paths:
        path = Path(path)
        if not path.is_dir():
            raise FileNotFoundError(f"Directory does not exist: {path}")


def find_files(
    directory: str | os.PathLike[str],
    *,
    ends_with: str,
    recursive: bool = False,
    bids_filters: list[str] | None = None,
) -> list[Path]:
    """
    Find files in a directory whose names end with the given string.

    When BIDS filters are provided, filters sharing the same key
    (e.g., run-1 and run-2) are OR'd, while filters with different
    keys (e.g., run-1 and sub-01) are AND'd.

    Parameters
    ----------
    directory : str or os.PathLike
        Directory to search for files.
    ends_with : str
        Filename ending to match (e.g., ".csv", ".nii.gz", "_bold.func.gii").
    recursive : bool, optional
        If True, search subdirectories recursively. Default is False.
    bids_filters : list of str, optional
        BIDS entities to filter filenames by (e.g., ["run-1", "sub-01"]).

    Returns
    -------
    list of Path
        Matching files, sorted by name.
    """
    directory = Path(directory)

    files = directory.rglob("*") if recursive else directory.iterdir()

    if not bids_filters:
        return sorted(f for f in files if f.is_file() and f.name.endswith(ends_with))

    validate_bids_entities(*bids_filters)

    groups: dict[str, list[str]] = {}
    for entity in bids_filters:
        key = entity.split("-", 1)[0]
        groups.setdefault(key, []).append(entity)

    return sorted(
        file
        for file in files
        if file.is_file()
        and file.name.endswith(ends_with)
        and all(
            any(entity in file.name for entity in group) for group in groups.values()
        )
    )


class DillProcess(Process):
    """
    Extend the `Process` class to support serialization
    of closures and local functions.

    Notes
    -----
    Adapted from https://stackoverflow.com/a/72776044.
    """

    def __init__(self, *args, **kwargs):
        import dill

        super().__init__(*args, **kwargs)
        self._target = dill.dumps(self._target)
        self._args, self._kwargs = self._args, self._kwargs  # For type checker

    def run(self):
        import dill

        if self._target:
            self._target = dill.loads(self._target)
            self._target(*self._args, **self._kwargs)
