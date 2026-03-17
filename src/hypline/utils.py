import re
from multiprocessing import Process
from pathlib import Path

import dill

_BIDS_ENTITY_RE = re.compile(r"^[a-z]+-[a-zA-Z0-9]+$")


def validate_dirs(*paths: Path) -> None:
    for path in paths:
        if not path.is_dir():
            raise FileNotFoundError(f"Directory does not exist: {path}")


def validate_bids_entities(*tags: str) -> None:
    for tag in tags:
        if not _BIDS_ENTITY_RE.match(tag):
            raise ValueError(f"Invalid BIDS entity tag: {tag!r}")


def find_files(
    directory: Path,
    ext: str,
    *,
    filters: list[str] | None = None,
) -> list[Path]:
    ext = ext.strip()
    if not ext.startswith("."):
        ext = f".{ext}"

    tags = filters or []
    return sorted(
        file
        for file in directory.iterdir()
        if file.suffix == ext and all(tag in file.name for tag in tags)
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
        super().__init__(*args, **kwargs)
        self._target = dill.dumps(self._target)
        self._args, self._kwargs = self._args, self._kwargs  # For type checker

    def run(self):
        if self._target:
            self._target = dill.loads(self._target)
            self._target(*self._args, **self._kwargs)
