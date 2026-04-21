import os
from multiprocessing import Process
from pathlib import Path

import polars as pl

from hypline.bids import BIDSPath, validate_bids_entities
from hypline.enums import SurfaceSpace, VolumeSpace

_BOLD_IDENTITY_ENTITIES = frozenset(
    ("sub", "ses", "task", "acq", "ce", "rec", "dir", "run")
)

# Mapping between a BOLD data space name and its enum variant
_BOLD_SPACES = {
    space.value: space
    for space_type in (SurfaceSpace, VolumeSpace)
    for space in space_type
}

# Mapping from BOLD space type to its file extension
BOLD_EXTENSIONS = {
    SurfaceSpace: ".func.gii",
    VolumeSpace: ".nii.gz",
}


def parse_bold_space(value: str) -> SurfaceSpace | VolumeSpace:
    bold_space = _BOLD_SPACES.get(value)
    if bold_space is None:
        valid = ", ".join(_BOLD_SPACES.keys())
        raise ValueError(
            f"Unsupported BOLD data space: {value}. Must be one of: {valid}"
        )
    return bold_space


def _strip_bold_extension(bold_path: Path) -> str:
    """Return the filename stem with the imaging extension removed."""
    name = bold_path.name
    for ext in BOLD_EXTENSIONS.values():
        if name.endswith(ext):
            return name[: -len(ext)]
    raise ValueError(f"Unrecognized BOLD file extension: {name}")


def get_repetition_time(bold_path: str | os.PathLike[str]) -> float:
    """
    Extract the repetition time (TR) in seconds from a BOLD file.

    Reads TR from the BIDS JSON sidecar if available, otherwise
    falls back to the NIfTI header for volume data or GIfTI darray
    metadata for surface data.

    Parameters
    ----------
    bold_path : str or os.PathLike
        Path to a BOLD file (.nii, .nii.gz, or .func.gii).

    Returns
    -------
    float
        Repetition time in seconds.

    Raises
    ------
    ValueError
        If TR cannot be determined from any source.
    """
    import json

    import nibabel as nib

    bold_path = Path(bold_path)

    # Primary: BIDS JSON sidecar
    stem = _strip_bold_extension(bold_path)
    sidecar = bold_path.with_name(stem + ".json")
    if sidecar.exists():
        with open(sidecar) as f:
            TR = json.load(f).get("RepetitionTime")
        if TR is not None:
            return float(TR)

    # Fallback: format-specific header/metadata
    img = nib.load(bold_path)
    if isinstance(img, nib.Nifti1Image):
        TR = img.header.get_zooms()[3]
        if TR > 0:
            return float(TR)
        raise ValueError(f"TR is zero or unset in NIfTI header: {bold_path.name}")
    if isinstance(img, nib.GiftiImage):
        time_step = img.darrays[0].meta.get("TimeStep")
        if time_step is not None:
            return float(time_step) / 1000  # milliseconds to seconds
        raise ValueError(f"TimeStep missing from GIfTI metadata: {bold_path.name}")
    raise ValueError(f"Unsupported image format: {type(img).__name__}")


def load_events(bold_path: str | os.PathLike[str]) -> pl.DataFrame | None:
    """Load the events TSV colocated with a BOLD file, or return None.

    Events must be named with identity entities only (no space, desc, etc.)
    in canonical BIDS order. If any file in the same directory shares this
    run's identity entities but is not the canonical name, raises rather than
    silently ignoring it — same run implies same stimulus timeline, so such
    variants are either redundant or divergent, and both cases warrant surfacing.
    """
    bids = BIDSPath(bold_path)
    shared = {k: v for k, v in bids.entities.items() if k in _BOLD_IDENTITY_ENTITIES}
    stem = "_".join(f"{k}-{v}" for k, v in shared.items())
    events_file = bids.path.parent / (stem + "_events.tsv")
    if events_file.exists():
        return pl.read_csv(events_file, separator="\t")

    misnamed_siblings = [
        p
        for p in bids.path.parent.glob("*_events.tsv")
        if p != events_file
        and all(BIDSPath(p).entities.get(k) == v for k, v in shared.items())
    ]
    if misnamed_siblings:
        raise ValueError(
            f"Expected {events_file.name!r} but found unexpected events file(s) "
            "colocated with this BOLD run: "
            f"{[p.name for p in sorted(misnamed_siblings)]}. "
            "Rename to use identity entities only in canonical BIDS order."
        )

    return None


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
