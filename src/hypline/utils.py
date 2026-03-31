import os
from multiprocessing import Process
from pathlib import Path

import dill

from hypline.bids import validate_bids_entities as validate_bids_entities
from hypline.enums import SurfaceSpace, VolumeSpace

# Mapping between a BOLD data space name and its enum variant
_BOLD_SPACES = {
    space.value: space
    for space_type in (SurfaceSpace, VolumeSpace)
    for space in space_type
}

_BOLD_EXTENSIONS = (".nii.gz", ".func.gii")


def parse_bold_space(value: str) -> SurfaceSpace | VolumeSpace:
    bold_space = _BOLD_SPACES.get(value)
    if bold_space is None:
        valid = ", ".join(_BOLD_SPACES.keys())
        raise ValueError(
            f"Unsupported BOLD data space: {value}. Must be one of: {valid}"
        )
    return bold_space


def _strip_bold_extension(filepath: Path) -> str:
    """Return the filename stem with the imaging extension removed."""
    name = filepath.name
    for ext in _BOLD_EXTENSIONS:
        if name.endswith(ext):
            return name[: -len(ext)]
    raise ValueError(f"Unrecognized BOLD file extension: {name}")


def get_repetition_time(filepath: str | os.PathLike[str]) -> float:
    """
    Extract the repetition time (TR) in seconds from a BOLD file.

    Reads TR from the BIDS JSON sidecar if available, otherwise
    falls back to the NIfTI header for volume data or GIfTI darray
    metadata for surface data.

    Parameters
    ----------
    filepath : str or os.PathLike
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

    filepath = Path(filepath)

    # Primary: BIDS JSON sidecar
    stem = _strip_bold_extension(filepath)
    sidecar = filepath.with_name(stem + ".json")
    if sidecar.exists():
        with open(sidecar) as f:
            TR = json.load(f).get("RepetitionTime")
        if TR is not None:
            return float(TR)

    # Fallback: format-specific header/metadata
    img = nib.load(filepath)
    if isinstance(img, nib.Nifti1Image):
        TR = img.header.get_zooms()[3]
        if TR > 0:
            return float(TR)
        raise ValueError(f"TR is zero or unset in NIfTI header: {filepath.name}")
    if isinstance(img, nib.GiftiImage):
        time_step = img.darrays[0].meta.get("TimeStep")
        if time_step is not None:
            return float(time_step) / 1000  # milliseconds to seconds
        raise ValueError(f"TimeStep missing from GIfTI metadata: {filepath.name}")
    raise ValueError(f"Unsupported image format: {type(img).__name__}")


def validate_dirs(*paths: str | os.PathLike[str]) -> None:
    for path in paths:
        path = Path(path)
        if not path.is_dir():
            raise FileNotFoundError(f"Directory does not exist: {path}")


def find_files(
    directory: str | os.PathLike[str],
    ext: str,
    *,
    bids_filters: list[str] | None = None,
) -> list[Path]:
    """
    Find files in a directory matching the given extension.

    When BIDS filters are provided, filters sharing the same key
    (e.g., run-1 and run-2) are OR'd, while filters with different
    keys (e.g., run-1 and sub-01) are AND'd.

    Parameters
    ----------
    directory : str or os.PathLike
        Directory to search for files.
    ext : str
        File extension to match (e.g., ".wav" or "wav").
    bids_filters : list of str, optional
        BIDS entities to filter filenames by (e.g., ["run-1", "sub-01"]).

    Returns
    -------
    list of Path
        Matching files, sorted by name.
    """
    directory = Path(directory)
    ext = ext.strip()
    if not ext.startswith("."):
        ext = f".{ext}"

    if not bids_filters:
        return sorted(f for f in directory.iterdir() if f.suffix == ext)

    validate_bids_entities(*bids_filters)

    groups: dict[str, list[str]] = {}
    for entity in bids_filters:
        key = entity.split("-")[0]
        groups.setdefault(key, []).append(entity)

    return sorted(
        file
        for file in directory.iterdir()
        if file.suffix == ext
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
        super().__init__(*args, **kwargs)
        self._target = dill.dumps(self._target)
        self._args, self._kwargs = self._args, self._kwargs  # For type checker

    def run(self):
        if self._target:
            self._target = dill.loads(self._target)
            self._target(*self._args, **self._kwargs)
