import os
from pathlib import Path

import polars as pl

from hypline.bids import BIDSPath
from hypline.enums import SurfaceSpace, VolumeSpace

_BOLD_IDENTITY_ENTITIES = frozenset(
    ("sub", "ses", "task", "acq", "ce", "rec", "dir", "run")
)

_BOLD_SPACES = {
    space.value: space
    for space_type in (SurfaceSpace, VolumeSpace)
    for space in space_type
}

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
