import json
from typing import NamedTuple

from hypline.bids import BIDSPath
from hypline.enums import SurfaceSpace, VolumeSpace
from hypline.events import Segment, load_segments
from hypline.layout import BIDSLayout

_BOLD_SPACES = {
    space.value: space
    for space_type in (SurfaceSpace, VolumeSpace)
    for space in space_type
}

BOLD_EXTENSIONS = {
    SurfaceSpace: ".func.gii",
    VolumeSpace: ".nii.gz",
}


def _validate_bold(bids: BIDSPath) -> None:
    if bids.suffix != "bold" or not bids.path.name.endswith(
        tuple(BOLD_EXTENSIONS.values())
    ):
        raise ValueError(
            f"Expected a BOLD file (suffix 'bold' with extension in "
            f"{sorted(BOLD_EXTENSIONS.values())}); got {bids.path.name!r}"
        )


def parse_bold_space(value: str) -> SurfaceSpace | VolumeSpace:
    bold_space = _BOLD_SPACES.get(value)
    if bold_space is None:
        valid = ", ".join(_BOLD_SPACES.keys())
        raise ValueError(
            f"Unsupported BOLD data space: {value}. Must be one of: {valid}"
        )
    return bold_space


def get_repetition_time(layout: BIDSLayout, bids: BIDSPath) -> float:
    """Extract TR in seconds for a BOLD run; never loads voxel data.

    Accepts any `bids` carrying the run's identity entities. TR is
    acquisition-level and identical across raw and derivative files, so
    resolution is anchored on the raw BIDS tree via `layout.path.raw`: tries
    `*_bold.json` first, then the raw BOLD image header. Raises ValueError if
    no source yields a usable TR or the format is unsupported.
    """
    sidecar = layout.path.raw(source=bids, suffix="bold", ext=".json")
    if sidecar.path.exists():
        with open(sidecar.path) as f:
            TR = json.load(f).get("RepetitionTime")
        if TR is not None:
            return float(TR)

    for ext in BOLD_EXTENSIONS.values():
        raw_bold = layout.path.raw(source=bids, suffix="bold", ext=ext)
        if raw_bold.path.exists():
            break
    else:
        raise ValueError(
            f"Cannot resolve TR for {bids.path.name}: "
            f"no raw sidecar or BOLD image found"
        )

    import nibabel as nib

    img = nib.load(raw_bold.path)
    if isinstance(img, nib.Nifti1Image):
        TR = img.header.get_zooms()[3]
        if TR > 0:
            return float(TR)
        raise ValueError(f"TR is zero or unset in NIfTI header: {raw_bold.path.name}")
    if isinstance(img, nib.GiftiImage):
        time_step = img.darrays[0].meta.get("TimeStep")
        if time_step is not None:
            return float(time_step) / 1000  # milliseconds to seconds
        raise ValueError(f"TimeStep missing from GIfTI metadata: {raw_bold.path.name}")
    raise ValueError(f"Unsupported image format: {type(img).__name__}")


def get_n_trs(bids: BIDSPath) -> int:
    """Extract volume count for a BOLD run; never loads voxel data."""
    _validate_bold(bids)

    import nibabel as nib

    img = nib.load(bids.path)
    if isinstance(img, nib.Nifti1Image):
        return int(img.header.get_data_shape()[3])
    if isinstance(img, nib.GiftiImage):
        return len(img.darrays)
    raise ValueError(f"Unsupported image format: {type(img).__name__}")


class BoldMeta(NamedTuple):
    bids: BIDSPath
    repetition_time: float
    n_trs: int
    segments: list[Segment]


def load_bold_meta(layout: BIDSLayout, bids: BIDSPath) -> BoldMeta:
    """Load TR, segments, and segment metadata for a BOLD run.

    Segment metadata comes from events.json `trial_type.Levels`: entries whose
    keys match the BIDS entity-value pattern are merged into `Segment.metadata`;
    other entries are ignored. Segments have empty metadata if events.json is
    absent.

    Sidecars (events.tsv, events.json) are resolved canonically from the raw
    BIDS tree via `layout.path.raw`; misnamed siblings are not inspected.

    When `bids` is a derivative, its volume count must match the raw BOLD —
    events.tsv onsets are raw-relative and hypline does not shift them.

    Raises ValueError if `bids` lacks a `task` entity (events sidecars are
    resolved by task), if a derivative `bids` has a different volume count
    than its raw counterpart, if events.tsv or events.json is invalid, if
    events.json declares segment entries that events.tsv does not, or if a
    `task` segment value disagrees with the filename's task entity.
    """
    _validate_bold(bids)

    if "task" not in bids.entities:
        raise ValueError(
            f"BOLD file {bids.path.name!r} missing required 'task' entity (BIDS)"
        )

    repetition_time = get_repetition_time(layout, bids)
    n_trs = get_n_trs(bids)

    # BOLD_EXTENSIONS pins volumetric raw to .nii.gz; surface derivatives
    # (.func.gii) inherit any trim applied during volumetric preprocessing.
    raw_bold_bids = layout.path.raw(source=bids, suffix="bold", ext=".nii.gz")
    if raw_bold_bids.path.exists() and raw_bold_bids.path != bids.path:
        n_trs_raw = get_n_trs(raw_bold_bids)
        if n_trs_raw != n_trs:
            raise ValueError(
                f"Derivative BOLD {bids.path.name!r} has {n_trs} volumes but "
                f"raw has {n_trs_raw}. events.tsv onsets are raw-relative; "
                f"hypline requires derivative n_trs to match raw."
            )

    segments = load_segments(layout, bids)
    if segments and segments[0].entity == "task":
        task_value = bids.entities.get("task")
        if segments[0].value != task_value:
            raise ValueError(
                f"events.tsv 'task-{segments[0].value}' does not match "
                f"filename 'task-{task_value}'"
            )

    return BoldMeta(
        bids=bids,
        repetition_time=repetition_time,
        n_trs=n_trs,
        segments=segments,
    )
