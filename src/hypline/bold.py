import json
from pathlib import Path
from typing import NamedTuple

from hypline.bids import BOLD_IDENTITY_ENTITIES, BIDSPath
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


def _is_bold(bids: BIDSPath) -> bool:
    return bids.suffix == "bold" and bids.path.name.endswith(
        tuple(BOLD_EXTENSIONS.values())
    )


def _validate_bold(bids: BIDSPath) -> None:
    if not _is_bold(bids):
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


def _tr_from_image(path: Path) -> float | None:
    """Read TR (seconds) from a NIfTI BOLD header, or None if unusable.

    Returns None when unusable — `path` absent, not NIfTI, or the header zoom
    missing/zero — so callers fall through to the next source rather than
    raising mid-ladder.
    """
    if not path.exists():
        return None

    import nibabel as nib

    img = nib.load(path)
    if isinstance(img, nib.Nifti1Image):
        TR = img.header.get_zooms()[3]
        return float(TR) if TR > 0 else None
    return None


def _tr_from_sidecar(path: Path) -> float | None:
    """Read RepetitionTime from a `*_bold.json` sidecar, or None if unusable.

    The sidecar's declared value is exact, unlike an image header's float32
    zoom; callers prefer it over the sibling image header.
    """
    if not path.exists():
        return None
    with open(path) as f:
        TR = json.load(f).get("RepetitionTime")
    return float(TR) if TR is not None else None


def _tr_for_bold(bids: BIDSPath) -> float | None:
    """Resolve TR for one BOLD `.nii.gz`: its sibling `.json` sidecar, then header."""
    sidecar = bids.path.with_name(bids.path.name.rsplit(".nii.gz", 1)[0] + ".json")
    TR = _tr_from_sidecar(sidecar)
    if TR is not None:
        return TR
    return _tr_from_image(bids.path)


def _tr_from_fmriprep_run(layout: BIDSLayout, bids: BIDSPath) -> float | None:
    """Resolve TR from any fmriprep BOLD `.nii.gz` for `bids`'s run.

    TR is acquisition-level, so any fmriprep variant (any `space`, `desc`,
    `res`, ...) carries the same value; `.nii.gz` only, since surface
    derivatives carry TR unreliably. Returns None if none yields a usable TR.

    Fast path: when `bids` is itself a NIfTI BOLD, read it directly, skip `find`.
    """
    if _is_bold(bids) and bids.path.name.endswith(".nii.gz"):
        TR = _tr_for_bold(bids)
        if TR is not None:
            return TR

    sub = bids.entities.get("sub")
    if sub is None:
        return None
    run_filters = [
        f"{k}-{bids.entities[k]}"
        for k in BOLD_IDENTITY_ENTITIES - {"sub"}
        if k in bids.entities
    ]
    try:
        candidates = layout.find.fmriprep(
            sub=sub, suffix="bold", ext=".nii.gz", bids_filters=run_filters
        )
    except FileNotFoundError:
        return None

    for candidate in candidates:
        TR = _tr_for_bold(candidate)
        if TR is not None:
            return TR
    return None


def get_repetition_time(layout: BIDSLayout, bids: BIDSPath) -> float:
    """Extract TR in seconds for a BOLD run; never loads voxel data.

    Accepts any `bids` carrying the run's identity entities. TR is
    acquisition-level and identical across raw and derivative files, so
    sources are tried by trust, requiring raw imaging files only as a last
    resort (fMRIPrep derivatives are commonly analyzed without the raw BOLD
    tree copied alongside):

    1. raw `*_bold.json` sidecar (exact, BIDS-declared, tiny — often retained
       even when raw BOLD images are not)
    2. any fmriprep BOLD `.nii.gz` for the run, preferring its sibling
       `*_bold.json` over the header (derivatives-only; reads `bids` directly
       when it already is that file, else finds a sibling)
    3. raw BOLD image header (last resort; the file most often absent)

    Raises ValueError if no source yields a usable TR.
    """
    raw_sidecar = layout.path.raw(source=bids, suffix="bold", ext=".json")
    TR = _tr_from_sidecar(raw_sidecar.path)
    if TR is not None:
        return TR

    TR = _tr_from_fmriprep_run(layout, bids)
    if TR is not None:
        return TR

    # Raw BOLD is volumetric .nii.gz by convention (surface is a derivative)
    raw_bold = layout.path.raw(source=bids, suffix="bold", ext=".nii.gz")
    TR = _tr_from_image(raw_bold.path)
    if TR is not None:
        return TR

    raise ValueError(
        f"Cannot resolve TR for {bids.path.name}: no raw sidecar yielded "
        f"RepetitionTime, and no BOLD image header carried a usable TR"
    )


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
