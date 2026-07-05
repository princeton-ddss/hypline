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


def run_identity_filters(source: BIDSPath) -> list[str]:
    """BIDS `key-value` filters pinning `source`'s run identity, excluding `sub`.

    `sub` is omitted because callers pass it as a separate `layout.find`
    argument; re-adding it here would duplicate the filter.
    """
    return [
        f"{k}-{source.entities[k]}"
        for k in BOLD_IDENTITY_ENTITIES - {"sub"}
        if k in source.entities
    ]


def _find_fmriprep_bold(layout: BIDSLayout, source: BIDSPath) -> list[BIDSPath]:
    """fmriprep BOLD `.nii.gz` files for `source`'s run (any space/desc variant).

    Returns [] if `source` lacks a `sub` entity or no fmriprep BOLD exists.
    `.nii.gz` only: surface derivatives carry TR unreliably, and n_trs is
    preserved across variants, so a volumetric file answers both.
    """
    sub = source.entities.get("sub")
    if sub is None:
        return []
    try:
        return layout.find.fmriprep(
            sub=sub,
            suffix="bold",
            ext=".nii.gz",
            bids_filters=run_identity_filters(source),
        )
    except FileNotFoundError:
        return []


def resolve_bold_image(layout: BIDSLayout, source: BIDSPath) -> BIDSPath:
    """Resolve an on-disk BOLD `.nii.gz` for `source`'s run, for loading n_trs.

    Unlike TR (resolvable from a tiny sidecar), the volume count requires a real
    image. Resolves by trust: any fmriprep BOLD for the run, then the raw image.
    Accepts any `source` carrying the run's identity entities (e.g. a feature).

    Raises FileNotFoundError if neither an fmriprep nor a raw BOLD image exists.
    """
    fmriprep = _find_fmriprep_bold(layout, source)
    if fmriprep:
        return fmriprep[0]

    raw_bold = layout.path.raw(source=source, suffix="bold", ext=".nii.gz")
    if raw_bold.path.exists():
        return raw_bold

    raise FileNotFoundError(
        f"No BOLD image found for {source.path.name}: neither an fmriprep nor a "
        f"raw `*_bold.nii.gz` exists for the run"
    )


def get_repetition_time(layout: BIDSLayout, source: BIDSPath) -> float:
    """Extract TR in seconds for a BOLD run; never loads voxel data.

    Accepts any `source` carrying the run's identity entities. TR is
    acquisition-level and identical across raw and derivative files, so
    sources are tried by trust, requiring raw imaging files only as a last
    resort (fMRIPrep derivatives are commonly analyzed without the raw BOLD
    tree copied alongside):

    1. raw `*_bold.json` sidecar (exact, BIDS-declared, tiny — often retained
       even when raw BOLD images are not)
    2. any fmriprep BOLD `.nii.gz` for the run, preferring its sibling
       `*_bold.json` over the header
    3. raw BOLD image header (last resort; the file most often absent)

    Raises ValueError if no source yields a usable TR.
    """
    raw_sidecar = layout.path.raw(source=source, suffix="bold", ext=".json")
    TR = _tr_from_sidecar(raw_sidecar.path)
    if TR is not None:
        return TR

    for candidate in _find_fmriprep_bold(layout, source):
        TR = _tr_for_bold(candidate)
        if TR is not None:
            return TR

    raw_bold = layout.path.raw(source=source, suffix="bold", ext=".nii.gz")
    TR = _tr_from_image(raw_bold.path)
    if TR is not None:
        return TR

    raise ValueError(
        f"Cannot resolve TR for {source.path.name}: no raw sidecar yielded "
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


# One run's voxel-bearing file(s): a lone volume, or the ordered (L, R) surface
# pair fmriprep emits per hemisphere. The two arities are the two BOLD spaces —
# a tuple never holds a volume, a bare path never holds one hemisphere.
VoxelSource = BIDSPath | tuple[BIDSPath, BIDSPath]


def resolve_voxel_source(images: list[BIDSPath]) -> VoxelSource:
    """Validate one run's discovered BOLD files and resolve them to a `VoxelSource`.

    `images` is whatever discovery grouped under one run's `BoldKey`; this returns
    the single `VoxelSource` those voxels live in, or raises if the files do not
    form a valid one.

    fmriprep emits surface BOLD per hemisphere, so one run has two `.func.gii`
    files differing only by `hemi`; a joint encoding model needs their voxels in a
    fixed column order across train and predict, or `Y`/`Y_hat` columns scramble
    silently, so the surface pair is returned ordered `(L, R)`. Volume BOLD is one
    file per run and returns as a bare `BIDSPath`.

    Dispatches on `hemi` presence, not extension: a file carries `hemi` iff it is
    surface. Requires exactly `{L, R}` for a surface run — a missing or repeated
    hemi raises rather than fitting on half a brain.
    """
    hemis = [img.entities.get("hemi") for img in images]
    if all(h is None for h in hemis):
        if len(images) != 1:
            raise ValueError(
                f"Expected one volume BOLD file per run; got {len(images)}: "
                f"{sorted(img.path.name for img in images)}"
            )
        return images[0]

    by_hemi = {h: img for h, img in zip(hemis, images)}
    if set(by_hemi) != {"L", "R"} or len(by_hemi) != len(images):
        raise ValueError(
            f"Surface run must have exactly one hemi-L and one hemi-R file; got "
            f"hemis {sorted(str(h) for h in hemis)}: "
            f"{sorted(img.path.name for img in images)}"
        )

    # The hemis get hstacked into one Y, so a variant mismatch (res, desc, ...)
    # would silently concatenate incompatible columns; require identical paths
    # modulo `hemi`. Equality holds only if entities share insertion order, which
    # they do here — both hemis come from one `find_bids_files` parse pass.
    if by_hemi["L"].without_entity("hemi") != by_hemi["R"].without_entity("hemi"):
        raise ValueError(
            f"Surface hemis must differ only in `hemi`; got "
            f"{by_hemi['L'].path.name} vs {by_hemi['R'].path.name}"
        )

    return (by_hemi["L"], by_hemi["R"])


class BoldMeta(NamedTuple):
    """Run-level BOLD metadata plus the image file(s) carrying its voxels.

    `source` is the run's voxel files: a lone volume `BIDSPath`, or the ordered
    `(hemi-L, hemi-R)` surface pair. TR, n_trs, and
    segments are run-level (hemi-invariant), resolved once from
    `representative_file()`; a voxel loader iterates `voxel_files()`.
    """

    source: VoxelSource
    repetition_time: float
    n_trs: int
    segments: list[Segment]

    def voxel_files(self) -> list[BIDSPath]:
        """Files in load order — `[volume]`, or `[L, R]`."""
        return [self.source] if isinstance(self.source, BIDSPath) else list(self.source)

    def representative_file(self) -> BIDSPath:
        """The run-identity file: the volume, or `hemi-L`.

        Carries the run's TR, n_trs, and segments; only voxel loading needs both
        hemis. Its entities are the run's filter axes except `hemi`, which is
        this file's own (`hemi-L`) and not run-level — filter on `run_entities`,
        not this file's raw entities.
        """
        return self.voxel_files()[0]

    def run_entities(self) -> dict[str, str]:
        """Run-level filter axes: the representative file's entities minus `hemi`.

        `hemi` is a per-file axis (this meta spans both hemis), so it is never a
        run-level filter target — see `representative_file`.
        """
        return {
            k: v for k, v in self.representative_file().entities.items() if k != "hemi"
        }


def load_bold_meta(layout: BIDSLayout, source: VoxelSource) -> BoldMeta:
    """Load TR, segments, and segment metadata for a BOLD run.

    `source` is the run's voxel files: a lone volume `BIDSPath`, or the ordered
    `(hemi-L, hemi-R)` surface pair. Run-level metadata
    is hemi-invariant, so it is read once from the representative (first) file.
    A surface pair is asserted to share the run's
    `n_trs` — an hstack over hemis with mismatched TR counts would misalign `Y`.

    Segment metadata comes from events.json `trial_type.Levels`: entries whose
    keys match the BIDS entity-value pattern are merged into `Segment.metadata`;
    other entries are ignored. Segments have empty metadata if events.json is
    absent.

    Sidecars (events.tsv, events.json) are resolved canonically from the raw
    BIDS tree via `layout.path.raw`; misnamed siblings are not inspected.

    When the representative file is a derivative, its volume count must match the
    raw BOLD — events.tsv onsets are raw-relative and hypline does not shift them.

    Raises ValueError if the representative file lacks a `task` entity (events
    sidecars are resolved by task), if a derivative representative file has a
    different volume count than its raw counterpart, if the two surface hemis
    disagree on `n_trs`, if events.tsv or events.json is invalid, if events.json
    declares segment entries that events.tsv does not, or if a `task` segment
    value disagrees with the filename's task entity.
    """
    files = [source] if isinstance(source, BIDSPath) else list(source)
    bids = files[0]
    _validate_bold(bids)

    if "task" not in bids.entities:
        raise ValueError(
            f"BOLD file {bids.path.name!r} missing required 'task' entity (BIDS)"
        )

    repetition_time = get_repetition_time(layout, bids)
    n_trs = get_n_trs(bids)

    # Hemis share a TR grid; an hstack on mismatched rows would misalign Y
    for img in files[1:]:
        if get_n_trs(img) != n_trs:
            raise ValueError(
                f"BOLD images for one run disagree on volume count: "
                f"{bids.path.name!r} has {n_trs}, {img.path.name!r} has "
                f"{get_n_trs(img)}"
            )

    # BOLD_EXTENSIONS pins volumetric raw to .nii.gz; surface derivatives
    # (.func.gii) inherit any trim applied during volumetric preprocessing.
    raw_bold = layout.path.raw(source=bids, suffix="bold", ext=".nii.gz")
    if raw_bold.path.exists() and raw_bold.path != bids.path:
        n_trs_raw = get_n_trs(raw_bold)
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
        source=source,
        repetition_time=repetition_time,
        n_trs=n_trs,
        segments=segments,
    )
