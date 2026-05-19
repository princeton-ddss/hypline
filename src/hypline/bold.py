import json
from dataclasses import dataclass, replace
from typing import NamedTuple

import polars as pl

from hypline.bids import (
    BIDS_ENTITY_KEY_RE,
    BIDS_ENTITY_RE,
    BIDS_ENTITY_VALUE_RE,
    RAW_BOLD_ENTITIES,
    BIDSPath,
)
from hypline.enums import SurfaceSpace, VolumeSpace
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
    """Extract the repetition time (TR) in seconds for a BOLD run.

    Reads TR from the raw `*_bold.json` sidecar (resolved under raw BIDS via
    `layout.path.raw`, regardless of whether `bids` itself points into raw or
    fmriprep). Falls back to the NIfTI/GIfTI header when the sidecar is
    absent, but only if `bids` is the BOLD image itself. Raises ValueError if
    no source yields a usable TR or the image format is unsupported.
    """
    sidecar = layout.path.raw(source=bids, suffix="bold", ext=".json")
    if sidecar.path.exists():
        with open(sidecar.path) as f:
            TR = json.load(f).get("RepetitionTime")
        if TR is not None:
            return float(TR)

    if bids.suffix != "bold":
        raise ValueError(
            f"No RepetitionTime in raw sidecar for {bids.path.name}, and "
            f"header fallback requires a BOLD image (got suffix {bids.suffix!r})"
        )

    import nibabel as nib

    img = nib.load(bids.path)
    if isinstance(img, nib.Nifti1Image):
        TR = img.header.get_zooms()[3]
        if TR > 0:
            return float(TR)
        raise ValueError(f"TR is zero or unset in NIfTI header: {bids.path.name}")
    if isinstance(img, nib.GiftiImage):
        time_step = img.darrays[0].meta.get("TimeStep")
        if time_step is not None:
            return float(time_step) / 1000  # milliseconds to seconds
        raise ValueError(f"TimeStep missing from GIfTI metadata: {bids.path.name}")
    raise ValueError(f"Unsupported image format: {type(img).__name__}")


@dataclass(frozen=True)
class Segment:
    entity: str
    value: str
    tr_slice: slice
    metadata: dict[str, str]


class BoldMeta(NamedTuple):
    bids: BIDSPath
    repetition_time: float
    segments: list[Segment]


def _parse_segments(
    events: pl.DataFrame | None,
    repetition_time: float,
) -> list[Segment]:
    """Parse BIDS key-value `trial_type` rows into segments.

    Rows matching `entity-value` with duration > 0 become segments. Flat
    labels (e.g. `rest`) are silently ignored. Returns [] for unsegmented runs.

    `task` is the only identity entity allowed as a segment entity, and only
    with a single segment row. Matching the segment value against the
    filename's task is the caller's responsibility.

    Raises ValueError if:
    - More than one distinct entity name is found across matching rows.
    - The segment entity is an identity entity other than `task`, or is
      `task` with more than one segment row.
    - The segment entity name also appears as a flat label in the same file.
    - Any segment value appears more than once.
    - Any two segment slices overlap.
    """
    if events is None:
        return []

    segment_rows = events.filter(
        pl.col("trial_type").str.contains(BIDS_ENTITY_RE.pattern)
        & (pl.col("duration") > 0.0)
    )

    if segment_rows.is_empty():
        return []

    segments: list[Segment] = []
    for row in segment_rows.iter_rows(named=True):
        entity, value = row["trial_type"].split("-", 1)
        onset_tr = round(row["onset"] / repetition_time)
        n_trs = round(row["duration"] / repetition_time)
        segments.append(
            Segment(
                entity=entity,
                value=value,
                tr_slice=slice(onset_tr, onset_tr + n_trs),
                metadata={},
            )
        )

    entity_names = {seg.entity for seg in segments}
    if len(entity_names) > 1:
        raise ValueError(
            f"events.tsv contains multiple BIDS key-value entities "
            f"{sorted(entity_names)} — only one segment entity allowed per run"
        )
    segment_entity = next(iter(entity_names))

    if segment_entity in RAW_BOLD_ENTITIES:
        if segment_entity != "task":
            raise ValueError(
                f"segment entity {segment_entity!r} is not allowed; "
                f"only 'task' is permitted among raw BOLD entities"
            )
        if len(segments) > 1:
            raise ValueError(
                f"segment entity 'task' allows at most one segment per run "
                f"(found {len(segments)})"
            )

    flat_labels = set(
        events.filter(~pl.col("trial_type").str.contains(BIDS_ENTITY_RE.pattern))[
            "trial_type"
        ].to_list()
    )
    if segment_entity in flat_labels:
        raise ValueError(
            f"Segment entity {segment_entity!r} also appears as a flat label in "
            f"events.tsv — rename either the entity or the flat label"
        )

    values_seen: set[str] = set()
    for seg in segments:
        if seg.value in values_seen:
            raise ValueError(
                f"Segment value {segment_entity}-{seg.value} appears more than once in "
                f"events.tsv — segment values must be unique within a run"
            )
        values_seen.add(seg.value)

    segments = sorted(segments, key=lambda seg: seg.tr_slice.start)

    for prev, curr in zip(segments, segments[1:]):
        if prev.tr_slice.stop > curr.tr_slice.start:
            raise ValueError(
                f"Segment slices overlap: {segment_entity}-{prev.value} "
                f"[{prev.tr_slice.start}:{prev.tr_slice.stop}] and "
                f"{segment_entity}-{curr.value} "
                f"[{curr.tr_slice.start}:{curr.tr_slice.stop}]"
            )

    return segments


def _validate_segment_records(
    segment_records: dict[str, dict],
    segments: list[Segment],
) -> None:
    """Validate events.json segment records against events.tsv segments.

    segment_records maps `entity-value` keys to Levels entries, each requiring
    a `metadata` dict. All records must share one entity, one metadata schema,
    and BIDS-valid keys/values that don't collide with BOLD identity entities.
    """
    expected_keys = {f"{seg.entity}-{seg.value}" for seg in segments}
    if set(segment_records) != expected_keys:
        missing = sorted(expected_keys - set(segment_records))
        extra = sorted(set(segment_records) - expected_keys)
        raise ValueError(
            f"events.json Levels keys do not match events.tsv segments "
            f"(missing: {missing}, extra: {extra})"
        )

    for key, record in segment_records.items():
        if "metadata" not in record:
            raise ValueError(
                f"events.json Levels entry {key!r} has no 'metadata' field"
            )

    schemas = [frozenset(record["metadata"]) for record in segment_records.values()]
    if len(set(schemas)) > 1:
        raise ValueError(
            "events.json segment metadata schemas differ across Levels — "
            "all entries must share the same keys"
        )
    schema = schemas[0]

    for field in schema:
        if not BIDS_ENTITY_KEY_RE.match(field):
            raise ValueError(
                f"events.json metadata key {field!r} must match "
                f"{BIDS_ENTITY_KEY_RE.pattern}"
            )
        if field in RAW_BOLD_ENTITIES:
            raise ValueError(
                f"events.json metadata key {field!r} collides with a raw "
                f"BOLD entity {sorted(RAW_BOLD_ENTITIES)}"
            )

    for key, record in segment_records.items():
        for field in schema:
            value = record["metadata"][field]
            if not isinstance(value, str) or not BIDS_ENTITY_VALUE_RE.match(value):
                raise ValueError(
                    f"events.json metadata {key}.{field} = {value!r} "
                    f"must be a string matching {BIDS_ENTITY_VALUE_RE.pattern}"
                )


def load_bold_meta(layout: BIDSLayout, bids: BIDSPath) -> BoldMeta:
    """Load TR, segments, and segment metadata for a BOLD run.

    Segment metadata comes from events.json `trial_type.Levels`: entries whose
    keys match the BIDS entity-value pattern are merged into `Segment.metadata`;
    other entries are ignored. Segments have empty metadata if events.json is
    absent.

    Sidecars (events.tsv, events.json) are resolved canonically from the raw
    BIDS tree via `layout.path.raw`; misnamed siblings are not inspected.

    Raises ValueError if `bids` lacks a `task` entity (events sidecars are
    resolved by task), if events.tsv or events.json is invalid, if events.json
    declares segment entries that events.tsv does not, or if a `task` segment
    value disagrees with the filename's task entity.
    """
    _validate_bold(bids)

    if "task" not in bids.entities:
        raise ValueError(
            f"BOLD file {bids.path.name!r} missing required 'task' entity (BIDS)"
        )

    repetition_time = get_repetition_time(layout, bids)

    events_bids = layout.path.raw(source=bids, suffix="events", ext=".tsv")
    events = (
        pl.read_csv(events_bids.path, separator="\t")
        if events_bids.path.exists()
        else None
    )

    events_meta_bids = layout.path.raw(source=bids, suffix="events", ext=".json")
    if events_meta_bids.path.exists():
        with open(events_meta_bids.path) as f:
            events_meta = json.load(f)
    else:
        events_meta = None

    segments = _parse_segments(events, repetition_time)

    if segments and segments[0].entity == "task":
        task_value = bids.entities.get("task")
        if segments[0].value != task_value:
            raise ValueError(
                f"events.tsv 'task-{segments[0].value}' does not match "
                f"filename 'task-{task_value}'"
            )

    levels: dict[str, dict] = (
        events_meta.get("trial_type", {}).get("Levels", {}) if events_meta else {}
    )
    segment_records: dict[str, dict] = {
        key: record for key, record in levels.items() if BIDS_ENTITY_RE.match(key)
    }

    if segment_records:
        if not segments:
            raise ValueError(
                "events.json declares segment entries in Levels but events.tsv "
                "has no segment rows — add rows to events.tsv or remove the "
                "entries from Levels"
            )
        _validate_segment_records(segment_records, segments)
        segments = [
            replace(
                seg,
                metadata=segment_records[f"{seg.entity}-{seg.value}"]["metadata"],
            )
            for seg in segments
        ]

    return BoldMeta(bids=bids, repetition_time=repetition_time, segments=segments)
