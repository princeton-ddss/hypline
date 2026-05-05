import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import NamedTuple

import polars as pl

from hypline.bids import (
    BIDS_ENTITY_KEY_RE,
    BIDS_ENTITY_RE,
    BIDS_ENTITY_VALUE_RE,
    BIDSPath,
)
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
    """Extract the repetition time (TR) in seconds from a BOLD file.

    Reads TR from the BIDS JSON sidecar if available, otherwise falls back to
    the NIfTI header for volume data or GIfTI darray metadata for surface data.
    Raises ValueError if TR is zero/unset in the NIfTI header, missing from
    GIfTI metadata, or the image format is unsupported.
    """
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


@dataclass(frozen=True)
class Segment:
    entity: str
    value: str
    slice: slice
    metadata: dict[str, str]


class BoldMeta(NamedTuple):
    bids: BIDSPath
    repetition_time: float
    segments: list[Segment]


def _resolve_run_sidecar(
    bold_path: str | os.PathLike[str],
    suffix: str,
    extension: str,
) -> Path:
    """Return the canonical run-level sidecar path for a BOLD file.

    Run-level sidecars (events.tsv, events.json, physio.tsv.gz, etc.) are named
    using the run's identity entities only — they describe the source data, not
    a specific image variant, and are invariant across `space`, `desc`, etc.
    Distinct from file-level sidecars like `*_bold.json` (which mirror the full
    image stem and are resolved separately by suffix swap).

    Returns the canonical path; may not exist (callers check). Raises if a file
    in the same directory matches identity entities and the same suffix/extension
    but is not the canonical name.
    """
    bids = BIDSPath(bold_path)
    shared = {k: v for k, v in bids.entities.items() if k in _BOLD_IDENTITY_ENTITIES}
    stem = "_".join(f"{k}-{v}" for k, v in shared.items())
    canonical = bids.path.parent / f"{stem}_{suffix}{extension}"

    misnamed = [
        p
        for p in bids.path.parent.glob(f"*_{suffix}{extension}")
        if p != canonical
        and all(BIDSPath(p).entities.get(k) == v for k, v in shared.items())
    ]
    if misnamed:
        raise ValueError(
            f"Expected {canonical.name!r} but found unexpected {suffix}{extension} "
            f"file(s) colocated with this BOLD run: "
            f"{[p.name for p in sorted(misnamed)]}. "
            "Rename to use identity entities only in canonical BIDS order."
        )

    return canonical


def load_events_tsv(bold_path: str | os.PathLike[str]) -> pl.DataFrame | None:
    """Load events TSV colocated with a BOLD file, or return None if absent.

    Raises ValueError if a misnamed sibling sharing this run's identity entities
    is found — same run implies same stimulus timeline, so variants are either
    redundant or divergent and both cases warrant surfacing.
    """
    events_tsv_file = _resolve_run_sidecar(bold_path, "events", ".tsv")
    if events_tsv_file.exists():
        return pl.read_csv(events_tsv_file, separator="\t")
    return None


def load_events_json(bold_path: str | os.PathLike[str]) -> dict | None:
    """Load events JSON sidecar colocated with a BOLD file, or return None if absent.

    Raises ValueError if a misnamed sibling sharing this run's identity entities
    is found — same run implies same stimulus timeline, so variants are either
    redundant or divergent and both cases warrant surfacing.
    """
    events_json_file = _resolve_run_sidecar(bold_path, "events", ".json")
    if events_json_file.exists():
        with open(events_json_file) as f:
            return json.load(f)
    return None


def _parse_segments(
    events: pl.DataFrame | None,
    repetition_time: float,
) -> list[Segment]:
    """Parse BIDS key-value `trial_type` rows into segments.

    Rows matching `entity-value` with duration > 0 become segments. Flat
    labels (e.g. `rest`) are silently ignored. Returns [] for unsegmented runs.

    Raises ValueError if:
    - More than one distinct entity name is found across matching rows.
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
                slice=slice(onset_tr, onset_tr + n_trs),
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

    segments = sorted(segments, key=lambda seg: seg.slice.start)

    for prev, curr in zip(segments, segments[1:]):
        if prev.slice.stop > curr.slice.start:
            raise ValueError(
                f"Segment slices overlap: {segment_entity}-{prev.value} "
                f"[{prev.slice.start}:{prev.slice.stop}] and "
                f"{segment_entity}-{curr.value} "
                f"[{curr.slice.start}:{curr.slice.stop}]"
            )

    return segments


def _validate_segment_records(
    segment_records: list[dict],
    segments: list[Segment],
) -> None:
    """Validate `SegmentMetadata` records in events.json against segments in events.tsv.

    Each record describes one segment and carries the segment-entity key plus
    arbitrary metadata keys. Raises ValueError on missing segment-entity key,
    value-set mismatch with events.tsv, inconsistent metadata schemas across
    records, malformed metadata keys, or collisions with BOLD identity entities.
    """
    segment_entity = segments[0].entity
    expected_values = {seg.value for seg in segments}

    for record in segment_records:
        if segment_entity not in record:
            raise ValueError(
                f"events.json SegmentMetadata record is missing segment entity key "
                f"{segment_entity!r}: {record}"
            )

    record_values = {record[segment_entity] for record in segment_records}
    if record_values != expected_values:
        missing = sorted(expected_values - record_values)
        extra = sorted(record_values - expected_values)
        raise ValueError(
            f"events.json SegmentMetadata values do not match events.tsv. "
            f"Missing: {missing}, extra: {extra}"
        )

    metadata_keys_per_record = [
        frozenset(k for k in record if k != segment_entity)
        for record in segment_records
    ]
    if len(set(metadata_keys_per_record)) > 1:
        raise ValueError(
            "events.json SegmentMetadata records have inconsistent metadata keys — "
            "all records must carry the same set of metadata keys"
        )
    metadata_keys = metadata_keys_per_record[0]

    for metadata_key in metadata_keys:
        if not BIDS_ENTITY_KEY_RE.match(metadata_key):
            raise ValueError(
                f"events.json metadata key {metadata_key!r} is invalid — "
                f"keys must match {BIDS_ENTITY_KEY_RE.pattern}"
            )
        if metadata_key in _BOLD_IDENTITY_ENTITIES:
            raise ValueError(
                f"events.json metadata key {metadata_key!r} collides with BOLD "
                f"identity entity — reserved keys: "
                f"{sorted(_BOLD_IDENTITY_ENTITIES)}"
            )

    for record in segment_records:
        for metadata_key in metadata_keys:
            metadata_value = record[metadata_key]
            if not isinstance(metadata_value, str) or not BIDS_ENTITY_VALUE_RE.match(
                metadata_value
            ):
                raise ValueError(
                    f"events.json metadata value {metadata_value!r} for key "
                    f"{metadata_key!r} is invalid — values must be strings "
                    f"matching {BIDS_ENTITY_VALUE_RE.pattern}"
                )


def load_bold_meta(bold_path: str | os.PathLike[str]) -> BoldMeta:
    """Load all metadata for a BOLD run: TR, segments, and segment metadata.

    Segment metadata is sourced from the colocated events.json `SegmentMetadata` field
    and merged into each `Segment.metadata`. If no events.json is present,
    segments have empty metadata dicts.

    Raises ValueError if events.tsv or events.json contents are invalid, or if
    an events.json is present but no segments were parsed from events.tsv.
    """
    bold_path = Path(bold_path)
    repetition_time = get_repetition_time(bold_path)
    events = load_events_tsv(bold_path)
    events_json = load_events_json(bold_path)

    segments = _parse_segments(events, repetition_time)
    segment_records: list[dict] = (
        events_json.get("SegmentMetadata", []) if events_json else []
    )
    if segment_records:
        if not segments:
            raise ValueError(
                "events.json has a SegmentMetadata field but events.tsv "
                "has no BIDS key-value rows — add segment rows to events.tsv "
                "or remove SegmentMetadata from events.json"
            )
        _validate_segment_records(segment_records, segments)
        segment_entity = segments[0].entity
        metadata_by_segment_value = {
            record[segment_entity]: {
                k: v for k, v in record.items() if k != segment_entity
            }
            for record in segment_records
        }
        segments = [
            replace(seg, metadata=metadata_by_segment_value[seg.value])
            for seg in segments
        ]

    return BoldMeta(
        bids=BIDSPath(bold_path),
        repetition_time=repetition_time,
        segments=segments,
    )
