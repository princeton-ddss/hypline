import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import polars as pl

from hypline.bids import (
    BIDS_ENTITY_KEY_RE,
    BIDS_ENTITY_RE,
    BIDS_ENTITY_VALUE_RE,
    RESERVED_BIDS_ENTITIES,
    STRUCTURAL_ENTITIES,
    BIDSPath,
)

if TYPE_CHECKING:
    from hypline.layout import BIDSLayout


@dataclass(frozen=True)
class Segment:
    entity: str
    value: str
    onset: float
    duration: float
    metadata: dict[str, str]


def segment_tr_slice(segment: Segment, repetition_time: float) -> slice:
    """Convert a segment's seconds-based span to TR indices.

    Preserves the left-inclusive `[t*TR, (t+1)*TR)` convention: an event at
    `onset=0.0` lands in TR 0, and an event at `onset=k*TR` lands in TR `k`.
    `segment.onset` is assumed already aligned to the target image (callers
    must shift for any dummy-scan trim before calling).
    """
    onset_tr = round(segment.onset / repetition_time)
    n_trs = round(segment.duration / repetition_time)
    return slice(onset_tr, onset_tr + n_trs)


def _parse_segments(events: pl.DataFrame | None) -> list[Segment]:
    """Parse BIDS key-value `trial_type` rows into segments.

    Rows matching `entity-value` with duration > 0 become segments; flat
    labels are silently ignored. `task` is the only identity entity allowed,
    and only with a single row; matching its value against the filename's
    task is the caller's responsibility. Returns [] for unsegmented runs.
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
        segments.append(
            Segment(
                entity=entity,
                value=value,
                onset=float(row["onset"]),
                duration=float(row["duration"]),
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

    if segment_entity in RESERVED_BIDS_ENTITIES:
        if segment_entity != "task":
            raise ValueError(
                f"segment entity {segment_entity!r} is not allowed; "
                f"only 'task' is permitted among BIDS-reserved entities"
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

    segments = sorted(segments, key=lambda seg: seg.onset)

    for prev, curr in zip(segments, segments[1:]):
        if prev.onset + prev.duration > curr.onset:
            raise ValueError(
                f"Segments overlap: {segment_entity}-{prev.value} "
                f"[{prev.onset}, {prev.onset + prev.duration}) and "
                f"{segment_entity}-{curr.value} "
                f"[{curr.onset}, {curr.onset + curr.duration})"
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
        if field in RESERVED_BIDS_ENTITIES:
            raise ValueError(
                f"events.json metadata key {field!r} collides with a "
                f"BIDS-reserved entity {sorted(RESERVED_BIDS_ENTITIES)}"
            )

    for key, record in segment_records.items():
        for field in schema:
            value = record["metadata"][field]
            if not isinstance(value, str) or not BIDS_ENTITY_VALUE_RE.match(value):
                raise ValueError(
                    f"events.json metadata {key}.{field} = {value!r} "
                    f"must be a string matching {BIDS_ENTITY_VALUE_RE.pattern}"
                )


def load_segments(layout: "BIDSLayout", source: BIDSPath) -> list[Segment]:
    """Load segments + Levels metadata for the run identified by `source`.

    No BOLD file required. `source` may be any BIDSPath carrying
    (sub, ses, task, run) — stimulus, feature, confound, or BOLD. Sidecars
    (events.tsv, events.json) are resolved canonically from the raw BIDS
    tree via `layout.path.raw`; misnamed siblings are not inspected.

    Returns [] when events.tsv is absent or has no segment rows.

    Raises ValueError if `source` lacks a `task` entity (events sidecars
    are resolved by task), if events.tsv or events.json is invalid, or if
    events.json declares segment entries that events.tsv does not.
    """
    if "task" not in source.entities:
        raise ValueError(
            f"Source path {source.path.name!r} missing required 'task' entity "
            f"(events sidecars are resolved by task)"
        )

    events_bids = layout.path.raw(source=source, suffix="events", ext=".tsv")
    events = (
        pl.read_csv(events_bids.path, separator="\t")
        if events_bids.path.exists()
        else None
    )

    events_meta_bids = layout.path.raw(source=source, suffix="events", ext=".json")
    if events_meta_bids.path.exists():
        with open(events_meta_bids.path) as f:
            events_meta = json.load(f)
    else:
        events_meta = None

    segments = _parse_segments(events)

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

    return segments


def merge_filename_and_sidecar(
    *,
    filename_entities: Mapping[str, str],
    sidecar_metadata: Mapping[str, str],
    structural_keys: frozenset[str],
) -> dict[str, str]:
    """Merge filename entities with sidecar metadata under the four-case contract.

    Cases:
      - sidecar-only (key in sidecar, absent from filename) → adopt sidecar value
      - both-same → keep one
      - both-differ → raise
      - filename-only descriptive (key on filename, absent from sidecar, not in
        `structural_keys`) → raise

    `structural_keys` are filename entities that may appear without a sidecar
    counterpart (e.g. {"ses", "run", segment_entity}).

    Precondition: sidecar keys are disjoint from raw BOLD identity entities
    (enforced upstream in events.json validation).
    """
    descriptive_filename_keys = (
        set(filename_entities) - structural_keys - set(sidecar_metadata)
    )
    if descriptive_filename_keys:
        raise ValueError(
            f"Filename carries entities {sorted(descriptive_filename_keys)} "
            f"absent from events.json Levels metadata — descriptive attributes "
            f"must live in events.json, not filenames"
        )

    merged: dict[str, str] = dict(filename_entities)
    for key, sidecar_value in sidecar_metadata.items():
        filename_value = filename_entities.get(key)
        if filename_value is None:
            merged[key] = sidecar_value
        elif filename_value != sidecar_value:
            raise ValueError(
                f"Filename and events.json disagree on {key!r}: "
                f"filename has {filename_value!r}, sidecar has {sidecar_value!r}"
            )
    return merged


def resolve_entities(layout: "BIDSLayout", source: BIDSPath) -> dict[str, str]:
    """Resolve a path's full entity set by merging filename with events.json.

    Locates the run's events sidecars via `(sub, ses, task, run)`, finds the
    segment matching the filename's segment-entity value, and merges its
    `metadata` onto the filename entities under the four-case contract
    (see `merge_filename_and_sidecar`).

    For unsegmented runs, only `STRUCTURAL_ENTITIES` are allowed on the filename.

    Raises ValueError if the filename lacks the segment entity declared in
    events.tsv, names a segment value not present in events.tsv, or if
    `merge_filename_and_sidecar` rejects the entity set.
    """
    filename_entities = dict(source.entities)
    segments = load_segments(layout, source)
    if not segments:
        return merge_filename_and_sidecar(
            filename_entities=filename_entities,
            sidecar_metadata={},
            structural_keys=STRUCTURAL_ENTITIES,
        )

    segment_entity = segments[0].entity
    segment_value = filename_entities.get(segment_entity)
    if segment_value is None:
        raise ValueError(
            f"Path is missing segment entity {segment_entity!r} declared in events.tsv"
        )

    matching = [s for s in segments if s.value == segment_value]
    if not matching:
        valid = sorted(s.value for s in segments)
        raise ValueError(
            f"Segment value {segment_entity}-{segment_value} not "
            f"found in events.tsv — valid values: {valid}"
        )

    return merge_filename_and_sidecar(
        filename_entities=filename_entities,
        sidecar_metadata=matching[0].metadata,
        structural_keys=STRUCTURAL_ENTITIES | {segment_entity},
    )
