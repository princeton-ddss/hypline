# Segment metadata

Design record for the `events.json` `Segments` field — per-segment descriptive entities
used for filtering and CV splits.

## Purpose

Feature filenames carry only structural identity: `ses`, `run`, and the segment entity value
(e.g. `trial-1`). Descriptive attributes — condition, stimulus item, counterbalance group —
live in `events.json` under the `Segments` key and are joined onto `CellKey` at enrichment
time.

## Wire format (events.json)

```json
{
  "Segments": [
    {"trial": "1", "cond": "R", "item": "101"},
    {"trial": "2", "cond": "L", "item": "102"},
    {"trial": "3", "cond": "R", "item": "103"}
  ]
}
```

- `Segments` is a list of records; each record is a flat dict of BIDS entity key-value pairs.
- One key per record must match the run's segment entity (e.g. `"trial"`); its value is the
  bare segment value (e.g. `"1"`, not `"trial-1"`).
- Remaining keys are metadata entities — must match `BIDS_ENTITY_KEY_RE` (`^[a-z]+$`).
- Values must be strings matching `BIDS_ENTITY_VALUE_RE` (`^[a-zA-Z0-9]+$`). Non-string JSON
  types (numbers, bools) are rejected — BIDS values are strings on filenames, and metadata
  values become filename-shaped tokens after enrichment.
- `Segments` is optional. Absence = no metadata for that run; `CellKey` carries filename
  entities only.

## Validation

Within a single events.json (enforced in `bold.load_bold_meta`):
- Every record has the segment-entity key.
- Segment entity values match events.tsv key-value rows exactly (set equality).
- All records have identical metadata key sets (schema invariance).
- No metadata key collides with BOLD identity entities (`sub`, `ses`, `task`, `acq`, `ce`,
  `rec`, `dir`, `run`). Encoding-pipeline reserved keys (`space`, `feature`) are rejected
  later by `CellKey.EXCLUDE` during `_resolve_cell_keys` — keeping `bold.py` agnostic of
  encoding-pipeline concerns.

Cross-run (enforced in `Encoding._discover_bold`):
- All segmented runs share the same metadata key set. Strict — a segmented run with no
  events.json (empty metadata) does not match a segmented run with populated metadata.
  Mixing the two raises rather than silently routing partial metadata downstream.

## In-memory representation

```python
@dataclass(frozen=True)
class Segment:
    entity: str           # e.g. "trial"
    value: str            # bare value, e.g. "1"
    slice: slice          # TR-slice derived from events.tsv onset/duration
    metadata: dict[str, str]  # e.g. {"cond": "R", "item": "101"}; empty if no Segments

class BoldMeta(NamedTuple):
    bids: BIDSPath
    repetition_time: float
    segments: list[Segment]   # empty list if unsegmented run
```

Slices use bare values as keys (matching `Segment.value`) — the segment entity name is
available on `Segment.entity` and is not redundantly embedded in the value.

## CellKey enrichment (`_resolve_cell_keys`)

For each feature cell, `Segment.metadata` is merged into the cell's `CellKey`:

```python
enriched = CellKey(**{**dict(cell_key.items()), **segment.metadata})
```

Source-of-truth rule: events.json is authoritative for descriptive segment metadata. The
feature filename carries only `ses`, `run`, and the segment entity. All other entities
either belong to `CellKey.EXCLUDE` (invariant-across-training or image-variant entities,
rejected at `CellKey` construction) or must come from events.json. Any descriptive entity
on a feature filename that is absent from `seg.metadata` is rejected.

Four filename × sidecar cases:
- Sidecar-only (key in `seg.metadata`, absent from filename): merged onto enriched `CellKey`.
- Both, same value: allowed; redundant but harmless.
- Both, different value: raise — the two sources of truth disagree.
- Filename-only descriptive (non-identity, non-segment, non-reserved key on filename, absent
  from `seg.metadata`): raise, pointing user to events.json.

After enrichment, all `bids_filters` (including metadata-entity filters like `cond-R`) apply
uniformly against the enriched `CellKey`. No separate metadata-aware filter path needed.

## Example — full picture

events.tsv:
```
onset   duration  trial_type
0.0     10.0      rest
10.0    20.0      trial-1
30.0    10.0      rest
40.0    20.0      trial-2
60.0    10.0      rest
70.0    20.0      trial-3
90.0    10.0      rest
```

events.json Segments as above. BOLD TR=2.0s, 50 TRs total.

Resulting BoldMeta.segments:
```python
[
    Segment(entity="trial", value="1", slice=slice(5, 15),  metadata={"cond": "R", "item": "101"}),
    Segment(entity="trial", value="2", slice=slice(20, 30), metadata={"cond": "L", "item": "102"}),
    Segment(entity="trial", value="3", slice=slice(35, 45), metadata={"cond": "R", "item": "103"}),
]
```

TRs 0–4, 15–19, 30–34, 45–49 are break TRs — not indexed by any segment, excluded from X/Y.

See [semantic-entity.md](semantic-entity.md) for segment entity inference rules.
See [feature-files.md](feature-files.md) for feature file naming conventions.
