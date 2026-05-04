# Segment metadata

Design record for the `events.json` `SegmentMetadata` field — per-segment descriptive entities
used for filtering and CV splits.

## Purpose

Feature filenames carry only structural identity: `ses`, `run`, and the segment entity value
(e.g. `trial-1`). Descriptive attributes — condition, stimulus item, counterbalance group —
live in `events.json` under the `SegmentMetadata` key and are joined onto `CellKey` at enrichment
time.

## Wire format (events.json)

```json
{
  "SegmentMetadata": [
    {"trial": "1", "cond": "R", "item": "101"},
    {"trial": "2", "cond": "L", "item": "102"},
    {"trial": "3", "cond": "R", "item": "103"}
  ]
}
```

- `SegmentMetadata` is a list of records; each record is a flat dict of BIDS entity key-value pairs.
- One key per record must match the run's segment entity (e.g. `"trial"`); its value is the
  bare segment value (e.g. `"1"`, not `"trial-1"`).
- Remaining keys are metadata entities — must match `BIDS_ENTITY_KEY_RE` (`^[a-z]+$`).
- Values must be strings matching `BIDS_ENTITY_VALUE_RE` (`^[a-zA-Z0-9]+$`). Non-string JSON
  types (numbers, bools) are rejected — BIDS values are strings on filenames, and metadata
  values become filename-shaped tokens after enrichment.
- `SegmentMetadata` is optional. Absence = no metadata for that run; `CellKey` carries filename
  entities only.

## Validation

Within a single events.json (enforced in `bold.load_bold_meta`):
- Every record has the segment-entity key.
- Segment entity values match events.tsv key-value rows exactly (set equality).
- All records have identical metadata key sets (schema invariance).
- No metadata key collides with BOLD identity entities (`sub`, `ses`, `task`, `acq`, `ce`,
  `rec`, `dir`, `run`). Encoding-pipeline reserved keys (`space`, `feature`) are rejected
  by `CellKey.EXCLUDE` at `CellKey.__init__` during `_discover_features` — keeping
  `bold.py` agnostic of encoding-pipeline concerns.

## Single-segment runs

A run with one segment row in events.tsv is **segmented** (segment count = 1), not unsegmented.
Feature filenames must carry the segment entity (e.g. `block-1`), same as multi-segment runs.

- **Unsegmented** = no events.tsv rows = use the entire BOLD run. Feature filenames carry `ses`/`run` only.
- **Single-segment** = one events.tsv row = slice the run by that segment. Feature filenames must identify the segment.

The distinction is whether a slice contract exists, not how many segments there are. Pre/post
run padding (instructions, fixation, scanner ramp-up) is near-universal in practice, so the
"whole-run-as-one-segment" case (onset=0, duration=full_run) is rare but permitted — declare
one explicit row in events.tsv covering the full duration.

Cross-run (enforced in `Encoding._discover_bold`):
- All segmented runs share the same metadata key set. Strict — a segmented run with no
  events.json (empty metadata) does not match a segmented run with populated metadata.
  Mixing the two raises rather than silently routing partial metadata downstream.

## CellKey enrichment (`_resolve_cell_keys`)

`Segment.metadata` is merged into each feature cell's `CellKey` at enrichment time. After
enrichment, all `bids_filters` (including metadata-entity filters like `cond-R`) apply
uniformly against the enriched `CellKey`. No separate metadata-aware filter path needed.

See [feature-files.md](feature-files.md) for the four filename × sidecar merge cases and
`CellKey` exclusion rules.
See [semantic-entity.md](semantic-entity.md) for segment entity inference rules.
