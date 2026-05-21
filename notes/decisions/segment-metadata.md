# Segment metadata

Design record for per-segment descriptive entities in `events.json` — used for filtering and CV splits.

## Purpose

Feature filenames carry only structural identity: `ses`, `run`, and the segment entity value
(e.g. `trial-1`). Descriptive attributes — condition, stimulus item, counterbalance group —
live in `events.json` under `trial_type.Levels` and are joined onto `CellKey` at enrichment
time.

## Wire format (events.json)

```json
{
  "trial_type": {
    "Levels": {
      "trial-1": {"metadata": {"cond": "R", "item": "101"}},
      "trial-2": {"metadata": {"cond": "L", "item": "102"}},
      "trial-3": {"metadata": {"cond": "R", "item": "103"}}
    }
  }
}
```

- Segment metadata lives in `trial_type.Levels` — a BIDS-standard field repurposed for this use.
- Keys are `entity-value` strings (e.g. `"trial-1"`); only keys matching the BIDS entity-value
  pattern (`BIDS_ENTITY_RE`) are treated as segment entries — all others are silently ignored,
  allowing coexistence with standard BIDS Levels annotations (e.g. `"rest"`, `"n/a"`).
- Each entry must have a `"metadata"` sub-dict. Entries lacking `"metadata"` raise `ValueError`.
- Metadata keys must match `BIDS_ENTITY_KEY_RE` (`^[a-z]+$`); values must be strings matching
  `BIDS_ENTITY_VALUE_RE` (`^[a-zA-Z0-9]+$`) — values become filename-shaped tokens after enrichment.
- `trial_type.Levels` is optional. Absence (or no entity-keyed entries) = no metadata for that run;
  `CellKey` carries filename entities only.

## Validation

Within a single events.json (enforced in `events.load_segments`, which
`bold.load_bold_meta` delegates to):
- Entity-keyed Levels entries match events.tsv segment `entity-value` keys exactly (set equality).
- All entries share identical metadata key sets (schema invariance).
- No metadata key collides with a raw BOLD entity. Encoding-pipeline reserved keys
  are rejected later during feature discovery — keeping the events module
  agnostic of encoding-pipeline concerns.

## Segmentation cases

The distinction is whether a slice contract exists, not how many segments there are.
A run with one events.tsv row is **segmented** (segment count = 1), not unsegmented.

- **Unsegmented** = no events.tsv, or events.tsv with no BIDS key-value rows. Whole run,
  no metadata. Filenames carry `ses`/`run` only.
- **Structurally segmented** = one or more events.tsv rows naming real internal units (blocks,
  trials). Use `block-N`/`trial-N`. Filenames carry the segment entity.
- **`task-<value>` escape hatch** = one events.tsv row when the run has no genuine internal
  structure but the user still needs either (a) a slice (e.g. trim instruction/fixation padding,
  so onset≠0 is expected) or (b) run-level metadata for filtering. Reuses `task` (which BOLD
  files always carry per BIDS) rather than inventing a structural entity like `block-1` that
  would falsely advertise multiple blocks. Value must match the filename's `task`.

Cross-run (enforced in `Encoding._discover_bold`):
- All segmented runs share the same metadata key set. Strict — a segmented run with no
  events.json (empty metadata) does not match a segmented run with populated metadata.
  Mixing the two raises rather than silently routing partial metadata downstream.

## CellKey enrichment (`_resolve_cell_keys`)

`Segment.metadata` is merged into each feature cell's `CellKey` at enrichment time. After
enrichment, all `bids_filters` (including metadata-entity filters like `cond-R`) apply
uniformly against the enriched `CellKey`. No separate metadata-aware filter path needed.

See [feature-files.md](feature-files.md) for the four filename × sidecar merge cases,
`CellKey` exclusion rules, and the one-file-per-segment requirement.
See [semantic-entity.md](semantic-entity.md) for segment entity inference rules.
