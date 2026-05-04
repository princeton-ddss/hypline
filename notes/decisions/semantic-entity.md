# Semantic segment entities

Design record for hypline's structural entity inference — how the segment entity is
identified from events.tsv, and the invariants it must satisfy.

## Motivation

Users think in terms of study design entities (`block`, `trial`). Hypline infers the segment
entity directly from events.tsv — no framework-specific naming required.

## events.tsv convention

Exactly one BIDS key-value entity name may appear across all `trial_type` rows (e.g. all
key-value rows use `trial-*`, never mixing `block-*` and `trial-*`). That entity is the
segment entity. Rows with more than one distinct BIDS entity name raise.

BIDS key-value rows declare segments — non-overlapping time windows within the run. Gaps
(pauses, breaks) before, between, and after segments are allowed; segments do not need to
tile the full run duration. Non-segment annotations must use flat (non-key-value) `trial_type`
labels (e.g. `rest`, `fixation`).

The segment entity name must not appear as a flat label in the same events.tsv (e.g. if
`trial` is the segment entity, `trial_type=trial` is forbidden — too easy to confuse with
`trial-1`).

This convention is layered on top of the BIDS spec, which treats `trial_type` as a free-text
label. Hypline restricts key-value format to segment declarations to keep structural intent
unambiguous.

## Definitions

- **Segment entity**: the single BIDS key-value entity name in events.tsv. Each value (e.g.
  `1`, `2`) identifies one time window. Feature files must carry this entity.
- **Unsegmented run**: a run with no events file or an events file with no BIDS key-value
  rows. The time window is the full run.

## Segment invariants

Every segment (BIDS key-value row in events.tsv) must satisfy:

1. `trial_type` matches `BIDS_ENTITY_RE` (`^[a-z]+-[a-zA-Z0-9]+$`)
2. Zero-duration rows are excluded
3. Segment values within a run are globally unique (no per-block-resetting IDs)
4. Slices are non-overlapping (gaps allowed anywhere — leading, trailing, between)
5. `max(slice.stop) ≤ BOLD array length` — validated in `_build_xy` after loading arrays

Full BOLD coverage is not required. Breaks consume TRs that are simply not indexed by any
segment slice and are excluded from X/Y.

## CellKey

`CellKey` is the open-schema row key for a feature time window. After enrichment it carries:
- Filename entities: `ses`, `run`, segment entity value (e.g. `trial=1`)
- Metadata entities from `events.json` `SegmentMetadata` (e.g. `cond=R`, `item=101`)

Excluded from `CellKey` (`CellKey.EXCLUDE`): `sub`, `task`, `acq`, `ce`, `rec`, `dir`
(invariant within a training call), `desc`, `res`, `den`, `echo` (BOLD image-variant
derivatives), `space`, `feature` (orthogonal axes). Entities are present or absent — no
`None` sentinel. Equality and hashing are order-independent.

CV splits are expressed by querying `CellKey` entities:
```python
train = [s for k, s in data.row_slices.items() if k["trial"] == "1"]
```

## Key invariants

- Feature files carry the segment entity — coarser or finer granularity is not supported.
- Exactly one BIDS key-value entity name per run — multiple segment-level entities (e.g.
  both `block-*` and `trial-*` rows) are a hard error. Use the finest granularity needed
  and express coarser groupings via `events.json` metadata.
- All BOLD runs in a training call must agree on the segment entity name, or all be
  unsegmented. Mixed levels are incoherent for a single encoding model.
- Unsegmented runs are valid — no key-value events required.
- For unsegmented runs, only `ses` and `run` are valid on feature filenames — any other
  entity raises. Descriptive attributes must live in `events.json`.

## Rationale

**Why key-value `trial_type` rather than side columns.** BIDS' idiomatic answer for
intra-run hierarchy is extra columns alongside `trial_type` (e.g. a `block` column on
each trial row). Hypline diverges because encoding requires explicit per-segment onset and
duration — deriving these from trial-level rows requires gap-filling heuristics that change
TR-to-segment assignment and therefore the science. Explicit segment rows make that judgment
author-controlled and reviewable.

**Why globally unique values.** Per-block-resetting trial IDs (`trial-1..trial-N` per block)
are permitted by BIDS and common in raw logs, but must be rewritten upstream before ingestion.

**Why single entity, not finest-granularity tiebreaker.** Multi-entity events.tsv under a
non-overlap-only rule would silently pick the wrong level if a descriptive entity happened to
be finer than the intended segment entity. Single-entity-only makes the intent unambiguous
without requiring tiling-strictness.

See [feature-files.md](feature-files.md) for feature file naming conventions.
See [segment-metadata.md](segment-metadata.md) for the events.json `SegmentMetadata` format.
See [../external/bids.md](../external/bids.md) for the BIDS entity reserved name list.
