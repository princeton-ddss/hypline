# Semantic partition entities

Design record for hypline's structural entity inference — how the partition entity is
identified from events.tsv, and the invariants it must satisfy.

## Motivation

Users think in terms of study design entities (`block`, `trial`). Hypline infers the partition
entity directly from the tiling structure of events.tsv — no framework-specific naming required.

## events.tsv convention

BIDS key-value `trial_type` rows (e.g. `block-1`, `trial-A`) declare partitions — they are
treated as structural intent. Every such entity must tile the run; partial tiling is a hard
error. Non-partition annotations must use flat (non-key-value) `trial_type` labels (e.g.
`rest`, `fixation`).

This convention is layered on top of the BIDS spec, which treats `trial_type` as a free-text
label. Hypline restricts key-value format to partition declarations to keep structural intent
unambiguous.

Friction case: descriptive labels that happen to match key-value format (e.g. `cond-A` with no
tiling intent) will be rejected as a partial-tiling error. Rename to a flat label (`condA`) if
descriptive, or add rows to make it tile if structural.

## Definitions

- **Partition entity**: the BIDS key-value entity in events.tsv whose values tile the run and
  have the smallest average slice duration. Feature files are always provided at this
  granularity. There must be exactly one per run — ties are a hard error.
- **Unpartitioned run**: a run with no events file or an events file with no BIDS key-value
  rows. The time window is the full run.

## Tiling invariants

Every BIDS key-value entity in events.tsv must satisfy all of the following:

1. All rows use valid BIDS `key-value` format (`BIDS_ENTITY_RE`: `^[a-z]+-[a-zA-Z0-9]+$`)
2. Zero-duration rows are excluded
3. Slices start at TR 0 (onset of first value = 0)
4. Slices are non-overlapping and contiguous (no gaps between consecutive slices)
5. Max slice endpoint equals the events span (`max(onset + duration)` across all valid rows)

Condition 5 uses the events file's own span as the reference — not the BOLD array length.
Full BOLD coverage is validated separately in `_build_xy` once arrays are loaded.

Partial tiling — where some key-value entities tile and others do not — is a hard error.
Use flat `trial_type` labels for any annotation that is not a partition.

## CellKey

`CellKey` is the open-schema row key for a feature time window. It stores all
BIDS entities from a feature filename except those in `CellKey.EXCLUDE`:
`sub` and `task` (invariant within a training call), `feature` (column axis,
not row axis), and `space` (BOLD-only). Entities are present or absent — there
is no `None` sentinel. Equality and hashing are order-independent.

CV splits are expressed by querying `CellKey` entities:
```python
train = [s for k, s in data.row_slices.items() if k["block"] == "1"]
```

## Key invariants

- Feature files are provided at the partition entity granularity — coarser or finer is not
  supported. If a user wants `block`-level features, `block` must be the partition entity.
- Multiple key-value entities in one run are allowed (e.g. `block` and `trial`); all must
  tile, and the finest (smallest average duration) is selected as the partition entity. Equal
  granularity is a degenerate design and raises.
- All BOLD runs in a training call must agree on the partition entity name, or all be
  unpartitioned. Mixed levels are incoherent for a single encoding model — TR-slice semantics
  differ across rows, making regularisation and CV splits incomparable.
- Unpartitioned runs are valid — no key-value events required.
- When all BOLD runs are unpartitioned, extra entities on feature filenames beyond `ses`/`run`
  are accepted as descriptive tags (e.g. `cond-A` for grouping or filtering). If those entities
  were intended as partition keys but events.tsv is absent or contains no BIDS key-value rows,
  the misalignment is not detectable and surfaces only as unexpected row counts in X/Y. Use
  flat `trial_type` labels in events.tsv if annotations are non-structural, or add a tiling
  events.tsv if partitioning is intended.

## Rationale

**Why key-value `trial_type` rather than side columns.** BIDS' idiomatic answer for
intra-run hierarchy is extra columns alongside `trial_type` (e.g. a `block` column on
each trial row), consumed by FSL/SPM/nilearn/fitlins as a flat DataFrame. Hypline
diverges because its model requires partitions to *tile the run* — every TR belongs
to exactly one slice. Deriving partition onset/duration from trial-level rows
requires gap-filling heuristics (does block-1 start at run start or first trial?
does it own inter-trial gaps?), and each choice changes TR-to-partition assignment
and therefore the science. Explicit tiling rows make that judgment author-controlled
and reviewable rather than hidden in inference.

**Multi-scale events.** Treating `block-1` as a 10s-duration row alongside trial rows
is a deliberate divergence from the dominant GLM-centric reading where `trial_type`
rows are stimulus occurrences. BIDS' minimal definition of "event" (anything with
onset + duration) accommodates both readings; hypline picks the multi-scale one. An
events.tsv authored for hypline is not a drop-in for nilearn/fitlins GLM calls.

**Why globally unique values.** Hypline requirement, not a BIDS norm. Per-block-resetting
trial IDs (`trial-1..trial-N` per block) are permitted by BIDS and common in raw logs,
but must be rewritten upstream before ingestion. Hierarchical tiling — `(block, trial)`
tuples unique but `trial` alone repeating — is therefore not supported.

See [feature-files.md](feature-files.md) for feature file naming conventions.
See [../external/bids.md](../external/bids.md) for the BIDS entity reserved name list.
