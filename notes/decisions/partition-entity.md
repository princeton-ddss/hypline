# The `partition` entity (custom)

Design record for hypline's custom `partition` BIDS entity — rationale, rejected alternatives, and invariants.

A hypline-specific BIDS entity for **sub-run partitions** — when a single
BOLD run contains multiple distinct stimulus blocks (e.g. two movie clips
played back-to-back) that should be modeled as separate feature streams.

## Properties

- **Not BIDS-standard.** Our own convention.
- Appears only on **feature filenames**, not BOLD filenames. A BOLD run is
  never itself split at the filename level; partitions are logical subdivisions
  within one continuous acquisition.
- Partition boundaries live in the **events file**, as rows with
  `trial_type = "partition-{value}"`, with `onset` and `duration` in seconds.
- Partitions must **exhaustively cover the run with no overlap** — every TR
  belongs to exactly one partition. This is what distinguishes "partition"
  from looser concepts like "segment" or "region."
- Enforced at two stages: contiguity and zero-start checked in `_discover_bold`
  at load time; full BOLD length coverage checked in `_build_xy` once arrays
  are loaded.
- The tiling invariant applies to how partitions are **declared in events**,
  not to what is selected for training. Users can filter a subset of
  partitions via `bids_filters` (e.g. `partition-1`) — only the selected
  partitions contribute to X/Y.
- **Numeric values are recommended** (`partition-1`, `partition-2`, ...),
  matching BIDS precedent for `run`. Ordering is intrinsic via `onset`, so
  the label does not need to encode order — but numeric labels make
  sort-by-label match sort-by-onset, which is useful in reports and UIs.
  Alphanumeric labels are allowed; not enforced.

## Naming considered and rejected

- `seg` — reserved (anatomical segmentation labels)
- `part` — reserved (magnitude/phase disambiguation)
- `chunk` — reserved (segmented acquisitions)
- `segment` — not reserved, but semantically loose; close enough to BIDS `seg`
  to cause confusion, and doesn't signal the exhaustive/disjoint constraint.

See [../external/bids.md](../external/bids.md) for the reserved list.

## Key conventions in code

- Internal keys use the entity **value only** (e.g. `"part1"`), matching how
  `ses`/`run` values are stored.
- Events are pre-processed at discovery time into `BoldMeta.partitions:
  dict[str, slice]` (TR-indexed slices). The raw events DataFrame is not
  retained. This keeps `_build_xy` free of events parsing logic and
  localizes the `partition-{value}` trial_type convention to `_discover_bold`.
- Partitioned cells require a colocated events file. Run-level
  (unpartitioned) runs do not.
