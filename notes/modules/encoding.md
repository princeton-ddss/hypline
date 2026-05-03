# Encoding — scope and assumptions

What a single encoding training run operates on, what it requires, and
what assumptions could break it.

The encoding pipeline fits a model from stimulus-derived features (X) to
BOLD responses (Y) — typically banded ridge regression.

## Scope of a single training run

A single `train(sub_id)` call is scoped to:

- **One subject.** Subjects are modeled independently — different brains,
  different voxel alignment.
- **One task (or no task).** Task is optional in BIDS — all files must
  either specify the same task or none at all. Cross-task encoding models
  are not meaningful. If inconsistent task values are discovered across
  BOLD or feature files, the pipeline raises.
- **Multiple sessions and runs: allowed and expected.** Same task, more
  data — concatenated into a single X/Y.
- **Multiple cells per run: allowed.** The segment entity is inferred from
  events.tsv at discovery time. Each segment value is a separate row block,
  identified by a `CellKey` carrying all non-excluded entities (filename +
  enriched metadata).
  See [../decisions/semantic-entity.md](../decisions/semantic-entity.md).

## Pipeline

`train(sub_id)` executes these steps in order:

1. **`_discover_features`** — scans feature filenames for `sub_id`; returns raw
   `dict[FeatureKey, Path]` with filename-only `CellKey`s. No events I/O. No user filters.
2. **`_discover_bold`** — scans BOLD filenames; reads sidecar JSON (TR), events.tsv
   (segment slices), and events.json `Segments` (metadata). Returns
   `dict[BoldKey, BoldMeta]`. Validates within-run and cross-run segment invariants. No user filters.
3. **`_resolve_cell_keys`** — validates filename entities against events.json and
   merges `Segment.metadata` onto each feature cell's `CellKey`. Rejects illegal
   filename entities. Raises on value mismatch between filename and sidecar. Returns
   resolved `dict[FeatureKey, Path]`.
4. **`_apply_filters`** — applies user `bids_filters` to both enriched feature cells
   and BOLD runs. Raises on typo (filter key absent from enriched schema). Returns
   filtered `(dict[FeatureKey, Path], dict[BoldKey, BoldMeta])`.
5. **`_validate_coverage`** — checks `sub`/`task` invariance across all files and
   bidirectional `ses`/`run` coverage between filtered features and BOLD. Raises if
   either side is empty. Void-returning.
6. **`_build_xy`** — loads BOLD arrays; validates `max(slice.stop) ≤ BOLD TRs`;
   assembles X (features) and Y (BOLD) matrices.

## Alignment contract

Features and BOLD runs are matched by stimulus-side identity entities
(`sub`, `ses`, `task`, `run`). Every BOLD run in scope must have feature
coverage for all requested features, and vice versa. Partial coverage is a
hard error — silently dropping runs would mask upstream bugs in feature
generation.

Acquisition entities (`acq`, `ce`, `rec`, `dir`) must be invariant across
all BOLD files in a training call. Mixing fMRIPrep output variants would
either cause a shape mismatch (`acq`) or silently corrupt alignment
(`ce`, `rec`, `dir`). Features are not required to carry these entities —
they are stimulus-derived and independent of acquisition parameters.

Coverage failures are reported fail-fast: the first missing (cell, feature)
pair raises, without aggregating the full list. Mismatches are typically
systematic (e.g. an entire feature missing across all cells from a failed
extraction run), so the first gap is enough signal for the user to fix the
root cause and rerun.

## Module layout

Feature I/O utilities (`read_feature`, `resample_feature`, `save_feature`) live in
`hypline/features/`. The `Encoding` class imports from `features.utils` directly;
external callers should too.

`find_files` is called with `recursive=True` throughout encoding discovery. This is
intentional — fMRIPrep derivatives are often organized in per-subject subdirectories
(`sub-01/func/`), so flat-directory search would miss them.

## `bids_filters` routing

All user-supplied `bids_filters` are applied post-resolution in `_apply_filters` against
resolved `CellKey`s (features) and BOLD filename entities (BOLD). Neither `_discover_features`
nor `_discover_bold` apply user filters — both use only hard-coded structural filters
(`sub`, `feature`, `space`).

Rationale: metadata entities (e.g. `cond-R`) do not exist on filenames and cannot be routed
to `find_files`. Applying all filters uniformly post-resolution ensures consistent behaviour
whether the filter targets a structural entity (`ses-1`) or a descriptive one (`cond-R`).

- **Reserved entities** (`sub`, `space`, `feature`): rejected at construction — use the
  dedicated arguments instead.

**Match semantics:** multiple filters sharing the same entity key OR-match within that group;
different entity keys AND-match across groups (e.g. `["task-rest", "task-nback", "ses-1"]`
selects runs where task is rest or nback, AND session is 1). This mirrors `find_files` behaviour.

**Asymmetric schemas:** feature cells and BOLD files do not share the same entity key set
(e.g. `task` is excluded from `CellKey` but present on BOLD filenames). A filter key absent
from one side is silently skipped on that side — it applies only where the entity is meaningful.
A filter key absent from *both* sides raises `ValueError` (typo diagnostic).

Filter entity validation is **fail-then-diagnose**: every filter entity is checked against the
union of resolved cell keys and BOLD filename keys — any absent entity raises `ValueError`
before any empty-result `FileNotFoundError`.

## Assumptions that could break

- **Consistent TR across all runs** for a subject. Mixed-TR datasets would
  need per-run TR handling.
- **Feature files mirror BOLD identity entities.** See
  [../decisions/feature-files.md](../decisions/feature-files.md).
- **Segment contract** (single entity, non-overlap, cross-run agreement, cell schema
  invariance): see [../decisions/semantic-entity.md](../decisions/semantic-entity.md).
- **Segment metadata contract** (events.json `Segments` format, enrichment, filtering):
  see [../decisions/segment-metadata.md](../decisions/segment-metadata.md).
- **One feature file per (CellKey, feature) pair.** Multiple files for the same pair is
  ambiguous provenance, not something to merge.
