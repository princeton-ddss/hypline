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
   (segment slices), and events.json `trial_type.Levels` (metadata) from the **raw** BIDS
   tree via `BIDSLayout.path.raw` (sidecars are identity-keyed, not per-variant). Returns
   `dict[BoldKey, BoldMeta]`. Validates within-run and cross-run segment invariants. No user filters.
3. **`_resolve_cell_keys`** — precondition: every feature's `(ses, run)` must map to a
   `BoldMeta` (raises `FileNotFoundError` if not — required to read segments for
   enrichment). Then merges `Segment.metadata` onto each feature cell's `CellKey`.
   Rejects illegal filename entities. Raises on value mismatch between filename and
   sidecar. Returns resolved `dict[FeatureKey, Path]`.
4. **`_apply_filters`** — applies user `bids_filters` to both enriched feature cells
   and BOLD runs. Raises on typo (filter key absent from enriched schema). Returns
   filtered `(dict[FeatureKey, Path], dict[BoldKey, BoldMeta])`.
5. **`_validate_coverage`** — checks `sub`/`task` invariance across all files.
   Bidirectional `ses`/`run` coverage: every filtered BOLD run must have feature
   coverage and vice versa. Raises if either filtered set is empty. Void-returning.
6. **`_build_xy`** — loads BOLD arrays; validates `max(slice.stop) ≤ BOLD TRs`;
   assembles X (features) and Y (BOLD) matrices.

## Alignment contract

Features and BOLD runs are matched by stimulus-side identity entities
(`sub`, `ses`, `task`, `run`). Every BOLD run in scope must have feature
coverage for all requested features, and vice versa. Partial coverage is a
hard error — silently dropping runs would mask upstream bugs in feature
generation.

**X/Y temporal alignment**: both X and Y are framed segment-locally —
features bin into TRs 0…n-1 of the segment, and BOLD is sliced to the same
segment — so the two meet at the segment boundary without offset arithmetic.
This relies on `start_time` being source-relative; see
[../decisions/feature-files.md](../decisions/feature-files.md#temporal-alignment).

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

Feature I/O utilities (`read_feature`, `save_feature`) are exposed from
`hypline.features`. TR-alignment lives in `hypline.downsample` (shared with
confound generation — see [../decisions/confound-files.md](../decisions/confound-files.md)).

Feature and BOLD discovery uses `BIDSLayout` (see `hypline.layout`), which walks the
derivatives tree recursively. This handles fMRIPrep's per-subject subdirectories
(`sub-01/func/`) without flat-directory blind spots.

## `bids_filters` routing

All user-supplied `bids_filters` are applied post-resolution in `_apply_filters` against
resolved `CellKey`s (features) and BOLD filename entities (BOLD). Neither `_discover_features`
nor `_discover_bold` apply user filters — both use only hard-coded structural filters
(`sub`, `feat`, `space`).

Rationale: metadata entities (e.g. `cond-R`) do not exist on filenames and cannot be routed
to `BIDSLayout` queries. Applying all filters uniformly post-resolution ensures consistent
behaviour whether the filter targets a structural entity (`ses-1`) or a descriptive one (`cond-R`).

- **Reserved entities** (`sub`, `space`, `feat`): rejected at construction — use the
  dedicated arguments instead.

**Match semantics:** multiple filters sharing the same entity key OR-match within that group;
different entity keys AND-match across groups (e.g. `["task-rest", "task-nback", "ses-1"]`
selects runs where task is rest or nback, AND session is 1).

**Asymmetric schemas:** feature cells and BOLD files do not share the same entity key set
(e.g. `task` is excluded from `CellKey` but present on BOLD filenames). A filter key absent
from one side is silently skipped on that side — it applies only where the entity is meaningful.
A filter key absent from _both_ sides raises `ValueError` (typo diagnostic).

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
- **Segment metadata contract** (events.json `trial_type.Levels` format, enrichment, filtering):
  see [../decisions/segment-metadata.md](../decisions/segment-metadata.md).
- **One feature file per segment** — feature files must be split at segment boundaries
  upstream; encoding never bins across them. See
  [../decisions/feature-files.md](../decisions/feature-files.md#one-file-per-segment).
- **One feature file per (CellKey, feature) pair.** Multiple files for the same pair is
  ambiguous provenance, not something to merge.
