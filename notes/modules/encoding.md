# Encoding — scope and assumptions

What a single encoding training run operates on, what it requires, and
what assumptions could break it.

The encoding pipeline fits a model from stimulus-derived features (X) to
BOLD responses (Y) — typically banded ridge regression.

## Scope of a single training run

A single `train(sub_id)` call is scoped to:

- **One subject.** Subjects are modeled independently — different brains,
  different voxel alignment.
- **Tasks are an explicit input.** `Encoding(tasks=[...])` declares which
  task values are in scope; others are excluded at discovery. `task` is a
  `CellKey` axis: `tasks=["A", "B"]` opts into a multi-task fit where
  A-cells and B-cells are distinct rows sharing regression weights.
  Single-task is the norm; the explicit list makes any deviation visible
  at the call site.
- **Multiple sessions and runs: allowed and expected.** More data,
  concatenated into a single X/Y.
- **Features are named with optional variant.** Each `Encoding(features=[...])`
  entry is `<kind>` (canonical, reads the bare `<kind>/` folder) or
  `<kind>-<desc>` (reads the `<kind>-<desc>/` variant folder). Two entries
  sharing a `kind` are rejected at construction — variants of one kind are
  near-collinear and belong in separate models, not bands in one fit. There is
  no implicit fallback: a named variant missing on disk raises rather than
  reading the canonical folder. (Confound generators instead source all
  variants; see [../decisions/confound-files.md](../decisions/confound-files.md).)
- **Multiple cells per run: allowed.** The segment entity is inferred from
  events.tsv at discovery time. Each segment value is a separate row block,
  identified by a `CellKey` carrying all non-excluded entities (filename +
  enriched metadata).
  See [../decisions/semantic-entity.md](../decisions/semantic-entity.md).

## Pipeline

`train(sub_id)` executes these steps in order:

1. **`_discover_features`** — resolves `sub_id`'s dyad via `dyad_of`
   (participants.tsv; features are dyad-keyed — see
   [../decisions/dyad-keyed.md](../decisions/dyad-keyed.md)), then scans
   dyad-keyed feature filenames; returns raw `dict[FeatureKey, Path]` with
   filename-only `CellKey`s. No events I/O. No user filters.
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
5. **`_validate_coverage`** — bidirectional `(ses, task, run)` coverage: every
   filtered BOLD run must have feature coverage and vice versa. Raises if either
   filtered set is empty. Void-returning.
6. **`_build_xy`** — loads BOLD arrays; validates `max(slice.stop) ≤ BOLD TRs`;
   assembles X (features) and Y (BOLD) matrices.

## Alignment contract

Features and BOLD runs are matched by their shared run entities
(`ses`, `task`, `run`). The leading identity is *not* shared (features
`dyad`-keyed, BOLD `sub`-keyed — see
[../decisions/dyad-keyed.md](../decisions/dyad-keyed.md)). Every BOLD run in
scope must have feature coverage for all requested features, and vice versa.
Partial coverage is a hard error — silently dropping runs would mask upstream
bugs in feature generation.

**X/Y temporal alignment**: both X and Y are framed segment-locally —
features bin into TRs 0…n-1 of the segment, and BOLD is sliced to the same
segment — so the two meet at the segment boundary without offset arithmetic.
This relies on `start_time` being source-relative; see
[../decisions/feature-files.md](../decisions/feature-files.md#temporal-alignment).

Acquisition entities (`acq`, `ce`, `rec`, `dir`, `echo`, `part`, `chunk`) are
disallowed project-wide — see [../decisions/unsupported-entities.md](../decisions/unsupported-entities.md).
Encoding therefore assumes a single coherent acquisition without per-call
invariance checks.

Coverage failures are reported fail-fast: the first missing (cell, feature)
pair raises, without aggregating the full list. Mismatches are typically
systematic (e.g. an entire feature missing across all cells from a failed
extraction run), so the first gap is enough signal for the user to fix the
root cause and rerun.

## Module layout

Feature I/O lives in `hypline.io` — see
[../decisions/feature-files.md](../decisions/feature-files.md) for the API
shape. Encoding operates on `BIDSPath` objects discovered via `BIDSLayout`
and reads feature files via `read_feature` / `read_feature_metadata`.
TR-alignment lives in `hypline.downsample` (shared with confound
generation — see [../decisions/confound-files.md](../decisions/confound-files.md)).

Feature and BOLD discovery uses `BIDSLayout` (see `hypline.layout`), which walks the
derivatives tree recursively. This handles fMRIPrep's per-subject subdirectories
(`sub-01/func/`) without flat-directory blind spots.

`_discover_bold` reads from `derivatives/hypline` (`find.hypline`), the home of
hypline's BOLD postprocessing outputs. `bold_desc` selects the variant flavor:
`denoised` (the default, see `layout.path.denoised`) today, and any future
postprocessing variant (e.g. hyperalignment) keyed by `desc-<flavor>`.

Encoding never consumes fMRIPrep's raw `desc-preproc` BOLD; it reads BOLD that
the upstream denoise step has already cleaned of nuisance signal. Nuisance
denoising and encoding are disjoint concerns over different regressor sets:
denoise removes signals of no interest (motion, aCompCor/tCompCor, WM/CSF, drift),
while encoding predicts BOLD from stimulus features of interest (phonemic onsets,
MFCC, etc.). The two never mix — cleaning BOLD is a prior, separate stage, not
something the encoding fit does.

## `bids_filters` routing

All user-supplied `bids_filters` are applied post-resolution in `_apply_filters` against
resolved `CellKey`s (features) and BOLD filename entities (BOLD). Neither `_discover_features`
nor `_discover_bold` apply user filters. `_discover_features` uses only structural
filters (`dyad` — resolved from `sub` via `dyad_of` — and `feat`) plus the
per-feature `desc` variant selector; `_discover_bold` uses `sub`, `space`, `desc`
(the `bold_desc` derivative flavor), and `task`.

Rationale: metadata entities (e.g. `cond-R`) do not exist on filenames and cannot be routed
to `BIDSLayout` queries. Applying all filters uniformly post-resolution ensures consistent
behaviour whether the filter targets a structural entity (`ses-1`) or a descriptive one (`cond-R`).

**Reserved entities** (`sub`, `task`, `space`, `feat`, `desc`) are rejected at construction —
use the dedicated arguments instead. `desc` is reserved because it is overloaded:
feature-file variant selector (`features=[...]`) vs. BOLD derivative flavor (`bold_desc`,
default `"denoised"`); neither routes through `bids_filters`.

**Match semantics:** multiple filters sharing the same entity key OR-match within that group;
different entity keys AND-match across groups (e.g. `["run-1", "run-2", "ses-1"]` selects
runs where run is 1 or 2, AND session is 1).

**Asymmetric schemas:** feature cells and BOLD files do not always share the same entity
key set (e.g. enrichment metadata lives on `CellKey` but not BOLD filenames). A filter key
absent from one side is silently skipped on that side — it applies only where the entity is
meaningful. A filter key absent from _both_ sides raises `ValueError` (typo diagnostic).

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
