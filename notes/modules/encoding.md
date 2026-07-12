# Encoding — scope and assumptions

What encoding training and inference operate on, what they require, and
what assumptions could break them.

The encoding pipeline fits a model from stimulus-derived regressors (X) to
BOLD responses (Y) — typically banded ridge regression. X carries two regressor
streams: **features** (the signal of interest) plus optional **confounds**
(stimulus-derived nuisance, partialled out in the same fit). Each feature gets
its own ridge band; all confounds collapse into a single trailing band regardless
of how many are configured. Confounds are optional — with none configured X is
features only. Training (`EncodingTrainer`) and out-of-sample inference
(`EncodingPredictor`) are separate roles over a shared X-build path; see
[Predict](#predict--out-of-sample-inference-across-subjects).

## Scope of a single training run

A single `train(sub_id)` call is scoped to:

- **One subject.** Subjects are modeled independently — different brains,
  different voxel alignment.
- **Tasks are an explicit input.** `EncodingTrainer(tasks=[...])` declares which
  task values are in scope; others are excluded at discovery. `task` is a
  `CellKey` axis: `tasks=["A", "B"]` opts into a multi-task fit where
  A-cells and B-cells are distinct rows sharing regression weights.
  Single-task is the norm; the explicit list makes any deviation visible
  at the call site.
- **Multiple sessions and runs: allowed and expected.** More data,
  concatenated into a single X/Y.
- **Features are named with optional variant.** Each `EncodingTrainer(features=[...])`
  entry is `<kind>` (canonical, reads the bare `<kind>/` folder) or
  `<kind>-<desc>` (reads the `<kind>-<desc>/` variant folder). Two entries
  sharing a `kind` are rejected at construction — variants of one kind are
  near-collinear and belong in separate models, not bands in one fit. There is
  no implicit fallback: a named variant missing on disk raises rather than
  reading the canonical folder. (Confound generators instead source all
  variants; see [../decisions/confound-files.md](../decisions/confound-files.md).)
- **Confounds dedup differently from features.** `EncodingTrainer(confounds=[...])`
  entries share **one** ridge band, so two variants of one kind (e.g.
  `phonemic-onset` and `phonemic-rate`) may coexist in a single fit — they are
  deduped on the full `<kind>-<desc>` ref, not on `kind`. This is the opposite of
  features, which reject two entries sharing a kind. Order within the band is
  irrelevant (one alpha). A ref may not appear in both `features` and `confounds`
  of one fit — it resolves to one file in one role — so the overlap is rejected at
  construction.
- **Multiple cells per run: allowed.** The segment entity is inferred from
  events.tsv at discovery time. Each segment value is a separate row block,
  identified by a `CellKey` carrying all non-excluded entities (filename +
  enriched metadata).
  See [../decisions/semantic-entity.md](../decisions/semantic-entity.md).

## Pipeline

Features and confounds are two independent regressor streams. Each is discovered
and validated on its own (schema, generator-metadata, per-stream cell coverage),
resolved against BOLD **before** merging, then combined into one regressor dict
that filter/enrich/build run over once. The per-stream resolve is load-bearing:
resolution's unsegmented-run guard allows only one file per run, but a feature and
a confound legitimately share one run — resolving the merged dict would miscount
and falsely reject. `train(sub_id)` executes these steps in order:

1. **`_discover_features`** — resolves `sub_id`'s dyad via `dyad_of`
   (participants.tsv; features are dyad-keyed — see
   [../decisions/dyad-keyed.md](../decisions/dyad-keyed.md)), then scans
   dyad-keyed feature filenames; returns raw `dict[RegressorKey, Path]` with
   filename-only `CellKey`s. No events I/O. No user filters.
   **`_discover_confounds`** mirrors it against `confounds/` (same schema and
   per-cell coverage checks); returns an empty dict when no confounds are configured.
2. **`_discover_bold`** — scans BOLD filenames; reads sidecar JSON (TR), events.tsv
   (segment slices), and events.json `trial_type.Levels` (metadata) from the **raw** BIDS
   tree via `BIDSLayout.path.raw` (sidecars are identity-keyed, not per-variant). Returns
   `dict[BoldKey, BoldMeta]`. A surface run maps two files to one `BoldKey`;
   `resolve_voxel_source` orders the hemis `(L, R)` so their voxels concatenate into one Y,
   while a volume run stays one file (see [../external/fmriprep.md](../external/fmriprep.md#surface-vs-volume)).
   Validates within-run and cross-run segment invariants. No user filters.
3. **`_resolve_cell_keys`** — precondition: every cell's `(ses, run)` must map to a
   `BoldMeta` (raises `FileNotFoundError` if not — required to read segments for
   enrichment). Then merges `Segment.metadata` onto each cell's `CellKey`.
   Rejects illegal filename entities. Raises on value mismatch between filename and
   sidecar. Applied to each stream separately (see above), then the two resolved dicts
   are merged into one `dict[RegressorKey, Path]`.
4. **`_apply_filters`** — applies user `bids_filters` to both enriched regressor cells
   and BOLD runs. Raises on typo (filter key absent from enriched schema). Returns
   filtered `(dict[RegressorKey, Path], dict[BoldKey, BoldMeta])`.
5. **`_validate_coverage`** — bidirectional `(ses, task, run)` coverage: every
   filtered BOLD run must have regressor coverage and vice versa. Raises if either
   filtered set is empty. Void-returning. **Train-only** — predict needs only the
   regressor→BOLD half (already enforced by `_resolve_cell_keys`), so it skips this.
   **`_validate_confound_alignment`** (train-only) then asserts the feature and
   confound streams cover the same cell set — filters can narrow one stream and leave
   the other with an orphan cell that would misalign the confound band against X's rows.
6. **`_enrich_regressor_metas`** — the single regressor↔BOLD-timeline crossover, role-neutral
   (placement depends only on the cell's BOLD timeline): derives each cell's TR-grid
   placement (`onset_tr`, `n_trs`, `repetition_time`) from `bold_metas` (segment TR-slice,
   or the run header for unsegmented runs). Reads no BOLD voxel data. Once this runs,
   X-building never touches `bold_metas` again.
7. **`_build_x`** — assembles the X regressor matrix and its row/column geometry from
   the enriched metas alone; no Y, no BOLD. Shared by train and predict. Features are
   downsampled onto the TR grid at read time; confounds are read TR-level and asserted
   to already span the cell's `n_trs` (no downsample). Each feature is its own column
   band; confounds share one trailing band.
8. **`_build_training_data`** (train-only) — wraps `_build_x` and appends Y via
   `_align_y`, which slices each cell's BOLD onto X's row geometry and validates
   `n_trs` against the array (drift guard + bounds check) before assembling.

## Prod/comp turn split

In a dyadic conversation a subject alternates between **producing** speech (own
turn) and **comprehending** the partner's, and one word feature can drive
different responses in those states. `_build_x` supports splitting every
regressor into a prod copy and a comp copy so the ridge learns separate weights.

- **Screen band is always present, unscaled.** Two intercept-like task boxcars
  (`prod`, `comp`) ride a reserved shared-alpha band that skips `StandardScaler`
  — standardizing maps their off-state `0 -> -mean` and destroys the flat-zero
  semantics that let them absorb the prod-vs-comp mean BOLD offset, keeping that
  block-level task variance out of ridge-penalized feature weights. Present
  regardless of `split`.
- **`split` (on `XRecipe`, default true) gates only regressor duplication.**
  When set, each feature array and the confound array is duplicated into a prod
  copy (kept on the modeled subject's speaking TRs) and a comp copy (kept on the
  partner's). The two copies share their regressor's band (one alpha), so each
  feature band and the single confound band doubles in width; band *keys* in
  `col_slices` are unchanged. Silence TRs are zero in both (implicit baseline).
- **Three-state mask, not `comp = ~prod`.** The per-TR (prod, comp) boxcar is
  re-derived from `load_turns`/`stamp_turns` on a synthetic TR-cadence grid — not
  read off the feature files, whose per-word `turn_sub` is dropped at downsample.
  `prod = turn_sub == sub`; `comp = turn_sub is not None and != sub`; silence
  (`turn_sub is None`) is false in both. See
  [events.md](events.md#speaking-turns).
- **Subject-relative mask invariant.** `_build_x(sub_id, ...)` resolves prod/comp
  against whichever subject's data is being built — the train subject at train,
  the *source* subject at predict — so train and predict Xs stay identical in
  column meaning ("prod copy = the fed subject speaking") and the `col_slices`
  drift guard passes. The recipe stores only the boolean `split`, never a
  subject-specific mask. See
  [../decisions/dyad-keyed.md](../decisions/dyad-keyed.md).
- **Requires a 2-subject dyad.** The split raises if the dyad is not exactly two
  subjects (partner/comp ambiguous) or the subject is not in it, and if the
  screens are all-zero across every cell (no TR in any speaker window).

## Predict — out-of-sample inference across subjects

Predict reuses the same discovery + `_build_x` path (no parallel builder) so the
rebuilt X is byte-identical in layout to train's; a `col_slices` mismatch against
the stored recipe is a hard error (schema-drift guard). It diverges from train in
three durable ways:

- **Three independent subjects.** *Model* subject = whose trained weights (which
  artifact). *Source* subject = whose features build X (the prediction inputs).
  *Target* subject = whose actual BOLD is Y, for comparison (analyze only — predict
  takes no target). All three are orthogonal; any combination is valid within one
  study. See [../decisions/dyad-keyed.md](../decisions/dyad-keyed.md).
- **Cell selection, not coverage.** Each model predicts on its out-of-sample cells
  by default; an explicit `test_on` selector overrides (honored even for trained-on
  cells, with a warning, never rejected). `test_on` is a list of `<entity>-<value>`
  refs sharing the `bids_filters` AND/OR grammar (same-entity OR, cross-entity AND —
  see [../decisions/layout.md](../decisions/layout.md#bids_filters--structural-vs-descriptive));
  a named entity absent from the cell schema raises. OOS is `available − train` for a single
  model (unbounded — a new subject's extra runs are fair game) but `universe − train`
  for a K-fold model (bounded to the train corpus). Empty selection is an error,
  not a silent no-op.
- **No train-time filters replayed.** Predict discovers the source's *full* cell
  set; recipe `bids_filters` (the train-corpus bound) are deliberately not re-applied,
  so `test_on` can name cells outside the original corpus.

The package predicts per-model and never stitches the K predictions — consolidating
them is the consumer's concern.

## Analyze — scoring a prediction against a target brain

`EncodingPredictor.analyze` compares a source-driven prediction against a *target*
subject's actual BOLD, producing per-fold, per-band, per-role, per-voxel correlations.
It wraps `predict` (so the source builds X and the loaded model supplies weights), then
recovers the target's Y and turns to score. All three subject roles are orthogonal;
which triples are *scientifically* meaningful depends on whether the source and
target share a conversation:

- **Same person** (source == target) — X and Y are one conversation, one brain: a
  genuine within-subject fit.
- **Partners in one dyad** — shared conversation, different brains. Self and partner
  build *different* X (subject-relative screens/split — see
  [Prod/comp turn split](#prodcomp-turn-split)) over the same stimulus, so the
  row-wise pairing stays coherent.
- **Different dyads** — X and Y are different conversations, paired by run index
  only: a scramble / null control, mechanically valid but not a fit. `analyze` only
  **warns** here; `_align_y`'s length-drift guard is the hard net if runs mismatch.

### Prod/comp/both roles

Correlations are computed over three row subsets, all derived from the **target's**
turns (the target's own dyad, not the source's — the source's conversation may never
give the target the floor). Each role's raw per-TR mask is FIR-smeared with the
model's `delays` before selecting rows:

- **`prod`** / **`comp`** — *exclusive* pure-production / pure-comprehension rows.
  Raw prod/comp are already disjoint (one floor-holder per TR). The exclusive AND-NOT
  matters only because smearing spills each mask onto boundary rows; it drops those
  delay-contaminated rows.
- **`both`** — the smeared *union*: any speech-active row, re-including the boundary
  rows exclusive drops. Pure-silence rows (no speech within the delay window) stay
  excluded — hypline conversations have silence TRs, so `both` is not the whole run.

The *inclusive* prod/comp (smeared without the AND-NOT) are intentionally not
computed. A role with zero rows scores **NaN**, not zero — a correlation over zero
varying pairs is undefined, and NaN is skipped by `nanmean` across folds instead of
masquerading as a real low correlation.

### Eval-result persistence

Analyze returns a pure in-memory `xr.Dataset` (`corr` over `fold`/`band`/`role`/
`voxel`) and writes nothing — persistence is the caller's job, mirroring `train`'s
write-free `EncodingArtifact` return. `save_eval`/`load_eval` are the storage seam:
each takes a raw `path`, not a layout `(sub, kind, desc)` triple. The `analyze` CLI
binds evals to `results/sub-<target>/encodingEval-<desc>/…nc`, but the seam stays
path-based so an eval file is self-describing wherever it is written. An eval
result is **archival scientific data**, not a Python-runtime object like the fitted
model, so it is stored as self-contained **netCDF-4** any tool can read — not
joblib/pickle. Provenance (`model_sub`, `source_sub`, `target_sub`, `delays`,
`bold_space`, `test_on`, `fold_cells`, `hypline_version`) rides in the Dataset
`attrs`, so the file is self-describing regardless of where the caller writes it.

## Same-study assumption (load-bearing)

Predict and analyze run across subjects *within one study* sharing the design
(paradigm, acquisition, TR). Encoding weights tie to a feature space and voxel grid,
so crossing studies breaks feature alignment regardless of TR. This is a usage
contract, not a recipe guard: `XRecipe` carries no `repetition_time`, and the
per-build TR check (`_discover_bold`) plus `_align_y`'s drift guard enforce
*consistency*, not cross-study tolerance.

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
the upstream denoise step has already cleaned of **run-level nuisance** signal
(motion, aCompCor/tCompCor, WM/CSF, drift). That cleaning is a prior, separate
stage — not something the encoding fit does.

Encoding-time confounds are a distinct concern from denoise nuisance, split by
what they model and when they act. Denoise removes run-level nuisance from BOLD
before encoding ever sees it. Encoding-time confounds are **stimulus-derived**
regressors (e.g. phonemic onsets/rate) that ride *in the encoding fit itself* —
partialled out within the same ridge so a feature band cannot claim variance a
confound explains. "Of interest vs. nuisance" is therefore a per-fit role split,
not a denoise-vs-encoding split: a stimulus-derived ref is a feature band in one
fit and a confound band in another, never intrinsically one or the other. Within
a single fit a ref is one role only (features and confounds may not overlap).
See [../decisions/confound-files.md](../decisions/confound-files.md).

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
  need per-run TR handling. (Cross-subject predict/analyze additionally assume
  one shared study — see [Same-study assumption](#same-study-assumption-load-bearing).)
- **Feature files mirror BOLD identity entities.** See
  [../decisions/feature-files.md](../decisions/feature-files.md).
- **Prod/comp split relies on turn-derivable data for the dyad** — a 2-subject
  dyad with non-degenerate `turn_speaker` windows. See
  [Prod/comp turn split](#prodcomp-turn-split).
- **Segment contract** (single entity, non-overlap, cross-run agreement, cell schema
  invariance): see [../decisions/semantic-entity.md](../decisions/semantic-entity.md).
- **Segment metadata contract** (events.json `trial_type.Levels` format, enrichment, filtering):
  see [../decisions/segment-metadata.md](../decisions/segment-metadata.md).
- **One feature file per segment** — feature files must be split at segment boundaries
  upstream; encoding never bins across them. See
  [../decisions/feature-files.md](../decisions/feature-files.md#one-file-per-segment).
- **One feature file per (CellKey, feature) pair.** Multiple files for the same pair is
  ambiguous provenance, not something to merge.
