# Denoise — scope and assumptions

What nuisance regression reads and what assumptions could break it. The
pipeline regresses nuisance signals out of preprocessed BOLD into a cleaned
derivative; one `denoise(sub_id)` call cleans every matching run for a subject.

The cleaned output (`desc-clean`) is what encoding reads by default
(`bold_desc="clean"`).

## Output

Cleaned BOLD (`desc-clean`) is written **in place** in the fmriprep `func/`
dir beside its `desc-preproc` source, differing only in the `desc` entity.
Input is fixed to `desc-preproc` so denoise never re-cleans its own output.
Surface BOLD is per-hemisphere (`hemi-L`/`hemi-R`); each hemi file is cleaned
independently — no L/R coupling.

## Nuisance source

Denoise sources **run-level nuisance regressors**, not `confounds/`. It does not
read the `confounds/` area at all — that folder is reserved for feature-granular
encoding confounds. Regressors come from two channels, h-concatenated into one
`(n_trs, n_regressors)` matrix:

1. **fmriprep tsv** — `--columns` / `--compcor` select columns from the run's
   `desc-confounds_timeseries.tsv` natively (no intermediate hypline file). The
   format knowledge lives in `hypline.fmriprep`; denoise resolves the run's tsv
   via `find.fmriprep` (one match or raise), reads, and selects.
2. **custom nuisance** — `--custom-sources <kind>[-<desc>]` names sources in
   `nuisance/`; `--custom-columns` selects columns from their horizontal concat.
   The two must be given together. Each source resolves via `find.nuisance`.

At least one of `--columns` / `--compcor` / `--custom-sources` is required — an
all-empty invocation raises rather than silently no-opping. See
[../decisions/nuisance-files.md](../decisions/nuisance-files.md) for the nuisance
file contract and [../external/fmriprep.md](../external/fmriprep.md) for the tsv
read.

## `--columns` selection grammar

`--columns` tokens resolve as literals or as one of two hardcoded group
prefixes (`cosine`, `motion_outlier` — the variable-count families, see
[../external/fmriprep.md](../external/fmriprep.md)); see
`select_fmriprep_columns`. Gotcha: a group matching nothing contributes zero
columns **silently** — no raise, unlike a missing literal. No glob support
(`trans_*` is not a thing); `--custom-columns` is likewise literal-only.

## Assumptions that could break

- **Column names must be unique across all channels.** Validated on the raw
  column list (custom-vs-custom and custom-vs-fmriprep) before any name→column
  mapping; a collision raises rather than silently dropping or prefixing.
- **Row count must match across all channels and the BOLD.** Inter-channel row
  equality is checked before h-concat (h-concat would null-pad and defeat the
  finiteness guarantee); confounds-vs-BOLD-TRs is checked at clean time. Either
  mismatch raises.
- **One file per resolved ref** — for both the fmriprep tsv and each custom
  source, zero or multiple matches raise.
- **fmriprep vs. custom NaN policy differs.** fmriprep derivative columns carry
  leading `n/a` by design, so the native read backfills them; custom nuisance
  TSV has no such convention, so `read_nuisance` raises on any non-finite cell.
