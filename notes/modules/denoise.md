# Denoise — scope and assumptions

What nuisance regression reads and what assumptions could break it. The
pipeline regresses nuisance signals out of preprocessed BOLD into a denoised
derivative; one `denoise(sub_id)` call cleans every matching run for a subject.

## Output

Denoised BOLD (`desc-denoised`) is written to the **`derivatives/hypline/`**
tree (see [../decisions/layout.md](../decisions/layout.md)), mirroring fmriprep's
`sub-XX/[ses-YY/]func/` shape with full BOLD identity preserved — only
`desc=denoised` and the root differ from the `desc-preproc` source. Input is
fixed to `desc-preproc` so denoise never re-cleans its own output. Surface BOLD
is per-hemisphere (`hemi-L`/`hemi-R`); each hemi file is cleaned independently —
no L/R coupling.

Each output gets a per-file `..._desc-denoised_bold.json` sidecar: `Sources`
(a `bids:fmriprep:` URI to the source BOLD), resolved cleaning settings
(post-expansion fmriprep/custom columns, compcor specs, `clean_params`,
`n_regressors`, `RepetitionTime`/`TaskName`/`SkullStripped`), and
`hypline_version`. On first output, a write-if-absent
`derivatives/hypline/dataset_description.json` (`GeneratedBy: hypline`,
`DatasetType: derivative`, `DatasetLinks`) stamps the tree.

**Denoise output and encoding's default input must agree — and currently do
not.** Encoding reads its input BOLD via `find.fmriprep` defaulting to
`bold_desc="clean"`, so it looks for `desc-clean` in the fmriprep tree. Denoise
now writes `desc-denoised` under `derivatives/hypline/`, so neither the area nor
the descriptor matches: the denoise→encoding default chain finds nothing. Closing
the gap means pointing encoding's read path at the hypline area with a `denoised`
default. Until then this block documents a live break, not a settled contract.

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
