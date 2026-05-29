# Denoise — scope and assumptions

What confound regression reads and what assumptions could break it. The
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

## Confound source

One origin: hypline confound files in the `confounds/` parquet area. The
`Denoiser` takes `confounds: list[str]` of `<kind>` or `<kind>-<desc>` refs and
loads each into one regressor matrix. fmriprep's `desc-confounds_timeseries.tsv`
is no longer read in place — `confoundgen fmriprep` imports selected tsv columns
into `conf-fmriprep_desc-*` bundles, so every confound reaches denoise through
the single parquet path. A bare `<kind>` resolves to the bare derivation only,
not all variants — name each derivation explicitly. See
[../decisions/confound-files.md](../decisions/confound-files.md) for the
selector contract.

## Assumptions that could break

- **Row count must match across all confound bundles and the BOLD.** Two
  checks: inter-bundle TR (row) equality at load, and confounds-vs-BOLD-TRs at
  clean time. Either mismatch raises.
- **One confound file per resolved ref** — multiple matches raise.
- **Confound parquet is pre-validated.** Finite-values and TR-alignment are
  enforced at write time (see
  [../decisions/confound-files.md](../decisions/confound-files.md)), so denoise
  does not re-check them.
