# Denoise — scope and assumptions

What confound regression reads and what assumptions could break it. The
pipeline regresses nuisance signals out of preprocessed BOLD into a cleaned
derivative; one `denoise(sub_id)` call cleans every matching run for a subject.

The cleaned output (`desc-clean`) is what encoding reads by default
(`bold_desc="clean"`).

## Confound sources

Two origins, loaded together into one regressor matrix:

- **Standard confounds** — fmriprep's `desc-confounds_timeseries.tsv`, read
  **in place** from the fmriprep tree (TSV + JSON sidecar). This is an interim
  deviation from the target [confound-files.md](../decisions/confound-files.md)
  design, where external sources are converted into hypline confound files via
  `confoundgen` rather than read in place. Until that import exists, denoise
  reads the TSV directly.
- **Custom confounds** — hypline confound files in the `confounds/` parquet
  area, selected by `ModelSpec.custom_confounds` entries of the form `<kind>`
  or `<kind>-<desc>`. A bare `<kind>` resolves to the bare derivation only, not
  all variants — name each derivation explicitly. See
  [../decisions/confound-files.md](../decisions/confound-files.md) for the
  selector contract.

## Assumptions that could break

- **Row count must match across BOLD, standard, and custom confounds.** Two
  separate checks: standard-vs-custom at load, and confounds-vs-BOLD-TRs at
  clean time. Either mismatch raises.
- **One confounds file per run** for both standard (fmriprep TSV) and each
  resolved custom-confound selector — multiple matches raise.
- **Custom confound parquet is pre-validated.** Finite-values and TR-alignment
  are enforced at write time (see
  [../decisions/confound-files.md](../decisions/confound-files.md)), so denoise
  does not re-check them.
