# Nuisance files

Design record for hypline's nuisance file format — schema and naming rules.
Parallel to [confound-files.md](confound-files.md); the two are deliberately
distinct (see below).

Nuisance regressors are **run-level**: they clean a BOLD run, so they are always
one row per TR. Confounds track **feature granularity** and serve the
encoding-side consumer; the two are split by granularity into separate folders
and formats — see [confound-files.md](confound-files.md), [layout.md](layout.md).

This format covers nuisance files on disk in `nuisance/`. Denoise also pulls
nuisance regressors directly from fmriprep's native tsv, which does *not* use
this format — see [../modules/denoise.md](../modules/denoise.md) for the full set
of channels denoise reads.

## Format

- Extension: `.tsv` — fmriprep's native confounds format, and a recognized BIDS
  timeseries form. Chosen over the `confounds/`/`features/` parquet convention
  because nuisance columns are **scalar-per-row** and users select columns **by
  name** — a different shape than the vector-payload `confound`/`feature` array
  columns. Parquet would also be hostile to the bring-your-own-physio case.
- A wide table: one scalar column per regressor, one row per TR.
- Finite values only — empty, non-numeric, and non-finite cells are rejected at
  read.
- No metadata footer and no array column — the caller selects columns by name.

## Naming

```
nuisance/sub-XX/[ses-YY/]<kind>[-<desc>]/
    sub-XX_ses-YY_task-T_run-N_nuis-<kind>[_desc-<desc>]_timeseries.tsv
```

- `nuis-<kind>` is the category entity — BIDS-free (`nuis` collides with no
  reserved BIDS short label). Optional `desc-<desc>` names the derivation.
- Carries the `_timeseries` BIDS suffix.
- Mirrors source-BOLD identity entities (`sub`, `ses`, `task`, `run`), so each
  BOLD run resolves to exactly one file per source ref.
- **Strictly full-run-level** — denoise regresses against the whole BOLD run, so
  a nuisance file carries no segment entity. `--data-filters` selects *which
  runs* to clean, not sub-run segments.

See [bidspath-validation.md](bidspath-validation.md) /
[unsupported-entities.md](unsupported-entities.md) for the `nuis-` entity and
suffixed-derived-file validation rules.

## I/O

The reader is **BOLD-agnostic**: it validates internal consistency (non-empty,
numeric, finite) but not row count, because it has no BOLD to compare against —
the row-count-vs-BOLD check lives at denoise. It is intentionally not a public
export; it serves denoise, not external callers.
