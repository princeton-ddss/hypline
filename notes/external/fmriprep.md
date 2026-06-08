# fMRIPrep — external spec reference

fMRIPrep output behaviors that hypline code specifically depends on.

fMRIPrep outputs follow the BIDS-Derivatives spec. Relevant behaviors we
depend on:

## Repetition time (TR)

Stored in a colocated JSON sidecar (`..._bold.json`) under `"RepetitionTime"`,
both in raw BIDS and next to each fmriprep `desc-preproc_bold`. The declared
sidecar value is exact; the image header is a less-trusted fallback. Surface
`.func.gii` derivatives carry TR unreliably (`TimeStep` is often absent), so
hypline never reads TR from them. Since TR is acquisition-level, any fmriprep
`space`/`desc` variant of a run yields the same value. See
[../modules/bold.md](../modules/bold.md) for the full resolution order and why
the sidecar is preferred (raw imaging files are not required).

## Events files

fMRIPrep does not produce events.tsv — they pass through from raw BIDS.
Colocated with BOLD, named with shared **identity entities** (`sub`, `ses`,
`task`, `run`) only:

```
sub-01_ses-01_task-movie_run-1_events.tsv
```

`space`, `desc`, and other derivative-specific entities are **not** part of
the events filename — events describe the stimulus, not the derivative.
If events carry non-standard derivative entities (e.g. `..._space-X_desc-preproc_events.tsv`),
hypline raises rather than silently ignoring them — siblings sharing the run's identity
entities but not matching the canonical identity-only name are rejected. Files for a different
run or subject are ignored. Only the canonical file is read; the raise is specifically for
ambiguous same-run variants.

## Derivative variants

Multiple outputs share the same identity entities, distinguished by `desc`:

- `desc-preproc_bold` — preprocessed BOLD. fMRIPrep *can* produce fewer
  volumes than the raw image (`--dummy-scans` / auto non-steady-state
  trimming), but hypline rejects runs where derivative and raw counts differ —
  see [../modules/bold.md](../modules/bold.md).
- `desc-brain_mask` — brain mask
- `desc-confounds_timeseries` — confound regressors. Denoise reads this tsv
  (+ its JSON sidecar, needed for CompCor metadata) **directly** via
  `hypline.fmriprep`, selecting columns by `--columns`/`--compcor`; it is not
  converted into a hypline file first. Leading `n/a` cells in derivative columns
  (`*_derivative1`, `framewise_displacement`, `dvars`, `std_dvars`) are parsed as
  null and backfilled. See [../modules/denoise.md](../modules/denoise.md).
  - **Variable-count column families.** Most confound columns are a fixed,
    named set per fmriprep version (`trans_x`, motion derivatives/powers, etc.).
    The exceptions are `cosine*` (discrete-cosine drift basis; term count scales
    with run length / TR) and `motion_outlier*` (one column per scrubbed volume;
    count varies per run). These two cannot be enumerated a priori — their count
    is a property of the run, not the schema. CompCor (`a_comp_cor*`/`t_comp_cor*`)
    is the third variable-count family but is selected by its own `--compcor`
    grammar, not by name.
Denoise does **not** write into the fmriprep tree. Its denoised output
(`desc-denoised_bold`) lands in the separate `derivatives/hypline/` area to keep
fmriprep's `GeneratedBy` provenance honest — see
[../modules/denoise.md](../modules/denoise.md) and
[../decisions/layout.md](../decisions/layout.md).

## Surface vs volume

- Surface: `.func.gii`, one file per hemisphere (`hemi-L` / `hemi-R`).
- Volume: `.nii.gz`.

## BOLD file identification in bold_dir

A `bold_dir` contains many `.nii.gz` (and `.func.gii`) files that are **not**
BOLD time series — `boldref`, `brain_mask`, `dseg`, confound TSVs, etc. All
share the same `space-*` and identity entities. The BIDS suffix `_bold`
immediately before the extension is the only reliable discriminator:

- `..._bold.nii.gz` / `..._bold.func.gii` — BOLD time series
- `..._boldref.nii.gz`, `..._desc-brain_mask.nii.gz`, etc. — not BOLD

Discovery must match on the full trailing `_bold{ext}`, not just the extension.
