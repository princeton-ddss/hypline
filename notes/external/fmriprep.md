# fMRIPrep — external spec reference

fMRIPrep output behaviors that hypline code specifically depends on.

fMRIPrep outputs follow the BIDS-Derivatives spec. Relevant behaviors we
depend on:

## Repetition time (TR)

Stored in a colocated JSON sidecar (`..._bold.json`) under `"RepetitionTime"`.
The sidecar is the authoritative source — reading it avoids loading the
NIfTI/GIfTI header. Fall back to the image header only if the sidecar is
missing.

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
`load_events` raises rather than silently ignoring them — it scans siblings in the same
directory and raises if any share the run's identity entities but are not the canonical name.
Files for a different run or subject are ignored. Only the canonical file is read; the raise
is specifically for ambiguous same-run variants.

## Derivative variants

Multiple outputs share the same identity entities, distinguished by `desc`:

- `desc-preproc_bold` — preprocessed BOLD
- `desc-brain_mask` — brain mask
- `desc-confounds_timeseries` — confound regressors

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
