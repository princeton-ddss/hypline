# `hypline denoise`

Regress nuisance confounds out of preprocessed BOLD, writing a cleaned
derivative. This is where the two pipeline branches meet: `denoise` reads
fMRIPrep's preprocessed BOLD and the [confound files](confoundgen.md) you
generated, and produces a `desc-clean` BOLD ready for an encoding model.

```bash
hypline denoise <dataset-root> --confounds <refs> [OPTIONS]
```

## Inputs

Two things, both already in your dataset:

- **Preprocessed BOLD** (`desc-preproc`) under `derivatives/fmriprep/`.
- **Confound files** under `confounds/`, named by `--confounds`.

```
<dataset-root>/
├── derivatives/fmriprep/sub-01/func/
│   ├── sub-01_task-conv_run-1_space-fsaverage6_hemi-L_desc-preproc_bold.func.gii
│   └── sub-01_task-conv_run-1_space-fsaverage6_hemi-R_desc-preproc_bold.func.gii
└── confounds/sub-01/
    ├── fmriprep-minimal/...conf-fmriprep_desc-minimal.parquet
    └── phonemic-onset/...conf-phonemic_desc-onset.parquet
```

## Options

| Option           | Description                                                                | Default               |
| ---------------- | -------------------------------------------------------------------------- | --------------------- |
| `--confounds`    | Comma-separated confound refs to regress out (see below) **(required)**    | —                     |
| `--space`        | BOLD space to clean: `fsaverage5`, `fsaverage6`, `MNI152NLin6Asym`, `MNI152NLin2009cAsym` | `MNI152NLin2009cAsym` |
| `--sub-ids`      | Comma-separated subject IDs to process; omit for all                       | all                   |
| `--data-filters` | Narrow to specific runs/conditions — see [Segments and metadata](../concepts/segments.md) | none |
| `--force`        | Overwrite existing outputs (default skips them)                            | off                   |

**`--confounds`** selects which confound files to load, by reference. Each
reference is a `<kind>` or `<kind>-<desc>` token matching the files
[`confoundgen`](confoundgen.md) wrote:

| Ref                 | Resolves to                                  |
| ------------------- | -------------------------------------------- |
| `fmriprep-minimal`  | `conf-fmriprep_desc-minimal`                 |
| `phonemic-onset`    | `conf-phonemic_desc-onset`                   |
| `phonemic`          | the bare `conf-phonemic` (no `desc`)         |

All referenced confounds are stacked into one regressor matrix. A bare `<kind>`
resolves only to the `desc`-less file, so name each derivation explicitly.

!!! warning "Two things must line up"

    - **Space must exist.** `--space` must name a space present in your fMRIPrep
      outputs. Surface spaces (`fsaverage*`) are cleaned per hemisphere
      automatically.
    - **TR counts must match.** Every referenced confound file and the BOLD must
      have the same number of TRs, or `denoise` raises.

## Example

Clean surface BOLD for all subjects, regressing out a motion bundle plus the
phonemic onset confound:

```bash
hypline denoise data/ \
  --space fsaverage6 \
  --confounds fmriprep-minimal,phonemic-onset
```

Clean only run 1 of subjects 01 and 02:

```bash
hypline denoise data/ \
  --confounds fmriprep-full \
  --sub-ids 01,02 \
  --data-filters run-1
```

## Outputs

The cleaned BOLD is written **in place**, in the same fMRIPrep `func/` directory
as its `desc-preproc` source, differing only in the `desc` entity
(`desc-clean`):

```
<dataset-root>/derivatives/fmriprep/sub-01/func/
├── sub-01_task-conv_run-1_space-fsaverage6_hemi-L_desc-preproc_bold.func.gii   # input
├── sub-01_task-conv_run-1_space-fsaverage6_hemi-L_desc-clean_bold.func.gii     # output
├── sub-01_task-conv_run-1_space-fsaverage6_hemi-R_desc-preproc_bold.func.gii   # input
└── sub-01_task-conv_run-1_space-fsaverage6_hemi-R_desc-clean_bold.func.gii     # output
```

The output keeps the input's dimensions and metadata — only the signal values
change. Cleaned BOLD lives beside its source because it continues the fMRIPrep
pipeline and shares the run's identity. An encoding model reads `desc-clean` by
default.

## Common errors

| What you see | Cause | Fix |
| ------------ | ----- | --- |
| `Expected one '<ref>' confound for ..., found 0` | A `--confounds` ref names a file `confoundgen` never wrote (wrong `desc`, or that step was skipped). | Check the ref spelling against the [`confoundgen` outputs](confoundgen.md); a bare `phonemic` matches only the `desc`-less file, so use `phonemic-onset` / `phonemic-rate`. |
| `Unequal number of TRs between BOLD and confounds` | A confound table has a different row count than the BOLD it is paired with. | Regenerate that confound against the same run; fMRIPrep and phonemic confounds must both span every TR. |
| Command finishes, but no `desc-clean` files appear | `--space` names a valid space that is **absent** from your fMRIPrep outputs, so nothing matched. | Pass a `--space` you actually preprocessed (check the `space-` entity on your `desc-preproc` files). |
| `No subjects found — nothing to denoise` | No subjects under `derivatives/fmriprep/`, or `--sub-ids` / `--data-filters` excluded them all. | Confirm fMRIPrep outputs exist and that your filters are not too narrow. |
