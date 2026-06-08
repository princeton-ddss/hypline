# `hypline denoise`

Regress nuisance regressors out of preprocessed BOLD, writing a denoised
derivative. `denoise` reads fMRIPrep's preprocessed BOLD together with the
nuisance regressors you select, and produces a `desc-denoised` BOLD under the
`derivatives/hypline/` tree.

```bash
hypline denoise <dataset-root> (--columns … | --compcor … | --custom-sources …) [OPTIONS]
```

Nuisance regressors come from two channels, stacked into one regressor matrix:

1. **fMRIPrep's confounds table**, read natively — `--columns` / `--compcor`
   select columns directly from each run's `desc-confounds_timeseries.tsv`. No
   intermediate hypline file is generated.
2. **Custom nuisance files** under `nuisance/` — `--custom-sources` /
   `--custom-columns` select run-level regressors you supply yourself (e.g.
   physiological recordings).

## Inputs

- **Preprocessed BOLD** (`desc-preproc`) under `derivatives/fmriprep/`.
- **fMRIPrep confounds table** (`desc-confounds_timeseries.tsv`, with its `.json`
  sidecar) beside the BOLD — read directly when you pass `--columns` /
  `--compcor`. The sidecar is always read (it carries the CompCor metadata), so
  it must be present even for a `--columns`-only run.
- **Custom nuisance files** under `nuisance/` — read only when you pass
  `--custom-sources`.

```
<dataset-root>/
├── derivatives/fmriprep/sub-01/func/
│   ├── sub-01_task-conv_run-1_space-fsaverage6_hemi-L_desc-preproc_bold.func.gii
│   ├── sub-01_task-conv_run-1_space-fsaverage6_hemi-R_desc-preproc_bold.func.gii
│   ├── sub-01_task-conv_run-1_desc-confounds_timeseries.tsv
│   └── sub-01_task-conv_run-1_desc-confounds_timeseries.json   # required with --columns/--compcor
└── nuisance/sub-01/physio-v1/                                  # optional, user-supplied
    └── sub-01_task-conv_run-1_nuis-physio_desc-v1_timeseries.tsv
```

## Options

| Option             | Description                                                                | Default               |
| ------------------ | -------------------------------------------------------------------------- | --------------------- |
| `--columns`        | Comma-separated fMRIPrep confound columns to regress out (see below)       | none                  |
| `--compcor`        | Comma-separated CompCor selectors (see below)                              | none                  |
| `--custom-sources` | Comma-separated `nuisance/` sources as `<kind>[-<desc>]`; requires `--custom-columns` | none       |
| `--custom-columns` | Column names to select from the `--custom-sources` files; requires `--custom-sources` | none       |
| `--space`          | BOLD space to clean: `fsaverage5`, `fsaverage6`, `MNI152NLin6Asym`, `MNI152NLin2009cAsym` | `MNI152NLin2009cAsym` |
| `--sub-ids`        | Comma-separated subject IDs to process; omit for all                       | all                   |
| `--data-filters`   | Narrow to specific runs/conditions — see [Segments and metadata](../concepts/segments.md) | none |
| `--force`          | Overwrite existing outputs (default skips them)                            | off                   |

At least one of `--columns`, `--compcor`, or `--custom-sources` is required — an
all-empty invocation raises rather than silently doing nothing.

**`--columns`** accepts exact column names from the fMRIPrep table (`trans_x`,
`rot_x`, …) plus *group prefixes* that expand to every matching column
(`cosine` → all cosine regressors, `motion_outlier` → all motion-outlier
columns).

**`--compcor`** selects principal components using `type:mask:n` tokens:

| Token       | Meaning                                                |
| ----------- | ------------------------------------------------------ |
| `a:CSF:5`   | top-5 **a**natomical CompCor components in the CSF mask |
| `a:WM:5`    | top-5 anatomical components in the white-matter mask   |
| `t::10`     | top-10 **t**emporal CompCor components (no mask)       |
| `a:CSF:0.5` | enough CSF components to explain 50% of variance       |

`type` is `a` (anatomical — mask required) or `t` (temporal — no mask). `n` is a
top-N integer, or a fraction between 0 and 1 for a variance threshold.

**`--custom-sources` / `--custom-columns`** pull run-level regressors from the
`nuisance/` area. Each source is a `<kind>` or `<kind>-<desc>` token naming a
`nuisance/<kind>[-<desc>]/` directory; `--custom-columns` then selects columns
from the horizontal concat of all named sources. The two must be given together.

!!! note "Authoring a `nuisance/` file"

    Custom nuisance files are yours to create — hypline never writes them. Each
    must be a **tab-separated `.tsv`** with the `_timeseries` suffix and a
    `nuis-<kind>` entity, e.g.
    `sub-01_task-conv_run-1_nuis-physio_desc-v1_timeseries.tsv`, placed in
    `nuisance/sub-01/<kind>[-<desc>]/`. It is a **wide** table: one named column
    per regressor, one row per TR (row count must match the BOLD run). Every
    value must be **finite** — unlike the fMRIPrep table, there is no `n/a`
    convention, so a blank or non-numeric cell raises rather than being filled.

!!! warning "Things that must line up"

    - **Space must exist.** `--space` must name a space present in your fMRIPrep
      outputs. Surface spaces (`fsaverage*`) are cleaned per hemisphere
      automatically.
    - **Column names must be unique** across all channels (fMRIPrep and custom);
      a collision raises rather than silently dropping one.
    - **TR counts must match.** Every selected regressor channel and the BOLD
      must have the same number of TRs, or `denoise` raises.

## Example

Clean surface BOLD for all subjects, regressing out a motion + drift model:

```bash
hypline denoise data/ \
  --space fsaverage6 \
  --columns trans_x,trans_y,trans_z,rot_x,rot_y,rot_z,cosine
```

Add CompCor components, and a custom physiological regressor from `nuisance/`:

```bash
hypline denoise data/ \
  --space fsaverage6 \
  --columns trans_x,trans_y,trans_z,rot_x,rot_y,rot_z,cosine \
  --compcor a:CSF:5,a:WM:5 \
  --custom-sources physio-v1 --custom-columns resp,cardiac
```

Clean only run 1 of subjects 01 and 02:

```bash
hypline denoise data/ \
  --columns trans_x,trans_y,trans_z,rot_x,rot_y,rot_z,cosine \
  --sub-ids 01,02 \
  --data-filters run-1
```

## Outputs

The denoised BOLD is written to its own **`derivatives/hypline/`** tree, mirroring
fMRIPrep's `sub-XX/[ses-YY/]func/` shape and preserving the source's full BOLD
identity — only the `desc` entity (`desc-denoised`) and the root differ from the
`desc-preproc` source:

```
<dataset-root>/derivatives/hypline/sub-01/func/
├── sub-01_task-conv_run-1_space-fsaverage6_hemi-L_desc-denoised_bold.func.gii    # output
├── sub-01_task-conv_run-1_space-fsaverage6_hemi-L_desc-denoised_bold.json        # sidecar
├── sub-01_task-conv_run-1_space-fsaverage6_hemi-R_desc-denoised_bold.func.gii    # output
└── sub-01_task-conv_run-1_space-fsaverage6_hemi-R_desc-denoised_bold.json        # sidecar
```

The output keeps the input's dimensions — only the signal values change. Each run
gets a per-file `.json` sidecar recording its `Sources` (a `bids:` URI to the
`desc-preproc` input), the resolved cleaning settings, and the hypline version.
On first output, hypline stamps a `derivatives/hypline/dataset_description.json`
(written once, left alone thereafter).

Denoised BOLD lives in its own tree rather than beside its fMRIPrep source
because denoising is hypline's own pipeline, not a continuation of fMRIPrep — a
separate tree carries an honest `GeneratedBy: hypline` provenance instead of
inheriting fMRIPrep's.

## Common errors

| What you see | Cause | Fix |
| ------------ | ----- | --- |
| `at least one of --columns, --compcor, or --custom-sources must be given` | The command was called with no regressor channel selected. | Pass at least one of `--columns`, `--compcor`, or `--custom-sources`. |
| `--custom-sources and --custom-columns must be given together` | One of the custom-nuisance options was passed without the other. | Supply both, or neither. |
| A `--custom-sources` source resolves to 0 (or multiple) files | A source names a `nuisance/<kind>[-<desc>]/` directory that does not exist (or matches more than one file per run). | Check the source spelling against your `nuisance/` directories. |
| `Unequal number of TRs between BOLD and nuisance` | A regressor channel has a different row count than the BOLD it is paired with. | Confirm the fMRIPrep confounds table and any custom nuisance files span every TR of the run. |
| Command finishes, but no `desc-denoised` files appear | `--space` names a valid space that is **absent** from your fMRIPrep outputs, so nothing matched. | Pass a `--space` you actually preprocessed (check the `space-` entity on your `desc-preproc` files). |
| `No subjects found — nothing to denoise` | No subjects under `derivatives/fmriprep/`, or `--sub-ids` / `--data-filters` excluded them all. | Confirm fMRIPrep outputs exist and that your filters are not too narrow. |
