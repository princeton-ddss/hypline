# `hypline confoundgen`

Generate **confound files** — TR-aligned nuisance regressors that
[`denoise`](denoise.md) regresses out of the BOLD signal. `confoundgen` is a
group of subcommands, one per confound source.

```bash
hypline confoundgen <source> <dataset-root> [OPTIONS]
```

| Subcommand | Source                                                  |
| ---------- | ------------------------------------------------------- |
| `fmriprep` | columns from fMRIPrep's confounds table                 |
| `phonemic` | timing of phonemic features (onset indicator, rate)     |

All confound files share one on-disk format regardless of source, so `denoise`
can mix them freely. Each is a Parquet table already aligned to the BOLD TR grid.

---

## `confoundgen fmriprep`

Import selected columns from fMRIPrep's `desc-confounds_timeseries.tsv` into a
named hypline confound bundle (`conf-fmriprep`). A bundle is a single Parquet
file per run that stacks all the columns you selected into one regressor table.
This converts fMRIPrep's regressors into hypline's format so they reach
`denoise` through the same path as every other confound. It reads only the
confounds table — never the BOLD.

### Inputs

fMRIPrep's per-run confounds table under `derivatives/fmriprep/`:

```
<dataset-root>/derivatives/fmriprep/sub-01/func/
├── sub-01_task-conv_run-1_desc-confounds_timeseries.tsv
└── sub-01_task-conv_run-1_desc-confounds_timeseries.json   # needed for CompCor
```

### Options

| Option           | Description                                                                 | Default |
| ---------------- | --------------------------------------------------------------------------- | ------- |
| `--desc`         | Label naming this bundle (alphanumeric), e.g. `--desc minimal` → `conf-fmriprep_desc-minimal` **(required)** | — |
| `--columns`      | Comma-separated confound columns to pull (see below)                        | none    |
| `--compcor`      | Comma-separated CompCor selectors (see below)                               | none    |
| `--sub-ids`      | Comma-separated subject IDs to process; omit for all                        | all     |
| `--data-filters` | Narrow to specific runs/conditions — see [Segments and metadata](../concepts/segments.md) | none |
| `--force`        | Overwrite existing outputs (default skips them)                             | off     |

At least one of `--columns` or `--compcor` is required.

**`--columns`** accepts exact column names from the fMRIPrep table
(`trans_x`, `rot_x`, …) plus *group prefixes* that expand to every matching
column (`cosine` → all cosine regressors, `motion_outlier` → all motion-outlier
columns).

**`--compcor`** selects principal components using `type:mask:n` tokens:

| Token       | Meaning                                              |
| ----------- | ---------------------------------------------------- |
| `a:CSF:5`   | top-5 **a**natomical CompCor components in the CSF mask |
| `a:WM:5`    | top-5 anatomical components in the white-matter mask |
| `t::10`     | top-10 **t**emporal CompCor components (no mask)     |
| `a:CSF:0.5` | enough CSF components to explain 50% of variance     |

`type` is `a` (anatomical — mask required) or `t` (temporal — no mask). `n` is a
top-N integer, or a fraction between 0 and 1 for a variance threshold.

### Example

A minimal motion model (6 motion parameters + cosine drift):

```bash
hypline confoundgen fmriprep data/ --desc minimal \
  --columns trans_x,trans_y,trans_z,rot_x,rot_y,rot_z,cosine
```

A fuller model adding CompCor components:

```bash
hypline confoundgen fmriprep data/ --desc full \
  --columns trans_x,trans_y,trans_z,rot_x,rot_y,rot_z,cosine \
  --compcor a:CSF:5,a:WM:5
```

### Outputs

One bundle per run, tagged `conf-fmriprep` and the `--desc` label. Each bundle
lives in its own `fmriprep-<label>/` subdirectory:

```
<dataset-root>/confounds/sub-01/
├── fmriprep-minimal/
│   └── sub-01_task-conv_run-1_conf-fmriprep_desc-minimal.parquet
└── fmriprep-full/
    └── sub-01_task-conv_run-1_conf-fmriprep_desc-full.parquet
```

---

## `confoundgen phonemic`

Generate phonemic confounds from phonemic feature files. These capture *when*
speech occurred, not its content — an **onset** indicator and a speech **rate**
regressor, derived purely from phoneme timing.

!!! tip "Usually automatic"

    [`featuregen phonemic`](featuregen.md) runs this step for you by default.
    Call `confoundgen phonemic` directly only to regenerate confounds without
    regenerating features.

### Inputs

Phonemic feature files produced by [`featuregen phonemic`](featuregen.md):

```
<dataset-root>/features/sub-01/phonemic/
└── sub-01_task-conv_run-1_feat-phonemic.parquet
```

### Options

| Option           | Description                                                       | Default |
| ---------------- | ---------------------------------------------------------------- | ------- |
| `--sub-ids`      | Comma-separated subject IDs to process; omit for all             | all     |
| `--data-filters` | Narrow to specific runs/conditions — see [Segments and metadata](../concepts/segments.md) | none |
| `--force`        | Overwrite existing outputs (default skips them)                  | off     |

### Example

```bash
hypline confoundgen phonemic data/
```

### Outputs

Two derivations per run, tagged `conf-phonemic` and distinguished by `desc`.
Each `desc` lands in its own subdirectory:

```
<dataset-root>/confounds/sub-01/
├── phonemic-onset/
│   └── sub-01_task-conv_run-1_conf-phonemic_desc-onset.parquet   # speech-onset indicator
└── phonemic-rate/
    └── sub-01_task-conv_run-1_conf-phonemic_desc-rate.parquet    # speech rate per TR
```

Refer to a derivation by name when denoising: `phonemic-onset`, `phonemic-rate`
(see [`denoise --confounds`](denoise.md)).
