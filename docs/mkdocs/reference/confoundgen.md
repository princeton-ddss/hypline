# `hypline confoundgen`

Generate **confound files** — TR-aligned, stimulus-derived regressors keyed to
BOLD runs. `confoundgen` is a group of subcommands, one per confound source.

```bash
hypline confoundgen <source> <dataset-root> [OPTIONS]
```

| Subcommand | Source                                              |
| ---------- | --------------------------------------------------- |
| `phonemic` | timing of phonemic features (onset indicator, rate) |

Each confound file is a Parquet table already aligned to the BOLD TR grid.

!!! note "fMRIPrep regressors are not generated here"

    Motion, drift, and CompCor regressors are **not** confound files. `denoise`
    reads them straight from fMRIPrep's confounds table — see
    [`denoise --columns` / `--compcor`](denoise.md). `confounds/` holds only
    stimulus-derived, feature-granular confounds.

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
<dataset-root>/features/dyad-101/phonemic/
└── dyad-101_task-conv_run-1_feat-phonemic.parquet
```

### Options

| Option           | Description                                                       | Default |
| ---------------- | ---------------------------------------------------------------- | ------- |
| `--dyad-ids`     | Comma-separated dyad IDs to process; omit for all                | all     |
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
<dataset-root>/confounds/dyad-101/
├── phonemic-onset/
│   └── dyad-101_task-conv_run-1_conf-phonemic_desc-onset.parquet   # speech-onset indicator
└── phonemic-rate/
    └── dyad-101_task-conv_run-1_conf-phonemic_desc-rate.parquet    # speech rate per TR
```

Each derivation is referred to by name (`phonemic-onset`, `phonemic-rate`). These
are feature-granular, encoding-side confounds; they are not read by
[`denoise`](denoise.md), whose nuisance regressors come from fMRIPrep and the
`nuisance/` area instead.
