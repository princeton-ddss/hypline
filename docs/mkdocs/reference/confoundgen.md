# `hypline confoundgen`

Generate **confound files** — TR-aligned, stimulus-derived regressors keyed to
BOLD runs. `confoundgen` is a group of subcommands, one per confound source.

```bash
hypline confoundgen <source> <dataset-root> [OPTIONS]
```

| Subcommand | Source                                              |
| ---------- | --------------------------------------------------- |
| `phonemic` | timing of phonemic features (onset indicator, rate) |
| `semantic` | timing of semantic features (onset indicator, rate) |

Each confound file is a Parquet table already aligned to the BOLD TR grid.

!!! note "fMRIPrep regressors are not generated here"

    Motion, drift, and CompCor regressors are **not** confound files. `denoise`
    reads them straight from fMRIPrep's confounds table — see
    [`denoise --columns` / `--compcor`](denoise.md). `confounds/` holds only
    stimulus-derived, feature-granular confounds.

!!! note "Why there is no `desc` option"

    Generated confounds capture *when* speech occurred, not its content — they
    depend only on feature timing, never the feature values. So semantic features
    from two different models, `semantic-gpt2xl` and `semantic-llama`, yield the
    same `semantic-onset` and `semantic-rate`; there is no variant to choose.

    Need a confound that depends on variant-specific data (e.g. the feature values
    themselves)? Derive it yourself and write it with
    [`save_confound`](python-api.md#hypline.save_confound), choosing your own
    `desc`.

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
<dataset-root>/features/dyad-103/ses-1/phonemic/
└── dyad-103_ses-1_task-conv_run-1_feat-phonemic.parquet
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
<dataset-root>/confounds/dyad-103/ses-1/
├── phonemic-onset/
│   └── dyad-103_ses-1_task-conv_run-1_conf-phonemic_desc-onset.parquet   # speech-onset indicator
└── phonemic-rate/
    └── dyad-103_ses-1_task-conv_run-1_conf-phonemic_desc-rate.parquet    # speech rate per TR
```

Each derivation is referred to by name (`phonemic-onset`, `phonemic-rate`). These
are feature-granular, encoding-side confounds; they are not read by
[`denoise`](denoise.md), whose nuisance regressors come from fMRIPrep and the
`nuisance/` area instead.

---

## `confoundgen semantic`

Generate semantic confounds from semantic feature files — the same **onset**
indicator and speech **rate** regressor as `phonemic`, derived from semantic
feature timing rather than phoneme timing.

!!! tip "Usually automatic"

    [`featuregen semantic`](featuregen.md) runs this step for you by default.
    Call `confoundgen semantic` directly only to regenerate confounds without
    regenerating features.

### Inputs

Semantic feature files produced by [`featuregen semantic`](featuregen.md):

```
<dataset-root>/features/dyad-103/ses-1/semantic/
└── dyad-103_ses-1_task-conv_run-1_feat-semantic.parquet
```

### Options

Identical to [`confoundgen phonemic`](#confoundgen-phonemic): `--dyad-ids`,
`--data-filters`, `--force`.

### Example

```bash
hypline confoundgen semantic data/
```

### Outputs

Two derivations per run, tagged `conf-semantic` and distinguished by `desc`,
each in its own subdirectory:

```
<dataset-root>/confounds/dyad-103/ses-1/
├── semantic-onset/
│   └── dyad-103_ses-1_task-conv_run-1_conf-semantic_desc-onset.parquet   # speech-onset indicator
└── semantic-rate/
    └── dyad-103_ses-1_task-conv_run-1_conf-semantic_desc-rate.parquet    # speech rate per TR
```
