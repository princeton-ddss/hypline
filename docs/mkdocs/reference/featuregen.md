# `hypline featuregen`

Generate stimulus-derived **features** — the predictors (X) an encoding model
maps onto the BOLD signal. `featuregen` is a group of subcommands, one per
feature kind.

```bash
hypline featuregen <kind> <dataset-root> [OPTIONS]
```

| Subcommand | Generates                                          |
| ---------- | -------------------------------------------------- |
| `phonemic` | phoneme-level articulatory features from transcripts |

---

## `featuregen phonemic`

Derive phoneme-level features from word-level transcripts. Each word is looked
up in [CMUdict](http://www.speech.cs.cmu.edu/cgi-bin/cmudict) to get its
phonemes, and each phoneme is represented by its articulatory features (place,
manner, voicing, …).

### Inputs

Transcripts produced by [`transcribe`](transcribe.md), under `stimuli/`:

```
<dataset-root>/stimuli/sub-01/transcript/
└── sub-01_task-conv_run-1_stim-transcript.csv
```

### Options

| Option              | Description                                                       | Default |
| ------------------- | ---------------------------------------------------------------- | ------- |
| `--no-articulatory` | Use a plain phoneme identity vector instead of articulatory features | off |
| `--desc`            | Tag outputs as a named variant (alphanumeric), e.g. `--desc v2` → `desc-v2` | none |
| `--skip-confoundgen`| Write features only; do **not** also generate phonemic confounds | off     |
| `--sub-ids`         | Comma-separated subject IDs to process; omit for all             | all     |
| `--data-filters`    | Narrow to specific runs/conditions — see [Segments and metadata](../concepts/segments.md) | none |
| `--force`           | Overwrite existing outputs (default skips them)                  | off     |

!!! tip "Features and confounds together"

    By default, `featuregen phonemic` also runs [`confoundgen phonemic`](confoundgen.md)
    on the features it just wrote — generating the timing-based phonemic
    confounds in the same pass. This is the common path; pass
    `--skip-confoundgen` if you want features without confounds.

### Example

Generate phonemic features (and their confounds) for all subjects:

```bash
hypline featuregen phonemic data/
```

Generate a second variant, features only — `--no-articulatory` swaps the
articulatory vector for a plain phoneme identity vector, and `--desc identity`
names the variant after it so it sits beside the default features:

```bash
hypline featuregen phonemic data/ --desc identity --no-articulatory --skip-confoundgen
```

### Outputs

A phonemic feature file per transcript, tagged `feat-phonemic`, under
`features/`. With `--skip-confoundgen` omitted, the matching `conf-phonemic`
confounds appear too (see [`confoundgen`](confoundgen.md)):

```
<dataset-root>/
├── features/sub-01/phonemic/
│   └── sub-01_task-conv_run-1_feat-phonemic.parquet
└── confounds/sub-01/                                    # from the chained confoundgen
    ├── phonemic-onset/
    │   └── sub-01_task-conv_run-1_conf-phonemic_desc-onset.parquet
    └── phonemic-rate/
        └── sub-01_task-conv_run-1_conf-phonemic_desc-rate.parquet
```

A `--desc` label lands as `desc-<label>` and lives in its own subdirectory
(`phonemic-<label>/`), keeping variants separate. See
[The hypline dataset layout](../concepts/layout.md#variants-with-desc).

!!! note "Feature file format"

    Feature files are Parquet tables. Each row is one phoneme with a
    `start_time` (seconds from the start of the stimulus) and a `feature` vector.
    Timing is the phoneme's word onset — hypline does not yet have sub-word
    audio alignment, so phonemes within a word share that word's onset.
