# `hypline featuregen`

Generate stimulus-derived **features** — the predictors (X) an encoding model
maps onto the BOLD signal. `featuregen` is a group of subcommands, one per
feature kind.

```bash
hypline featuregen <kind> <dataset-root> [OPTIONS]
```

| Subcommand | Generates                                            |
| ---------- | ---------------------------------------------------- |
| `phonemic` | phoneme-level articulatory features from transcripts |
| `semantic` | contextual word embeddings from a Hugging Face causal LM |

---

## `featuregen phonemic`

Derive phoneme-level features from word-level transcripts. Each word is looked
up in [CMUdict](http://www.speech.cs.cmu.edu/cgi-bin/cmudict) to get its
phonemes, and each phoneme is represented by its articulatory features (place,
manner, voicing, …).

### Inputs

Transcripts produced by [`transcribe`](transcribe.md), under `stimuli/`:

```
<dataset-root>/stimuli/dyad-103/ses-1/transcript/
└── dyad-103_ses-1_task-conv_run-1_transcript.csv
```

### Options

| Option              | Description                                                       | Default |
| ------------------- | ---------------------------------------------------------------- | ------- |
| `--no-articulatory` | Use a plain phoneme identity vector instead of articulatory features | off |
| `--desc`            | Tag outputs as a named variant (alphanumeric), e.g. `--desc v2` → `desc-v2` | none |
| `--skip-confoundgen`| Write features only; do **not** also generate phonemic confounds | off     |
| `--dyad-ids`        | Comma-separated dyad IDs to process; omit for all                | all     |
| `--data-filters`    | Narrow to specific runs/conditions — see [Segments and metadata](../concepts/segments.md) | none |
| `--force`           | Overwrite existing outputs (default skips them)                  | off     |

!!! tip "Features and confounds together"

    By default, `featuregen phonemic` also runs [`confoundgen phonemic`](confoundgen.md)
    on the features it just wrote — generating the timing-based phonemic
    confounds in the same pass. This is the common path; pass
    `--skip-confoundgen` if you want features without confounds.

### Example

Generate phonemic features (and their confounds) for all dyads:

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
├── features/dyad-103/ses-1/phonemic/
│   └── dyad-103_ses-1_task-conv_run-1_feat-phonemic.parquet
└── confounds/dyad-103/ses-1/                            # from the chained confoundgen
    ├── phonemic-onset/
    │   └── dyad-103_ses-1_task-conv_run-1_conf-phonemic_desc-onset.parquet
    └── phonemic-rate/
        └── dyad-103_ses-1_task-conv_run-1_conf-phonemic_desc-rate.parquet
```

A `--desc` label lands as `desc-<label>` and lives in its own subdirectory
(`phonemic-<label>/`), keeping variants separate. See
[The hypline dataset layout](../concepts/layout.md#variants-with-desc).

!!! note "Feature file format"

    Feature files are Parquet tables. Each row is one phoneme with a
    `start_time` (seconds from the start of the stimulus) and a `feature` vector.
    Timing is the phoneme's word onset — hypline does not yet have sub-word
    audio alignment, so phonemes within a word share that word's onset.

---

## `featuregen semantic`

Derive **contextual word embeddings** from any Hugging Face causal (decoder) LM.
The transcript is tokenized and run through the model; each sub-word token's
hidden state at a chosen `--layer` becomes its `feature` vector. The model id is
passed verbatim to `from_pretrained`, so the hub is open and unbounded — any
causal LM with a fast (Rust) tokenizer and a BOS token works.

### Inputs

The same transcripts as [`featuregen phonemic`](#featuregen-phonemic), under
`stimuli/`. Untimed words (null `start_time`) are retained as real LM context
but carry their null timing into the output; null-`word` rows are skipped.

The whole transcript is encoded in one forward pass, so it must fit the model's
context window (tokens + a BOS prefix ≤ `max_position_embeddings`). A long
transcript on a short-context model (gpt-2 caps at 1024) raises rather than
truncating — reach for a longer-context LM instead.

### Options

| Option               | Description                                                       | Default |
| -------------------- | ---------------------------------------------------------------- | ------- |
| `--model`            | **Required.** Hugging Face causal-LM id (e.g. `gpt2-xl`, `meta-llama/Llama-3.2-1B`) | — |
| `--model-dir`        | Cache dir for downloaded weights                                 | `~/.cache/hypline/huggingface` |
| `--device`           | Hardware target (`cpu` or `cuda`)                                | `cpu`   |
| `--layer`            | Hidden-layer index in `0..num_hidden_layers`; omit for the middle layer | middle |
| `--desc`             | Tag outputs as a named variant (alphanumeric), e.g. `--desc v2` → `desc-v2` | none |
| `--skip-confoundgen` | Write features only; do **not** also generate semantic confounds | off     |
| `--dyad-ids`         | Comma-separated dyad IDs to process; omit for all                | all     |
| `--data-filters`     | Narrow to specific runs/conditions — see [Segments and metadata](../concepts/segments.md) | none |
| `--force`            | Overwrite existing outputs (default skips them)                  | off     |

!!! tip "Features and confounds together"

    As with `phonemic`, `featuregen semantic` also runs
    [`confoundgen semantic`](confoundgen.md) by default. Pass `--skip-confoundgen`
    for features without confounds.

### Example

Generate gpt-2 semantic features (and their confounds) for all dyads:

```bash
hypline featuregen semantic data/ --model gpt2-xl
```

### Outputs

A semantic feature file per transcript, tagged `feat-semantic`, under
`features/`. Each row carries `start_time`, `word`, `token`, the `feature`
vector, and — for any non-zero layer — per-token LM metrics (`rank`,
`true_prob`, `entropy`). The Parquet footer records `hf_model`, `hf_tokenizer`
(equal to `hf_model` unless overridden via the [Python API](python-api.md)), and
`layer`. With `--skip-confoundgen` omitted, the matching `conf-semantic`
confounds appear too (see [`confoundgen`](confoundgen.md)):

```
<dataset-root>/
├── features/dyad-103/ses-1/semantic/
│   └── dyad-103_ses-1_task-conv_run-1_feat-semantic.parquet
└── confounds/dyad-103/ses-1/                            # from the chained confoundgen
    ├── semantic-onset/
    │   └── dyad-103_ses-1_task-conv_run-1_conf-semantic_desc-onset.parquet
    └── semantic-rate/
        └── dyad-103_ses-1_task-conv_run-1_conf-semantic_desc-rate.parquet
```

!!! note "Causal LMs only"

    The model must be a causal/decoder LM with a fast tokenizer and a BOS token.
    Encoder checkpoints (BERT and the like) are rejected up front — `from_pretrained`
    would silently load them and emit garbage.
