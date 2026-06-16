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
| `spectral` | Whisper log-Mel spectrogram from stimulus audio, aligned to the BOLD TR grid |
| `syntactic` | per-token POS, dependency, and stopword features from transcripts |

---

## `featuregen phonemic`

Derive phoneme-level features from word-level transcripts. Each word is looked
up in [CMUdict](http://www.speech.cs.cmu.edu/cgi-bin/cmudict) to get its
phonemes, and each phoneme is represented by its articulatory features (place,
manner, voicing, …).

### Inputs

Transcripts produced by [`transcribe`](transcribe.md), under `stimuli/`:

```
<dataset-root>/stimuli/dyad-030/ses-1/transcript/
└── dyad-030_ses-1_task-conv_run-1_transcript.csv
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
├── features/dyad-030/ses-1/phonemic/
│   └── dyad-030_ses-1_task-conv_run-1_feat-phonemic.parquet
└── confounds/dyad-030/ses-1/                            # from the chained confoundgen
    ├── phonemic-onset/
    │   └── dyad-030_ses-1_task-conv_run-1_conf-phonemic_desc-onset.parquet
    └── phonemic-rate/
        └── dyad-030_ses-1_task-conv_run-1_conf-phonemic_desc-rate.parquet
```

A `--desc` label lands as `desc-<label>` and lives in its own subdirectory
(`phonemic-<label>/`), keeping variants separate. See
[The hypline dataset layout](../concepts/layout.md#variants-with-desc).

!!! note "Feature file format"

    Feature files are Parquet tables. Each row is one phoneme with a
    `start_time` (seconds from the start of the stimulus), a `turn_sub` label
    (forward-filled from the transcript; carried through unchanged), and a
    `feature` vector.
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
`features/`. Each row carries `start_time`, `turn_sub` (forward-filled from the
transcript; carried through unchanged), `word`, `token`, the `feature`
vector, and — for any non-zero layer — per-token LM metrics (`rank`,
`true_prob`, `entropy`). The Parquet footer records `hf_model`, `hf_tokenizer`
(equal to `hf_model` unless overridden via the [Python API](python-api.md)), and
`layer`. With `--skip-confoundgen` omitted, the matching `conf-semantic`
confounds appear too (see [`confoundgen`](confoundgen.md)):

```
<dataset-root>/
├── features/dyad-030/ses-1/semantic/
│   └── dyad-030_ses-1_task-conv_run-1_feat-semantic.parquet
└── confounds/dyad-030/ses-1/                            # from the chained confoundgen
    ├── semantic-onset/
    │   └── dyad-030_ses-1_task-conv_run-1_conf-semantic_desc-onset.parquet
    └── semantic-rate/
        └── dyad-030_ses-1_task-conv_run-1_conf-semantic_desc-rate.parquet
```

!!! note "Causal LMs only"

    The model must be a causal/decoder LM with a fast tokenizer and a BOS token.
    Encoder checkpoints (BERT and the like) are rejected up front — `from_pretrained`
    would silently load them and emit garbage.

!!! note "Gated models"

    Some models (e.g. Llama) are license-gated. Request access on the model's
    Hugging Face page, then set an [access token](https://huggingface.co/settings/tokens)
    so the download can authenticate:

    ```bash
    export HF_TOKEN=hf_...
    hypline featuregen semantic data/ --model meta-llama/Llama-3.2-1B
    ```

    `transformers` reads `HF_TOKEN` automatically — no extra flag. Access must be
    granted to the same account that issued the token, or the download fails.

---

## `featuregen spectral`

Derive a **log-Mel spectrogram** from the stimulus audio using a
[Whisper](https://github.com/openai/whisper) feature extractor — the same
front-end that turns audio into the input Whisper's encoder sees. Unlike
`phonemic` and `semantic`, this reads audio directly (no transcript needed) and
its output is **pre-aligned to the run's BOLD TR grid**: one log-Mel vector per
TR, ready to feed an encoding model without a downstream binning step.

### Inputs

Stimulus audio under `stimuli/` — the same files [`transcribe`](transcribe.md)
reads, selected by `--audio-ext`:

```
<dataset-root>/stimuli/dyad-030/ses-1/audio/
└── dyad-030_ses-1_task-conv_run-1_audio.wav
```

Aligning to the TR grid needs the run's BOLD timing (TR and number of frames).
The dyad's partners share one simultaneous scan, so any partner's BOLD supplies
it. A dyad with no resolvable BOLD raises.

### Options

| Option           | Description                                                       | Default |
| ---------------- | ---------------------------------------------------------------- | ------- |
| `--audio-ext`    | Extension of the audio files, e.g. `.wav` **(required)**         | —       |
| `--model`        | Whisper model whose extractor produces the spectrogram: `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3` | `tiny` |
| `--model-dir`    | Cache dir for downloaded weights                                 | `~/.cache/hypline/huggingface` |
| `--desc`         | Tag outputs as a named variant (alphanumeric), e.g. `--desc v2` → `desc-v2` | none |
| `--dyad-ids`     | Comma-separated dyad IDs to process; omit for all                | all     |
| `--data-filters` | Narrow to specific runs/conditions — see [Segments and metadata](../concepts/segments.md) | none |
| `--force`        | Overwrite existing outputs (default skips them)                  | off     |

!!! note "No `--device`, no confounds"

    The Whisper feature extractor is a CPU Mel transform with no forward pass, so
    there is no `--device` option. And unlike `phonemic` and `semantic`, spectral
    has no chained `confoundgen` step — it writes features only.

### Example

Generate spectral features for all dyads with the default `tiny` extractor:

```bash
hypline featuregen spectral data/ --audio-ext .wav
```

### Outputs

A spectral feature file per stimulus, tagged `feat-spectral`, under `features/`:

```
<dataset-root>/features/dyad-030/ses-1/spectral/
└── dyad-030_ses-1_task-conv_run-1_feat-spectral.parquet
```

A `--desc` label lands as `desc-<label>` in its own subdirectory
(`spectral-<label>/`). See
[The hypline dataset layout](../concepts/layout.md#variants-with-desc).

!!! note "Feature file format"

    Unlike the per-word/per-phoneme feature files, spectral rows are
    **per-TR**: each row is one TR with its `start_time` (seconds from the start
    of the stimulus) and a log-Mel `feature` vector (length = the model's number
    of Mel bands). The Parquet footer records the `model`, `sampling_rate`,
    `hop_length`, `n_mels`, `chunk_length`, `repetition_time`, and
    `downsample_method`.

---

## `featuregen syntactic`

Derive **per-token syntactic-function features** from word-level transcripts with
[spaCy](https://spacy.io/). Each token's feature is a fixed-width one-hot of its
POS tag concatenated with its dependency relation, plus a final 0/1 stopword
dimension. The model is fixed (`en_core_web_lg`) and auto-downloaded on first use
— there is no `--model` option.

Words are tokenized and tagged **one conversational turn at a time**: the
dependency parser needs coherent utterances, so words are grouped by `turn_sub`
(which subject held the floor) and each maximal run is parsed as one document.
A word that spaCy splits into several tokens (`"don't"` → `do` + `n't`) yields
one row per piece, each inheriting the source word's `start_time` by char-span
overlap.

### Inputs

The same transcripts as [`featuregen phonemic`](#featuregen-phonemic), under
`stimuli/`. Untimed words (null `start_time`) are retained as parse context — a
dropped word would fragment its turn's sentence and mis-tag neighbors — but carry
their null timing into the output. Null-`word` rows are dropped and warned.

### Options

| Option           | Description                                                       | Default |
| ---------------- | ---------------------------------------------------------------- | ------- |
| `--desc`         | Tag outputs as a named variant (alphanumeric), e.g. `--desc v2` → `desc-v2` | none |
| `--dyad-ids`     | Comma-separated dyad IDs to process; omit for all                | all     |
| `--data-filters` | Narrow to specific runs/conditions — see [Segments and metadata](../concepts/segments.md) | none |
| `--force`        | Overwrite existing outputs (default skips them)                  | off     |

!!! note "Fixed model, no `--device`, no confounds"

    The spaCy model is fixed (`en_core_web_lg`), so there is no `--model` option;
    the parse runs on CPU, so there is no `--device` either. And unlike `phonemic`
    and `semantic`, syntactic has no chained `confoundgen` step — it writes
    features only.

### Example

Generate syntactic features for all dyads:

```bash
hypline featuregen syntactic data/
```

### Outputs

A syntactic feature file per transcript, tagged `feat-syntactic`, under
`features/`:

```
<dataset-root>/features/dyad-030/ses-1/syntactic/
└── dyad-030_ses-1_task-conv_run-1_feat-syntactic.parquet
```

A `--desc` label lands as `desc-<label>` in its own subdirectory
(`syntactic-<label>/`). See
[The hypline dataset layout](../concepts/layout.md#variants-with-desc).

!!! note "Feature file format"

    Each row is one spaCy **token** with its `start_time` (seconds from the start
    of the stimulus), its `turn_sub` label (the utterance the parse grouped on),
    the `token` text, its source `word`, and a one-hot
    `feature` vector (POS ⊕ dependency ⊕ stopword). Width and column order are fit
    to the model's full label vocabulary, so they are fixed across transcripts; a
    label outside that vocabulary leaves its block all-zero. The Parquet footer
    records `spacy_model` and the `feature_dim_labels` naming each dimension.
