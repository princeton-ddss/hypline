# A full run on the example dataset

This walkthrough takes a real example dataset — stimulus audio and fMRIPrep
outputs — to the two products an encoding model needs: **phonemic features** and
**denoised BOLD**, using one command per step. By the end you will have run the
whole hypline pipeline and seen exactly what each step reads and writes.

It assumes you have hypline installed (see [Installation](../index.md#installation),
including FFmpeg for transcription). No prior hypline experience is needed, but
skim [The hypline dataset layout](../concepts/layout.md) first if a path or
filename below is ever unclear — this tutorial shows the layout in action rather
than re-explaining it.

**What to expect:** about 15 minutes start to finish — most of it the one-time
~2.8 GB dataset download. Of the compute, transcription is the slow step —
roughly a minute on a laptop CPU; the rest run in seconds.

## 1. Get the example dataset

Download the example dataset and unpack it. Throughout this tutorial we call the
unpacked dataset root `data/`.

```bash
# TODO: replace with the published Zenodo record on release
curl -L -o hypline-tutorial-data.zip "<ZENODO-DOI>"
unzip hypline-tutorial-data.zip -d data/
```

[New paragraph about something else]

The dataset is a [BIDS](https://bids.neuroimaging.io/)-style tree for one dyad —
two partners (`sub-003` and `sub-103`) who held a conversation while both were
scanned. It already contains the inputs hypline needs: stimulus audio under
`stimuli/`, raw events and BOLD under each `sub-*/`, and fMRIPrep outputs under
`derivatives/fmriprep/`.

!!! info "What this example dataset is — and is not"

    It is a faithful subset of a real hyperscanning study (about 2.8 GB),
    trimmed so it is small enough to download and run quickly:

    - **Two of the study's five runs** are included (`run-1`, `run-2`). The
      per-run file set and the dyad structure are otherwise complete.
    - **Audio is released for the reading-condition (R) trials only**, for
      privacy. The `events.tsv` files still list every trial, so you will see
      audio for a *subset* of the events in each run — this is expected, not a
      packaging error (see [step 2](#2-transcribe-the-audio)).

    Otherwise the dataset mirrors the structure and design of the original study.

Every command below takes `data/` as its only positional argument and discovers
its inputs from the directory layout — you never pass individual file paths.

## 2. Transcribe the audio

`transcribe` turns each stimulus `.wav` into a word-level transcript using a
Whisper speech-recognition model.

```bash
hypline transcribe data/ --audio-ext .wav --model tiny
```

```text
Transcribing dyad-103_ses-1_task-conv_run-1_trial-1_audio.wav
Transcribing dyad-103_ses-1_task-conv_run-1_trial-3_audio.wav
Transcribing dyad-103_ses-1_task-conv_run-2_trial-1_audio.wav
Transcribing dyad-103_ses-1_task-conv_run-2_trial-3_audio.wav
```

(Log lines are abridged here; a first run also prints a one-time model download
and voice-activity-detection messages.)

!!! tip "Why `--model tiny`"

    `tiny` keeps this tutorial fast — about a minute on a laptop CPU, with a small
    one-time model download. It mis-hears some words, which is fine here: you are
    learning the workflow, not analyzing the transcripts. For a real analysis,
    omit `--model` to use the default `large-v2` — far more accurate, but a
    multi-GB download and much slower on CPU (pass `--device cuda` if you have a
    GPU).

Notice only **four** files are transcribed, not one per run. That is the
reading-condition subset from [step 1](#1-get-the-example-dataset): each run's
audio covers only its R trials (`trial-1`, `trial-3`), so transcripts exist for
those trials and not the others. The run's `events.tsv` still describes every
trial — hypline simply transcribes the audio that is present.

The transcripts land beside the audio, under a new `transcript/` subdirectory:

```text
data/stimuli/dyad-103/ses-1/transcript/
├── dyad-103_ses-1_task-conv_run-1_trial-1_transcript.csv
├── dyad-103_ses-1_task-conv_run-1_trial-3_transcript.csv
├── dyad-103_ses-1_task-conv_run-2_trial-1_transcript.csv
└── dyad-103_ses-1_task-conv_run-2_trial-3_transcript.csv
```

Each CSV is one row per word, with its timing and the partner who spoke it:

```csv
word,start_time,end_time,confidence_score,turn_sub
Thank,5.714,6.095,0.368,003
you.,6.195,6.416,0.326,003
```

These transcripts are **dyad-keyed** (`dyad-103`), because the conversation
belongs to the pair, not to either partner. See
[Subject vs. dyad](../concepts/layout.md#subject-vs-dyad) for why.

!!! success "Check"

    `ls data/stimuli/dyad-103/ses-1/transcript/` lists **four** `_transcript.csv`
    files, and each opens with the `word,start_time,end_time,…` header above. No
    transcripts means the audio was not found — confirm you passed `--audio-ext
    .wav` and that `data/` is the unpacked dataset root.

## 3. Generate phonemic features

`featuregen phonemic` reads those transcripts and computes a **phonemic feature**
for each — a per-word representation that becomes a predictor in the encoding
model.

```bash
hypline featuregen phonemic data/
```

```text
Generating phonemic features for dyad-103_ses-1_task-conv_run-1_trial-1_transcript.csv
...
Generating phonemic confounds for dyad-103_ses-1_task-conv_run-1_trial-1_feat-phonemic.parquet
...
```

By default this step also generates the matching **phonemic confounds** —
speech-onset and speech-rate regressors derived from the same features — so you
get both in one command. (Pass `--skip-confoundgen` to suppress that, or run
[`confoundgen phonemic`](../reference/confoundgen.md) on its own later.)

Two new areas appear, both dyad-keyed:

```text
data/
├── features/dyad-103/ses-1/phonemic/
│   └── dyad-103_ses-1_task-conv_run-1_trial-1_feat-phonemic.parquet   # … one per transcript (4)
└── confounds/dyad-103/ses-1/
    ├── phonemic-onset/
    │   └── dyad-103_ses-1_task-conv_run-1_trial-1_conf-phonemic_desc-onset.parquet   # … (4)
    └── phonemic-rate/
        └── dyad-103_ses-1_task-conv_run-1_trial-1_conf-phonemic_desc-rate.parquet    # … (4)
```

The two confound flavors live in their own subdirectories because they are
`desc` _variants_ of the same `conf-phonemic` kind — see
[Variants with `desc`](../concepts/layout.md#variants-with-desc).

That completes the **stimulus branch**: from audio to the features (and
confounds) the encoding model uses as predictors.

!!! success "Check"

    You should have **four** `feat-phonemic.parquet` files under `features/`, plus
    four files in *each* of the `phonemic-onset/` and `phonemic-rate/` confound
    subdirectories — one per transcript from step 2.

## 4. Denoise the BOLD

The other branch cleans the BOLD signal — the encoding model's _target_.
`denoise` reads fMRIPrep's preprocessed BOLD and regresses out nuisance signals
you select from fMRIPrep's own confounds table.

```bash
hypline denoise data/ \
  --columns trans_x,trans_y,trans_z,rot_x,rot_y,rot_z,cosine
```

```text
Denoising starting: sub-003_ses-1_task-conv_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz
Denoising complete: sub-003_ses-1_task-conv_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz
...
```

Here `--columns` names confound columns from fMRIPrep's table: the six head-motion
parameters (`trans_*`, `rot_*`) plus `cosine`, a prefix that expands to every
cosine-drift regressor. We did not pass `--space`, so `denoise` cleaned the
default **volumetric** space (`MNI152NLin2009cAsym`) — the main target for most
analyses.

This step is **sub-keyed**: it processes each partner's brain (`sub-003`,
`sub-103`) independently, so all four run × subject combinations are denoised.
The output goes to hypline's own derivatives tree, leaving fMRIPrep's untouched:

```text
data/derivatives/hypline/sub-003/ses-1/func/
├── sub-003_ses-1_task-conv_run-1_space-MNI152NLin2009cAsym_desc-denoised_bold.nii.gz
└── sub-003_ses-1_task-conv_run-1_space-MNI152NLin2009cAsym_desc-denoised_bold.json
```

The same pair is written for each run and subject — `sub-003` and `sub-103`,
`run-1` and `run-2`.

Each denoised BOLD carries a `.json` sidecar recording exactly how it was made —
the `desc-preproc` source it came from, the resolved regressor columns, and the
hypline version — so the result is reproducible. See the
[`denoise` reference](../reference/denoise.md) for CompCor selectors, custom
`nuisance/` regressors, and surface spaces.

!!! success "Check"

    `derivatives/hypline/` now holds a `desc-denoised` `.nii.gz` + `.json` pair
    for each subject and run — **eight** files total (2 subjects × 2 runs × 2). If
    the command logged `No subjects found`, check that `derivatives/fmriprep/`
    unpacked correctly under `data/`.

## What you have now

`data/` now holds both sides an encoding model joins:

| Side       | Where                               | From      |
| ---------- | ----------------------------------- | --------- |
| Predictors | `features/dyad-103/…/phonemic/`     | steps 2–3 |
| Target     | `derivatives/hypline/sub-*/…/func/` | step 4    |

Each command read only what the previous steps wrote — no file paths, just the
dataset root. To regenerate a step after changing an option, re-run it with
`--force`; without it, hypline skips outputs that already exist.

## Where to go next

- **Process only some runs or conditions** — [Filter to specific runs or
  conditions](../how-to/filter.md).
- **Regenerate outputs after a fix** — [Regenerate outputs](../how-to/regenerate.md).
- **Per-command options** — the [Reference](../reference/transcribe.md) pages.
