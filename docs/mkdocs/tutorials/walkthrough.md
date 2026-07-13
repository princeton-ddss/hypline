# A full run on the example dataset

This walkthrough takes a real example dataset — stimulus audio and fMRIPrep
outputs — through the whole hypline pipeline: **phonemic features** and
**denoised BOLD**, then a fitted **encoding model** joining the two, using one
command per step. By the end you will have run hypline end to end and seen
exactly what each step reads and writes.

It assumes you have hypline installed (see [Installation](../index.md#installation),
including FFmpeg for transcription). No prior hypline experience is needed, but
skim [The hypline dataset layout](../concepts/layout.md) first if a path or
filename below is ever unclear; this tutorial shows the layout in action rather
than re-explaining it.

Expect about 25 minutes start to finish. Two steps dominate: the
one-time ~2.8 GB dataset download, and the encoding fit ([step 6](#6-fit-the-encoding-model)),
a whole-brain ridge model that runs ~4–5 minutes per subject on CPU. Every
other step runs in seconds to about a minute (transcription). The fit runs on
CPU so the tutorial works anywhere; if you have a GPU, `--device cuda` makes
it far faster.

## 1. Get the example dataset

Download the example dataset and unpack it. Throughout this tutorial we call the
unpacked dataset root `data/`.

```bash
# TODO: replace with the published Zenodo record on release
curl -L -o hypline-tutorial-data.zip "<ZENODO-DOI>"
unzip hypline-tutorial-data.zip -d data/
```

The dataset is a [BIDS](https://bids.neuroimaging.io/)-style tree for one dyad:
two partners (`sub-031` and `sub-032`) who held a conversation while both were
scanned. It already contains the inputs hypline needs: stimulus audio under
`stimuli/`, raw events and BOLD under each `sub-*/`, and fMRIPrep outputs under
`derivatives/fmriprep/`.

!!! info "What the example dataset covers"

    It is a faithful subset of a real hyperscanning study (about 2.8 GB),
    trimmed so it is small enough to download and run quickly:

    - **Two of the study's five runs** are included (`run-1`, `run-2`). The
      per-run file set and the dyad structure are otherwise complete.
    - **Audio is released for the reading-condition (R) trials only**, for
      privacy. The `events.tsv` files still list every trial, so each run's
      audio covers only a subset of its events (see
      [step 2](#2-transcribe-the-audio)).

    Otherwise the dataset mirrors the structure and design of the original study.

Every command below takes `data/` as its only positional argument and discovers
its inputs from the directory layout; you never pass individual file paths.

## 2. Transcribe the audio

`transcribe` turns each stimulus `.wav` into a word-level transcript using a
Whisper speech-recognition model.

```bash
hypline transcribe data/ --audio-ext .wav --model tiny
```

```text
Transcribing dyad-030_ses-1_task-conv_run-1_trial-1_audio.wav
Transcribing dyad-030_ses-1_task-conv_run-1_trial-3_audio.wav
Transcribing dyad-030_ses-1_task-conv_run-2_trial-1_audio.wav
Transcribing dyad-030_ses-1_task-conv_run-2_trial-3_audio.wav
```

(Log lines are abridged here; a first run also prints a one-time model download
and voice-activity-detection messages.)

!!! tip "Why `--model tiny`"

    `tiny` keeps this tutorial fast: about a minute on a laptop CPU, with a small
    one-time model download. It mis-hears some words, which is fine here, since you are
    learning the workflow rather than analyzing the transcripts. For a real analysis,
    omit `--model` to use the default `large-v2`, which is far more accurate but a
    multi-GB download and much slower on CPU (pass `--device cuda` if you have a
    GPU).

Only four files are transcribed, not one per run. These are the
reading-condition subset from [step 1](#1-get-the-example-dataset): each run's
audio covers only its R trials (`trial-1`, `trial-3`), so transcripts exist for
those trials and not the others. The run's `events.tsv` still describes every
trial; hypline transcribes whatever audio is present.

The transcripts land beside the audio, under a new `transcript/` subdirectory:

```text
data/stimuli/dyad-030/ses-1/transcript/
├── dyad-030_ses-1_task-conv_run-1_trial-1_transcript.csv
├── dyad-030_ses-1_task-conv_run-1_trial-3_transcript.csv
├── dyad-030_ses-1_task-conv_run-2_trial-1_transcript.csv
└── dyad-030_ses-1_task-conv_run-2_trial-3_transcript.csv
```

Each CSV is one row per word, with its timing and the partner who spoke it:

```csv
word,start_time,end_time,confidence_score,turn_sub
Thank,5.714,6.095,0.368,031
you.,6.195,6.416,0.326,031
```

These transcripts are **dyad-keyed** (`dyad-030`), because the conversation
belongs to the pair rather than to either partner. See
[Subject vs. dyad](../concepts/layout.md#subject-vs-dyad) for why.

!!! success "Check"

    `ls data/stimuli/dyad-030/ses-1/transcript/` lists four `_transcript.csv`
    files, and each opens with the `word,start_time,end_time,…` header above. If
    no transcripts appear, the audio was not found: confirm you passed `--audio-ext
    .wav` and that `data/` is the unpacked dataset root.

## 3. Generate phonemic features

`featuregen phonemic` reads those transcripts and computes a **phonemic feature**
for each: a per-word representation that becomes a predictor in the encoding
model.

```bash
hypline featuregen phonemic data/
```

```text
Generating phonemic features for dyad-030_ses-1_task-conv_run-1_trial-1_transcript.csv
...
Generating phonemic confounds for dyad-030_ses-1_task-conv_run-1_trial-1_feat-phonemic.parquet
...
```

By default this step also generates the matching **phonemic confounds** —
speech-onset and speech-rate regressors derived from the same features — so you
get both in one command. Pass `--skip-confoundgen` to suppress that, or run
[`confoundgen phonemic`](../reference/confoundgen.md) on its own later.

Two new areas appear, both dyad-keyed:

```text
data/
├── features/dyad-030/ses-1/phonemic/
│   └── dyad-030_ses-1_task-conv_run-1_trial-1_feat-phonemic.parquet   # … one per transcript (4)
└── confounds/dyad-030/ses-1/
    ├── phonemic-onset/
    │   └── dyad-030_ses-1_task-conv_run-1_trial-1_conf-phonemic_desc-onset.parquet   # … (4)
    └── phonemic-rate/
        └── dyad-030_ses-1_task-conv_run-1_trial-1_conf-phonemic_desc-rate.parquet    # … (4)
```

The two confound flavors live in their own subdirectories because they are
`desc` variants of the same `conf-phonemic` kind — see
[Variants with `desc`](../concepts/layout.md#variants-with-desc).

That completes the **stimulus branch**: from audio to the features (and
confounds) the encoding model uses as predictors.

!!! success "Check"

    You should have four `feat-phonemic.parquet` files under `features/`, plus
    four files in each of the `phonemic-onset/` and `phonemic-rate/` confound
    subdirectories, one per transcript from step 2.

## 4. Denoise the BOLD

The other branch cleans the BOLD signal, the encoding model's target.
`denoise` reads fMRIPrep's preprocessed BOLD and regresses out nuisance signals
you select from fMRIPrep's own confounds table.

```bash
hypline denoise data/ \
  --columns trans_x,trans_y,trans_z,rot_x,rot_y,rot_z,cosine
```

```text
Denoising starting: sub-031_ses-1_task-conv_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz
Denoising complete: sub-031_ses-1_task-conv_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz
...
```

Here `--columns` names confound columns from fMRIPrep's table: the six head-motion
parameters (`trans_*`, `rot_*`) plus `cosine`, a prefix that expands to every
cosine-drift regressor. We did not pass `--space`, so `denoise` cleaned the
default volumetric space (`MNI152NLin2009cAsym`), the main target for most
analyses.

This step is **sub-keyed**: it processes each partner's brain (`sub-031`,
`sub-032`) independently, so all four run × subject combinations are denoised.
The output goes to hypline's own derivatives tree, leaving fMRIPrep's untouched:

```text
data/derivatives/hypline/sub-031/ses-1/func/
├── sub-031_ses-1_task-conv_run-1_space-MNI152NLin2009cAsym_desc-denoised_bold.nii.gz
└── sub-031_ses-1_task-conv_run-1_space-MNI152NLin2009cAsym_desc-denoised_bold.json
```

The same pair is written for each run and subject — `sub-031` and `sub-032`,
`run-1` and `run-2`.

Each denoised BOLD carries a `.json` sidecar recording exactly how it was made —
the `desc-preproc` source it came from, the resolved regressor columns, and the
hypline version — so the result is reproducible. See the
[`denoise` reference](../reference/denoise.md) for CompCor selectors, custom
`nuisance/` regressors, and surface spaces.

!!! success "Check"

    `derivatives/hypline/` now holds a `desc-denoised` `.nii.gz` + `.json` pair
    for each subject and run: eight files total (2 subjects × 2 runs × 2). If
    the command logged `No subjects found`, check that `derivatives/fmriprep/`
    unpacked correctly under `data/`.

## 5. (Optional) Add a custom nuisance regressor

So far `denoise` pulled every regressor from fMRIPrep's confounds table via
`--columns`. The other channel is **custom nuisance files** under `nuisance/`:
run-level regressors you supply yourself that fMRIPrep never produced (e.g.
physiological recordings). The example dataset ships a small set so you can try
this path:

```text
data/nuisance/sub-031/ses-1/demo/
└── sub-031_ses-1_task-conv_run-1_nuis-demo_timeseries.tsv   # … one per subject × run (4)
```

!!! info "These are synthetic"

    The shipped `nuis-demo` files hold **synthetic placeholder regressors**
    (`demo_regressor1`, `demo_regressor2`) rather than real signals, so the tutorial
    can exercise `--custom-sources` without needing physiological data. In a real
    analysis you author these yourself; the
    [`denoise` reference](../reference/denoise.md#options) documents the
    `nuisance/` file format under `--custom-sources`.

Re-run `denoise` adding the custom source. `--custom-sources` names the
`nuisance/<kind>/` directory and `--custom-columns` selects columns from it; the
two go together:

```bash
hypline denoise data/ \
  --columns trans_x,trans_y,trans_z,rot_x,rot_y,rot_z,cosine \
  --custom-sources demo \
  --custom-columns demo_regressor1,demo_regressor2 \
  --force
```

The synthetic regressors are now stacked with the fMRIPrep columns into one
regressor matrix. We pass `--force` because [step 4](#4-denoise-the-bold) already
wrote `desc-denoised` outputs; without it, `denoise` skips files that exist. The
`.json` sidecar now also records `custom_sources` and `custom_columns`, so the
result stays reproducible.

!!! success "Check"

    The same eight `desc-denoised` files are rewritten, and each sidecar's
    `custom_columns` lists `demo_regressor1`, `demo_regressor2`. A
    `Nuisance column name collision across channels` error means a custom column
    name collides with a selected fMRIPrep column; rename or drop one.

## 6. Fit the encoding model

Both sides are now in place: phonemic features (the predictors) and denoised
BOLD (the target). `encoding train` joins them, fitting a voxelwise ridge model
per subject:

```bash
hypline encoding train data/ \
  --tasks conv \
  --features phonemic \
  --desc v1 \
  --fold-by run \
  --n-folds loo
```

```text
Fitting starting: sub-031 fold 1/2 — training on 1 cells / … rows
Fitting starting: sub-031 fold 2/2 — training on 1 cells / … rows
Fitting complete: sub-031 (2 folds)
Fitting starting: sub-032 fold 1/2 — training on 1 cells / … rows
Fitting starting: sub-032 fold 2/2 — training on 1 cells / … rows
Fitting complete: sub-032 (2 folds)
```

`--features phonemic` uses the features from step 3 as the model's predictors,
and `--tasks conv` scopes the fit to the `conv` task. `--fold-by run --n-folds
loo` cross-validates by run, leaving one run out per fold. This is the common
setup, and the one that lets you score held-out data in
[step 7](#7-score-a-model-within-a-subject).
With two runs, `loo` yields two folds. `--desc v1` tags this model variant so its
output lands in its own subdirectory.

!!! note "The fit is the slow step"

    This is the tutorial's one compute-heavy step: a whole-brain ridge over a
    grid of regularization strengths, ~4–5 minutes per subject on CPU. The
    `Fitting starting/complete` lines above mark each fold's progress, so a
    long pause between them is normal. The fit runs on CPU so the tutorial works
    anywhere; pass `--device cuda` if you have a GPU.

The step is **sub-keyed** like `denoise`, one model per brain. Outputs go to a
new `results/` area:

```text
data/results/sub-031/encodingModel-v1/
├── sub-031_result-encodingModel_desc-v1.joblib   # the fitted model
└── sub-031_result-encodingModel_desc-v1.json     # provenance sidecar
```

!!! success "Check"

    `results/` now holds an `encodingModel-v1/` directory for `sub-031` and
    `sub-032`, each with a `.joblib` + `.json` pair. If the command logged
    `No subjects found`, check that step 4 wrote `desc-denoised` BOLD.

### Load the result back

The model saves as a `.joblib` blob you load back into Python for downstream
analysis, the same way [`read_feature`](../reference/python-api.md) reads a
feature file:

```python
from hypline.encoding import load_artifact

artifact = load_artifact(
    "data/results/sub-031/encodingModel-v1/sub-031_result-encodingModel_desc-v1.joblib"
)
artifact.recipe   # the features, delays, alphas, and split the model was fit with
artifact.models   # the fitted pipeline(s)
```

The fitted model is a starting point: the next two steps use it, scoring its
predictions against a real brain.

## 7. Score a model within a subject

`encoding analyze` scores a model's predictions against a subject's actual BOLD.
It takes three subject roles, independent by design:

- **target** (`--target-sub`) — whose real BOLD is the comparison, and whose
  speaking turns define the roles below.
- **model** (`--model-sub`) — whose trained weights are loaded.
- **source** (`--source-sub`) — whose features build the prediction inputs.

Start with the simplest case, all three the same subject: `sub-031`'s own model,
scored against `sub-031`'s own brain.

```bash
hypline encoding analyze data/ \
  --target-sub 031 \
  --model-sub self \
  --model-desc v1 \
  --desc selfeval
```

```text
Analyzing: target sub-031, model sub-031, source sub-031 (OOS)
Analysis complete: target sub-031 — scored 2 folds
```

`--model-desc v1` names the model from [step 6](#6-fit-the-encoding-model), and
`--desc selfeval` tags this eval. We passed no `--test-on`, so `analyze` scores
each fold's **held-out** run, the run that fold did not train on. This is why
step 6 folded: a single unfolded model has no held-out data to score against
itself.

Each score is broken out by **role**, derived from the target's turns in the
conversation: `prod` (the target is speaking), `comp` (the partner is speaking,
the target listening), and `both` (either). One eval thus reports how well the
model predicts the brain during production, comprehension, and overall.

The eval lands in its own `results/` subdirectory, keyed by the target subject:

```text
data/results/sub-031/encodingEval-selfeval/
└── sub-031_result-encodingEval_desc-selfeval.nc   # per-voxel scores (netCDF-4)
```

It is a self-describing netCDF-4 file; load it back as an
[`xarray.Dataset`](https://docs.xarray.dev/):

```python
from hypline.encoding import load_eval

ds = load_eval(
    "data/results/sub-031/encodingEval-selfeval/sub-031_result-encodingEval_desc-selfeval.nc"
)
ds["corr"].sel(role="prod")   # scores during the target's own speech
ds.attrs["model_sub"], ds.attrs["target_sub"]   # provenance rides along
```

!!! note "These scores are not `[-1, 1]` correlations"

    `corr` holds himalaya *split* scores — each feature band's own contribution
    to the joint prediction — so a value is not a plain Pearson correlation and
    can fall outside `[-1, 1]`. Read them as relative encoding scores rather than
    accuracy fractions.

!!! success "Check"

    `results/sub-031/` gains an `encodingEval-selfeval/` directory with one
    `.nc` file, and the log reads `scored 2 folds`. An `empty out-of-sample set`
    error means the model wasn't folded; re-run step 6 with `--fold-by run
    --n-folds loo`.

## 8. Score across brains

The within-subject eval is the warm-up. What hypline is built for is the
**cross-brain** case: because the two partners shared one conversation, you can
drive one partner's model with the partner's speech and score it against the
other partner's brain. Same command, different subject wiring:

```bash
hypline encoding analyze data/ \
  --target-sub 031 \
  --model-sub partner \
  --source-sub partner \
  --model-desc v1 \
  --desc crosseval
```

```text
Analyzing: target sub-031, model sub-032, source sub-032 (OOS)
Analysis complete: target sub-031 — scored 2 folds
```

`--target-sub 031` keeps `sub-031`'s brain as the comparison, but `--model-sub
partner` and `--source-sub partner` swap in `sub-032`'s model and features (hypline
resolves `partner` through `participants.tsv`). The output structure is identical
to step 7 — a `.nc` under `encodingEval-crosseval/` — so `load_eval` reads it the
same way.

!!! info "What this step demonstrates"

    This is how you run the cross-brain analysis that motivates hypline: the
    mechanics of pointing one partner's model at the other's brain. It does not
    demonstrate the *effect*, since two runs of `--model tiny` transcripts are far
    too little data for the cross-brain scores to mean anything. On a full study
    they carry the shared-representation signal; here they only confirm the
    command runs end to end.

!!! success "Check"

    `results/sub-031/` now also holds `encodingEval-crosseval/`, and the log
    shows `model sub-032, source sub-032` — the partner's model and features
    scored against `sub-031`'s brain.

## What you have now

`data/` now holds a full encoding run, end to end:

| Side       | Where                                  | From          |
| ---------- | -------------------------------------- | ------------- |
| Predictors | `features/dyad-030/…/phonemic/`        | steps 2–3     |
| Target     | `derivatives/hypline/sub-*/…/func/`    | step 4        |
| Model      | `results/sub-*/encodingModel-v1/`      | step 6        |
| Eval       | `results/sub-031/encodingEval-*/`      | steps 7–8     |

Each command read only what the previous steps wrote — no file paths, just the
dataset root. To regenerate a step after changing an option, re-run it with
`--force`. Without it, hypline skips outputs that already exist.

## Where to go next

- **Process only some runs or conditions** — [Filter to specific runs or
  conditions](../how-to/filter.md).
- **Regenerate outputs after a fix** — [Regenerate outputs](../how-to/regenerate.md).
- **Per-command options** — the Reference pages:
  [transcribe](../reference/transcribe.md) ·
  [featuregen](../reference/featuregen.md) ·
  [confoundgen](../reference/confoundgen.md) ·
  [denoise](../reference/denoise.md) ·
  [encoding](../reference/encoding.md).
