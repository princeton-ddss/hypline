# Prepare your own dataset

The [tutorial](../tutorials/walkthrough.md) hands you a ready-made dataset. This
guide is the step that comes before it on your own data: arranging your
recordings into the tree hypline expects, so that every command can find its
inputs by convention. Once the tree is right, the commands run exactly as the
tutorial shows.

The [dataset layout](../data-prep/layout.md) describes the tree in full; this page
is the practical checklist for building one from scratch.

## What you supply, and what hypline fills in

Hypline reads a few things you must provide and writes everything else. Knowing
the line between the two is most of the work:

| You supply | Hypline generates |
| ---------- | ----------------- |
| `participants.tsv` — the dyad ↔ subject map | `stimuli/…/transcript/` — transcripts |
| Raw BOLD and `events.tsv` under `sub-*/` | `features/` — features |
| fMRIPrep outputs under `derivatives/fmriprep/` | `confounds/` — stimulus confounds |
| Stimulus audio under `stimuli/…/audio/` | `derivatives/hypline/` — denoised BOLD |
| `events.json` sidecars (optional metadata) | `results/` — models and evals |
| `nuisance/` regressors (optional) | |

You never create anything in the right-hand column by hand. Get the left-hand
column in place and the pipeline produces the rest.

## 1. Map subjects to dyads: `participants.tsv`

Hypline is a hyperscanning pipeline, so it needs to know which two subjects make
up each dyad. That mapping lives in `participants.tsv` at the dataset root — a
standard BIDS table with the required `participant_id` column plus a custom
`dyad_id` column:

```tsv
participant_id	dyad_id
sub-031	dyad-030
sub-032	dyad-030
sub-033	dyad-034
sub-034	dyad-034
```

This is the single source of truth that lets a dyad-keyed feature reach a
sub-keyed brain. Two subjects share a `dyad_id` exactly when they belong 
to the same scanning pair.

!!! warning "Use real tabs"

    Every `.tsv` hypline reads must be separated by actual tab characters, not
    spaces. A space-separated row collapses into one column and fails with a
    misleading "missing column" error. This bites most often in
    `participants.tsv`, since it is the first file hypline reads.

## 2. Add event information under `sub-*/`

Hypline reads each run’s structure from BIDS `events.tsv` files stored in the
subject’s `func` directory:

```
sub-031/ses-1/func/
├── sub-031_ses-1_task-conv_run-1_bold.nii.gz
└── sub-031_ses-1_task-conv_run-1_events.tsv
```

The raw BOLD image may remain as part of your original BIDS dataset, but hypline
does not read it directly. Hypline reads the accompanying `events.tsv` and
obtains its imaging data from the fMRIPrep derivatives described in the next
section.

An `events.tsv` file can describe segments such as trials, blocks, or
conditions, as well as the subject’s speaking turns. Hypline uses these
annotations when generating segment-level features, filtering runs or
conditions, and assigning transcript words to speakers.

!!! important "Learn how to create your events files"

    See [Segments and metadata](segments.md) before creating your `events.tsv`
    files. That page explains the required columns, hypline’s segment-labeling
    convention, speaking-turn annotations, and how to attach descriptive
    metadata through `events.json`.

    For an unsegmented whole-run dataset, `events.tsv` may be omitted unless you
    want to include speaking-turn annotations.

!!! info "Sessions are optional"

    The `ses-1/` level is optional. A dataset without sessions omits it entirely
    (`sub-031/func/`), and hypline handles both. Keep it consistent across the
    dataset.

## 3. Add your fMRIPrep outputs

Hypline does not preprocess BOLD; it consumes the output of
[fMRIPrep](https://fmriprep.org/). Run fMRIPrep yourself and place its
derivatives under `derivatives/fmriprep/`, in the per-subject shape it already
produces:

```
derivatives/fmriprep/sub-031/ses-1/func/
├── sub-031_ses-1_task-conv_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz
└── sub-031_ses-1_task-conv_run-1_desc-confounds_timeseries.tsv
```

[`denoise`](../how-to/denoise.md) reads the preprocessed BOLD and pulls its
nuisance regressors from fMRIPrep's own `desc-confounds` table, so both must be
present. The BOLD `space` you preprocessed into is the one you will pass to
`denoise` and `encoding` later.

## 4. Lay out the stimulus audio

The conversation audio is dyad-keyed (it belongs to the pair, not either
partner), so it goes under `stimuli/`, keyed by dyad:

```
stimuli/dyad-030/ses-1/audio/
└── dyad-030_ses-1_task-conv_run-1_audio.wav
```

This is the only stimulus area you fill by hand. From here
[`transcribe`](../how-to/transcribe.md) writes the transcripts and
[`featuregen`](../how-to/featuregen.md) writes the features, both back under
`stimuli/` and `features/` at the same dyad key.

## 5. (Optional) Describe conditions and custom nuisance

Two optional inputs round out a dataset:

- **`events.json` sidecars** attach descriptive metadata (condition, item,
  counterbalance group) to the segments declared in `events.tsv`. This is what
  lets you filter on `cond-R` even though `cond` never appears in a filename. See
  [Attaching metadata](../FAQ/filter.md#attaching-metadata-eventsjson).
- **`nuisance/` files** hold run-level regressors you supply yourself that
  fMRIPrep never produced (physiological recordings, say) for `denoise` to
  regress out alongside the fMRIPrep columns. See the
  [`denoise` reference](../how-to/denoise.md).

Both are optional. A dataset with neither still runs the full pipeline.

## Check the tree before you run

Laid out, a minimal single-dyad dataset looks like this:

```
data/
├── participants.tsv
├── sub-031/ses-1/func/                        # raw BOLD + events (you supply)
├── sub-032/ses-1/func/
├── derivatives/fmriprep/                      # fMRIPrep outputs (you supply)
└── stimuli/dyad-030/ses-1/audio/              # conversation audio (you supply)
```

Everything else (`features/`, `confounds/`, `derivatives/hypline/`, `results/`)
appears as you run the commands. With this in place, follow the
[tutorial](../tutorials/walkthrough.md) from its transcription step onward; every
command takes `data/` as its main positional argument and discovers its inputs from there.

!!! success "Check"

    `hypline transcribe data/ --audio-ext .wav` should log one line per audio
    file it finds. `No dyads found` means the `stimuli/…/audio/` layout or
    `participants.tsv` is off; `No subjects found` from `denoise` means the
    `derivatives/fmriprep/` tree did not land where hypline looks.
