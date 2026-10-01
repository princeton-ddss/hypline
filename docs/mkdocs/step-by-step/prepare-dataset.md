# Prepare your own dataset

The [tutorial](../tutorials/walkthrough.md) hands you a ready-made dataset. This
guide is the step that comes before it on your own data: arranging your
recordings into the tree hypline expects, so that every command can find its
inputs by convention. Once the tree is right, the commands run exactly as the
tutorial shows.

The [dataset layout](layout.md) describes the tree in full; this page
is the practical checklist for building one from scratch.

## What you supply, and what hypline fills in

Hypline reads a few things you must provide and writes everything else. Knowing
the line between the two is most of the work:

| You supply | Hypline generates |
| ---------- | ----------------- |
| `participants.tsv` — the dyad ↔ subject map | `stimuli/…/transcript/` — transcripts |
| Raw BOLD (optional) and `events.tsv` under `sub-*/` | `features/` — features |
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
sub-041	dyad-040
sub-042	dyad-040
sub-051	dyad-050
sub-052	dyad-050
```

This is the single source of truth that lets a dyad-keyed feature reach a
sub-keyed brain. Two subjects share a `dyad_id` exactly when they belong 
to the same scanning pair.

In these examples the tens digit numbers the pair and the last digit marks its
role: `dyad-040` is the pair, `sub-041` and `sub-042` its two partners. Hypline
does not require this; any IDs work as long as `participants.tsv` maps them.

!!! warning "Use real tabs"

    Every `.tsv` hypline reads must be separated by actual tab characters, not
    spaces. A space-separated row collapses into one column and fails with a
    misleading "missing column" error. This bites most often in
    `participants.tsv`, since it is the first file hypline reads.

Hypline reads this participant-dyad mapping `participants.tsv` file from the
dataset root folder.

## 2. Add event information under `sub-*/`

Hypline reads each run’s structure from BIDS `events.tsv` files stored in the
subject’s `func` directory:

```
sub-041/ses-1/func/
├── sub-041_ses-1_task-conv_run-1_bold.nii.gz
└── sub-041_ses-1_task-conv_run-1_events.tsv
```

!!! info "Sessions are optional"

    The `ses-1/` level is optional. A dataset without sessions omits it entirely
    (`sub-041/func/`), and hypline handles both. Keep it consistent across the
    dataset.

The raw BOLD image may remain as part of your original BIDS dataset, but hypline
does not read it directly. Hypline reads the accompanying `events.tsv` and
obtains its imaging data from the fMRIPrep derivatives described in the next
section.

An `events.tsv` file can describe segments such as trials, blocks, or
conditions, as well as the subject’s speaking turns. Hypline uses these
annotations when generating segment-level features, filtering runs or
conditions, and assigning transcript words to speakers. For an
unsegmented whole-run dataset, `events.tsv` may be omitted for every step
except `encoding`, which requires speaking-turn (`turn_speaker`) rows.

See [Segments and metadata](segments.md) to learn how to create your `events.tsv`
files. That page explains the required columns, hypline’s segment-labeling
convention, speaking-turn annotations, and how to attach descriptive
metadata through `events.json`.

## 3. Add your fMRIPrep outputs

Hypline does not preprocess BOLD; it takes in the output of
[fMRIPrep](https://fmriprep.org/). Run fMRIPrep yourself and place its
derivatives under `derivatives/fmriprep/`, in the per-subject shape it already
produces:

```
derivatives/fmriprep/sub-041/ses-1/func/
├── sub-041_ses-1_task-conv_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz
└── sub-041_ses-1_task-conv_run-1_desc-confounds_timeseries.tsv
```

[`denoise`](../step-by-step/denoise.md) reads the preprocessed BOLD and pulls its
nuisance regressors from fMRIPrep's own `desc-confounds` table, so both must be
present. The BOLD `space` you preprocessed into is the one you will pass to
`denoise` and `encoding` later.

## 4. Add the stimulus audio

The conversation audio is dyad-keyed (it belongs to the pair, not either
partner), so it goes under `stimuli/`, keyed by dyad:

```
stimuli/dyad-040/ses-1/audio/
└── dyad-040_ses-1_task-conv_run-1_audio.wav
```

This is the only stimulus area you fill by hand. From here
[`transcribe`](../step-by-step/transcribe.md) writes the transcripts and
[`featuregen`](../step-by-step/featuregen.md) writes the features, both back under
`stimuli/` and `features/` at the same dyad key.

## 5. (Optional) Describe conditions and custom nuisance

Two optional inputs round out a dataset:

- **`events.json` sidecars** attach descriptive metadata (condition, item,
  counterbalance group) to the segments declared in `events.tsv`. This is what
  lets you filter on `condition-R` even though `condition` never appears in a filename. See
  [Attaching metadata](segments.md#attaching-metadata-eventsjson).
- **`nuisance/` files** hold run-level regressors you supply yourself that
  fMRIPrep never produced (physiological recordings, say) for `denoise` to
  regress out alongside the fMRIPrep columns. See the
  [`denoise` reference](../step-by-step/denoise.md).

Both are optional. A dataset with neither still runs the full pipeline.

## Check the tree before you run

Laid out, a minimal single-dyad dataset looks like this:

```
data/
├── participants.tsv
├── sub-041/ses-1/func/                        # raw BOLD + events (you supply; raw BOLD not required)
├── sub-042/ses-1/func/
├── derivatives/fmriprep/                      # fMRIPrep outputs (you supply)
│   ├── sub-041/ses-1/func/                   
│   └── sub-042/ses-1/func/
└── stimuli/dyad-040/ses-1/audio/              # conversation audio (you supply)
```

Everything else (`features/`, `confounds/`, `derivatives/hypline/`, `results/`)
appears as you run the commands. With this in place, follow the
[tutorial](../tutorials/walkthrough.md) from its transcription step onward; every
command takes `data/` as its main positional argument and discovers its inputs from there.

!!! success "Check"

    `hypline transcribe data/ --audio-ext .wav` should log one line per audio
    file it finds. `No dyads found` means `stimuli/` holds no `dyad-*`
    directories; `No subjects found` from `denoise` means the
    `derivatives/fmriprep/` tree did not land where hypline looks.
