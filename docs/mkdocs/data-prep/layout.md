# Hypline dataset layout

Hypline discovers inputs and writes outputs according to a fixed dataset structure. 
Each data-processing command takes the **dataset root** as its main positional argument; 
you organize files according to hypline’s conventions instead of passing each input 
and output path separately.

This page explains that structure, including how hypline organizes subject-level 
brain data and dyad-level conversation data. To assemble your own data in this format, 
see [Prepare your own dataset](../data-prep/prepare-dataset.md). Once your dataset is 
ready, continue to the command-specific guides, beginning with [hypline transcribe](../how-to/transcribe.md). 

## The root tree

A hypline dataset extends the [BIDS](https://bids.neuroimaging.io/) standard with
a few extra areas. A representative dataset tree looks like this:

```
<dataset-root>/
├── participants.tsv                         # required: dyad ↔ subject mapping
├── sub-031/ses-1/func/                      # raw BIDS (events files live here)
├── derivatives/
│   ├── fmriprep/sub-031/ses-1/func/         # fMRIPrep outputs (preprocessed BOLD)
│   └── hypline/sub-031/ses-1/func/          # hypline imaging derivatives (denoised BOLD)
├── stimuli/dyad-030/ses-1/                 
│   ├── audio/                               # audio files you supply
│   └── transcript/                          # generated transcripts
├── features/dyad-030/ses-1/phonemic/        # generated features
├── confounds/dyad-030/ses-1/phonemic/       # generated confounds
├── results/sub-031/
│   ├── encodingModel-v1/                    # fitted encoding models
│   └── encodingEval-v1/                     # evaluation results
└── nuisance/sub-031/ses-1/physio-v1/        # optional, user-supplied nuisance regressors
```

- **`sub-031/`, `derivatives/fmriprep/`** are BIDS and BIDS-derivatives areas
  that you provide. Hypline reads event timing from the raw BIDS tree and
  preprocessed BOLD data from fMRIPrep; it does not require the raw BOLD images themselves.
- **`derivatives/hypline/`** is a BIDS derivatives tree hypline fills with its
  imaging derivatives — currently the [`denoise`](../how-to/denoise.md)
  output. It mirrors fMRIPrep's `sub-XX/[ses-YY/]func/` shape and carries its own
  `dataset_description.json`.
- **`stimuli/`, `features/`, `confounds/`** are hypline additions. Hypline
  creates and fills these as you run commands. They are keyed by **dyad**
  (`dyad-030/`), not subject — see [Subject vs. Dyad](#subject-vs-dyad) below.
- **`results/`** is where [`encoding`](../how-to/encoding.md) writes its
  analysis outputs — fitted models (`encodingModel-<desc>/`) and evaluation results
  (`encodingEval-<desc>/`). It is keyed by **subject**, since one output
  consumes many runs across sessions.
- **`nuisance/`** is optional and you fill it — run-level regressors (e.g.
  physiological recordings) for [`denoise`](../how-to/denoise.md) to regress
  out alongside fMRIPrep's confounds.
- **`participants.tsv`** is a standard BIDS table at the dataset root, required
  to map subjects to dyads — see [Subject vs. Dyad](#subject-vs-dyad).

!!! info "Sessions are optional"

    Examples here use a `ses-1/` level under each subject
    (`sub-031/ses-1/func/`) to match the tutorial dataset. Datasets without
    sessions omit the level entirely (`sub-031/func/`). Hypline handles both cases.

## Subject vs. Dyad

An artifact is keyed by what it is derived from:

- **`sub`-keyed** — derived from one *brain*: raw BOLD, `derivatives/fmriprep/`,
  `derivatives/hypline/` (denoised), `nuisance/`, and `results/` (a subject's
  fitted encoding model, whose weights tie to that brain's voxel grid).
- **`dyad`-keyed** — derived from the *shared conversation* between two partners:
  `stimuli/`, `features/`, `confounds/`. Each dyadic conversation run produces one set of stimuli, features, 
  and confounds, which can later be used to fit a separate encoding model 
  for each partner. A `dyad-030` audio file represents the shared recording, not
  either partner individually.

Because the two worlds use different identity entities, hypline bridges them
through **`participants.tsv`** — a standard BIDS table at the dataset root with
the required `participant_id` column plus a custom **`dyad_id`** column:

```tsv
participant_id   dyad_id
sub-031          dyad-030
sub-032          dyad-030
```

This is the single source of truth for which subjects make up which dyad. Here
subjects `031` and `032` are partners in `dyad-030` (a real study has many such
pairs). It is read lazily: a purely `sub`-keyed workflow (e.g. `denoise`
alone) never needs it, but any step that joins a dyad-keyed stimulus artifact to
a sub-keyed BOLD requires it and errors if it is missing.

So a `dyad-030` feature file does not match a BOLD file by sharing `sub`;
the two carry different identity entities. The join goes through
`participants.tsv`: a subject's encoding model looks up its dyad, then reads that
dyad's features.

## File-naming convention

Hypline follows BIDS filename conventions: a filename is a chain of
`entity-value` pairs joined by `_`, ending in a suffix and extension.

```
sub-031_task-conv_run-1_space-T1w_desc-preproc_bold.nii.gz
\____________________________________________/ \__/ \_____/
                   entities                   suffix   ext
```
### Identity entities

The **identity entities** at the front name which recording a file belongs to.
A file leads with exactly one of `sub` or `dyad` (never both), followed by
the BOLD-identity entities `ses`, `task`, `run`. A `sub`-keyed file belongs to
one brain; a `dyad`-keyed file belongs to one shared conversation. Generated
files mirror the identity entities of the source they came from.

### Category entities

Files stored under `features/`, `confounds/`, `nuisance/`, and `results/` 
carry one **category entity** that identifies their contents:

| Entity        | Area          | Example                         |
| ------------- | ------------- | ------------------------------- |
| `feat-<kind>`   | `features/`   | `feat-phonemic`, `feat-semantic`, `feat-spectral`, `feat-syntactic` |
| `conf-<kind>`   | `confounds/`  | `conf-phonemic`, `conf-semantic` |
| `nuis-<kind>`   | `nuisance/`   | `nuis-physio`                   |
| `result-<kind>` | `results/`    | `result-encodingModel`, `result-encodingEval` |

The `<kind>` matches the subdirectory the file lives in. A phonemic feature
(`feat-phonemic`) lives under `features/dyad-030/ses-1/phonemic/`. A result's
`<kind>-<desc>` subdirectory pairs the entity with its `--desc` variant tag —
`result-encodingModel_desc-v1` under `results/sub-031/encodingModel-v1/`.

Denoised BOLD files under `derivatives/hypline/` instead follow standard 
BIDS-derivatives naming and use `desc-denoised`.

Stimuli carry no category entity. Their kind is a trailing filename suffix
(`_audio`, `_transcript`) instead — e.g. `dyad-030_ses-1_task-conv_run-1_audio.wav`
under `stimuli/dyad-030/ses-1/audio/`.

### `desc` variants

Some commands accept a `--desc` label that tags an output as one *variant* among
several. Variants live in their own subdirectory so they stay physically
separate:

```
confounds/dyad-030/ses-1/
├── phonemic-onset/    # conf-phonemic_desc-onset — speech-onset indicator
└── phonemic-rate/     # conf-phonemic_desc-rate  — speech rate per TR
```

This lets you keep several derivations of the same source side by side and pick
between them later by name.

## Selecting subjects and runs

Because commands discover files by convention, you select what to process using 
identity options and `--data-filters`, rather than individual file paths. Dyad-keyed 
commands (`transcribe`, `featuregen`, and `confoundgen`) take **`--dyad-ids`**, whereas 
subj-keyed commands (`denoise` and `encoding`) take **`--sub-ids`**. Use --force to 
overwrite existing outputs; otherwise, hypline skips outputs it has already generated.

For how to combine these, see [Filter to specific runs or
conditions](../FAQ/filter.md).

## Why the convention matters

Centralizing discovery in one convention means commands compose cleanly: each
reads what earlier steps wrote, with no configuration file wiring inputs to
outputs. It also keeps your dataset self-describing, since the directory tree itself
records what has been generated and from what.
