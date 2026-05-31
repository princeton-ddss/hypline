# The hypline dataset layout

Every hypline command takes a single argument — the **dataset root** — and finds
its inputs and writes its outputs by following a fixed directory convention.
Understanding that convention is the key to using hypline: you never pass file
paths, you organize your files where hypline expects them.

This page describes the layout once. The [reference pages](../reference/transcribe.md)
assume it.

## The root tree

A hypline dataset extends the [BIDS](https://bids.neuroimaging.io/) standard with
a few extra areas. A complete tree looks like this:

```
<dataset-root>/
├── sub-01/func/                              # raw BIDS (events files live here)
├── derivatives/
│   └── fmriprep/sub-01/func/                 # fMRIPrep outputs (preprocessed BOLD)
├── stimuli/sub-01/audio/                     # stimulus audio, transcripts
├── features/sub-01/phonemic/                 # generated features
└── confounds/sub-01/phonemic/                # generated confounds
```

- **`sub-01/`, `derivatives/fmriprep/`** are standard BIDS areas. You provide
  these — your raw recordings and your fMRIPrep run.
- **`stimuli/`, `features/`, `confounds/`** are hypline additions. Hypline
  creates and fills these as you run commands.

!!! info "Sessions are optional"

    Datasets with multiple sessions add a `ses-YY/` level under each subject
    (`sub-01/ses-1/func/`). Datasets without sessions omit it entirely
    (`sub-01/func/`). Hypline handles both — examples here are sessionless for
    brevity.

## How files are named

Hypline follows BIDS filename conventions: a filename is a chain of
`entity-value` pairs joined by `_`, ending in a suffix and extension.

```
sub-01_task-conv_run-1_space-fsaverage6_hemi-L_bold.func.gii
\____________________________________________/ \__/  \_______/
                   entities                   suffix extension
```

The **identity entities** at the front — `sub`, `ses`, `task`, `run` — name
*which recording* a file belongs to. They are how hypline matches files across areas: a feature
file with `sub-01_task-conv_run-1` corresponds to the BOLD file with the same
`sub-01_task-conv_run-1`. Generated files always mirror the identity entities of
the source they came from.

### Category entities

Each hypline-generated file carries exactly one **category entity** naming what
kind of derivative it is:

| Entity        | Area          | Example                          |
| ------------- | ------------- | -------------------------------- |
| `stim-<kind>` | `stimuli/`    | `stim-audio`, `stim-transcript`  |
| `feat-<kind>` | `features/`   | `feat-phonemic`                  |
| `conf-<kind>` | `confounds/`  | `conf-phonemic`, `conf-fmriprep` |

The `<kind>` matches the subdirectory the file lives in. A phonemic feature
(`feat-phonemic`) lives under `features/sub-01/phonemic/`.

### Variants with `desc`

Some commands accept a `--desc` label that tags an output as one *variant* among
several. Variants live in their own subdirectory so they stay physically
separate:

```
confounds/sub-01/
├── fmriprep-minimal/    # conf-fmriprep_desc-minimal — e.g. motion only
└── fmriprep-full/       # conf-fmriprep_desc-full    — e.g. motion + CompCor
```

This lets you keep several derivations of the same source side by side and pick
between them later (for example, when choosing which confounds to regress out in
[`denoise`](../reference/denoise.md)).

## Selecting subjects and runs

Because commands discover files by convention, you select *what to process* with
options rather than paths — `--sub-ids` for subjects, `--data-filters` for runs
and conditions, both interpreted against the entities described above. A third
shared option, `--force`, overwrites existing outputs (by default hypline skips
files it has already generated, so reruns are cheap).

For how to combine these, see [Filter to specific runs or
conditions](../how-to/filter.md); for what `--data-filters` can match, see
[Segments and metadata](segments.md).

## Why this design

Centralizing discovery in one convention means commands compose cleanly: each
reads what earlier steps wrote, with no configuration file wiring inputs to
outputs. It also keeps your dataset self-describing — the directory tree itself
records what has been generated and from what.
