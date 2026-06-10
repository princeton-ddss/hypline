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
├── participants.tsv                         # required: dyad ↔ subject mapping
├── sub-001/func/                            # raw BIDS (events files live here)
├── derivatives/
│   ├── fmriprep/sub-001/func/               # fMRIPrep outputs (preprocessed BOLD)
│   └── hypline/sub-001/func/                # hypline imaging derivatives (denoised BOLD)
├── stimuli/dyad-101/audio/                  # stimulus audio, transcripts
├── features/dyad-101/phonemic/              # generated features
├── confounds/dyad-101/phonemic/             # generated confounds
└── nuisance/sub-001/physio-v1/              # optional, user-supplied nuisance regressors
```

- **`sub-001/`, `derivatives/fmriprep/`** are standard BIDS areas. You provide
  these — your raw recordings and your fMRIPrep run.
- **`derivatives/hypline/`** is a BIDS derivatives tree hypline fills with its
  imaging derivatives — currently the [`denoise`](../reference/denoise.md)
  output. It mirrors fMRIPrep's `sub-XX/[ses-YY/]func/` shape and carries its own
  `dataset_description.json`.
- **`stimuli/`, `features/`, `confounds/`** are hypline additions. Hypline
  creates and fills these as you run commands. They are keyed by **dyad**
  (`dyad-101/`), not subject — see [Subject vs. dyad](#subject-vs-dyad) below.
- **`nuisance/`** is optional and *you* fill it — run-level regressors (e.g.
  physiological recordings) for [`denoise`](../reference/denoise.md) to regress
  out alongside fMRIPrep's confounds.
- **`participants.tsv`** is a standard BIDS table at the dataset root, required
  to map subjects to dyads — see [Subject vs. dyad](#subject-vs-dyad).

!!! info "Sessions are optional"

    Datasets with multiple sessions add a `ses-YY/` level under each subject
    (`sub-001/ses-1/func/`). Datasets without sessions omit it entirely
    (`sub-001/func/`). Hypline handles both — examples here are sessionless for
    brevity.

## How files are named

Hypline follows BIDS filename conventions: a filename is a chain of
`entity-value` pairs joined by `_`, ending in a suffix and extension.

```
sub-001_task-conv_run-1_space-fsaverage6_hemi-L_bold.func.gii
\____________________________________________/ \__/  \_______/
                   entities                   suffix extension
```

The **identity entities** at the front name *which recording* a file belongs to.
A file leads with exactly one of `sub` **or** `dyad` (never both), followed by
the BOLD-identity entities `ses`, `task`, `run`. A `sub`-keyed file belongs to
one brain; a `dyad`-keyed file belongs to one shared conversation. Generated
files mirror the identity entities of the source they came from.

## Subject vs. dyad

Hypline is a hyperscanning pipeline: two partners hold one conversation while
both are scanned. An artifact is keyed by **what it is derived from**:

- **`sub`-keyed** — derived from one *brain*: raw BOLD, `derivatives/fmriprep/`,
  `derivatives/hypline/` (denoised), `nuisance/`.
- **`dyad`-keyed** — derived from the *shared conversation* between two partners:
  `stimuli/`, `features/`, `confounds/`. One conversation → one dyad → one set of
  stimuli/features/confounds, later consumed by *each* partner's per-subject
  encoding model. A `dyad-101` audio file is the dyad's shared recording, not
  either partner's.

Because the two worlds use different identity entities, hypline bridges them
through **`participants.tsv`** — a standard BIDS table at the dataset root with
the required `participant_id` column plus a custom **`dyad_id`** column:

```tsv
participant_id   dyad_id
sub-001          dyad-101
sub-101          dyad-101
sub-002          dyad-102
sub-102          dyad-102
```

This is the single source of truth for which subjects make up which dyad — here
subjects `001` and `101` are partners in `dyad-101`, `002` and `102` in
`dyad-102`. It is read lazily: a purely `sub`-keyed workflow (e.g. `denoise`
alone) never needs it, but any step that joins a dyad-keyed stimulus artifact to
a sub-keyed BOLD requires it and errors if it is missing.

!!! warning "Use real tabs"

    `participants.tsv` — and every `.tsv` hypline reads (`events.tsv`, custom
    `nuisance/` tables) — must be separated by **actual tab characters**, not
    spaces. Hypline splits on tabs, so a space-separated row collapses into one
    column and fails with a misleading "missing column" error.

So a `dyad-101` feature file does **not** match a BOLD file by sharing `sub` —
the two carry different identity entities. The join goes through
`participants.tsv`: a subject's encoding model looks up its dyad, then reads that
dyad's features.

### Category entities

Each hypline-generated file carries exactly one **category entity** naming what
kind of derivative it is:

| Entity        | Area          | Example                         |
| ------------- | ------------- | ------------------------------- |
| `stim-<kind>` | `stimuli/`    | `stim-audio`, `stim-transcript` |
| `feat-<kind>` | `features/`   | `feat-phonemic`                 |
| `conf-<kind>` | `confounds/`  | `conf-phonemic`                 |
| `nuis-<kind>` | `nuisance/`   | `nuis-physio`                   |

The `<kind>` matches the subdirectory the file lives in. A phonemic feature
(`feat-phonemic`) lives under `features/dyad-101/phonemic/`.

### Variants with `desc`

Some commands accept a `--desc` label that tags an output as one *variant* among
several. Variants live in their own subdirectory so they stay physically
separate:

```
confounds/dyad-101/
├── phonemic-onset/    # conf-phonemic_desc-onset — speech-onset indicator
└── phonemic-rate/     # conf-phonemic_desc-rate  — speech rate per TR
```

This lets you keep several derivations of the same source side by side and pick
between them later by name.

## Selecting subjects and runs

Because commands discover files by convention, you select *what to process* with
options rather than paths — an identity option plus `--data-filters` for runs and
conditions, both interpreted against the entities described above. The identity
option follows the area the command writes: dyad-keyed stimulus commands
(`transcribe`, `featuregen`, `confoundgen`) take **`--dyad-ids`**, while the
sub-keyed `denoise` takes **`--sub-ids`**. A third shared option, `--force`,
overwrites existing outputs (by default hypline skips files it has already
generated, so reruns are cheap).

For how to combine these, see [Filter to specific runs or
conditions](../how-to/filter.md); for what `--data-filters` can match, see
[Segments and metadata](segments.md).

## Why this design

Centralizing discovery in one convention means commands compose cleanly: each
reads what earlier steps wrote, with no configuration file wiring inputs to
outputs. It also keeps your dataset self-describing — the directory tree itself
records what has been generated and from what.
