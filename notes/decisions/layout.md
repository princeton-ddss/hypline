# Hypline root layout

Directory contract for the hypline-flavored BIDS root tree.

## Tree structure

```
<root>/
├── sub-XX/[ses-YY/]<datatype>/                        # raw BIDS
├── derivatives/fmriprep/sub-XX/[ses-YY/]<datatype>/   # fmriprep outputs
├── stimuli/sub-XX/[ses-YY/]<kind>/                    # audio, transcript, ...
└── features/sub-XX/[ses-YY/]<kind>/                   # phonemic, semantic, ...
```

`ses-YY/` is present only for datasets with sessions; sessionless datasets nest
`<datatype>` / `<kind>` directly under `sub-XX/`. Path resolution handles both.

`stimuli/` and `features/` are hypline extensions — not in the BIDS spec.
CLI commands take a single `<root_dir>`; path resolution is centralized.

## Vocabulary

Two terms name directory levels below `sub-XX/ses-YY/` and must not be conflated:

- **`datatype`** — BIDS spec directory (`func` / `anat` / `dwi` / ...). Applies to raw BIDS and fmriprep areas.
- **`kind`** — hypline category. For `stimuli/`: `audio`, `transcript`. For `features/`: `phonemic`, `semantic`. Matches the `feature-<kind>` entity on feature filenames.

## Cleaned BOLD contract

Cleaned BOLD lives **in-tree** under the fmriprep derivatives:

```
<root>/derivatives/fmriprep/sub-XX/ses-YY/func/
    sub-XX_ses-YY_..._desc-clean_bold.nii.gz
```

Any tool that writes cleaned BOLD must write to this location. The historical
convention (sibling tree `<fmriprep_dir>_cleaned/`) is deprecated and must not
be used for new writes.

## Area presence guarantees

- `sub-XX/`, `derivatives/fmriprep/` — assumed present; commands that require
  them raise if absent.
- `stimuli/`, `features/` — may be absent on first run; the command that
  creates outputs (`featuregen`, `transcribe`) materializes the directory before
  writing.

See [../external/bids.md](../external/bids.md) for BIDS entity conventions.
See [feature-files.md](feature-files.md) for feature file naming and path conventions.
