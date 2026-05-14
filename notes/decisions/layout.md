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
CLI commands take a single `<bids_root>`; path resolution is centralized.

## Vocabulary

Two terms name directory levels below `sub-XX/ses-YY/` and must not be conflated:

- **`datatype`** — BIDS spec directory (`func` / `anat` / `dwi` / ...). Applies to raw BIDS and fmriprep areas.
- **`kind`** — hypline category. For `stimuli/`: `audio`, `transcript` (matches the `stim-<kind>` entity on stimulus filenames). For `features/`: `phonemic`, `semantic` (matches the `feat-<kind>` entity on feature filenames).

## Post-processed BOLD placement

Post-processed BOLD (e.g. cleaned) lives in-tree alongside fmriprep outputs:

```
<root>/derivatives/fmriprep/sub-XX/ses-YY/func/
    sub-XX_ses-YY_..._desc-<label>_bold.nii.gz
```

Post-processing is a continuation of the fmriprep pipeline and shares its
identity entities, so outputs belong in the same derivatives tree rather than
a parallel one.

See [../external/bids.md](../external/bids.md) for BIDS entity conventions.
See [feature-files.md](feature-files.md) for feature file naming and path conventions.
