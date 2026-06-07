# Hypline root layout

Directory contract for the hypline-flavored BIDS root tree.

## Tree structure

```
<root>/
├── sub-XX/[ses-YY/]<datatype>/                        # raw BIDS
├── derivatives/fmriprep/sub-XX/[ses-YY/]<datatype>/   # fmriprep outputs
├── stimuli/sub-XX/[ses-YY/]<kind>/                    # audio, transcript, ...
├── features/sub-XX/[ses-YY/]<kind>[-<desc>]/          # phonemic, semantic, ...
├── confounds/sub-XX/[ses-YY/]<kind>[-<desc>]/         # phonemic, semantic, ...
└── nuisance/sub-XX/[ses-YY/]<kind>[-<desc>]/          # physio, ... (run-level)
```

`ses-YY/` is present only for datasets with sessions; sessionless datasets nest
`<datatype>` / `<kind>` directly under `sub-XX/`. Path resolution handles both.

When a derived file carries a `desc-<desc>` entity, it lives in a
`<kind>-<desc>/` subdirectory (not the parent `<kind>/`), so variants are
physically separated on disk. Discovery selects bare vs. variant folders via the
`desc` argument; `desc="*"` aggregates all variants together. `desc` is a
dedicated argument, not a `bids_filters` key.

`desc` variants apply to `features/`, `confounds/`, and `nuisance/` only. A
stimulus is the experimental record (the audio, transcript, movie) with one
ground truth, so `stimuli/` has no variants — `path.stimulus` takes no `desc`. An
artifact that needs variants is a feature, not a stimulus.

`stimuli/`, `features/`, `confounds/`, and `nuisance/` are hypline extensions —
not in the BIDS spec. CLI commands take a single `<bids_root>`; path resolution
is centralized.

## Vocabulary

Two terms name directory levels below `sub-XX/ses-YY/` and must not be conflated:

- **`datatype`** — BIDS spec directory (`func` / `anat` / `dwi` / ...). Applies to raw BIDS and fmriprep areas.
- **`kind`** — hypline category. For `stimuli/`: `audio`, `transcript` (matches the `stim-<kind>` entity on stimulus filenames; no `desc` variants — see above). For `features/`: `phonemic`, `semantic` (matches the `feat-<kind>` entity on feature filenames). For `confounds/`: `phonemic`, `semantic` (matches the `conf-<kind>` entity on confound filenames; different derivations of a kind's source are differentiated by an optional `desc-<name>` entity). For `nuisance/`: user-named, e.g. `physio` (matches the `nuis-<kind>` entity on run-level nuisance filenames; optional `desc-<name>` derivation). Unlike the others, nuisance files carry a `_timeseries` suffix and are TSV — see [nuisance-files.md](nuisance-files.md).

## Category entities are mutually exclusive

A derived output carries exactly one of `stim-*`, `feat-*`, `conf-*`, `nuis-*`.
A confound derived from a `feat-phonemic` file is named `conf-phonemic`,
not `feat-phonemic_conf-phonemic`.

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
See [confound-files.md](confound-files.md) for confound file naming and path conventions.
See [nuisance-files.md](nuisance-files.md) for nuisance file naming and path conventions.

## Discovery contract — `_Find.*` raises on empty

`BIDSLayout.find.{stimuli,features,confounds,nuisance,fmriprep}` never return `[]`. When
nothing matches, they raise `FileNotFoundError` with a tier-specific diagnostic
pinpointing where the tree walk stopped.

Callers rely on this contract and do not perform their own emptiness checks.
Tools that need a soft "no data yet" path must catch `FileNotFoundError`
explicitly rather than expect an empty list.

## `bids_filters` — structural vs. descriptive

`find.{stimuli,features,confounds,nuisance,fmriprep}` accept both kinds of filter uniformly:

- **Structural** (keys in `STRUCTURAL_ENTITIES`: BOLD identity, category tags,
  image-variant descriptors) — matched against filenames during the on-disk walk.
- **Descriptive** — matched against each candidate's resolved entities (filename
  merged with `events.json` `Levels` metadata for the matching segment) via
  `events.resolve_entities`. Empty descriptive matches raise `FileNotFoundError`
  with a diagnostic; resolve failures re-raise as `ValueError`.

Same-key filter values OR-match; different keys AND-match.
