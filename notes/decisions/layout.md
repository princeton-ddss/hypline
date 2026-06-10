# Hypline root layout

Directory contract for the hypline-flavored BIDS root tree.

## Tree structure

```
<root>/
├── sub-XX/[ses-YY/]<datatype>/                        # raw BIDS
├── derivatives/fmriprep/sub-XX/[ses-YY/]<datatype>/   # fmriprep outputs
├── derivatives/hypline/sub-XX/[ses-YY/]func/          # hypline imaging derivatives (denoised BOLD)
├── stimuli/dyad-XX/[ses-YY/]<kind>/                   # audio, transcript, ... (shared conversation)
├── features/dyad-XX/[ses-YY/]<kind>[-<desc>]/         # phonemic, semantic, ... (shared conversation)
├── confounds/dyad-XX/[ses-YY/]<kind>[-<desc>]/        # phonemic, semantic, ... (shared conversation)
└── nuisance/sub-XX/[ses-YY/]<kind>[-<desc>]/          # physio, ... (run-level, per-brain)
```

`stimuli/`, `features/`, `confounds/` are **dyad-keyed** (derived from the
shared conversation); BOLD-derived areas (raw, fmriprep, hypline, `nuisance/`)
stay **sub-keyed**. See [dyad-keyed.md](dyad-keyed.md). The identity prefix is a
parameter threaded through discovery — shared areas pass `dyad`, per-brain areas
pass `sub`, one code path. `list.dyads(area=)` scans on-disk subdirs (distinct
from the participants.tsv mapping).

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

Two terms name directory levels below `<identity>/[ses-YY/]` (`<identity>` is
`sub-XX` or `dyad-XX`) and must not be conflated:

- **`datatype`** — BIDS spec directory (`func` / `anat` / `dwi` / ...). Applies to raw BIDS and fmriprep areas.
- **`kind`** — hypline category, naming both the `<kind>/` subdirectory and how
  the kind is recorded in the filename (see [Category entities are mutually
  exclusive](#category-entities-are-mutually-exclusive)):
    - `stimuli/`: `audio`, `transcript` — trailing `_<kind>` filename suffix; no `desc` variants (see above).
    - `features/`: `phonemic`, `semantic` — `feat-<kind>` entity.
    - `confounds/`: `phonemic`, `semantic` — `conf-<kind>` entity; source derivations differ by optional `desc-<name>`.
    - `nuisance/`: user-named, e.g. `physio` — `nuis-<kind>` entity; optional `desc-<name>`. Carries a `_timeseries` suffix and is TSV — see [nuisance-files.md](nuisance-files.md).

## Category entities are mutually exclusive

A derivative carries exactly one of `feat-*`, `conf-*`, `nuis-*`.
A confound derived from a `feat-phonemic` file is named `conf-phonemic`,
not `feat-phonemic_conf-phonemic`. Stimuli are not derivatives and carry no
category entity; their kind is a trailing `_<kind>` suffix (`_audio`,
`_transcript`).

## Denoised BOLD placement — `derivatives/hypline/`

Denoised BOLD lives in its own `derivatives/hypline/` tree, not in-tree under
fmriprep:

```
<root>/derivatives/hypline/sub-XX/[ses-YY/]func/
    sub-XX_..._desc-denoised_bold.nii.gz        # (or .func.gii for surface)
    sub-XX_..._desc-denoised_bold.json          # per-file sidecar
```

`hypline` is a layout area rooting at `derivatives/hypline/`; `area_root`
special-cases it the same way as `fmriprep`. The tree mirrors fmriprep's shape
(`sub-XX/[ses-YY/]func/`, full BOLD identity entities, `_bold` suffix) — only
`desc=denoised` and the root differ from the `desc-preproc` source. A
write-if-absent `dataset_description.json` (`GeneratedBy: hypline`) gives the
tree honest provenance: hypline's denoising is its own pipeline, not a
continuation of fmriprep, so claiming fmriprep's `GeneratedBy` would be false
metadata. A per-file sidecar can't override the tree-level `GeneratedBy`, which
is why the area is separate rather than in-tree.

### The split is by BIDS-conformance, not by generator

Both `derivatives/hypline/` and the root-level areas are hypline-generated, so
"who made it" is not the sorting rule:

- **`derivatives/hypline/`** — imaging derivatives that genuinely obey the BIDS
  derivative contract (full identity entities, recognized `_bold` suffix +
  nifti/gifti ext, `func` datatype). They earn a place under `derivatives/`.
- **Root-level `stimuli/`, `features/`, `confounds/`, `nuisance/`** —
  intentionally non-conformant (parquet vector payloads, non-reserved
  `feat-`/`conf-`/`nuis-` entities, kind-foldered, or wide TSV). They stay at
  root rather than masquerade as compliant derivatives.

Do not "fix the inconsistency" by forcing every hypline output under
`derivatives/`.

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
