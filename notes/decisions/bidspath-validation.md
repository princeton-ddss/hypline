# BIDSPath validation scope

What `BIDSPath` enforces vs. what it leaves to callers.

## Enforced invariants

- At least one entity must parse from the filename.
- Exactly one leading **identity** entity — `sub` xor `dyad`
  (`IDENTITY_ENTITIES`) — and it must lead the stem. `sub` keys per-brain files,
  `dyad` keys shared-conversation files (`stimuli`/`features`/`confounds`); see
  [dyad-keyed.md](dyad-keyed.md). `dyad` is **not** a BOLD identity entity
  (`BOLD_IDENTITY_ENTITIES` stays `sub`/`ses`/`task`/`run`). Re-key between the
  two with `with_identity(key, value)`; `with_entity`/`without_entity` reject
  identity keys.
- Entity keys/values, suffix, and extension match the BIDS character grammar
  (see `BIDS_ENTITY_RE`, `BIDS_SUFFIX_RE`, `EXTENSION_RE`).
- Extension must be present and non-empty. Every BIDS file has one; wrap
  directories as `Path`, not `BIDSPath`.
- Non-entity segments are only allowed as the trailing suffix.

## Entity value grammar is strict

Entity values must match `[a-zA-Z0-9]+` (per BIDS spec). Construction raises
`ValueError` for any violation — notably dots inside values (e.g. `item-1.0`).
Use `item-1` or `item-1p0` if a decimal-looking label is needed.

Why strict: a dot inside an entity value used to be swallowed by extension
splitting, causing `find_bids_files` to silently exclude the file and surface
a misleading "no files found" downstream. Loud rejection at construction
prevents that class of bug.

## Disallowed entities

`BIDSPath` rejects a subset of BIDS entities at construction —
`acq`, `ce`, `rec`, `dir`, `echo`, `part`, `chunk`. See
[unsupported-entities.md](unsupported-entities.md) for the list and rationale.

## Canonical entity order when constructing from kwargs

`BIDSPath.from_entities` places entities in a fixed order so derived paths
across the codebase look identical regardless of kwarg order:

1. Identity: the leading identity (`sub` xor `dyad`), then `ses`, `task`, `run`
2. Any other entities, alphabetically
3. Category (`stim`/`feat`/`conf`/`nuis`), then `desc`

`desc` trails the category so variant tagging stays adjacent to what it
modifies (e.g. `feat-llm_desc-v1`). At most one category entity is allowed.
Parsing (`BIDSPath(path)`) preserves whatever order the filename already has;
canonical ordering applies only on construction from kwargs.

## Deliberate relaxations from the BIDS spec

- **Suffix is optional.** Some hypline-internal paths (e.g., feature and confound
  files identified by `feat-<label>` / `conf-<label>`) have no BIDS suffix. See
  [feature-files.md](feature-files.md). Nuisance files are the exception: they
  carry the `_timeseries` suffix by design (a recognized BIDS timeseries form) —
  see [nuisance-files.md](nuisance-files.md). `BIDSPath` permits both — the
  suffix requirement lives in each reader, not in path construction.
- **`task` is not required.** BIDS mandates `task` for `bold`/`events` but not
  for anatomicals. BIDSPath stays general; callers that need `task` (BOLD
  loading, encoding) enforce it themselves — via direct entity access, an
  explicit `task-*` filter at discovery, or `validate_entity_invariance(...,
  ("task",))`.

## Why call-site enforcement for `task`

Pushing `task`-required into BIDSPath would block legitimate anatomical or
derivative paths. The type stays honest about what it can represent; consumers
that depend on `task` raise where the dependency actually exists.

Concrete sites:
- BOLD/events loading in `bold` — BIDS requires `task` on `bold` (events
  inherit it), so loaders raise when it's absent.
- Encoding in `encoding` — `Encoding(tasks=[...])` declares which tasks are in
  scope; discovery passes `task-*` filters to `layout.find`, so out-of-scope
  files never enter the pipeline (see [../modules/encoding.md](../modules/encoding.md)).
- `layout.find.stimuli` and `layout.find.features` — every file in these
  areas is tied to a BOLD run via `task`, and derivatives (transcripts,
  features, confounds) inherit it from their source. The check lives at the
  discovery boundary, so downstream consumers don't each re-validate.
  `layout.find.fmriprep` deliberately skips the check because
  fmriprep produces non-func artifacts (anat, xfm) that legitimately omit
  `task`; the `bold`-specific loaders enforce it where needed.
