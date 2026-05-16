# BIDSPath validation scope

What `BIDSPath` enforces vs. what it leaves to callers.

## Enforced invariants

- At least one entity must parse from the filename.
- First entity must be `sub` — guarantees `bp.sub` is always present and matches
  BIDS' rule that subject is the leading entity.
- Entity keys/values, suffix, and extension match the BIDS character grammar
  (see `BIDS_ENTITY_RE`, `BIDS_SUFFIX_RE`, `EXTENSION_RE`).
- Extension must be present and non-empty. Every BIDS file has one; wrap
  directories as `Path`, not `BIDSPath`.
- Non-entity segments are only allowed as the trailing suffix.

## Deliberate relaxations from the BIDS spec

- **Suffix is optional.** Some hypline-internal paths (e.g., feature files
  identified by `feat-<label>`) have no BIDS suffix. See
  [feature-files.md](feature-files.md).
- **`task` is not required.** BIDS mandates `task` for `bold`/`events` but not
  for anatomicals. BIDSPath stays general; callers that need `task` (BOLD
  loading, encoding) enforce it themselves — typically via
  `validate_entity_invariance(..., ("task",))` or direct entity access.

## Why call-site enforcement for `task`

Pushing `task`-required into BIDSPath would block legitimate anatomical or
derivative paths. The type stays honest about what it can represent; consumers
that depend on `task` raise where the dependency actually exists.

Concrete sites:
- BOLD/events loading in `bold` — events sidecars are resolved by task, so
  loaders raise when `task` is absent.
- Encoding invariance checks in `encoding` — feature and BOLD files must agree
  on `task` (see [../modules/encoding.md](../modules/encoding.md)).
