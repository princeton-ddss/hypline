# Confound files

Design record for hypline's confound file format — schema, naming rules, and
alignment contract with BOLD data.

Hypline's own derivative type — TR-aligned regressors paired to BOLD runs,
holding **stimulus-derived** confounds (e.g. phonemic onsets/rate). `confounds/`
is the home for feature-granular confounds; it is **not** denoise's input.

Consumer: confounds enter the **encoding** fit. `EncodingTrainer(confounds=[...])`
places them in X as a single ridge band (all configured confounds collapse into
one band regardless of count), partialled out alongside the feature bands within
the same fit — see [../modules/encoding.md](../modules/encoding.md). A ref is one
role per fit — feature xor confound, never both — enforced at construction. These
are stimulus-derived nuisance modeled *in* the encoding regression, distinct from
the run-level nuisance denoise strips from BOLD in a prior stage.

Run-level nuisance regressors used for denoising (motion, physio, CompCor,
fmriprep columns) live elsewhere: fmriprep's `desc-confounds_timeseries.tsv` is
read natively by denoise, and custom/hypline-generated nuisance lives in the
`nuisance/` folder. See [nuisance-files.md](nuisance-files.md) for that contract
and why the two are split by granularity rather than sharing one folder.

Confound I/O lives in `hypline.io` and is re-exported at `hypline.*`:
entity-based `save_confound`, path-based `read_confound` /
`read_confound_metadata` — same shape as feature I/O, see
[feature-files.md](feature-files.md) for rationale. TR alignment is handled
by `hypline.downsample` (shared with feature files).

Standard confounds are produced by `hypline confoundgen <kind>` (e.g.
`hypline confoundgen phonemic` for stimulus-derived confounds). Users may also
write custom confounds directly, as long as the conventions below are followed.

## Scope: standard generators produce variant-independent confounds only

Shipped standard generators (e.g. `phonemic`) derive confounds purely from
event **timing** (`start_time`), never from the feature **value** column. Every
`desc` variant of a feature shares an identical timing grid (feature-files
contract), so the confound output is provably independent of which variant is
read. A generator therefore sources all variants and collapses each to a single
timing source per cell, picked deterministically so repeated runs agree.

A shipped generator that reads the `feature` value column is a policy
violation — its output would silently depend on which variant was picked.
**Value-dependent regressors are out of scope for standard generation**; users
author those themselves via `save_confound`, choosing their own `desc`.

The collapse (`pick_timing_source`) **trusts** the shared-grid invariant rather
than validating it. Why it is unenforced — and what happens to a mislabeled
divergent variant — is covered in [feature-files.md](feature-files.md).

## Format

- Extension: `.parquet`
- Required columns:
  - `start_time` (numeric, seconds; must begin at `0.0` and have intervals
    equal to `repetition_time` — TR-aligned at save time)
  - `confound` (Array or List column — per-TR confound vector; normalized to
    `Array(Float64)` on write. Finite values only — NaN/inf are rejected at
    both write and read, so consumers may regress confound data without
    re-checking for non-finite values.)

## Naming

Confound files are **dyad-keyed** — like features, they describe the shared
conversation (see [dyad-keyed.md](dyad-keyed.md)). The stem carries the source
feature's identity entities (`dyad`, `ses`, `task`, `run`), any segment entity
value, and two hypline entities: `conf-<kind>` names the **source** a confound
is derived from (matches the `confoundgen` subcommand and the `confounds/<kind>/`
directory), and an **optional** `desc-<name>` names **which derivation** of that
source (phonemic timestamps → `desc-onset` indicator, `desc-rate` count). This is
the same source/derivation meaning and `<kind>[-<desc>]` template feature files
use — see [feature-files.md](feature-files.md). Like feature files, they carry
**no BIDS suffix** (none exists for derived confounds); validation rejects any
path with one.

A bare `conf-<kind>` is the kind's unnamed default derivation and a legitimate
confound on its own: a user computing their own single derivation of the
`phonemic` source writes a bare `conf-phonemic`, independent of the shipped
`desc-onset`/`desc-rate`. Bare and named forms coexist freely under one kind.
A selector token resolves to its matching derivation — `phonemic` to the bare
file, `phonemic-onset` to that named one, `*` to the whole kind.

One invariant: a `desc` labels *which* derivation, not *how* it was computed, so
one canonical method per `(conf, desc)` pair — two ways of computing `desc-onset`
cannot coexist under `conf-phonemic`.

Confound files live under `confounds/` — see [layout.md](layout.md) for the
root tree.

## Parquet metadata

Confound files carry a `hypline` JSON blob in the Parquet footer. Reserved
auto-injected keys (callers must not supply them):

- `confound_kind` — validated against the `conf` filename entity on read
- `confound_variant` — validated against the `desc` filename entity on read
  (`None` when the filename has no `desc` entity)
- `hypline_version` — package version provenance
- `tr_method` — free-form label for how TR-aligned rows were produced
  (downsampling method, upsampling method, or a marker for native-TR
  computation). `None` is allowed when no such label applies. Recorded
  verbatim; not validated by `save_confound`. Must be equal across files
  sharing the same `(conf, desc)` pair for consistency checks to pass
  (`None == None` passes).
- `repetition_time` — TR of the target BOLD acquisition, in seconds.
  Required at save time because a single-row DataFrame carries no spacing,
  and inferring TR from row spacing would silently disagree with the BOLD's
  true TR.
- `n_trs` — number of TR rows. Equal to the BOLD volume count; hypline
  enforces raw and preprocessed counts to match, so either anchor is correct —
  see [../modules/bold.md](../modules/bold.md).
- `confound_dim` — per-row confound vector width

Caller-supplied keys should let a consumer reproduce the regressor (generator
parameters that change what was written). Keys prefixed with `_` are reserved
for genuinely per-file metadata and exempt from cross-file equality checks.

## Relationship to feature files

Feature files (see [feature-files.md](feature-files.md)) hold X regressors at
the generator's natural unit and may be downsampled on the fly at read time.
Confound files hold confound regressors **already** at TR resolution — the
TR-alignment is part of the on-disk contract, not deferred to read time.
