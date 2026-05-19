# Confound files

Design record for hypline's confound file format — schema, naming rules, and
alignment contract with BOLD data.

Hypline's own derivative type — TR-aligned nuisance regressors paired to BOLD
runs. `confounds/` is the single home for **any** TR-aligned nuisance
regressor used in hypline, regardless of origin: stimulus-derived (e.g.
phonemic onsets/rate), motion or physio, CompCor components, or regressors
imported from external pipelines. Files are TR-aligned at save time so the
on-disk contract is uniform across origins.

External sources like fmriprep's `desc-confounds_timeseries.tsv` are not
read in place; they are converted into hypline confound files via
`confoundgen` so provenance lives in the Parquet footer rather than relying
on the source pipeline's column conventions at read time.

Confound I/O lives in `hypline.confounds`: `save_confound` writes,
`read_confound` reads, `read_confound_metadata` reads the footer without
loading data. TR alignment is handled by `hypline.downsample` (shared with
feature files — see [feature-files.md](feature-files.md)).

Standard confounds are produced by `hypline confoundgen <kind>` (e.g.
`hypline confoundgen phonemic` for stimulus-derived, `hypline confoundgen
fmriprep` for imports from fmriprep outputs). Users may also write custom
confounds directly, as long as the conventions below are followed.

## Format

- Extension: `.parquet`
- Required columns:
  - `start_time` (numeric, seconds; must begin at `0.0` and have intervals
    equal to `repetition_time` — TR-aligned at save time)
  - `confound` (Array or List column — per-TR confound vector; normalized to
    `Array(Float64)` on write)

## Naming

Confound filenames carry **stimulus-side identity entities** from the source
BOLD (`sub`, `ses`, `task`, `run`), any segment entity value, hypline's own
`conf-<kind>` entity (e.g. `conf-phonemic`), and an **optional** `desc-<name>`
entity discriminating variants within a kind (e.g. `desc-onset`, `desc-rate`).
Like feature files, they carry **no BIDS suffix** — no standard BIDS suffix
exists for derived confounds. Validation rejects any path with a suffix.

`conf-<kind>` mirrors `feat-<kind>` on feature files: it names the generator
kind (matches the `confoundgen` subcommand) and the `confounds/<kind>/`
directory. `desc-<name>` is used when a kind exposes multiple
individually-selectable regressors (phonemic → `desc-onset`, `desc-rate`);
omit it when a kind has a single canonical confound. Downstream selection
uses `kind` or `kind-name`, e.g. `--confounds=phonemic-onset,phonemic-rate`.

Within a single kind, files must either **all** carry `desc` or **none**
do — mixing `conf-phonemic.parquet` with `conf-phonemic_desc-onset.parquet`
is ambiguous and disallowed.

`desc` here means "which confound within the kind," **not** "variant of one
confound." Two different ways of computing `desc-onset` cannot coexist under
`conf-phonemic` — pick one canonical method per `(conf, desc)` pair.

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
- `n_trs` — number of TR rows. Source from the preprocessed BOLD the
  confounds will be regressed against, not the raw image — see
  [../modules/bold.md](../modules/bold.md).
- `confound_dim` — per-row confound vector width

Caller-supplied keys should let a consumer reproduce the regressor (generator
parameters that change what was written). Keys prefixed with `_` are reserved
for genuinely per-file metadata and exempt from cross-file equality checks.

## Relationship to feature files

Feature files (see [feature-files.md](feature-files.md)) hold X regressors at
the generator's natural unit and may be downsampled on the fly at read time.
Confound files hold nuisance regressors **already** at TR resolution — the
TR-alignment is part of the on-disk contract, not deferred to read time.
