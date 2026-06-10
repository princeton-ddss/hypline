# Dyad-keyed shared-conversation areas

Why some layout areas are keyed by `dyad` and others by `sub`, and how the two
worlds bridge.

## The split axis is derivation source, not convenience

hypline is a hyperscanning pipeline for dyadic conversation. An artifact is
keyed by **what it is derived from**:

- **dyad-keyed** — derived from the *shared conversation* between two partners:
  `stimuli/`, `features/`, `confounds/` (and `transcribe`, which writes
  `stimuli/`). One conversation → one dyad → one set of stimuli/features/
  confounds, consumed later by each partner's per-subject encoding model. A
  `dyad-003` audio file is the dyad's shared recording, not either partner's.
- **sub-keyed** — derived from one *brain*: raw BOLD, `derivatives/fmriprep`,
  `derivatives/hypline` (denoised), `nuisance/`, `results/`.

Rejected: keying everything by `dyad` (breaks per-subject BOLD/nuisance/results)
or leaving stimuli sub-keyed (the file describes the dyad, not one partner).

On-disk tree: `stimuli/dyad-XXX/…`, `features/dyad-XXX/…`,
`confounds/dyad-XXX/…`. See [layout.md](layout.md).

## Bridge = `participants.tsv` `dyad_id` column

The single source of truth for `dyad ↔ sub` is a standard BIDS
`participants.tsv` at `bids_root`, with `participant_id` plus a custom `dyad_id`
column. `read_participants` yields a bare `sub -> dyad` map (prefixes stripped);
`BIDSLayout.dyad_of(sub)` and `subjects_of(dyad)` expose it, read-once cached.
Missing file is a hard error (the mapping is required infra); an unmapped
sub/dyad raises `KeyError`. The file is read lazily, so sub-keyed-only workflows
that never cross worlds still run without it.

Rejected: a partner pointer in the filename (bloats every stem, duplicates a
fact that belongs in one table) or a bespoke mapping file (participants.tsv is
the BIDS-blessed home for subject-level metadata).

## Two dyad→sub crossings, one mechanism: pick-first-partner

A dyad-keyed artifact has no `sub`, but it must sometimes reach a real brain to
find sub-keyed raw-tree sidecars or BOLD. Two crossings exist, and both resolve
the same way — **pick the dyad's first subject** (`subjects_of(dyad)[0]`):

1. **Discovery sidecar resolution** (`events.py`). A dyad-keyed path has no
   `sub`, but `events.tsv`/`events.json` are sub-keyed. Any `--data-filters`
   carrying an events-backed entity (e.g. `cond-G`, `item-101`) must resolve
   against a real subject's sidecar.
2. **confoundgen TR source** (`phonemic.py`). Downsampling needs `n_trs`/TR from
   a real BOLD; a dyad feature has none. Synthesize a sub-keyed source from the
   first subject (`BIDSPath.with_identity`) and run the existing
   `resolve_bold_image`/`load_bold_meta`.

**Load-bearing invariant.** Pick-first is safe only because both partners scan
*simultaneously on one sequence*: TR/n_trs and `events.json` are identical
across partners by construction. The protocol forbids the alternative, so the
code does not assert invariance (a second BOLD resolution per file guarding a
condition that cannot occur). If hyperscanning ever allows partners on separate
sequences or timelines, pick-first breaks here.

**Exception — speaking turns are complementary, not identical.** `turn_speaker`
rows in each partner's events.tsv mark *that subject's own* turn windows, so the
partners' turn info differs by design. `load_turns` (events.py) therefore reads
*every* partner (`subjects_of(dyad)`) and unions the windows — pick-first does
not apply. The same simultaneous-timeline invariant still underwrites it: turn
windows from the two files share one clock, so cross-file overlap is genuine
cross-talk (raised) rather than a timeline-misalignment artifact.

## Encoding seam is a stopgap

`Encoding._discover_features` routes `sub → dyad` via `dyad_of` so dyad features
join to a sub-keyed BOLD model. `CellKey` keeps both `sub` and `dyad` as
**invariant identities, not cell axes** — dyad features join to sub-keyed BOLD
without `dyad` becoming a grouping axis. The full cross-subject join (train on
one partner, predict another) is a separate, larger change and is not yet
shipped. See [../modules/encoding.md](../modules/encoding.md).

See [bidspath-validation.md](bidspath-validation.md) for the `sub` xor `dyad`
identity rule and `with_identity` re-keying.
