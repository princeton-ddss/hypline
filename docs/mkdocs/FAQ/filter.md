# Filter to specific runs or conditions

Most hypline commands run over everything they discover by convention: every
run and segment, and every dyad or subject. When you want a command to touch only
part of your dataset (one dyad, two runs, a single condition), you narrow it with
two options that appear on nearly every command:

- **`--dyad-ids`** / **`--sub-ids`** — pick which dyads or subjects to process.
  Which one a command takes follows the area it writes: the dyad-keyed stimulus
  commands (`transcribe`, `featuregen`, `confoundgen`) take `--dyad-ids`, while
  the sub-keyed `denoise` takes `--sub-ids`. See [Subject vs.
  Dyad](../step-by-step/layout.md#subject-vs-dyad).
- **`--data-filters`** — pick which runs and conditions within them.

This guide shows the common selections. For why segments and conditions work
the way they do, see [Segments and metadata](../step-by-step/segments.md); for the
full option list of any single command, see its [step-by-step guide](../step-by-step/transcribe.md)
page.

## Select dyads or subjects

Pass comma-separated IDs (the value of the identity entity, without the prefix).
Use `--dyad-ids` on the stimulus commands and `--sub-ids` on `denoise`:

```bash
# stimulus command: only dyad 040 (pass more as a comma-separated list)
hypline featuregen phonemic data/ --dyad-ids 040

# denoise: only subjects 041 and 042 (the two partners of dyad-040)
hypline denoise data/ --columns trans_x,trans_y,trans_z --sub-ids 041,042
```

Omit the option entirely to process every dyad (or subject) found wherever that
command looks for its inputs.

## Select runs and conditions with `--data-filters`

`--data-filters` accepts comma-separated `entity-value` tokens. A token matches
against both filename entities (like `run`) and the metadata you defined in
`events.json` (like `condition`), so the same option filters structural and
descriptive attributes alike.

### One run

```bash
hypline denoise data/ --columns trans_x,trans_y,trans_z,rot_x,rot_y,rot_z --data-filters run-1
```

### Several runs (OR)

Repeating the same entity widens the match — any of the listed values qualifies:

```bash
# run 1 or run 2
hypline featuregen phonemic data/ --data-filters run-1,run-2
```

### A condition from `events.json` (metadata)

`condition` never appears in a filename (it lives in the `events.json` sidecar), but
you filter on it the same way:

```bash
# only the "R" condition
hypline featuregen phonemic data/ --data-filters condition-R
```

!!! warning "`denoise` and segment-level conditions"

    fMRIPrep BOLD files cover a whole run and carry no segment entity like
    `trial`. So on trial-segmented runs, a `condition` filter on `denoise` fails with
    `Path is missing segment entity 'trial' declared in events.tsv`. It works
    only when the run is declared with a single `task-<name>` row (see [Three
    kinds of run](../step-by-step/segments.md#three-kinds-of-run)). Otherwise,
    filter `denoise` by `run`.

### Runs AND a condition

Mixing different entities narrows the match — every named entity must hold:

```bash
# (run 1 or run 2) AND condition R
hypline featuregen phonemic data/ --data-filters run-1,run-2,condition-R
```

## The matching rule at a glance

| Tokens | Reads as |
| ------ | -------- |
| Same entity, multiple values | **OR** — `run-1,run-2` → run 1 or 2 |
| Different entities | **AND** — `run-1,condition-R` → run 1 and condition R |

So `run-1,run-2,condition-G` means *(run 1 or run 2) and condition G*. Combine the
identity option (`--dyad-ids` / `--sub-ids`) with `--data-filters` to slice on
both axes at once:

```bash
# dyad 040, run 1 only, condition R
hypline featuregen phonemic data/ \
  --dyad-ids 040 \
  --data-filters run-1,condition-R
```

## Combining with `--force`

The identity option (`--dyad-ids` / `--sub-ids`) and `--data-filters` decide
what is considered; `--force` decides whether already-generated outputs are
overwritten. They compose, so you can scope a rerun to one run and force just
that one:

```bash
hypline featuregen phonemic data/ --data-filters run-1 --force
```

See [Regenerate outputs after a fix](regenerate.md) for the rerun workflow.

## When a filter matches nothing

Three outcomes look similar but mean different things:

- **`No dyads found`** / **`No subjects found`** — no ids were discovered at all,
  because the input location is empty. Neither `--dyad-ids` / `--sub-ids` nor
  `--data-filters` triggers this: passed ids skip discovery, and discovery
  ignores filters. Confirm the files are in place. (Stimulus commands report
  dyads; `denoise` and `encoding train` report subjects.)
- **A per-id failure for an id you passed** (and the command exits `1`) — a
  `--dyad-ids` / `--sub-ids` value names an id that does not exist, e.g.
  `dyad 'dyad-999' not found under stimuli/`. Check the id against your dataset.
- **A per-id failure: `{id} failed: Files found but none matched …`**
  (and the command exits `1`) — ids were found, but a `--data-filters` token
  matched nothing for that id, whether because it names an entity that exists
  nowhere, or a value present on no file. Hypline treats an empty match as a typo
  and raises rather than silently skipping. Check the spelling against your
  filenames and `events.json` sidecars.
