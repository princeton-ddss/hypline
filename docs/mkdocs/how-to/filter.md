# Filter to specific runs or conditions

Most hypline commands run over *every* subject, run, and segment they discover by
convention. When you want a command to touch only part of your dataset — one
subject, two runs, a single condition — you narrow it with two options that
appear on nearly every command:

- **`--sub-ids`** — pick which subjects to process.
- **`--data-filters`** — pick which runs and conditions within them.

This guide shows the common selections. For *why* segments and conditions work
the way they do, see [Segments and metadata](../concepts/segments.md); for the
full option list of any single command, see its [Reference](../reference/transcribe.md)
page.

## Select subjects with `--sub-ids`

Pass comma-separated subject IDs (the value of the `sub-` entity, without the
prefix):

```bash
# only subjects 01 and 02
hypline featuregen phonemic data/ --sub-ids 01,02
```

Omit `--sub-ids` entirely to process every subject found wherever that command
looks for its inputs.

## Select runs and conditions with `--data-filters`

`--data-filters` accepts comma-separated `entity-value` tokens. A token matches
against **both** filename entities (like `run`) and the metadata you defined in
`events.json` (like `cond`), so the same option filters structural and
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

`cond` never appears in a filename — it lives in the `events.json` sidecar — but
you filter on it the same way:

```bash
# only the "R" condition
hypline denoise data/ --columns trans_x,trans_y,trans_z,rot_x,rot_y,rot_z --data-filters cond-R
```

### Runs AND a condition

Mixing *different* entities narrows the match — every named entity must hold:

```bash
# (run 1 or run 2) AND condition R
hypline denoise data/ \
  --columns trans_x,trans_y,trans_z,rot_x,rot_y,rot_z \
  --data-filters run-1,run-2,cond-R
```

## The matching rule, in one line

| Tokens | Reads as |
| ------ | -------- |
| Same entity, multiple values | **OR** — `run-1,run-2` → run 1 *or* 2 |
| Different entities | **AND** — `run-1,cond-R` → run 1 *and* condition R |

So `run-1,run-2,cond-G` means *(run 1 or run 2) and condition G*. Combine
`--sub-ids` with `--data-filters` to slice on both axes at once:

```bash
# subjects 01–02, run 1 only, condition R
hypline denoise data/ \
  --columns trans_x,trans_y,trans_z,rot_x,rot_y,rot_z \
  --sub-ids 01,02 \
  --data-filters run-1,cond-R
```

## Combining with `--force`

`--sub-ids` and `--data-filters` decide *what is considered*; `--force` decides
whether already-generated outputs are overwritten. They compose — scope a rerun
to one run and force just that one:

```bash
hypline featuregen phonemic data/ --data-filters run-1 --force
```

See [Regenerate outputs after a fix](regenerate.md) for the rerun workflow.

## When a filter matches nothing

Two outcomes look similar but mean different things:

- **`No subjects found`** — the input location is empty, or your `--sub-ids` /
  `--data-filters` excluded everything. Widen the filter, or confirm the files
  are in place.
- **An error naming an unknown entity** — a filter token names an entity that
  exists *nowhere*, neither on filenames nor in `events.json` metadata. Hypline
  treats this as a typo and raises, rather than silently matching nothing. Check
  the spelling against your filenames and sidecars.
