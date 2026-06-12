# Filter to specific runs or conditions

Most hypline commands run over _everything_ they discover by convention — every
run and segment, and every dyad or subject. When you want a command to touch only
part of your dataset — one dyad, two runs, a single condition — you narrow it with
two options that appear on nearly every command:

- **`--dyad-ids`** / **`--sub-ids`** — pick which dyads or subjects to process.
  Which one a command takes follows the area it writes: the dyad-keyed stimulus
  commands (`transcribe`, `featuregen`, `confoundgen`) take `--dyad-ids`, while
  the sub-keyed `denoise` takes `--sub-ids`. See [Subject vs.
  dyad](../concepts/layout.md#subject-vs-dyad).
- **`--data-filters`** — pick which runs and conditions within them.

## Select dyads or subjects

Pass comma-separated IDs (the value of the identity entity, without the prefix).
Use `--dyad-ids` on the stimulus commands and `--sub-ids` on `denoise`:

```bash
# stimulus command: only dyad 103 (pass more as a comma-separated list)
hypline featuregen phonemic data/ --dyad-ids 103

# denoise: only subjects 003 and 103 (the two partners of dyad-103)
hypline denoise data/ --columns trans_x,trans_y,trans_z --sub-ids 003,103
```

Omit the option entirely to process every dyad (or subject) found wherever that
command looks for its inputs.

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

Mixing _different_ entities narrows the match — every named entity must hold:

```bash
# (run 1 or run 2) AND condition R
hypline denoise data/ \
  --columns trans_x,trans_y,trans_z,rot_x,rot_y,rot_z \
  --data-filters run-1,run-2,cond-R
```

## The matching rule, in one line

| Tokens                       | Reads as                                           |
| ---------------------------- | -------------------------------------------------- |
| Same entity, multiple values | **OR** — `run-1,run-2` → run 1 _or_ 2              |
| Different entities           | **AND** — `run-1,cond-R` → run 1 _and_ condition R |

So `run-1,run-2,cond-G` means _(run 1 or run 2) and condition G_. Combine the
identity option (`--dyad-ids` / `--sub-ids`) with `--data-filters` to slice on
both axes at once:

```bash
# subjects 003 and 103, run 1 only, condition R
hypline denoise data/ \
  --columns trans_x,trans_y,trans_z,rot_x,rot_y,rot_z \
  --sub-ids 003,103 \
  --data-filters run-1,cond-R
```

## Combining with `--force`

The identity option (`--dyad-ids` / `--sub-ids`) and `--data-filters` decide
_what is considered_; `--force` decides whether already-generated outputs are
overwritten. They compose — scope a rerun to one run and force just that one:

```bash
hypline featuregen phonemic data/ --data-filters run-1 --force
```

See [Regenerate outputs after a fix](regenerate.md) for the rerun workflow.

## When a filter matches nothing

Two outcomes look similar but mean different things:

- **`No dyads found`** / **`No subjects found`** — the input location is empty, or
  your `--dyad-ids` / `--sub-ids` / `--data-filters` excluded everything. Widen the
  filter, or confirm the files are in place. (Stimulus commands report dyads;
  `denoise` reports subjects.)
- **An error naming an unknown entity** — a filter token names an entity that
  exists _nowhere_, neither on filenames nor in `events.json` metadata. Hypline
  treats this as a typo and raises, rather than silently matching nothing. Check
  the spelling against your filenames and sidecars.
