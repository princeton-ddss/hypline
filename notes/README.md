# Hypline notes

Domain knowledge behind the hypline codebase — assumptions, decisions, and
external contracts that guide development. Reference material for humans and
AI agents alike. Agent-agnostic by design.

## Scope — what goes where

| Content type                           | Location              |
|----------------------------------------|-----------------------|
| BIDS / fMRIPrep specs we rely on       | `notes/external/`     |
| Project decisions, custom conventions  | `notes/decisions/`    |
| Per-module scope and assumptions       | `notes/modules/`      |

## Adding notes

- **Record durable domain knowledge only** — if it will be true in six months
  regardless of who is editing the code, it belongs here.
- Keep entries lean: every line should earn its place.
- One topic per file. Split when a file exceeds ~150 lines or covers multiple
  concerns. Prefer extending an existing file over creating new ones.
- Kebab-case filenames, singular, descriptive.
- Start each file with a one-line purpose statement.
- Flag assumptions that could break under future changes.
- Link between files with relative paths. When a new decision affects an
  existing module or decision, update the consumer to link back.
- Create subfolders lazily — only when content clearly belongs in one.

## What NOT to record

- **Implementation details** already visible in the code. Docstrings and the
  code itself are authoritative.
- **Session or change history.** That belongs in commit messages and PR
  descriptions.
- **Obvious restatements** of the BIDS spec or library behavior. Only record
  external behavior we specifically depend on or that has bitten us.
- **Speculative conventions** not yet reflected in code. Write them down when
  they become real.

## Where to look by task

| Task                                   | Start here                                          |
|----------------------------------------|-----------------------------------------------------|
| Adding or renaming a BIDS entity       | `external/bids.md`, `decisions/partition-entity.md`  |
| Changing the feature file schema       | `decisions/feature-files.md`, `modules/encoding.md`|
| Editing encoding / regression logic    | `modules/encoding.md`                               |
| Consuming fMRIPrep outputs             | `external/fmriprep.md`                              |
| Introducing a new custom entity        | `decisions/partition-entity.md`, `external/bids.md`  |
