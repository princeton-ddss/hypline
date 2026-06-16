# Syntactic — scope and assumptions

How hypline derives per-token syntactic-function features (POS tag, dependency
relation, stopword flag) from word-level transcripts via spaCy.

## Source: word-level transcripts

Like phonemic ([phonemic.md](phonemic.md)) and semantic
([semantic.md](semantic.md)), syntactic features derive from word-level
transcripts. One row **per spaCy token**: a word that tokenizes into several
pieces (`"don't"` -> `do` + `n't`) yields several rows sharing that word's
`start_time`, attributed back by char-span overlap.

## Turn-grouped parsing

Words are parsed **one conversational turn at a time**, grouped by `turn_sub`
([events.md](events.md), [transcriber.md](transcriber.md)). The dependency
parser needs coherent utterances; concatenating both speakers' words would
corrupt parse trees across turn boundaries. Each maximal run of equal `turn_sub`
becomes one spaCy doc.

`turn_sub` is forward-filled **before** the null-word drop: stamp_turns leaves
untimed words null, but they belong to the surrounding utterance — leaving a null
would split a turn mid-sentence and corrupt its parse.

Transcript loading is shared across all three word-level features via
`features._utils.load_transcript_words` (which requires a `turn_sub` column).
Phonemic and semantic carry `turn_sub` through to their output too, though only
syntactic groups parsing on it.

## Retains untimed words as parse context

Untimed (null `start_time`) words stay in the input — the producer-retains /
consumer-drops contract ([../decisions/feature-files.md](../decisions/feature-files.md)),
same stake as semantic. The reason differs: not contextual embeddings but the
**parse tree** — dropping an untimed word would fragment its turn's sentence and
mis-tag its neighbors. Emitted with null `start_time` but real features.

A null **word** (vs. null `start_time`) is dropped and warned — it is not
faithful text.

## Fixed-width one-hot, model-vocab fit

The feature is a one-hot of the POS tag concatenated with the dependency
relation, plus a final 0/1 stopword dim. Width and column order are fit to the
**model's full label vocab** (`tagger`/`parser` `pipe_labels`), not the tags a
given transcript happens to use — a per-file fit would drift in both. A label
outside that vocab (e.g. `""` on an unparsed token) leaves its block all-zero
rather than erroring; a missing tag is a meaningful all-zero state. Dim labels
ship in the feature metadata.

## Model identity

`en_core_web_lg`, a fixed constant (contrast semantic's open HF id — the spaCy
model set is small and the parse quality depends on the specific model). Loaded
once as a class attribute and **auto-downloaded** in `__init__` on first use. The
CLI therefore discovers dyads *before* constructing `SyntacticFeature`, to avoid
a download when there is nothing to generate.

## No paired confound

Like spectral ([spectral.md](spectral.md)), a stimulus feature with no derived
nuisance — no paired confound generator. See spectral for the phonemic/semantic
contrast.
