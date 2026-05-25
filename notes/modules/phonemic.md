# Phonemic — scope and assumptions

How hypline derives phoneme-level features and the upstream-source caveat that
shapes the current implementation.

## Source: word-level transcripts

Phonemic features are currently derived from word-level transcripts (one row
per word, with a single `start_time` per word). We do not yet have true
phoneme-level alignment from audio.

Consequence: all phonemes belonging to the same transcribed word share that
word's `start_time`. Within-word phoneme order is preserved by row order, but
no sub-word timing exists.

Revisit if/when audio-level alignment (e.g. forced alignment via MFA) becomes
available — that would give each phoneme its own onset, and this caveat goes
away.

## CMUdict lookup

- We use the first listed variant (`pronunciations[0]`). Without phoneme-level
  audio alignment there is no signal to disambiguate variants; the first entry
  is the canonical form and alternates are numbered from 1.
- Stress digits (`0`, `1`, `2`) are stripped — stress is a prosodic property
  layered on top of phoneme identity and does not change articulatory
  features. If stress is needed downstream it should be a separate column,
  not folded into the phoneme inventory.

## OOV and punctuation handling

A transcript token may yield no phonemes in three cases:

- Word absent from CMUdict
- Word maps to an empty pronunciation
- Token is punctuation-only (becomes empty after `strip(PUNCTUATION)`)

All three follow the project-wide missing-unit convention — see
[../decisions/feature-files.md](../decisions/feature-files.md): one row at the
token's `start_time` with `phoneme=None` and a zero `feature` vector.

A token with a null `start_time` (un-alignable by WhisperX — see
[transcriber.md](transcriber.md)) is handled differently: it is **dropped**, not
emitted as a missing-unit row. A timing-less event cannot be binned to a TR, so
it carries no usable signal, and a kept null row would be rejected by
downstream downsampling, which fails fast on NaN timestamps (see
[../../src/hypline/downsample.py](../../src/hypline/downsample.py)).

## Feature vector

Articulatory features (place, manner, voicing, etc.) are intrinsic phoneme
properties, so multi-hot per phoneme is the natural representation.

## Confound generation collapses `desc-*` variants

Phonemic confounds (`desc-onset`, `desc-rate`) depend only on phoneme
`start_time` — values are discarded during downsampling. Since `desc-*`
variants share a `start_time` grid by convention
([../decisions/feature-files.md](../decisions/feature-files.md)), confound
generation collapses each non-`desc` identity group to one representative.
Output is therefore independent of which variant is present — the basis for the
timing-only scope rule
([../decisions/confound-files.md](../decisions/confound-files.md)).
