# Spectral — scope and assumptions

How hypline turns stimulus audio into a Whisper log-Mel spectrogram aligned to
the BOLD TR grid. The feature is the spectrogram itself — *not* the Whisper
model's hidden states.

## Source: stimulus audio

Unlike phonemic/semantic (which read word-level transcripts), spectral reads the
stimulus audio directly, decoded at the extractor's sampling rate via whisperX's
`load_audio` (FFmpeg path — see [transcriber.md](transcriber.md)). One feature
file per audio source, mirroring its identity entities.

## Extractor-derived geometry

All Whisper constants (sampling rate, Mel-bin count, hop, chunk length) are read
off the loaded `AutoFeatureExtractor`, never hardcoded, so a model swap (e.g.
`large-v3`'s 128 Mel bins vs. 80 for tiny–medium) is tracked automatically. The
`WhisperModel` enum (shared with transcribe — see [transcriber.md](transcriber.md))
is the guard: the 30-s chunk geometry is Whisper's contract, so a non-Whisper
extractor would have incompatible geometry, and the enum rejects it at the CLI
boundary.

## TR pre-alignment at generation time

Spectral is the first feature to take the producer's pre-alignment prerogative
([../decisions/feature-files.md](../decisions/feature-files.md)): it downsamples
to the run's BOLD TR grid (`method="mean"`) inside the generator, storing rows
at TR cadence rather than native ~10-ms frames.

TR and `n_trs` come from the run's BOLD, not a user parameter — deriving them
guarantees the storage grid equals the encoding grid, so the encoding
re-downsample is a pass-through (no mean-of-means over misaligned spans). The
feature is dyad-keyed and carries no `sub`; partners share one simultaneous
scan, so any partner's BOLD stands in for resolving TR/`n_trs` (the
[phonemic confound](../decisions/confound-files.md) pattern:
`resolve_bold_image` → `load_bold_meta` → `segment_n_trs`). Granularity-agnostic
via `segment_n_trs`, which handles per-trial and per-run audio without a
pre-decision.

## No paired confound

Spectral is a stimulus feature (the signal of interest), not a source of
nuisance regressors, so it has no paired confound generator. Phonemic and
semantic pair with confounds because those derive nuisance signals (onset rate,
etc.) from the feature; spectral has no such derived nuisance.
