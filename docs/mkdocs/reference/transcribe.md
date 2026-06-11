# `hypline transcribe`

Transcribe stimulus audio into word-level transcripts using a
[Whisper](https://github.com/openai/whisper) speech-recognition model. Each
transcript records, for every word, the time it was spoken — the timing that
later feeds [`featuregen`](featuregen.md).

```bash
hypline transcribe <dataset-root> --audio-ext <ext> [OPTIONS]
```

!!! note "FFmpeg required"

    Audio is decoded through [FFmpeg](https://ffmpeg.org/), which must be on your
    `PATH`. Any FFmpeg-supported format works (WAV, MP3, M4A, FLAC, video
    containers, …) — pass its extension to `--audio-ext`.

## Inputs

Stimulus audio files under the `stimuli/` area, with the `_audio` suffix:

```
<dataset-root>/stimuli/dyad-103/ses-1/audio/
└── dyad-103_ses-1_task-conv_run-1_audio.wav
```

See [The hypline dataset layout](../concepts/layout.md) for how files are
named and discovered.

## Options

| Option           | Description                                                      | Default          |
| ---------------- | ---------------------------------------------------------------- | ---------------- |
| `--audio-ext`    | Extension of the audio files, e.g. `.wav` **(required)**         | —                |
| `--model`        | Whisper model: `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3` | `large-v2` |
| `--model-dir`    | Where to find/download model weights                             | `~/.cache/hypline/whisperx` |
| `--device`       | Hardware target: `cpu` or `cuda`                                 | `cpu`            |
| `--dyad-ids`     | Comma-separated dyad IDs to process; omit for all                | all              |
| `--data-filters` | Narrow to specific runs/conditions — see [Segments and metadata](../concepts/segments.md) | none |
| `--force`        | Overwrite existing transcripts (default skips them)              | off              |

!!! tip "Model size vs. speed"

    Larger models are more accurate but slower. On a GPU, pass `--device cuda`.
    The first run downloads the chosen model to `--model-dir` (default
    `~/.cache/hypline/whisperx`); later runs reuse it.

## Example

Transcribe every dyad's WAV audio with the default model:

```bash
hypline transcribe data/ --audio-ext .wav
```

Transcribe only dyad 103 on a GPU (pass more as a comma-separated list):

```bash
hypline transcribe data/ --audio-ext .wav --dyad-ids 103 --device cuda
```

## Outputs

A word-level transcript per audio file, with the `_transcript` suffix, written
beside the audio under `stimuli/`:

```
<dataset-root>/stimuli/dyad-103/ses-1/
├── audio/
│   └── dyad-103_ses-1_task-conv_run-1_audio.wav
└── transcript/
    └── dyad-103_ses-1_task-conv_run-1_transcript.csv
```

Each transcript row is one word with its onset time. These onsets are what
`featuregen phonemic` reads to place features on the timeline.

!!! note "Un-timed words"

    Whisper occasionally emits a token it cannot place in time (some numerals
    and symbols). Such tokens appear in the transcript with a blank time and are
    dropped by downstream feature generation, since an event with no time cannot
    be aligned to the BOLD signal.

## Speaker turns

If your `events.tsv` files annotate speaking turns, each transcript gains a
`turn_sub` column naming which subject held the floor when each word began.

Mark turns in each subject's raw `events.tsv` with the flat `trial_type` label
`turn_speaker` — one row per window where **that subject** is the assigned
speaker:

```tsv
onset   duration   trial_type
0.0     12.5       turn_speaker
20.0    8.0        turn_speaker
```

- The label records **whose turn it is by study design**, not who was observed
  speaking — a turn window may still contain a word uttered by the other partner.
- Mark only your own turns (`turn_speaker`); transcribe reads **both** partners'
  events and combines them, so there is no separate "listening" label to keep in
  sync.
- Windows are `[onset, onset + duration)`. Gaps (silence) are allowed; windows
  must **not** overlap — within a subject or across partners. A cross-partner
  overlap is treated as cross-talk and raises an error.
- `turn_speaker` onsets are **run-relative** — the whole-run `events.tsv` clock,
  the same frame as your segment (e.g. `trial-1`) rows. Write them that way even
  when audio is split per trial; transcribe shifts each word by its segment's
  onset before matching, so you never annotate turns in per-trial time.

Each word's `turn_sub` is the bare subject label (`001`, `101`) whose window
contains the word's run-relative start. Words that are un-timed, or fall in a gap
between turns, get a blank `turn_sub`; gap hits are logged as a possible
timing/annotation mismatch. Transcripts whose runs carry no `turn_speaker` rows
omit the column entirely.
