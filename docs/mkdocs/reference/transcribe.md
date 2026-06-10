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

Stimulus audio files under the `stimuli/` area, tagged `stim-audio`:

```
<dataset-root>/stimuli/dyad-101/audio/
└── dyad-101_task-conv_run-1_stim-audio.wav
```

See [The hypline dataset layout](../concepts/layout.md) for how files are
named and discovered.

## Options

| Option           | Description                                                      | Default          |
| ---------------- | ---------------------------------------------------------------- | ---------------- |
| `--audio-ext`    | Extension of the audio files, e.g. `.wav` **(required)**         | —                |
| `--model`        | Whisper model: `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3` | `large-v2` |
| `--model-dir`    | Where to find/download model weights                             | system cache     |
| `--device`       | Hardware target: `cpu` or `cuda`                                 | `cpu`            |
| `--dyad-ids`     | Comma-separated dyad IDs to process; omit for all                | all              |
| `--data-filters` | Narrow to specific runs/conditions — see [Segments and metadata](../concepts/segments.md) | none |
| `--force`        | Overwrite existing transcripts (default skips them)              | off              |

!!! tip "Model size vs. speed"

    Larger models are more accurate but slower. On a GPU, pass `--device cuda`.
    The first run downloads the chosen model to `--model-dir` (or the system
    cache); later runs reuse it.

## Example

Transcribe every dyad's WAV audio with the default model:

```bash
hypline transcribe data/ --audio-ext .wav
```

Transcribe only dyads 101 and 102 on a GPU:

```bash
hypline transcribe data/ --audio-ext .wav --dyad-ids 101,102 --device cuda
```

## Outputs

A word-level transcript per audio file, tagged `stim-transcript`, written beside
the audio under `stimuli/`:

```
<dataset-root>/stimuli/dyad-101/
├── audio/
│   └── dyad-101_task-conv_run-1_stim-audio.wav
└── transcript/
    └── dyad-101_task-conv_run-1_stim-transcript.csv
```

Each transcript row is one word with its onset time. These onsets are what
`featuregen phonemic` reads to place features on the timeline.

!!! note "Un-timed words"

    Whisper occasionally emits a token it cannot place in time (some numerals
    and symbols). Such tokens appear in the transcript with a blank time and are
    dropped by downstream feature generation, since an event with no time cannot
    be aligned to the BOLD signal.
