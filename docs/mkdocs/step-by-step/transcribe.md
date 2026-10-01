# `hypline transcribe`

Transcribe stimulus audio into word-level transcripts using a
[Whisper](https://github.com/openai/whisper) speech-recognition model. For each word in each
transcript, this step records the time the word was spoken — the timing that
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
<dataset-root>/stimuli/dyad-040/ses-1/audio/
└── dyad-040_ses-1_task-conv_run-1_audio.wav
```

See [Hypline dataset layout](layout.md) for how files are
named and discovered.

## Options

| Option           | Description                                                      | Default          |
| ---------------- | ---------------------------------------------------------------- | ---------------- |
| `--audio-ext`    | Extension of the audio files, e.g. `.wav` — **required**        | —                |
| `--model`        | Whisper model: `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3` | `large-v2` |
| `--model-dir`    | Where to find/download model weights                             | `~/.cache/hypline/whisperx` |
| `--device`       | Hardware target: `cpu` or `cuda`                                 | `cpu`            |
| `--dyad-ids`     | Comma-separated dyad IDs to process; omit for all                | all              |
| `--data-filters` | Narrow to specific runs/conditions — see [Filter to specific runs or conditions](../FAQ/filter.md) | none |
| `--force`        | Overwrite existing transcripts (default skips them)              | off              |

!!! tip "Model size vs. speed"

    Larger models are more accurate but slower. On a GPU, pass `--device cuda`.
    The first run downloads the chosen model to `--model-dir` (default
    `~/.cache/hypline/whisperx`); later runs reuse it.

!!! note "Relocating the cache"

    The `~/.cache/hypline` root is shared by every command that downloads model
    weights (`transcribe`, `featuregen semantic`, `featuregen spectral`). Set
    `HYPLINE_CACHE` to move that root — e.g. `HYPLINE_CACHE=/scratch/$USER/hypline`
    on a cluster with a small home quota. An explicit `--model-dir` still wins
    over both.

!!! tip "When a command produces no output"

    `No dyads found` for stimulus commands or `No subjects found` for
    `denoise` and `encoding` means the command found no dyads or subjects
    where it looks for inputs. Check the dataset layout. A wrong
    `--dyad-ids` / `--sub-ids` value fails differently: you get a per-ID
    error and exit code `1`. (If
    `--data-filters` matches no data, see [Filter to specific runs or
    conditions](../FAQ/filter.md#when-a-filter-matches-nothing).)


## Example

Transcribe every dyad's WAV audio with the default model:

```bash
hypline transcribe data/ --audio-ext .wav
```

Transcribe only dyad 040 on a GPU (pass more as a comma-separated list):

```bash
hypline transcribe data/ --audio-ext .wav --dyad-ids 040 --device cuda
```

## Outputs

A word-level transcript per audio file, with the `_transcript` suffix, written
beside the audio under `stimuli/`:

```
<dataset-root>/stimuli/dyad-040/ses-1/
├── audio/
│   └── dyad-040_ses-1_task-conv_run-1_audio.wav
└── transcript/
    └── dyad-040_ses-1_task-conv_run-1_transcript.csv
```

Each transcript row is one word with its onset time. These onsets are what
`featuregen phonemic` reads to place features on the timeline.

!!! note "Untimed words"

    Whisper occasionally emits a token it cannot place in time (some numerals
    and symbols). Such tokens appear in the transcript with a blank time.
    Feature generation keeps them, since they still give context to the
    language model and parser. They are dropped later, when `confoundgen` and
    `encoding` place rows on the TR grid, since a word with no time cannot be
    aligned to the BOLD signal.

## Speaker turns

`encoding` requires speaking-turn annotations in your `events.tsv` files; the
other steps work without them. When they are present, each transcript gains a
`turn_sub` column naming which subject held the floor when each word began.

Mark turns in each subject's `events.tsv` with the flat `trial_type` label
`turn_speaker` — one row per window where that subject is the assigned
speaker:

```tsv
onset   duration   trial_type
0.0     12.5       turn_speaker
20.0    8.0        turn_speaker
```

See [Speaker turns](segments.md#speaker-turns) for the full rules on writing
them.
