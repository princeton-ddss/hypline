# Transcriber — scope and assumptions

How hypline turns stimulus audio files into word-level transcripts, and the
audio-decode contract that determines which formats are supported.

## Audio decode path

whisperx's `load_audio` shells out to the FFmpeg CLI to decode each input
file, downmix to mono, and resample to 16 kHz, returning an in-memory numpy
array. That array is then fed to:

- Whisper (transcription)
- Pyannote's Silero VAD (voice activity detection)

Pyannote never receives a file path — only the already-decoded tensor.

## Implications

- **Any FFmpeg-supported format works** (WAV, MP3, M4A, FLAC, OGG, video
  containers, etc.). FFmpeg is a hard runtime dep, checked at
  `Transcriber.__init__`.
- **`torchcodec` is unused** in our pipeline. Pyannote imports it
  defensively for its own file-decode path, which we do not exercise.
- **The loud `torchcodec is not installed correctly` warning at import
  time is filtered out.** Root cause is a torchcodec/torch ABI mismatch
  in pyannote's transitive deps — upstream and not worth pinning around
  for a pipeline that never touches torchcodec.

## Null-timed tokens

WhisperX's forced aligner occasionally cannot place a real spoken word in audio
— from a CTC `backtrack` failure, or a word whose characters were all left
unaligned. Whisper still emits the word, but `word_segments` carries it with
`start`/`end` absent, which becomes a `null` `start_time` row in the transcript
CSV, sitting in correct speech order with valid text. This is **not** the
digits/symbols case: numerals and symbols align via whisperX's wildcard path and
do get a `start_time`. Null-timed rows are rare and cannot be fixed upstream, so
downstream consumers must tolerate them — see [phonemic.md](phonemic.md) for how
phonemic features handle them.

## Speaker turns

When the dyad's `events.tsv` files carry `turn_speaker` rows, each transcript
gains a `turn_sub` column naming which subject held the floor when each word
began (assigned by study design, not observed speech). Transcriber delegates
all turn logic to `hypline.events` (`load_turns`, `stamp_turns`); see
[events.md](events.md). A non-zero count of timed words landing in no turn
window is logged as a possible timing/annotation mismatch rather than raised.

## Caveat

If a future change routes audio through pyannote by file path (e.g.
swapping Silero VAD for pyannote's segmentation models that read files
directly), torchcodec re-enters the path and the suppression becomes
misleading. Revisit at that point.
