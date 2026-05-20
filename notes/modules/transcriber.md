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

## Caveat

If a future change routes audio through pyannote by file path (e.g.
swapping Silero VAD for pyannote's segmentation models that read files
directly), torchcodec re-enters the path and the suppression becomes
misleading. Revisit at that point.
