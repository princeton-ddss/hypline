# Semantic — scope and assumptions

How hypline derives token-level contextual embeddings from a Hugging Face (HF)
causal language model (LM), and the assumptions that shape the forward pass.

## Source: word-level transcripts

Like phonemic ([phonemic.md](phonemic.md)), semantic features derive from
word-level transcripts. One row **per sub-word token**: a word that tokenizes
into several pieces yields several rows sharing that word's `start_time`.

## Full-sequence forward pass retains untimed words

The model runs over the **whole** token sequence, including words with a null
`start_time` (un-alignable by WhisperX — see
[transcriber.md](transcriber.md)). These rows are emitted with null
`start_time` but a real embedding.

This is a stronger stake in the producer-retains / consumer-drops contract
([../decisions/feature-files.md](../decisions/feature-files.md)) than phonemic's:
phonemic vectorizes each phoneme independently, so a dropped untimed word costs
only its own row; semantic embeddings are **contextual**, so dropping an untimed
word would corrupt its neighbors' embeddings and the LM metrics. The untimed
word must stay in the input even though TR-aligning consumers later drop its row.

A null **word** (vs. null `start_time`) is dropped and warned — it cannot be
tokenized and is not faithful text.

## Single-pass, no windowing

The whole transcript goes through the model in one forward pass — no chunking.
Safe **only because** transcripts are short (~3 min audio, ~400–550 words) and
sit well under any candidate model's context limit. A fail-fast guard aborts the
run if a transcript's token count (plus the BOS prefix) ever exceeds the model's
context window. Windowing (overlapping chunks, stitched hidden states) is the
noted future extension if longer stimuli arrive.

## Layer 0 is a static lookup

Selecting `layer=0` reads the input embedding table directly and skips the
forward pass, so the LM metrics (`rank`, `true_prob`, `entropy`) are undefined
and those columns are omitted. Every other layer runs the model and emits them.

## Model identity

Any HF causal LM loadable via `AutoModelForCausalLM.from_pretrained` is in
scope. The HF id is an open string, not an enum or name map — the hub is
unbounded, so `from_pretrained` validates the id better than a curated set
could. Contrast `WhisperModel`, an enum, because whisperX supports a genuinely
*closed* checkpoint list: enum for closed sets, open string for open sets.

## Gated models authenticate via `HF_TOKEN`

License-gated checkpoints (e.g. Llama) download only when the `HF_TOKEN` env var
holds a token whose account has been granted access. `transformers` reads it
automatically, so hypline exposes no auth flag.
