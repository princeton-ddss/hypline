# Feature families

An encoding model is only as interesting as the features you give it. Each feature
describes the speech a participant heard along one dimension, and the model's
score tells you how well the brain tracks *that* dimension. So choosing features
is choosing a scientific question. Hypline generates four families, each aimed at
a different level of speech.

This page is the map for picking among them. For how to generate each — options,
inputs, outputs — see the [`featuregen` reference](../reference/featuregen.md).

## The four families at a glance

| Family | Captures | The question it answers |
| ------ | -------- | ----------------------- |
| **Phonemic** | The articulatory makeup of each speech sound | How does the brain track low-level phonetic structure? |
| **Spectral** | The acoustic spectrum of the audio itself | How does the brain track the raw sound of speech? |
| **Syntactic** | Each word's grammatical role | How does the brain track sentence structure? |
| **Semantic** | Each word's meaning in context | How does the brain track meaning and prediction? |

They run roughly low-level to high: spectral and phonemic describe the *sound*,
syntactic and semantic describe the *language*. You can generate several and give
each its own [band](how-encoding-works.md#bands-one-ridge-strength-per-feature) in
one model, letting the fit separate what each level explains.

## Phonemic: the sounds of speech

Phonemic features describe each word as its sequence of phonemes, and each phoneme
by its articulatory properties: where in the mouth it is formed, how, and whether
it is voiced. They come from a dictionary lookup, not a learned model — words map
to phonemes through the CMU Pronouncing Dictionary, and phonemes to a fixed table
of articulatory features. Reach for phonemic features to ask how the brain
represents the fine phonetic structure of speech, below the level of the word.

## Spectral: the raw acoustics

Spectral features are the log-Mel spectrogram of the audio: the energy in each
frequency band over time, the same front-end a speech model sees before it
understands anything. This is the only family read straight from the audio rather
than the transcript, and the only one delivered already aligned to the BOLD TR
grid — `featuregen` bins its fine-grained frames onto the TR grid for you, so no
downstream downsampling step is needed. Use it to model the brain's response to
the acoustic signal itself, independent of what the words mean.

## Syntactic: grammatical structure

Syntactic features tag each word with its part of speech and its grammatical
relation to the rest of the sentence, from a spaCy parse of each conversational
turn. They isolate structure from meaning: two sentences can share syntax while
differing in content, and vice versa. Use them to ask how the brain encodes
grammatical form, separable from the semantic content the words carry.

## Semantic: meaning in context

Semantic features are contextual word embeddings from a language model — each
word's representation as the model understands it *in that sentence*, not in
isolation. This is the family that carries meaning, and the one the analysis in
Zada et al. (2026) is built on. Because the embedding is contextual, it also
reflects prediction: how expected a word was given everything before it.

Unlike the other three, semantic features have **no default model** — you choose
which language model produces them with `--model`, and the choice matters. A
larger model gives richer representations; a longer-context model handles longer
transcripts. See the
[`featuregen semantic` reference](../reference/featuregen.md#featuregen-semantic)
for the model requirements and how to select a layer.

## Choosing among them

There is no single right feature — it depends on the question. A few starting
points:

- **Meaning and prediction** — semantic. The workhorse for representation-level
  questions, and the paper's basis for the cross-brain analysis.
- **Sound, not meaning** — spectral or phonemic, depending on whether you want the
  raw acoustics or the phonetic structure abstracted from them.
- **Structure, not content** — syntactic.
- **Separating levels** — generate several families and give each its own band in
  one model, so the fit apportions variance across them and each family's score
  reflects its unique contribution.

Once you have features, [How the encoding model works](how-encoding-works.md)
explains how they become a fitted model, and
[Reading an encoding result](reading-an-eval.md) explains how to read each band's
score back out.
