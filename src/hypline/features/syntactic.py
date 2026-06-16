from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from loguru import logger

from hypline.bids import BIDS_ENTITY_VALUE_RE, normalize_bids_filters
from hypline.features._utils import build_word_spans, first_overlapping_word
from hypline.io import skip_existing, write_feature
from hypline.layout import BIDSLayout

if TYPE_CHECKING:
    from spacy.language import Language
    from spacy.tokens import Token

    from hypline.bids import BIDSPath

SPACY_MODEL = "en_core_web_lg"


class SyntacticFeature:
    """Per-token syntactic-function feature: POS tag, dependency relation, stopword.

    Each transcript word is tokenized by spaCy (so `"don't"` -> `do` + `n't`) and
    tagged in the context of its conversational turn — the dependency parser needs
    coherent utterances, so words are grouped by `turn_sub` (which subject held the
    floor) and parsed one turn at a time. Every token becomes one output row sharing
    its source word's `start_time`; the embedding is a fixed-width one-hot of the
    POS tag concatenated with the dependency relation, plus a final 0/1 stopword dim.
    """

    _nlp: Language | None = None
    _tag_index: dict[str, int]
    _dep_index: dict[str, int]
    _dim_labels: list[str]

    def __init__(
        self,
        *,
        bids_root: str | Path,
        bids_filters: list[str] | None = None,
        desc: str | None = None,
        force: bool = False,
    ):
        if desc is not None and not BIDS_ENTITY_VALUE_RE.match(desc):
            raise ValueError(f"Invalid desc: {desc!r}")

        self._load()

        self._layout = BIDSLayout(bids_root)
        self._bids_filters = normalize_bids_filters(bids_filters, reserved={"dyad"})
        self._desc = desc
        self._force = force

    @classmethod
    def _load(cls):
        if cls._nlp is not None:
            return

        import spacy

        try:
            nlp = spacy.load(SPACY_MODEL)
        except OSError:
            logger.info("Downloading spaCy model {} (one-time)", SPACY_MODEL)
            spacy.cli.download(SPACY_MODEL)
            nlp = spacy.load(SPACY_MODEL)

        # Fit dims to the model's full label vocab so output width and column
        # order are fixed regardless of which tags a given transcript happens to
        # use (a per-file fit would drift in both).
        tag_labels = list(nlp.pipe_labels["tagger"])
        dep_labels = list(nlp.pipe_labels["parser"])
        cls._nlp = nlp
        cls._tag_index = {t: i for i, t in enumerate(tag_labels)}
        cls._dep_index = {d: i for i, d in enumerate(dep_labels)}
        cls._dim_labels = (
            [f"pos:{t}" for t in tag_labels]
            + [f"dep:{d}" for d in dep_labels]
            + ["is_stop"]
        )

    def generate(self, dyad_id: str):
        transcripts = self._layout.find.stimuli(
            dyad=dyad_id,
            kind="transcript",
            ext=".csv",
            bids_filters=self._bids_filters,
        )

        for transcript in transcripts:
            out = self._layout.path.feature(
                source=transcript, kind="syntactic", desc=self._desc
            )
            if skip_existing(out.path, force=self._force):
                continue
            logger.info("Generating syntactic features for {}", transcript.path.name)
            self._generate_one(transcript, out.path)

    def _generate_one(self, source: BIDSPath, out_path: Path) -> None:
        kept_starts, kept_words, kept_turns = self._load_words(source)

        nlp = self._nlp
        assert nlp is not None  # narrow Optional; _load runs in __init__
        dim = len(self._dim_labels)

        rows_start, rows_token, rows_word, rows_feature = [], [], [], []
        for words, word_starts, spans in self._iter_turns(
            kept_starts, kept_words, kept_turns
        ):
            text = " ".join(words)
            # Full-sentence tokenization (not per-word): spaCy resolves
            # abbreviations and punctuation using neighbors, so token boundaries
            # can differ from a word-by-word pass. A multi-token word's pieces all
            # overlap its span, so each inherits its start_time.
            doc = nlp(text)
            for token in doc:
                idx = first_overlapping_word(token.idx, token.idx + len(token), spans)
                rows_start.append(word_starts[idx])
                rows_token.append(token.text)
                rows_word.append(words[idx])
                rows_feature.append(self._token_vector(token))

        out_df = pl.DataFrame(
            {
                "start_time": rows_start,
                "token": rows_token,
                "word": rows_word,
                "feature": pl.Series(
                    "feature",
                    np.vstack(rows_feature).tolist(),
                    dtype=pl.Array(pl.Float64, dim),
                ),
            }
        )
        metadata = {
            "spacy_model": SPACY_MODEL,
            "feature_dim_labels": self._dim_labels,
        }
        write_feature(out_df, out_path, metadata=metadata)
        logger.debug("Wrote syntactic feature to {}", out_path)

    def _load_words(
        self, source: BIDSPath
    ) -> tuple[list[float | None], list[str], list[str | None]]:
        """Return `(starts, words, turns)` for timed and untimed words.

        Drops null-word rows but keeps untimed (null-start_time) words as parse
        context (see class docstring). `turn_sub` is forward-filled *before* the
        drop: stamp_turns leaves untimed words null, but they belong to the
        surrounding utterance — leaving a null would fragment a turn mid-sentence
        and corrupt its parse tree. Raises if no usable words remain.
        """
        df = pl.read_csv(source.path).with_columns(
            pl.col("turn_sub").cast(pl.Utf8).fill_null(strategy="forward")
        )

        null_words = df.get_column("word").is_null().sum()
        if null_words:
            logger.warning(
                "Skipped {} null-word row(s) in {}", null_words, source.path.name
            )

        df = df.filter(pl.col("word").is_not_null())
        if df.is_empty():
            raise ValueError(
                f"{source.path.name}: no usable words (all null); cannot "
                "generate syntactic features"
            )

        return (
            df.get_column("start_time").to_list(),
            df.get_column("word").cast(pl.Utf8).to_list(),
            df.get_column("turn_sub").to_list(),
        )

    def _iter_turns(
        self,
        starts: list[float | None],
        words: list[str],
        turns: list[str | None],
    ) -> Iterator[tuple[list[str], list[float | None], list[tuple[int, int]]]]:
        """Yield `(words, word_starts, spans)` per consecutive-turn run.

        A turn boundary is any change in `turn_sub` between adjacent kept words;
        each maximal run of equal `turn_sub` is parsed as one document.
        `build_word_spans` rebuilds char spans local to each run's joined text, so
        offsets index that run, not the whole file.
        """
        start = 0
        for i in range(1, len(words) + 1):
            if i == len(words) or turns[i] != turns[start]:
                run_words, run_starts = words[start:i], starts[start:i]
                yield run_words, run_starts, build_word_spans(run_words)
                start = i

    def _token_vector(self, token: Token) -> np.ndarray:
        vec = np.zeros(len(self._dim_labels))
        # A label outside the fitted vocab (e.g. "" on an unparsed token) has no
        # column; leave its block all-zero rather than KeyError, since the one-hot
        # cannot represent it and a missing tag is a meaningful all-zero state.
        if (i := self._tag_index.get(token.tag_)) is not None:
            vec[i] = 1.0
        if (j := self._dep_index.get(token.dep_)) is not None:
            vec[len(self._tag_index) + j] = 1.0
        vec[-1] = float(token.is_stop)
        return vec
