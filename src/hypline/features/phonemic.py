from importlib.resources import files
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger

from hypline.io import write_feature
from hypline.layout import BIDSLayout

ARPABET_PHONEMES = [
    "B",
    "CH",
    "D",
    "DH",
    "F",
    "G",
    "HH",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "V",
    "W",
    "Y",
    "Z",
    "ZH",
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "EH",
    "ER",
    "EY",
    "IH",
    "IY",
    "OW",
    "OY",
    "UH",
    "UW",
]

PUNCTUATION = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"


class PhonemicFeature:
    _pronunciations: dict
    _phoneme_index: dict[str, int]
    _articulatory_vectors: dict[str, np.ndarray]
    _articulatory_feature_names: list[str]

    def __init__(
        self,
        *,
        bids_root: str | Path,
        use_articulatory: bool = True,
        bids_filters: list[str] | None = None,
        desc: str | None = None,
    ):
        self._load()

        self._layout = BIDSLayout(bids_root)
        self._use_articulatory = use_articulatory
        self._bids_filters = bids_filters
        self._desc = desc

    @classmethod
    def _load(cls):
        if hasattr(cls, "_pronunciations"):
            return

        import nltk

        try:
            nltk.corpus.cmudict.entries()
        except LookupError:
            nltk.download("cmudict", quiet=True)
        cls._pronunciations = nltk.corpus.cmudict.dict()
        cls._phoneme_index = {ph: i for i, ph in enumerate(ARPABET_PHONEMES)}
        articulatory_df = pl.read_csv(
            str(files("hypline.data") / "articulatory_features.csv")
        )
        feature_values = articulatory_df[:, 2:].to_numpy().flatten()
        unique_features = np.unique(feature_values[feature_values != None])  # noqa: E711
        feature_index = {f: i for i, f in enumerate(unique_features)}
        cls._articulatory_feature_names = unique_features.tolist()
        cls._articulatory_vectors = {}
        for row in articulatory_df.iter_rows(named=True):
            vec = np.zeros(len(cls._articulatory_feature_names))
            for col in articulatory_df.columns[2:]:
                if row[col] is not None:
                    vec[feature_index[row[col]]] = 1
            cls._articulatory_vectors[row["phoneme"]] = vec

    def generate(self, sub_id: str):
        dim = (
            len(self._articulatory_feature_names)
            if self._use_articulatory
            else len(ARPABET_PHONEMES)
        )

        transcripts = self._layout.find.stimuli(
            sub=sub_id,
            kind="transcript",
            ext=".csv",
            bids_filters=self._bids_filters,
        )

        for transcript in transcripts:
            logger.info("Generating phonemic features for {}", transcript.path.name)
            df = pl.read_csv(transcript.path)
            start_times = df.get_column("start_time").to_list()
            raw_words = df.get_column("word").cast(pl.Utf8).to_list()

            rows_start, rows_phoneme, rows_word, rows_feature = [], [], [], []
            for start, word in zip(start_times, raw_words):
                phonemes = self._get_phonemes(word.strip(PUNCTUATION)) or [None]
                for ph in phonemes:
                    rows_start.append(start)
                    rows_phoneme.append(ph)
                    rows_word.append(word)
                    rows_feature.append(
                        np.zeros(dim) if ph is None else self._phoneme_vector(ph)
                    )

            out_df = pl.DataFrame(
                {
                    "start_time": rows_start,
                    "phoneme": rows_phoneme,
                    "word": rows_word,
                    "feature": pl.Series(
                        "feature",
                        np.vstack(rows_feature).tolist(),
                        dtype=pl.Array(pl.Float64, dim),
                    ),
                }
            )

            out = self._layout.path.feature(
                source=transcript, kind="phonemic", desc=self._desc
            )
            out.path.parent.mkdir(parents=True, exist_ok=True)

            metadata = {
                "use_articulatory": self._use_articulatory,
                "feature_dim_labels": (
                    self._articulatory_feature_names
                    if self._use_articulatory
                    else ARPABET_PHONEMES
                ),
            }

            write_feature(out_df, out.path, metadata=metadata)
            logger.debug("Wrote phonemic feature to {}", out.path)

    def _get_phonemes(self, word: str) -> list[str]:
        if pronunciations := self._pronunciations.get(word.lower()):
            return [ph.strip("012") for ph in pronunciations[0]]
        return []

    def _phoneme_vector(self, phoneme: str) -> np.ndarray:
        if self._use_articulatory:
            return self._articulatory_vectors[phoneme]
        vec = np.zeros(len(ARPABET_PHONEMES))
        vec[self._phoneme_index[phoneme]] = 1
        return vec
