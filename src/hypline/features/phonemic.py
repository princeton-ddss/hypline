from importlib.resources import files

import numpy as np
import polars as pl
from loguru import logger

from hypline.features._utils import save_feature
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
        layout: BIDSLayout,
        use_articulatory: bool = True,
        bids_filters: list[str] | None = None,
    ):
        self._load()

        self._layout = layout
        self._use_articulatory = use_articulatory
        self._bids_filters = bids_filters

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
        feature_fn = (
            self._get_word_articulatory_vector
            if self._use_articulatory
            else self._get_word_phoneme_vector
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
            words = df.get_column("word").cast(pl.Utf8).str.strip_chars(PUNCTUATION)
            features = np.vstack([feature_fn(w) for w in words.to_list()])
            df = df.with_columns(
                pl.Series(
                    "feature",
                    features.tolist(),
                    dtype=pl.Array(pl.Float64, features.shape[1]),
                )
            )

            out = self._layout.path.feature(source=transcript, kind="phonemic")
            out.path.parent.mkdir(parents=True, exist_ok=True)

            metadata = {
                "use_articulatory": self._use_articulatory,
                "dim_labels": (
                    self._articulatory_feature_names
                    if self._use_articulatory
                    else ARPABET_PHONEMES
                ),
            }

            save_feature(df, out.path, metadata=metadata)
            logger.debug("Wrote phonemic feature to {}", out.path)

    def _get_word_phoneme_vector(self, word: str) -> np.ndarray:
        vec = np.zeros(len(ARPABET_PHONEMES))
        if pronunciations := self._pronunciations.get(word.lower()):
            for phoneme in pronunciations[0]:
                vec[self._phoneme_index[phoneme.strip("012")]] = 1
        return vec

    def _get_word_articulatory_vector(self, word: str) -> np.ndarray:
        vec = np.zeros(len(self._articulatory_feature_names))
        if pronunciations := self._pronunciations.get(word.lower()):
            for phoneme in pronunciations[0]:
                vec += self._articulatory_vectors[phoneme.strip("012")]
        return vec
