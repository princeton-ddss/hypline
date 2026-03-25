import os
from importlib.resources import files
from pathlib import Path

import numpy as np
import polars as pl

from hypline.bids import BIDSPath
from hypline.featuregen.utils import save_feature
from hypline.utils import find_files, validate_dirs

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
    _n_articulatory_features: int

    def __init__(self):
        self._load()

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
        cls._n_articulatory_features = len(feature_index)
        cls._articulatory_vectors = {}
        for row in articulatory_df.iter_rows(named=True):
            vec = np.zeros(cls._n_articulatory_features)
            for col in articulatory_df.columns[2:]:
                if row[col] is not None:
                    vec[feature_index[row[col]]] = 1
            cls._articulatory_vectors[row["phoneme"]] = vec

    def _get_word_phoneme_vector(self, word: str) -> np.ndarray:
        vec = np.zeros(len(ARPABET_PHONEMES))
        if pronunciations := self._pronunciations.get(word.lower()):
            for phoneme in pronunciations[0]:
                vec[self._phoneme_index[phoneme.strip("012")]] = 1
        return vec

    def _get_word_articulatory_vector(self, word: str) -> np.ndarray:
        vec = np.zeros(self._n_articulatory_features)
        if pronunciations := self._pronunciations.get(word.lower()):
            for phoneme in pronunciations[0]:
                vec += self._articulatory_vectors[phoneme.strip("012")]
        return vec

    def generate(
        self,
        input_dir: str | os.PathLike[str],
        output_dir: str | os.PathLike[str],
        *,
        use_articulatory: bool = True,
        bids_filters: list[str] | None = None,
    ):
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        validate_dirs(input_dir, output_dir)

        feature_fn = (
            self._get_word_articulatory_vector
            if use_articulatory
            else self._get_word_phoneme_vector
        )

        transcripts = find_files(input_dir, ".csv", bids_filters=bids_filters)
        for transcript in transcripts:
            df = pl.read_csv(transcript)
            words = df.get_column("word").cast(pl.Utf8).str.strip_chars(PUNCTUATION)
            features = np.vstack([feature_fn(w) for w in words.to_list()])
            df = df.with_columns(
                pl.Series(
                    "feature",
                    features.tolist(),
                    dtype=pl.Array(pl.Float64, features.shape[1]),
                )
            )

            bids_path = BIDSPath(transcript).with_entity("feature", "phonemic")
            out_path = output_dir / (bids_path.path.stem + ".parquet")
            save_feature(df, out_path)
