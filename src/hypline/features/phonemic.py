import os
from importlib.resources import files
from pathlib import Path

import numpy as np
import polars as pl

from hypline.bids import BIDSPath, validate_bids_entities
from hypline.features.utils import save_feature
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

# Entities provided via dedicated arguments — not allowed in bids_filters
_RESERVED_ENTITIES = frozenset({"sub"})


class PhonemicFeature:
    _pronunciations: dict
    _phoneme_index: dict[str, int]
    _articulatory_vectors: dict[str, np.ndarray]
    _articulatory_feature_names: list[str]

    def __init__(
        self,
        input_dir: str | os.PathLike[str],
        output_dir: str | os.PathLike[str],
        *,
        use_articulatory: bool = True,
        bids_filters: list[str] | None = None,
    ):
        self._load()

        validate_dirs(input_dir, output_dir)
        self._input_dir = Path(input_dir)
        self._output_dir = Path(output_dir)

        self._use_articulatory = use_articulatory

        bids_filters = list(bids_filters or [])
        validate_bids_entities(*bids_filters)
        for entity in bids_filters:
            key = entity.split("-", 1)[0]
            if key in _RESERVED_ENTITIES:
                raise ValueError(
                    f"bids_filters cannot contain {key!r} "
                    "— use the dedicated argument instead"
                )
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

        transcripts = find_files(
            self._input_dir,
            ends_with=".csv",
            bids_filters=[f"sub-{sub_id}", *self._bids_filters],
        )

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
            out_path = self._output_dir / (bids_path.path.stem + ".parquet")
            metadata = {
                "use_articulatory": self._use_articulatory,
                "dim_labels": (
                    self._articulatory_feature_names
                    if self._use_articulatory
                    else ARPABET_PHONEMES
                ),
            }
            save_feature(df, out_path, metadata=metadata)

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
