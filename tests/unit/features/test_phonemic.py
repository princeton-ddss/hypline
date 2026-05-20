from pathlib import Path

import numpy as np
import polars as pl
import pytest

from hypline.features import PhonemicFeature, read_feature, read_feature_metadata
from hypline.features.phonemic import ARPABET_PHONEMES
from hypline.layout import BIDSLayout

from ..conftest import BIDSTree

SUB = "001"
TASK = "conv"

CACHED_ATTRS = (
    "_pronunciations",
    "_phoneme_index",
    "_articulatory_vectors",
    "_articulatory_feature_names",
)


@pytest.fixture()
def fake_cmudict(monkeypatch):
    """Run real `_load` (for articulatory data), then swap in pronunciation dict."""
    for attr in CACHED_ATTRS:
        if hasattr(PhonemicFeature, attr):
            monkeypatch.delattr(PhonemicFeature, attr)

    PhonemicFeature._load()
    monkeypatch.setattr(
        PhonemicFeature,
        "_pronunciations",
        {
            "cat": [["K", "AE1", "T"]],
            "dog": [["D", "AO1", "G"]],
            "the": [["DH", "AH0"]],
        },
    )


def _write_transcript(
    tree: BIDSTree, rows: list[tuple[float, str]], *, sub: str = SUB, task: str = TASK
) -> Path:
    path = tree.add_stimulus(sub=sub, task=task, kind="transcript", ext=".csv")
    df = pl.DataFrame(
        {"start_time": [r[0] for r in rows], "word": [r[1] for r in rows]}
    )
    df.write_csv(path)
    return path


class TestPhonemicGenerateArpabet:
    def test_writes_one_row_per_phoneme(self, tree: BIDSTree, fake_cmudict):
        _write_transcript(tree, [(0.0, "cat"), (1.0, "dog")])
        layout = BIDSLayout(tree.root)
        PhonemicFeature(layout=layout, use_articulatory=False).generate(SUB)
        out = layout.find.features(sub=SUB, kind="phonemic")[0]
        df = read_feature(out.path)
        assert df.get_column("phoneme").to_list() == ["K", "AE", "T", "D", "AO", "G"]

    def test_phonemes_share_word_start_time(self, tree: BIDSTree, fake_cmudict):
        _write_transcript(tree, [(0.0, "cat"), (1.5, "dog")])
        layout = BIDSLayout(tree.root)
        PhonemicFeature(layout=layout, use_articulatory=False).generate(SUB)
        df = read_feature(layout.find.features(sub=SUB, kind="phonemic")[0].path)
        assert df.get_column("start_time").to_list() == [0.0, 0.0, 0.0, 1.5, 1.5, 1.5]

    def test_one_hot_vectors_match_arpabet_indices(self, tree: BIDSTree, fake_cmudict):
        _write_transcript(tree, [(0.0, "cat")])
        layout = BIDSLayout(tree.root)
        PhonemicFeature(layout=layout, use_articulatory=False).generate(SUB)
        df = read_feature(layout.find.features(sub=SUB, kind="phonemic")[0].path)
        feats = np.array(df.get_column("feature").to_list())
        assert feats.shape == (3, len(ARPABET_PHONEMES))
        for row, ph in zip(feats, ["K", "AE", "T"]):
            (hit,) = np.flatnonzero(row)
            assert hit == ARPABET_PHONEMES.index(ph)
            assert row.sum() == 1.0


class TestPhonemicGenerateArticulatory:
    def test_vectors_are_articulatory_multi_hot(self, tree: BIDSTree, fake_cmudict):
        _write_transcript(tree, [(0.0, "cat")])
        layout = BIDSLayout(tree.root)
        feat = PhonemicFeature(layout=layout, use_articulatory=True)
        feat.generate(SUB)
        dim = len(PhonemicFeature._articulatory_feature_names)
        df = read_feature(layout.find.features(sub=SUB, kind="phonemic")[0].path)
        feats = np.array(df.get_column("feature").to_list())
        assert feats.shape == (3, dim)
        assert (feats.sum(axis=1) > 0).all()
        assert feats.sum(axis=1).max() > 1  # K/AE/T have multiple articulatory features

    def test_matches_articulatory_vectors_lookup(self, tree: BIDSTree, fake_cmudict):
        _write_transcript(tree, [(0.0, "cat")])
        layout = BIDSLayout(tree.root)
        PhonemicFeature(layout=layout, use_articulatory=True).generate(SUB)
        df = read_feature(layout.find.features(sub=SUB, kind="phonemic")[0].path)
        feats = np.array(df.get_column("feature").to_list())
        for row, ph in zip(feats, ["K", "AE", "T"]):
            expected = PhonemicFeature._articulatory_vectors[ph]
            assert np.allclose(row, expected)


class TestPhonemicMissingUnits:
    def test_oov_word_emits_null_row(self, tree: BIDSTree, fake_cmudict):
        _write_transcript(tree, [(0.0, "cat"), (1.0, "qwerty")])
        layout = BIDSLayout(tree.root)
        PhonemicFeature(layout=layout, use_articulatory=False).generate(SUB)
        df = read_feature(layout.find.features(sub=SUB, kind="phonemic")[0].path)
        phonemes = df.get_column("phoneme").to_list()
        assert phonemes == ["K", "AE", "T", None]
        starts = df.get_column("start_time").to_list()
        assert starts[-1] == 1.0
        last_feat = np.array(df.get_column("feature").to_list())[-1]
        assert np.allclose(last_feat, 0.0)

    def test_punctuation_stripped_before_lookup(self, tree: BIDSTree, fake_cmudict):
        _write_transcript(tree, [(0.0, "cat,")])
        layout = BIDSLayout(tree.root)
        PhonemicFeature(layout=layout, use_articulatory=False).generate(SUB)
        df = read_feature(layout.find.features(sub=SUB, kind="phonemic")[0].path)
        assert df.get_column("phoneme").to_list() == ["K", "AE", "T"]

    def test_punctuation_only_token_emits_null_row(self, tree: BIDSTree, fake_cmudict):
        _write_transcript(tree, [(0.0, "cat"), (1.0, "...")])
        layout = BIDSLayout(tree.root)
        PhonemicFeature(layout=layout, use_articulatory=False).generate(SUB)
        df = read_feature(layout.find.features(sub=SUB, kind="phonemic")[0].path)
        phonemes = df.get_column("phoneme").to_list()
        assert phonemes[-1] is None
        last_feat = np.array(df.get_column("feature").to_list())[-1]
        assert np.allclose(last_feat, 0.0)


class TestPhonemicMetadata:
    def test_arpabet_metadata(self, tree: BIDSTree, fake_cmudict):
        _write_transcript(tree, [(0.0, "cat")])
        layout = BIDSLayout(tree.root)
        PhonemicFeature(layout=layout, use_articulatory=False).generate(SUB)
        meta = read_feature_metadata(
            layout.find.features(sub=SUB, kind="phonemic")[0].path
        )
        assert meta["use_articulatory"] is False
        assert meta["dim_labels"] == ARPABET_PHONEMES

    def test_articulatory_metadata(self, tree: BIDSTree, fake_cmudict):
        _write_transcript(tree, [(0.0, "cat")])
        layout = BIDSLayout(tree.root)
        PhonemicFeature(layout=layout, use_articulatory=True).generate(SUB)
        meta = read_feature_metadata(
            layout.find.features(sub=SUB, kind="phonemic")[0].path
        )
        assert meta["use_articulatory"] is True
        assert meta["dim_labels"] == PhonemicFeature._articulatory_feature_names


class TestPhonemicDesc:
    def test_desc_lands_on_output_filename(self, tree: BIDSTree, fake_cmudict):
        _write_transcript(tree, [(0.0, "cat")])
        layout = BIDSLayout(tree.root)
        PhonemicFeature(
            layout=layout,
            use_articulatory=False,
            desc="ver1",
        ).generate(SUB)
        out = layout.find.features(sub=SUB, kind="phonemic")[0]
        assert out.entities.get("desc") == "ver1"
        assert out.path.name.endswith("_feat-phonemic_desc-ver1.parquet")

    def test_no_desc_omits_entity(self, tree: BIDSTree, fake_cmudict):
        _write_transcript(tree, [(0.0, "cat")])
        layout = BIDSLayout(tree.root)
        PhonemicFeature(layout=layout, use_articulatory=False).generate(SUB)
        out = layout.find.features(sub=SUB, kind="phonemic")[0]
        assert "desc" not in out.entities
        assert out.path.name.endswith("_feat-phonemic.parquet")


class TestPhonemicMultiTranscript:
    def test_generates_one_feature_per_transcript(self, tree: BIDSTree, fake_cmudict):
        tree.add_stimulus(sub=SUB, task=TASK, run="1", kind="transcript", ext=".csv")
        tree.add_stimulus(sub=SUB, task=TASK, run="2", kind="transcript", ext=".csv")
        layout = BIDSLayout(tree.root)
        df = pl.DataFrame({"start_time": [0.0], "word": ["cat"]})
        for transcript in layout.find.stimuli(sub=SUB, kind="transcript", ext=".csv"):
            df.write_csv(transcript.path)
        PhonemicFeature(layout=layout, use_articulatory=False).generate(SUB)
        outs = layout.find.features(sub=SUB, kind="phonemic")
        assert len(outs) == 2
