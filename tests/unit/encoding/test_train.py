from collections.abc import Callable
from typing import Literal

import numpy as np
import polars as pl
import pytest

from hypline.encoding import (
    EncodingArtifact,
    EncodingTrainer,
    load_artifact,
    save_artifact,
)
from hypline.encoding._artifact import FoldSpec
from hypline.encoding._context import _CONFOUND_BAND, _SCREEN_BAND
from hypline.encoding._schema import BoldKey, CellKey, TrainingData
from hypline.encoding._train import (
    _group_cells_by,
    _inner_cv,
    _partition_groups,
    _select_rows,
)

from ..conftest import DEFAULT_BOLD_N_TRS, BIDSTree
from .conftest import DYAD, PARTNER, SPACE, SUB, TASK, _add_dyad_turns, _make_encoding

# make_train_setup return type: a stubbed trainer plus the data it fits on
_TrainSetupFactory = Callable[..., tuple[EncodingTrainer, TrainingData]]


def _add_block_segmented_run(tree: BIDSTree) -> None:
    """Toy single-run BIDS whose events split BOLD into two blocks.

    10-TR BOLD at TR=2.0 (20 s); two 10 s blocks (onsets 0 / 10) => 5 TRs each,
    with per-block mfcc features carrying one distinct timed row per TR. Both
    subjects speak once per block (`SUB` first 5 s, `PARTNER` next 5 s), so each
    block yields non-empty prod and comp masks. The turn rows share `SUB`'s events
    file with the block rows in one write — a second `add_events` would clobber
    them — so this fixture writes the pair itself rather than via `_add_dyad_turns`.
    """
    feature_df = pl.DataFrame(
        {
            "start_time": [0.0, 2.0, 4.0, 6.0, 8.0],
            "feature": [[float(i)] for i in range(5)],
        },
        schema={"start_time": pl.Float64, "feature": pl.Array(pl.Float64, 1)},
    )
    tree.add_participants({SUB: DYAD, PARTNER: DYAD})
    blocks = [
        {"trial_type": "block-1", "onset": 0.0, "duration": 10.0},
        {"trial_type": "block-2", "onset": 10.0, "duration": 10.0},
    ]
    sub_turns = [
        {"trial_type": "turn_speaker", "onset": 0.0, "duration": 5.0},
        {"trial_type": "turn_speaker", "onset": 10.0, "duration": 5.0},
    ]
    partner_turns = [
        {"trial_type": "turn_speaker", "onset": 5.0, "duration": 5.0},
        {"trial_type": "turn_speaker", "onset": 15.0, "duration": 5.0},
    ]
    # PARTNER needs its own turn_speaker rows — turns are complementary and
    # load_turns reads every partner's file. The block rows in PARTNER's file are
    # realism scaffolding, not load-bearing: segments resolve from
    # subjects_of(dyad)[0] == SUB, so PARTNER's blocks are never read.
    tree.add_events(sub=SUB, task=TASK, run="1", rows=[*sub_turns, *blocks])
    tree.add_events(sub=PARTNER, task=TASK, run="1", rows=[*partner_turns, *blocks])
    tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="denoised")
    for block in ("1", "2"):
        tree.add_feature(
            dyad=DYAD,
            task=TASK,
            kind="mfcc",
            run="1",
            extra_entities={"block": block},
            df=feature_df,
        )


class TestEncodingInit:
    def test_valid_config_succeeds(self, tree: BIDSTree):
        enc = _make_encoding(tree, ["mfcc"])
        assert list(enc._recipe.features) == ["mfcc"]

    def test_empty_features_raises(self, tree: BIDSTree):
        with pytest.raises(ValueError, match="non-empty"):
            _make_encoding(tree, [])

    def test_duplicate_features_raises(self, tree: BIDSTree):
        with pytest.raises(ValueError, match="Duplicate"):
            _make_encoding(tree, ["mfcc", "mfcc"])

    def test_duplicate_kind_across_variants_raises(self, tree: BIDSTree):
        with pytest.raises(ValueError, match="Duplicate feature kind"):
            _make_encoding(tree, ["semantic", "semantic-gpt2"])

    def test_confounds_default_empty(self, tree: BIDSTree):
        enc = _make_encoding(tree, ["mfcc"])
        assert enc._recipe.confounds == {}

    def test_confounds_parsed(self, tree: BIDSTree):
        enc = _make_encoding(tree, ["mfcc"], confounds=["phonemic-onset"])
        assert enc._recipe.confounds == {"phonemic-onset": ("phonemic", "onset")}

    def test_duplicate_confounds_raises(self, tree: BIDSTree):
        with pytest.raises(ValueError, match="Duplicate entries in confounds"):
            _make_encoding(tree, ["mfcc"], confounds=["phonemic", "phonemic"])

    def test_confound_variants_share_kind_allowed(self, tree: BIDSTree):
        enc = _make_encoding(
            tree, ["mfcc"], confounds=["phonemic-onset", "phonemic-rate"]
        )
        assert set(enc._recipe.confounds) == {"phonemic-onset", "phonemic-rate"}

    def test_feature_confound_overlap_raises(self, tree: BIDSTree):
        with pytest.raises(ValueError, match="appear in both features and confounds"):
            _make_encoding(tree, ["mfcc"], confounds=["mfcc"])

    @pytest.mark.parametrize("entry", ["a_b", "a-", "-b", "a-b-c", ""])
    def test_malformed_feature_entry_raises(self, tree: BIDSTree, entry: str):
        with pytest.raises(ValueError, match="Invalid (kind|desc)"):
            _make_encoding(tree, [entry])

    def test_variant_entry_parsed(self, tree: BIDSTree):
        enc = _make_encoding(tree, ["semantic-gpt3"])
        assert enc._recipe.features == {"semantic-gpt3": ("semantic", "gpt3")}

    def test_desc_reserved_in_filters_raises(self, tree: BIDSTree):
        with pytest.raises(ValueError, match="desc"):
            _make_encoding(tree, ["mfcc"], bids_filters=["desc-gpt3"])

    def test_reserved_entity_in_filters_raises(self, tree: BIDSTree):
        with pytest.raises(ValueError, match="sub"):
            _make_encoding(tree, ["mfcc"], bids_filters=["sub-001"])

    def test_unknown_entity_accepted_at_init(self, tree: BIDSTree):
        enc = _make_encoding(tree, ["mfcc"], bids_filters=["xyz-foo"])
        assert enc._recipe.bids_filters == ["xyz-foo"]

    def test_invalid_bold_space_raises(self, tree: BIDSTree):
        with pytest.raises(ValueError, match="Unsupported BOLD data space"):
            _make_encoding(tree, ["mfcc"], bold_space="notaspace")

    def test_invalid_bold_desc_raises(self, tree: BIDSTree):
        with pytest.raises(ValueError, match="Invalid bold_desc"):
            _make_encoding(tree, ["mfcc"], bold_desc="not-valid")

    def test_fold_by_and_n_folds_set_together_succeeds(self, tree: BIDSTree):
        enc = _make_encoding(tree, ["mfcc"], fold_by="run", n_folds=2)
        assert enc._fold == FoldSpec(by="run", n=2)

    def test_fold_by_without_n_folds_raises(self, tree: BIDSTree):
        with pytest.raises(ValueError, match="set together"):
            _make_encoding(tree, ["mfcc"], fold_by="run")

    def test_n_folds_without_fold_by_raises(self, tree: BIDSTree):
        with pytest.raises(ValueError, match="set together"):
            _make_encoding(tree, ["mfcc"], n_folds=2)

    def test_fold_by_excluded_entity_raises(self, tree: BIDSTree):
        with pytest.raises(ValueError, match="not a cell axis"):
            _make_encoding(tree, ["mfcc"], fold_by="space", n_folds=2)

    def test_n_folds_one_raises(self, tree: BIDSTree):
        with pytest.raises(ValueError, match=">= 2 or 'loo'"):
            _make_encoding(tree, ["mfcc"], fold_by="run", n_folds=1)

    def test_n_folds_loo_accepted(self, tree: BIDSTree):
        enc = _make_encoding(tree, ["mfcc"], fold_by="run", n_folds="loo")
        assert enc._fold == FoldSpec(by="run", n="loo")

    def test_fold_by_task_allowed(self, tree: BIDSTree):
        enc = _make_encoding(tree, ["mfcc"], fold_by="task", n_folds=2)
        assert enc._fold == FoldSpec(by="task", n=2)


class TestApplyFilters:
    def test_no_filters_returns_unchanged(self, tree: BIDSTree):
        tree.add_participants({SUB: DYAD})
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="denoised")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=[
                {"trial_type": "block-1", "onset": 0.0, "duration": 100.0},
                {"trial_type": "block-2", "onset": 100.0, "duration": 100.0},
            ],
        )
        tree.add_feature(
            dyad=DYAD, task=TASK, kind="mfcc", run="1", extra_entities={"block": "1"}
        )
        tree.add_feature(
            dyad=DYAD, task=TASK, kind="mfcc", run="1", extra_entities={"block": "2"}
        )
        enc = _make_encoding(tree, ["mfcc"])
        feature_paths = enc._discover_features(SUB)
        bold_metas = enc._discover_bold(SUB)
        feature_paths = enc._resolve_cell_keys(SUB, feature_paths, bold_metas)
        filtered_features, filtered_bold = enc._apply_filters(
            SUB, feature_paths, bold_metas
        )
        assert filtered_features == feature_paths
        assert filtered_bold == bold_metas

    def test_filter_narrows_features(self, tree: BIDSTree):
        tree.add_participants({SUB: DYAD})
        for run in ("1", "2"):
            tree.add_bold(
                sub=SUB, task=TASK, space=SPACE, run=run, tr=2.0, desc="denoised"
            )
            tree.add_events(
                sub=SUB,
                task=TASK,
                run=run,
                rows=[
                    {"trial_type": "block-1", "onset": 0.0, "duration": 100.0},
                    {"trial_type": "block-2", "onset": 100.0, "duration": 100.0},
                ],
            )
        tree.add_feature(
            dyad=DYAD, task=TASK, kind="mfcc", run="1", extra_entities={"block": "1"}
        )
        tree.add_feature(
            dyad=DYAD, task=TASK, kind="mfcc", run="1", extra_entities={"block": "2"}
        )
        tree.add_feature(
            dyad=DYAD, task=TASK, kind="mfcc", run="2", extra_entities={"block": "1"}
        )
        tree.add_feature(
            dyad=DYAD, task=TASK, kind="mfcc", run="2", extra_entities={"block": "2"}
        )
        enc = _make_encoding(tree, ["mfcc"], bids_filters=["run-1"])
        feature_paths = enc._discover_features(SUB)
        bold_metas = enc._discover_bold(SUB)
        feature_paths = enc._resolve_cell_keys(SUB, feature_paths, bold_metas)
        feature_paths, bold_metas = enc._apply_filters(SUB, feature_paths, bold_metas)
        assert all(feature_key.cell.get("run") == "1" for feature_key in feature_paths)
        assert all(bold_key.run == "1" for bold_key in bold_metas)

    def test_or_within_entity_and_across_entities(self, tree: BIDSTree):
        tree.add_participants({SUB: DYAD})
        for ses, run in (("1", "1"), ("1", "2"), ("2", "1")):
            tree.add_bold(
                sub=SUB,
                task=TASK,
                space=SPACE,
                ses=ses,
                run=run,
                tr=2.0,
                desc="denoised",
            )
            tree.add_events(
                sub=SUB,
                task=TASK,
                ses=ses,
                run=run,
                rows=[
                    {"trial_type": "block-1", "onset": 0.0, "duration": 100.0},
                ],
            )
            tree.add_feature(
                dyad=DYAD,
                task=TASK,
                kind="mfcc",
                ses=ses,
                run=run,
                extra_entities={"block": "1"},
            )
        enc = _make_encoding(
            tree,
            ["mfcc"],
            bids_filters=["ses-1", "run-1", "run-2"],  # ses-1 AND (run-1 OR run-2)
        )
        feature_paths = enc._discover_features(SUB)
        bold_metas = enc._discover_bold(SUB)
        feature_paths = enc._resolve_cell_keys(SUB, feature_paths, bold_metas)
        feature_paths, bold_metas = enc._apply_filters(SUB, feature_paths, bold_metas)
        assert BoldKey(ses="1", task=TASK, run="1") in bold_metas
        assert BoldKey(ses="1", task=TASK, run="2") in bold_metas
        assert BoldKey(ses="2", task=TASK, run="1") not in bold_metas

    def test_filter_on_cell_only_entity_skipped_on_bold(self, tree: BIDSTree):
        tree.add_participants({SUB: DYAD})
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="denoised")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=[
                {"trial_type": "block-1", "onset": 0.0, "duration": 100.0},
                {"trial_type": "block-2", "onset": 100.0, "duration": 100.0},
            ],
            sidecar_json={
                "trial_type": {
                    "Levels": {
                        "block-1": {"metadata": {"cond": "R"}},
                        "block-2": {"metadata": {"cond": "L"}},
                    }
                }
            },
        )
        tree.add_feature(
            dyad=DYAD, task=TASK, kind="mfcc", run="1", extra_entities={"block": "1"}
        )
        tree.add_feature(
            dyad=DYAD, task=TASK, kind="mfcc", run="1", extra_entities={"block": "2"}
        )
        enc = _make_encoding(
            tree,
            ["mfcc"],
            bids_filters=["cond-R"],  # on CellKey but not on BOLD filenames
        )
        feature_paths = enc._discover_features(SUB)
        bold_metas = enc._discover_bold(SUB)
        feature_paths = enc._resolve_cell_keys(SUB, feature_paths, bold_metas)
        feature_paths, bold_metas = enc._apply_filters(SUB, feature_paths, bold_metas)
        assert len(feature_paths) == 1
        assert len(bold_metas) == 1

    def test_typo_filter_entity_raises(self, tree: BIDSTree):
        tree.add_participants({SUB: DYAD})
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="denoised")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=[
                {"trial_type": "block-1", "onset": 0.0, "duration": 100.0},
                {"trial_type": "block-2", "onset": 100.0, "duration": 100.0},
            ],
        )
        tree.add_feature(
            dyad=DYAD, task=TASK, kind="mfcc", run="1", extra_entities={"block": "1"}
        )
        tree.add_feature(
            dyad=DYAD, task=TASK, kind="mfcc", run="1", extra_entities={"block": "2"}
        )
        enc = _make_encoding(tree, ["mfcc"], bids_filters=["typo-foo"])
        feature_paths = enc._discover_features(SUB)
        bold_metas = enc._discover_bold(SUB)
        feature_paths = enc._resolve_cell_keys(SUB, feature_paths, bold_metas)
        with pytest.raises(ValueError, match="typo"):
            enc._apply_filters(SUB, feature_paths, bold_metas)

    def test_valid_entity_wrong_value_passes_filter_raises_at_coverage(
        self, tree: BIDSTree
    ):
        tree.add_participants({SUB: DYAD})
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="denoised")
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=[
                {"trial_type": "block-1", "onset": 0.0, "duration": 100.0},
                {"trial_type": "block-2", "onset": 100.0, "duration": 100.0},
            ],
        )
        tree.add_feature(
            dyad=DYAD, task=TASK, kind="mfcc", run="1", extra_entities={"block": "1"}
        )
        tree.add_feature(
            dyad=DYAD, task=TASK, kind="mfcc", run="1", extra_entities={"block": "2"}
        )
        enc = _make_encoding(tree, ["mfcc"], bids_filters=["block-99"])
        feature_paths = enc._discover_features(SUB)
        bold_metas = enc._discover_bold(SUB)
        feature_paths = enc._resolve_cell_keys(SUB, feature_paths, bold_metas)
        feature_paths, bold_metas = enc._apply_filters(SUB, feature_paths, bold_metas)
        with pytest.raises(FileNotFoundError, match="No regressor files match"):
            enc._validate_coverage(SUB, feature_paths, bold_metas)


class TestValidateCoverage:
    def test_valid_alignment_passes(self, tree: BIDSTree):
        tree.add_participants({SUB: DYAD})
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", desc="denoised")
        tree.add_feature(dyad=DYAD, task=TASK, kind="mfcc", run="1")
        enc = _make_encoding(tree, ["mfcc"])
        feature_paths = enc._discover_features(SUB)
        bold_metas = enc._discover_bold(SUB)
        feature_paths = enc._resolve_cell_keys(SUB, feature_paths, bold_metas)
        feature_paths, bold_metas = enc._apply_filters(SUB, feature_paths, bold_metas)
        enc._validate_coverage(SUB, feature_paths, bold_metas)

    def test_empty_features_after_filter_raises(self, tree: BIDSTree):
        tree.add_participants({SUB: DYAD})
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", desc="denoised")
        tree.add_feature(dyad=DYAD, task=TASK, kind="mfcc", run="1")
        enc = _make_encoding(tree, ["mfcc"])
        feature_paths = enc._discover_features(SUB)
        bold_metas = enc._discover_bold(SUB)
        feature_paths = enc._resolve_cell_keys(SUB, feature_paths, bold_metas)
        feature_paths, bold_metas = enc._apply_filters(SUB, feature_paths, bold_metas)
        with pytest.raises(FileNotFoundError, match="No regressor files match"):
            enc._validate_coverage(SUB, {}, bold_metas)

    def test_empty_bold_after_filter_raises(self, tree: BIDSTree):
        tree.add_participants({SUB: DYAD})
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", desc="denoised")
        tree.add_feature(dyad=DYAD, task=TASK, kind="mfcc", run="1")
        enc = _make_encoding(tree, ["mfcc"])
        feature_paths = enc._discover_features(SUB)
        bold_metas = enc._discover_bold(SUB)
        feature_paths = enc._resolve_cell_keys(SUB, feature_paths, bold_metas)
        feature_paths, bold_metas = enc._apply_filters(SUB, feature_paths, bold_metas)
        with pytest.raises(FileNotFoundError, match="No BOLD files match"):
            enc._validate_coverage(SUB, feature_paths, {})

    def test_bold_without_features_raises(self, tree: BIDSTree):
        tree.add_participants({SUB: DYAD})
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", desc="denoised")
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="2", desc="denoised")
        tree.add_feature(dyad=DYAD, task=TASK, kind="mfcc", run="1")
        enc = _make_encoding(tree, ["mfcc"])
        feature_paths = enc._discover_features(SUB)
        bold_metas = enc._discover_bold(SUB)
        feature_paths = enc._resolve_cell_keys(SUB, feature_paths, bold_metas)
        feature_paths, bold_metas = enc._apply_filters(SUB, feature_paths, bold_metas)
        with pytest.raises(FileNotFoundError, match="No regressor files found"):
            enc._validate_coverage(SUB, feature_paths, bold_metas)

    def test_features_without_bold_after_filter_raises(self, tree: BIDSTree):
        # `res` is a BOLD-only entity (features carry no res), so filtering on it
        # narrows BOLDs but not features, leaving run-2 features without a match
        tree.add_participants({SUB: DYAD})
        tree.add_bold(
            sub=SUB,
            task=TASK,
            space=SPACE,
            run="1",
            desc="denoised",
            extra_entities={"res": "2"},
        )
        tree.add_bold(
            sub=SUB,
            task=TASK,
            space=SPACE,
            run="2",
            desc="denoised",
            extra_entities={"res": "3"},
        )
        tree.add_feature(dyad=DYAD, task=TASK, kind="mfcc", run="1")
        tree.add_feature(dyad=DYAD, task=TASK, kind="mfcc", run="2")
        enc = _make_encoding(tree, ["mfcc"], bids_filters=["res-2"])
        feature_paths = enc._discover_features(SUB)
        bold_metas = enc._discover_bold(SUB)
        feature_paths = enc._resolve_cell_keys(SUB, feature_paths, bold_metas)
        feature_paths, bold_metas = enc._apply_filters(SUB, feature_paths, bold_metas)
        with pytest.raises(FileNotFoundError, match="No BOLD file found for regressor"):
            enc._validate_coverage(SUB, feature_paths, bold_metas)

    def test_multiple_bold_gaps_reports_count(self, tree: BIDSTree):
        tree.add_participants({SUB: DYAD})
        for run in ("1", "2", "3"):
            tree.add_bold(sub=SUB, task=TASK, space=SPACE, run=run, desc="denoised")
            tree.add_feature(dyad=DYAD, task=TASK, kind="mfcc", run=run)
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="4", desc="denoised")
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="5", desc="denoised")
        enc = _make_encoding(tree, ["mfcc"])
        feature_paths = enc._discover_features(SUB)
        bold_metas = enc._discover_bold(SUB)
        feature_paths = enc._resolve_cell_keys(SUB, feature_paths, bold_metas)
        feature_paths, bold_metas = enc._apply_filters(SUB, feature_paths, bold_metas)
        with pytest.raises(FileNotFoundError, match="other coverage gaps"):
            enc._validate_coverage(SUB, feature_paths, bold_metas)


class TestFoldHelpers:
    def test_group_cells_by_descriptive_entity(self):
        # cond spans multiple runs — groups collapse runs into shared cond buckets
        cells = {
            CellKey(task="a", run="1", cond="x"),
            CellKey(task="a", run="2", cond="x"),
            CellKey(task="a", run="1", cond="y"),
        }
        groups = _group_cells_by(cells, "cond")
        assert set(groups) == {"x", "y"}
        assert len(groups["x"]) == 2 and len(groups["y"]) == 1

    def test_group_cells_by_missing_key_raises(self):
        cells = {CellKey(task="a", run="1"), CellKey(task="a")}
        with pytest.raises(ValueError, match="not present on cell"):
            _group_cells_by(cells, "run")

    def test_partition_groups_contiguous_chunks(self):
        groups = {str(i): {CellKey(task="a", run=str(i))} for i in range(4)}
        held = _partition_groups(groups, 2)
        all_cells = set().union(*groups.values())
        assert len(held) == 2
        assert held[0].isdisjoint(held[1])
        assert held[0] | held[1] == all_cells

    def test_partition_groups_uneven_remainder_in_earlier_buckets(self):
        groups = {str(i): {CellKey(task="a", run=str(i))} for i in range(5)}
        held = _partition_groups(groups, 2)
        # 5 groups / 2 folds → 3 + 2 (earlier bucket absorbs the remainder)
        assert sorted(len(b) for b in held) == [2, 3]

    def test_partition_groups_loo_one_group_per_fold(self):
        groups = {str(i): {CellKey(task="a", run=str(i))} for i in range(3)}
        held = _partition_groups(groups, "loo")
        assert len(held) == 3
        assert all(len(b) == 1 for b in held)

    def test_partition_groups_n_folds_exceed_groups_raises(self):
        groups = {"1": {CellKey(task="a", run="1")}}
        with pytest.raises(ValueError, match="exceeds"):
            _partition_groups(groups, 2)

    def test_partition_groups_loo_one_group_raises(self):
        groups = {"1": {CellKey(task="a", run="1")}}
        with pytest.raises(ValueError, match="needs >= 2 groups"):
            _partition_groups(groups, "loo")

    def test_select_rows_preserves_build_x_order(self):
        # row_slices insertion order is _sort_key order; _select_rows must keep it
        data = TrainingData(
            X=np.arange(30).reshape(6, 5).astype(np.float64),
            Y=np.arange(6).reshape(6, 1).astype(np.float64),
            row_slices={
                CellKey(task="a", run="1"): slice(0, 2),
                CellKey(task="a", run="2"): slice(2, 4),
                CellKey(task="a", run="3"): slice(4, 6),
            },
            col_slices={"mfcc": slice(0, 5)},
        )
        # subset given out of order; result must follow row_slices order, not arg order
        subset = {CellKey(task="a", run="3"), CellKey(task="a", run="1")}
        X_sub, Y_sub, ordered_cells = _select_rows(data, subset)
        assert ordered_cells == [CellKey(task="a", run="1"), CellKey(task="a", run="3")]
        # rows 0,1 (run 1) then 4,5 (run 3) — run 2 dropped, order preserved
        np.testing.assert_array_equal(Y_sub.ravel(), [0, 1, 4, 5])
        np.testing.assert_array_equal(X_sub, data.X[[0, 1, 4, 5]])


class TestInnerCv:
    """The 3-step inner-unit rule (`_inner_cv`), exercised per branch.

    The selector is pure (cells in, splitter out), so these tests assert on the
    returned `PredefinedSplit.test_fold` / `KFold` directly without any fitting.
    """

    from sklearn.model_selection import KFold, PredefinedSplit

    @staticmethod
    def _group_ids_per_cell(
        test_fold: np.ndarray, cell_lengths: list[int]
    ) -> list[int]:
        # collapse per-row group ids to one id per cell; a constant id within a cell
        # proves leave-one-value-out splits whole cells, never straddling a boundary
        ids = []
        offset = 0
        for length in cell_lengths:
            block = test_fold[offset : offset + length]
            assert len(set(block)) == 1, "group id must be constant within a cell"
            ids.append(block[0])
            offset += length
        return ids

    def test_inner_cv_uses_fold_by_when_multivalued(self):
        # step 1: structural fold_by with >=2 train values → leave-one-run-out
        cells = [CellKey(task="a", run="1"), CellKey(task="a", run="2")]
        cv = _inner_cv(
            ordered_cells=cells,
            cell_lengths=[3, 3],
            segment_entity=None,
            fold_by="run",
        )
        assert isinstance(cv, self.PredefinedSplit)
        assert self._group_ids_per_cell(cv.test_fold, [3, 3]) == [0, 1]

    def test_inner_cv_uses_descriptive_fold_by(self):
        # step 1: fold_by=cond → leave-one-cond-out (descriptive axis is valid)
        cells = [
            CellKey(task="a", run="1", cond="x"),
            CellKey(task="a", run="2", cond="y"),
        ]
        cv = _inner_cv(
            ordered_cells=cells,
            cell_lengths=[2, 2],
            segment_entity=None,
            fold_by="cond",
        )
        assert isinstance(cv, self.PredefinedSplit)
        assert self._group_ids_per_cell(cv.test_fold, [2, 2]) == [0, 1]

    def test_inner_cv_descends_to_coarser_entity_when_fold_by_constant(self):
        # step 2, BIDS non-nesting: fold_by=run constant but coarser ses varies →
        # leave-one-ses-out (the chain does not skip entities coarser than fold_by)
        cells = [
            CellKey(ses="1", task="a", run="1"),
            CellKey(ses="2", task="a", run="1"),
        ]
        cv = _inner_cv(
            ordered_cells=cells,
            cell_lengths=[2, 2],
            segment_entity=None,
            fold_by="run",
        )
        assert isinstance(cv, self.PredefinedSplit)
        assert self._group_ids_per_cell(cv.test_fold, [2, 2]) == [0, 1]

    def test_inner_cv_descends_below_structural_fold_by(self):
        # step 2: fold_by=run single run with trials → leave-one-trial-out
        cells = [
            CellKey(task="a", run="1", trial="1"),
            CellKey(task="a", run="1", trial="2"),
        ]
        cv = _inner_cv(
            ordered_cells=cells,
            cell_lengths=[2, 2],
            segment_entity="trial",
            fold_by="run",
        )
        assert isinstance(cv, self.PredefinedSplit)
        assert self._group_ids_per_cell(cv.test_fold, [2, 2]) == [0, 1]

    def test_inner_cv_excludes_descriptive_from_chain(self):
        # step 2→3: only descriptive cond varies → no structural unit → step 3 KFold
        cells = [
            CellKey(task="a", run="1", cond="x"),
            CellKey(task="a", run="1", cond="y"),
        ]
        cv = _inner_cv(
            ordered_cells=cells,
            cell_lengths=[2, 2],
            segment_entity=None,
            fold_by=None,
        )
        assert isinstance(cv, self.KFold)
        assert cv.n_splits == 2 and cv.shuffle is False

    def test_inner_cv_single_cell_contiguous_split(self):
        # step 3: single cell → contiguous KFold(2, shuffle=False), no seed
        cells = [CellKey(task="a", run="1")]
        cv = _inner_cv(
            ordered_cells=cells,
            cell_lengths=[6],
            segment_entity=None,
            fold_by=None,
        )
        assert isinstance(cv, self.KFold)
        assert cv.n_splits == 2
        assert cv.shuffle is False
        assert cv.random_state is None

    def test_inner_cv_single_cell_too_few_rows_raises(self):
        # tiny-fold guard: single cell with <2 rows cannot form a 2-way split
        cells = [CellKey(task="a", run="1")]
        with pytest.raises(ValueError, match="single cell"):
            _inner_cv(
                ordered_cells=cells,
                cell_lengths=[1],
                segment_entity=None,
                fold_by=None,
            )

    def test_inner_cv_never_straddles_cell_boundary(self):
        # invariant: every cell's rows carry one group id across uneven lengths
        cells = [
            CellKey(task="a", run="1"),
            CellKey(task="a", run="2"),
            CellKey(task="a", run="3"),
        ]
        cv = _inner_cv(
            ordered_cells=cells,
            cell_lengths=[2, 3, 4],
            segment_entity=None,
            fold_by="run",
        )
        assert isinstance(cv, self.PredefinedSplit)
        assert self._group_ids_per_cell(cv.test_fold, [2, 3, 4]) == [0, 1, 2]
        assert cv.get_n_splits() == 3


class TestAssembleTrainingData:
    """End-to-end `_assemble_training_data`: discovers, reads, downsamples, assembles X.

    Drives the real `sub -> TrainingData` orchestration (the same method `train()`
    calls) up to but excluding the fit, so a reordered or dropped step surfaces here
    rather than passing against a stale hand-listed chain.
    """

    def _assemble(self, tree: BIDSTree, feature_df: pl.DataFrame) -> TrainingData:
        _add_dyad_turns(tree)
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="denoised")
        tree.add_feature(dyad=DYAD, task=TASK, kind="mfcc", run="1", df=feature_df)
        enc = _make_encoding(tree, ["mfcc"])
        return enc._assemble_training_data(SUB)

    @staticmethod
    def _feature_df() -> pl.DataFrame:
        return pl.DataFrame(
            {"start_time": [0.0, 4.0, 8.0], "feature": [[1.0], [2.0], [3.0]]},
            schema={"start_time": pl.Float64, "feature": pl.Array(pl.Float64, 1)},
        )

    @staticmethod
    def _confound_df(n_trs: int, width: int = 1) -> pl.DataFrame:
        # Confounds are TR-level: one row per TR (spaced by the 2.0 TR), no downsample
        return pl.DataFrame(
            {
                "start_time": [2.0 * i for i in range(n_trs)],
                "confound": [[float(i)] * width for i in range(n_trs)],
            },
            schema={"start_time": pl.Float64, "confound": pl.Array(pl.Float64, width)},
        )

    def test_untimed_row_dropped_x_matches_all_timed(
        self, tree: BIDSTree, tmp_path_factory: pytest.TempPathFactory
    ):
        # Distinct per-row feature values: a stray start_time drop that left the
        # feature column unfiltered would shift X (or mismatch length and raise)
        schema = {"start_time": pl.Float64, "feature": pl.Array(pl.Float64, 1)}
        null_df = pl.DataFrame(
            {
                "start_time": [0.0, None, 4.0, 8.0],
                "feature": [[1.0], [2.0], [3.0], [4.0]],
            },
            schema=schema,
        )
        timed_df = pl.DataFrame(
            {
                "start_time": [0.0, 4.0, 8.0],
                "feature": [[1.0], [3.0], [4.0]],
            },
            schema=schema,
        )
        timed_tree = BIDSTree(tmp_path_factory.mktemp("timed"))
        x_null = self._assemble(tree, null_df).X
        x_timed = self._assemble(timed_tree, timed_df).X
        np.testing.assert_array_equal(x_null, x_timed)

    def test_segment_entity_derived_from_block_segmented_run(self, tree: BIDSTree):
        # Localizes the derivation half: real key-value events => segmented BOLD =>
        # segment_entity == 'block'. Points straight at the derivation if it breaks.
        _add_block_segmented_run(tree)
        enc = _make_encoding(tree, ["mfcc"])
        data = enc._assemble_training_data(SUB)
        assert data.segment_entity == "block"

    def test_confounds_appended_as_single_trailing_band(self, tree: BIDSTree):
        # Two confounds collapse into one band; the feature keeps its own named band
        _add_dyad_turns(tree)
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="denoised")
        tree.add_feature(
            dyad=DYAD, task=TASK, kind="mfcc", run="1", df=self._feature_df()
        )
        tree.add_confound(
            dyad=DYAD,
            task=TASK,
            kind="phonemic",
            run="1",
            desc="onset",
            df=self._confound_df(DEFAULT_BOLD_N_TRS),
        )
        tree.add_confound(
            dyad=DYAD,
            task=TASK,
            kind="phonemic",
            run="1",
            desc="rate",
            df=self._confound_df(DEFAULT_BOLD_N_TRS),
        )
        enc = _make_encoding(
            tree, ["mfcc"], confounds=["phonemic-onset", "phonemic-rate"]
        )
        data = enc._assemble_training_data(SUB)
        assert list(data.col_slices) == [_SCREEN_BAND, "mfcc", _CONFOUND_BAND]
        assert data.col_slices[_SCREEN_BAND] == slice(0, 2)
        # Two 1-wide confounds collapse into one trailing band
        assert data.col_slices["mfcc"] == slice(2, 3)
        assert data.col_slices[_CONFOUND_BAND] == slice(3, 5)
        assert data.X.shape == (DEFAULT_BOLD_N_TRS, 5)

    def test_surface_y_joins_both_hemis_on_voxel_axis(self, tree: BIDSTree):
        # End-to-end surface path: `_align_y` must load both hemis and join them,
        # so Y spans 2 * n_vertices columns over the run's TR rows.
        n_vertices = 4
        _add_dyad_turns(tree)
        tree.add_surface_bold(
            sub=SUB,
            task=TASK,
            run="1",
            space="fsaverage6",
            desc="denoised",
            n_vertices=n_vertices,
        )
        tree.add_feature(
            dyad=DYAD, task=TASK, kind="mfcc", run="1", df=self._feature_df()
        )
        enc = _make_encoding(tree, ["mfcc"], bold_space="fsaverage6")
        data = enc._assemble_training_data(SUB)
        assert data.Y.shape == (DEFAULT_BOLD_N_TRS, 2 * n_vertices)

    def test_confound_wrong_tr_count_raises(self, tree: BIDSTree):
        # Confounds must already span the cell's TRs; a short file raises rather
        # than being silently binned
        tree.add_participants({SUB: DYAD})
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="denoised")
        tree.add_feature(
            dyad=DYAD, task=TASK, kind="mfcc", run="1", df=self._feature_df()
        )
        tree.add_confound(
            dyad=DYAD,
            task=TASK,
            kind="phonemic",
            run="1",
            df=self._confound_df(DEFAULT_BOLD_N_TRS - 1),
        )
        enc = _make_encoding(tree, ["mfcc"], confounds=["phonemic"])
        with pytest.raises(ValueError, match="confounds must be saved"):
            enc._assemble_training_data(SUB)

    def test_regressor_shape_error_fires_before_coverage_error(self, tree: BIDSTree):
        # A regressor metadata mismatch (mfcc model v1 vs v2) and a coverage gap
        # (clip only at run-1) coexist under uniform BOLD. `_assemble_training_data`
        # runs regressor-shape before coverage, so the metadata error must win — and
        # this also guards that the relocated shape check is wired into assembly at
        # all (a coverage-only fixture would slip a dropped shape line past the suite).
        _add_dyad_turns(tree, run="1")
        _add_dyad_turns(tree, run="2")
        for run in ("1", "2"):
            tree.add_bold(
                sub=SUB, task=TASK, space=SPACE, run=run, tr=2.0, desc="denoised"
            )
        tree.add_feature(
            dyad=DYAD,
            task=TASK,
            kind="mfcc",
            run="1",
            df=self._feature_df(),
            metadata={"model": "v1"},
        )
        tree.add_feature(
            dyad=DYAD,
            task=TASK,
            kind="mfcc",
            run="2",
            df=self._feature_df(),
            metadata={"model": "v2"},
        )
        tree.add_feature(
            dyad=DYAD, task=TASK, kind="clip", run="1", df=self._feature_df()
        )
        enc = _make_encoding(tree, ["mfcc", "clip"])
        with pytest.raises(ValueError, match="Inconsistent metadata for feat=mfcc"):
            enc._assemble_training_data(SUB)

    def test_missing_regressor_at_cell_raises_in_assembly(self, tree: BIDSTree):
        # Guards that `_require_regressor_at_every_cell` is wired into assembly: clip
        # is present at run-1 but missing at run-2 (mfcc covers both), a gap
        # `_validate_coverage` does not catch — it would slip into `_build_x`.
        _add_dyad_turns(tree, run="1")
        _add_dyad_turns(tree, run="2")
        for run in ("1", "2"):
            tree.add_bold(
                sub=SUB, task=TASK, space=SPACE, run=run, tr=2.0, desc="denoised"
            )
            tree.add_feature(
                dyad=DYAD, task=TASK, kind="mfcc", run=run, df=self._feature_df()
            )
        tree.add_feature(
            dyad=DYAD, task=TASK, kind="clip", run="1", df=self._feature_df()
        )
        enc = _make_encoding(tree, ["mfcc", "clip"])
        with pytest.raises(FileNotFoundError, match="Missing regressor=clip"):
            enc._assemble_training_data(SUB)

    def test_mixed_tr_bold_raises_in_assembly(self, tree: BIDSTree):
        # Guards that `_require_uniform_bold_shape` is wired into assembly: run-1 and
        # run-2 carry different TRs with no filter to discard either, so the mixed set
        # reaches assembly and must raise before pooling into one X/Y.
        _add_dyad_turns(tree, run="1")
        _add_dyad_turns(tree, run="2")
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="denoised")
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="2", tr=1.5, desc="denoised")
        for run in ("1", "2"):
            tree.add_feature(
                dyad=DYAD, task=TASK, kind="mfcc", run=run, df=self._feature_df()
            )
        enc = _make_encoding(tree, ["mfcc"])
        with pytest.raises(ValueError, match="Inconsistent repetition times"):
            enc._assemble_training_data(SUB)

    def test_filter_narrows_heterogeneous_subject_to_uniform_subset(
        self, tree: BIDSTree
    ):
        # The motivating case for relocating shape checks past the filter: run-1 and
        # run-2 have different TRs (heterogeneous subject), but a run-1 filter narrows
        # to a shape-uniform subset, so assembly must pass rather than raise on the
        # discarded run-2 TR.
        _add_dyad_turns(tree, run="1")
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="denoised")
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="2", tr=1.5, desc="denoised")
        tree.add_feature(
            dyad=DYAD, task=TASK, kind="mfcc", run="1", df=self._feature_df()
        )
        tree.add_feature(
            dyad=DYAD, task=TASK, kind="mfcc", run="2", df=self._feature_df()
        )
        enc = _make_encoding(tree, ["mfcc"], bids_filters=["run-1"])
        data = enc._assemble_training_data(SUB)
        assert list(data.row_slices) == [CellKey(task=TASK, run="1")]


class TestSplit:
    """The prod/comp screens (always on) and the opt-in regressor split.

    The fixture `_add_dyad_turns` gives a 10-TR run at TR=2.0 (so TR i covers
    second 2*i) where SUB speaks over seconds [0, 10) and PARTNER over [10, 20).
    SUB speaking is "production", the partner speaking is "comprehension", so:
    SUB's prod mask is on for TRs 0..4, its comp mask is on for TRs 5..9. Every
    assertion below rests on that 5-on/5-off layout, with no silent (unspoken) TRs.
    """

    @staticmethod
    def _feature_df() -> pl.DataFrame:
        # One distinct timed value per TR, so a mis-zeroed split column is visible
        return pl.DataFrame(
            {
                "start_time": [2.0 * i for i in range(DEFAULT_BOLD_N_TRS)],
                "feature": [[float(i + 1)] for i in range(DEFAULT_BOLD_N_TRS)],
            },
            schema={"start_time": pl.Float64, "feature": pl.Array(pl.Float64, 1)},
        )

    def _assemble(self, tree: BIDSTree, *, split: bool) -> TrainingData:
        _add_dyad_turns(tree)
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="denoised")
        tree.add_feature(
            dyad=DYAD, task=TASK, kind="mfcc", run="1", df=self._feature_df()
        )
        enc = _make_encoding(tree, ["mfcc"], split=split)
        return enc._assemble_training_data(SUB)

    # Screen band (always on): the two prod/comp boxcars the masks feed into.

    def test_screen_band_always_present_when_unsplit(self, tree: BIDSTree):
        # The screen band is always present; only feature and confound bands
        # change with `split`
        data = self._assemble(tree, split=False)
        assert list(data.col_slices) == [_SCREEN_BAND, "mfcc"]
        assert data.col_slices[_SCREEN_BAND] == slice(0, 2)
        assert data.col_slices["mfcc"] == slice(2, 3)
        assert data.X.shape == (DEFAULT_BOLD_N_TRS, 3)

    def test_screen_columns_hold_prod_comp_boxcars(self, tree: BIDSTree):
        # Screen col 0 = prod (SUB, TRs 0..4), col 1 = comp (PARTNER, TRs 5..9)
        data = self._assemble(tree, split=False)
        screens = data.X[:, data.col_slices[_SCREEN_BAND]]
        expected = np.zeros((DEFAULT_BOLD_N_TRS, 2))
        expected[:5, 0] = 1.0
        expected[5:, 1] = 1.0
        np.testing.assert_array_equal(screens, expected)

    # Split (opt-in): duplicates each regressor into prod/comp copies via the masks.

    def test_split_doubles_feature_band_not_screen_band(self, tree: BIDSTree):
        # Split never touches the screen band; only feature/confound bands double
        data = self._assemble(tree, split=True)
        assert list(data.col_slices) == [_SCREEN_BAND, "mfcc"]
        assert data.col_slices[_SCREEN_BAND] == slice(0, 2)
        assert data.col_slices["mfcc"] == slice(2, 4)
        assert data.X.shape == (DEFAULT_BOLD_N_TRS, 4)

    def test_split_zeroes_each_copy_off_its_turn(self, tree: BIDSTree):
        # Each copy keeps the feature only on its own turn's TRs. A TR where
        # neither speaks would be zero in both copies, but this fixture has none.
        data = self._assemble(tree, split=True)
        feature = self._feature_df().get_column("feature").to_numpy().ravel()
        prod_copy, comp_copy = data.X[:, 2], data.X[:, 3]
        expected_prod = feature.copy()
        expected_prod[5:] = 0.0
        expected_comp = feature.copy()
        expected_comp[:5] = 0.0
        np.testing.assert_array_equal(prod_copy, expected_prod)
        np.testing.assert_array_equal(comp_copy, expected_comp)

    # Guards (through assembly): failure modes when the dyad or masks are unusable.

    def test_non_dyad_of_two_raises(self, tree: BIDSTree):
        # "the partner" (and thus comp) is undefined for a 1- or 3-subject dyad.
        tree.add_participants({SUB: DYAD})
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="denoised")
        tree.add_feature(
            dyad=DYAD, task=TASK, kind="mfcc", run="1", df=self._feature_df()
        )
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=[
                {"trial_type": "turn_speaker", "onset": 0.0, "duration": 20.0},
            ],
        )
        enc = _make_encoding(tree, ["mfcc"])
        with pytest.raises(ValueError, match="requires a 2-subject dyad"):
            enc._assemble_training_data(SUB)

    def test_all_zero_screens_raises(self, tree: BIDSTree):
        tree.add_participants({SUB: DYAD, PARTNER: DYAD})
        tree.add_bold(sub=SUB, task=TASK, space=SPACE, run="1", tr=2.0, desc="denoised")
        tree.add_feature(
            dyad=DYAD, task=TASK, kind="mfcc", run="1", df=self._feature_df()
        )
        # Both turns sit past the BOLD's last TR onset (18.0 s), so every mask is empty
        tree.add_events(
            sub=SUB,
            task=TASK,
            run="1",
            rows=[
                {"trial_type": "turn_speaker", "onset": 100.0, "duration": 5.0},
            ],
        )
        tree.add_events(
            sub=PARTNER,
            task=TASK,
            run="1",
            rows=[
                {"trial_type": "turn_speaker", "onset": 105.0, "duration": 5.0},
            ],
        )
        enc = _make_encoding(tree, ["mfcc"])
        with pytest.raises(ValueError, match="all-zero"):
            enc._assemble_training_data(SUB)


class TestTrainWiring:
    def test_train_fits_and_returns_artifact(
        self, tree: BIDSTree, monkeypatch: pytest.MonkeyPatch
    ):
        # Exercise the train() wiring (backend, float32 cast, cell_lengths,
        # fit, return) without _discover_*/BIDS plumbing. A hyphenated feature
        # name checks it survives ColumnKernelizer's transformer-name handling.
        n_rows, n_voxels = 20, 5
        rng = np.random.RandomState(0)
        data = TrainingData(
            X=rng.randn(n_rows, 7).astype(np.float64),
            Y=rng.randn(n_rows, n_voxels).astype(np.float64),
            row_slices={
                CellKey(task="a", run="1"): slice(0, 10),
                CellKey(task="a", run="2"): slice(10, 20),
            },
            col_slices={"phonemic-gpt3": slice(0, 3), "mfcc": slice(3, 7)},
        )
        enc = _make_encoding(tree, ["phonemic-gpt3", "mfcc"])
        monkeypatch.setattr(enc, "_assemble_training_data", lambda *a, **k: data)

        from sklearn.pipeline import Pipeline

        artifact = enc.train(SUB)
        assert isinstance(artifact, EncodingArtifact)
        assert len(artifact.models) == 1
        pipeline = artifact.models[0].pipeline
        assert isinstance(pipeline, Pipeline)
        pred = np.asarray(pipeline.predict(data.X.astype(np.float32)))
        assert pred.shape == (n_rows, n_voxels)
        # train records the cells it fit on and leaves universe unbound
        assert artifact.models[0].train_cells == set(data.row_slices)
        assert artifact.universe is None
        assert artifact.recipe.col_slices == data.col_slices


class TestFoldedTrain:
    """Outer folding fits K models, each on universe minus one held-out group."""

    # Four runs, one cell each; rows are contiguous per cell in _sort_key order
    CELLS = [CellKey(task="a", run=str(i)) for i in range(1, 5)]

    @pytest.fixture
    def make_train_setup(
        self, tree: BIDSTree, monkeypatch: pytest.MonkeyPatch
    ) -> _TrainSetupFactory:
        def _build(
            *,
            fold_by: str,
            n_folds: int | Literal["loo"],
            cells: list[CellKey] | None = None,
        ) -> tuple[EncodingTrainer, TrainingData]:
            cells = cells if cells is not None else self.CELLS
            n_per, n_cols, n_voxels = 5, 7, 3
            rng = np.random.RandomState(0)
            n_rows = n_per * len(cells)
            data = TrainingData(
                X=rng.randn(n_rows, n_cols).astype(np.float64),
                Y=rng.randn(n_rows, n_voxels).astype(np.float64),
                row_slices={
                    cell: slice(i * n_per, (i + 1) * n_per)
                    for i, cell in enumerate(cells)
                },
                col_slices={"phonemic-gpt3": slice(0, 3), "mfcc": slice(3, 7)},
            )
            enc = _make_encoding(
                tree, ["phonemic-gpt3", "mfcc"], fold_by=fold_by, n_folds=n_folds
            )
            monkeypatch.setattr(enc, "_assemble_training_data", lambda *a, **k: data)
            return enc, data

        return _build

    def test_partition_correctness(self, make_train_setup: _TrainSetupFactory):
        enc, data = make_train_setup(fold_by="run", n_folds=2)
        artifact = enc.train(SUB)
        universe = set(data.row_slices)

        assert len(artifact.models) == 2
        assert artifact.universe == universe
        held_out = [universe - m.train_cells for m in artifact.models]
        assert all(m.train_cells < universe for m in artifact.models)
        assert held_out[0].isdisjoint(held_out[1])
        assert held_out[0] | held_out[1] == universe

    def test_loo_one_group_held_out_per_model(
        self, make_train_setup: _TrainSetupFactory
    ):
        enc, data = make_train_setup(fold_by="run", n_folds="loo")
        artifact = enc.train(SUB)
        universe = set(data.row_slices)

        assert len(artifact.models) == len(self.CELLS)
        assert all(len(universe - m.train_cells) == 1 for m in artifact.models)

    def test_fold_records_fold_config(self, make_train_setup: _TrainSetupFactory):
        enc, _ = make_train_setup(fold_by="run", n_folds=2)
        artifact = enc.train(SUB)
        assert artifact.fold is not None
        assert artifact.fold.by == "run"
        assert artifact.fold.n == 2

    def test_fold_by_descriptive_spans_runs(self, make_train_setup: _TrainSetupFactory):
        # two conds, each spanning two runs → folds=2 over cond holds out whole conds
        cells = [
            CellKey(task="a", run="1", cond="x"),
            CellKey(task="a", run="2", cond="x"),
            CellKey(task="a", run="3", cond="y"),
            CellKey(task="a", run="4", cond="y"),
        ]
        enc, data = make_train_setup(fold_by="cond", n_folds=2, cells=cells)
        artifact = enc.train(SUB)
        universe = set(data.row_slices)
        held_out = [universe - m.train_cells for m in artifact.models]
        # each held-out set is one whole cond (two runs)
        assert all(len(h) == 2 for h in held_out)
        assert all(len({c["cond"] for c in h}) == 1 for h in held_out)

    def test_missing_fold_by_key_raises_in_train(
        self, make_train_setup: _TrainSetupFactory
    ):
        # 'cond' is absent from every cell — late, data-dependent failure
        enc, _ = make_train_setup(fold_by="cond", n_folds=2)
        with pytest.raises(ValueError, match="no 'cond' axis"):
            enc.train(SUB)

    def test_n_folds_exceed_groups_raises_in_train(
        self, make_train_setup: _TrainSetupFactory
    ):
        enc, _ = make_train_setup(fold_by="run", n_folds=2, cells=[self.CELLS[0]])
        with pytest.raises(ValueError, match="exceeds"):
            enc.train(SUB)

    def test_loo_single_group_raises_in_train(
        self, make_train_setup: _TrainSetupFactory
    ):
        enc, _ = make_train_setup(fold_by="run", n_folds="loo", cells=[self.CELLS[0]])
        with pytest.raises(ValueError, match="needs >= 2 groups"):
            enc.train(SUB)

    def test_runless_dataset_raises_in_train(
        self, make_train_setup: _TrainSetupFactory
    ):
        # run-less: 'run' uniformly absent → clear "no axis" error, not a collapse
        enc, _ = make_train_setup(
            fold_by="run",
            n_folds=2,
            cells=[CellKey(task="a"), CellKey(task="b")],
        )
        with pytest.raises(ValueError, match="no 'run' axis"):
            enc.train(SUB)

    def test_round_trip_preserves_models_and_cells(
        self, make_train_setup: _TrainSetupFactory
    ):
        from himalaya.backend import set_backend

        enc, data = make_train_setup(fold_by="run", n_folds=2)
        artifact = enc.train(SUB)
        out = enc._layout.path.result(sub=SUB, kind="encodingModel", desc="v1")
        save_artifact(artifact, out.path)

        loaded = load_artifact(out.path)
        assert len(loaded.models) == len(artifact.models)
        assert loaded.universe == artifact.universe
        assert [m.train_cells for m in loaded.models] == [
            m.train_cells for m in artifact.models
        ]

        # A reloaded per-fold pipeline predicts — proves the K-model write/read
        # path end to end, not just metadata survival. Each model's CellDelayer
        # is sized to its own training rows, so predict on that fold's slice.
        set_backend("numpy")
        for orig, got in zip(artifact.models, loaded.models):
            X_fold, _, _ = _select_rows(data, orig.train_cells)
            X_fold = X_fold.astype(np.float32)
            ref = np.asarray(orig.pipeline.predict(X_fold))
            np.testing.assert_array_equal(np.asarray(got.pipeline.predict(X_fold)), ref)

    def test_per_fold_cells_follow_sort_order(
        self, make_train_setup: _TrainSetupFactory
    ):
        # Guards the CellDelayer-boundary bug: _select_rows must return cells in
        # row_slices (_sort_key) order regardless of train_cells set iteration, so
        # per-cell row counts derived downstream stay aligned with the row layout
        _, data = make_train_setup(fold_by="run", n_folds=2)
        universe = set(data.row_slices)
        held = _partition_groups(_group_cells_by(universe, "run"), 2)
        train_cells = universe - held[0]
        _, _, ordered_cells = _select_rows(data, train_cells)
        expected_order = [c for c in data.row_slices if c in train_cells]
        assert ordered_cells == expected_order

    def test_inner_cv_set_per_model(self, make_train_setup: _TrainSetupFactory):
        # Inner CV is rebuilt per model, confined to that model's train cells.
        from sklearn.model_selection import PredefinedSplit

        # loo over 5 runs → each model holds out one run, trains on 4 → 4 inner splits
        cells = [CellKey(task="a", run=str(i)) for i in range(1, 6)]
        enc, _ = make_train_setup(fold_by="run", n_folds="loo", cells=cells)
        artifact = enc.train(SUB)
        cvs = [m.pipeline.named_steps["model"].cv for m in artifact.models]
        assert all(isinstance(cv, PredefinedSplit) for cv in cvs)
        assert all(cv.get_n_splits() == 4 for cv in cvs)
        # distinct instances → rebuilt per model, not built once and shared
        assert cvs[0] is not cvs[1]


class TestTrainIntegration:
    """End-to-end train() over toy BIDS files — no monkeypatching.

    Covers the disk -> _discover_bold -> bold_metas -> segment_entity -> _inner_cv
    chain that the stubbed TestTrainWiring/TestFoldedTrain tests bypass: there
    _apply_filters/_build_training_data are stubbed, forcing bold_metas empty and
    segment_entity None.
    """

    def test_segmented_run_threads_segment_entity_into_inner_cv(self, tree: BIDSTree):
        from sklearn.model_selection import PredefinedSplit

        _add_block_segmented_run(tree)
        enc = _make_encoding(tree, ["mfcc"])
        artifact = enc.train(SUB)

        # segment_entity='block' descended into the inner CV: a single run with
        # two blocks yields leave-one-block-out (PredefinedSplit), not the
        # contiguous KFold that segment_entity=None would have forced here
        cv = artifact.models[0].pipeline.named_steps["model"].cv
        assert isinstance(cv, PredefinedSplit)
        assert cv.get_n_splits() == 2
