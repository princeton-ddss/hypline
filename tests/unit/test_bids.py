from pathlib import Path

import pytest

from hypline.bids import (
    BIDSPath,
    find_bids_files,
    normalize_bids_filters,
    validate_bids_entities,
    validate_entity_invariance,
    validate_extension,
)


class TestBIDSPathParsing:
    def test_basic_entities(self):
        bp = BIDSPath("sub-01_ses-1_task-rest_bold.nii.gz")
        assert bp.entities == {"sub": "01", "ses": "1", "task": "rest"}
        assert bp.suffix == "bold"
        assert bp.extension == ".nii.gz"

    def test_entities_without_suffix(self):
        bp = BIDSPath("sub-01_ses-1.json")
        assert bp.entities == {"sub": "01", "ses": "1"}
        assert bp.suffix is None
        assert bp.extension == ".json"

    def test_no_extension(self):
        bp = BIDSPath("sub-01_ses-1_bold")
        assert bp.suffix == "bold"
        assert bp.extension == ""

    def test_single_entity_with_suffix(self):
        bp = BIDSPath("sub-01_bold.nii.gz")
        assert bp.entities == {"sub": "01"}
        assert bp.suffix == "bold"

    def test_single_entity_no_suffix(self):
        bp = BIDSPath("sub-01.tsv")
        assert bp.entities == {"sub": "01"}
        assert bp.suffix is None

    def test_many_entities(self):
        bp = BIDSPath(
            "sub-004_ses-1_task-Black_space-fsaverage6_hemi-L_desc-clean_bold.func.gii"
        )
        assert bp.entities == {
            "sub": "004",
            "ses": "1",
            "task": "Black",
            "space": "fsaverage6",
            "hemi": "L",
            "desc": "clean",
        }
        assert bp.suffix == "bold"
        assert bp.extension == ".func.gii"

    def test_path_preserved(self):
        bp = BIDSPath("/data/raw/sub-01/ses-1/sub-01_ses-1_bold.nii.gz")
        assert bp.path == Path("/data/raw/sub-01/ses-1/sub-01_ses-1_bold.nii.gz")

    def test_path_object_input(self):
        p = Path("sub-01_bold.nii")
        bp = BIDSPath(p)
        assert bp.path == p


class TestBIDSPathEntityAccess:
    def test_getattr(self):
        bp = BIDSPath("sub-01_task-rest_bold.nii")
        assert bp.sub == "01"
        assert bp.task == "rest"

    def test_getattr_missing_entity(self):
        bp = BIDSPath("sub-01_bold.nii")
        with pytest.raises(AttributeError, match="ses"):
            _ = bp.ses

    def test_getattr_private_attr(self):
        bp = BIDSPath("sub-01_bold.nii")
        with pytest.raises(AttributeError):
            _ = bp._nonexistent


class TestBIDSPathWithEntity:
    def test_add_new_entity(self):
        bp = BIDSPath("sub-01_bold.nii.gz")
        bp2 = bp.with_entity("desc", "clean")
        assert bp2.entities == {"sub": "01", "desc": "clean"}
        assert bp2.suffix == "bold"
        assert bp2.extension == ".nii.gz"

    def test_replace_existing_entity(self):
        bp = BIDSPath("sub-01_ses-1_bold.nii")
        bp2 = bp.with_entity("ses", "2")
        assert bp2.ses == "2"
        assert bp2.sub == "01"

    def test_with_entity_preserves_suffix(self):
        bp = BIDSPath("sub-01_bold.nii")
        bp2 = bp.with_entity("run", "02")
        assert bp2.suffix == "bold"

    def test_with_entity_preserves_no_suffix(self):
        bp = BIDSPath("sub-01_ses-1.json")
        bp2 = bp.with_entity("ses", "2")
        assert bp2.suffix is None

    def test_original_unchanged(self):
        bp = BIDSPath("sub-01_ses-1_bold.nii")
        _ = bp.with_entity("ses", "2")
        assert bp.ses == "1"

    def test_new_entity_appended_at_end(self):
        bp = BIDSPath("sub-01_ses-1_bold.nii")
        bp2 = bp.with_entity("desc", "clean")
        assert list(bp2.entities) == ["sub", "ses", "desc"]
        assert bp2.path.name == "sub-01_ses-1_desc-clean_bold.nii"

    def test_replaced_entity_keeps_position(self):
        bp = BIDSPath("sub-01_ses-1_run-1_bold.nii")
        bp2 = bp.with_entity("ses", "2")
        assert list(bp2.entities) == ["sub", "ses", "run"]
        assert bp2.path.name == "sub-01_ses-2_run-1_bold.nii"

    def test_with_entity_invalid_value(self):
        bp = BIDSPath("sub-01_bold.nii")
        with pytest.raises(ValueError, match="Invalid BIDS entity"):
            bp.with_entity("desc", "has spaces")


class TestBIDSPathValidation:
    def test_no_entities(self):
        with pytest.raises(ValueError, match="No BIDS entities"):
            BIDSPath("bold.nii.gz")

    def test_invalid_entity_key_uppercase(self):
        with pytest.raises(ValueError, match="Invalid BIDS entity"):
            BIDSPath("Sub-01_bold.nii")

    def test_invalid_entity_value_special_chars(self):
        with pytest.raises(ValueError, match="Invalid BIDS entity"):
            BIDSPath("sub-01!_bold.nii")

    def test_duplicate_entity_key(self):
        with pytest.raises(ValueError, match="Duplicate BIDS entity key"):
            BIDSPath("sub-01_sub-02_bold.nii")

    def test_non_entity_segment_not_last(self):
        with pytest.raises(ValueError, match="must be the last segment"):
            BIDSPath("sub-01_notentity_ses-1_bold.nii")

    def test_invalid_suffix(self):
        with pytest.raises(ValueError, match="Invalid BIDS suffix"):
            BIDSPath("sub-01_bold!.nii")


class TestBIDSPathRepr:
    def test_repr(self):
        bp = BIDSPath("sub-01_bold.nii")
        assert repr(bp) == "BIDSPath('sub-01_bold.nii')"


class TestBIDSPathComparison:
    def test_equal_same_path(self):
        assert BIDSPath("sub-01_bold.nii") == BIDSPath("sub-01_bold.nii")

    def test_not_equal_different_path(self):
        assert BIDSPath("sub-01_bold.nii") != BIDSPath("sub-02_bold.nii")

    def test_hashable_in_set(self):
        bp1 = BIDSPath("sub-01_bold.nii")
        bp2 = BIDSPath("sub-01_bold.nii")
        assert len({bp1, bp2}) == 1

    def test_sortable(self):
        bp1 = BIDSPath("sub-01_bold.nii")
        bp2 = BIDSPath("sub-02_bold.nii")
        assert bp1 < bp2
        assert sorted([bp2, bp1]) == [bp1, bp2]

    def test_lt_raises_for_non_bidspath(self):
        bp = BIDSPath("sub-01_bold.nii")
        with pytest.raises(TypeError):
            _ = bp < "not-a-bidspath"  # type: ignore[arg-type]

    def test_eq_returns_not_implemented_for_non_bidspath(self):
        bp = BIDSPath("sub-01_bold.nii")
        assert bp.__eq__("not-a-bidspath") is NotImplemented

    def test_hash_consistent_with_eq(self):
        bp1 = BIDSPath("sub-01_bold.nii")
        bp2 = BIDSPath("sub-01_bold.nii")
        assert bp1 == bp2
        assert hash(bp1) == hash(bp2)


class TestValidateExtension:
    def test_valid_single(self):
        validate_extension(".nii")

    def test_valid_compound(self):
        validate_extension(".nii.gz")
        validate_extension(".func.gii")

    def test_invalid_no_dot(self):
        with pytest.raises(ValueError, match="Invalid extension"):
            validate_extension("nii")

    def test_invalid_dot_only(self):
        with pytest.raises(ValueError, match="Invalid extension"):
            validate_extension(".")

    def test_invalid_special_chars(self):
        with pytest.raises(ValueError, match="Invalid extension"):
            validate_extension(".nii!")


class TestNormalizeBidsFilters:
    def test_returns_list_copy(self):
        orig = ["sub-01"]
        result = normalize_bids_filters(orig)
        assert result == orig
        assert result is not orig

    def test_none_returns_empty_list(self):
        assert normalize_bids_filters(None) == []

    def test_rejects_invalid_entity(self):
        with pytest.raises(ValueError, match="Invalid BIDS entity"):
            normalize_bids_filters(["BAD"])

    def test_rejects_reserved_key(self):
        with pytest.raises(ValueError, match="sub"):
            normalize_bids_filters(["sub-01"], reserved=["sub"])

    def test_allows_non_reserved(self):
        result = normalize_bids_filters(["task-rest", "ses-1"], reserved=["sub"])
        assert result == ["task-rest", "ses-1"]


class TestValidateEntityInvariance:
    def test_passes_when_consistent(self):
        paths = [BIDSPath("sub-01_ses-1_bold.nii"), BIDSPath("sub-01_ses-1_bold.nii")]
        validate_entity_invariance(paths, ["sub", "ses"])

    def test_raises_on_mismatch(self):
        paths = [BIDSPath("sub-01_bold.nii"), BIDSPath("sub-02_bold.nii")]
        with pytest.raises(ValueError, match="sub"):
            validate_entity_invariance(paths, ["sub"])

    def test_includes_none_in_display(self):
        paths = [BIDSPath("sub-01_ses-1_bold.nii"), BIDSPath("sub-01_bold.nii")]
        with pytest.raises(ValueError, match="ses"):
            validate_entity_invariance(paths, ["ses"])


class TestFindBidsFiles:
    def test_finds_matching_files(self, tmp_path: Path):
        (tmp_path / "sub-01_task-rest_bold.nii.gz").touch()
        results = find_bids_files(tmp_path, ".nii.gz")
        assert len(results) == 1
        assert results[0].sub == "01"

    def test_suffix_filter(self, tmp_path: Path):
        (tmp_path / "sub-01_bold.nii.gz").touch()
        (tmp_path / "sub-01_mask.nii.gz").touch()
        results = find_bids_files(tmp_path, ".nii.gz", suffix="bold")
        assert len(results) == 1
        assert results[0].suffix == "bold"

    def test_bids_filter_and(self, tmp_path: Path):
        (tmp_path / "sub-01_task-rest_bold.nii.gz").touch()
        (tmp_path / "sub-02_task-conv_bold.nii.gz").touch()
        results = find_bids_files(
            tmp_path, ".nii.gz", bids_filters=["sub-01", "task-rest"]
        )
        assert len(results) == 1
        assert results[0].sub == "01"

    def test_bids_filter_or_within_key(self, tmp_path: Path):
        (tmp_path / "sub-01_bold.nii.gz").touch()
        (tmp_path / "sub-02_bold.nii.gz").touch()
        (tmp_path / "sub-03_bold.nii.gz").touch()
        results = find_bids_files(
            tmp_path, ".nii.gz", bids_filters=["sub-01", "sub-02"]
        )
        assert len(results) == 2

    def test_bids_filter_and_with_or(self, tmp_path: Path):
        (tmp_path / "sub-01_task-rest_bold.nii.gz").touch()
        (tmp_path / "sub-02_task-rest_bold.nii.gz").touch()
        (tmp_path / "sub-01_task-conv_bold.nii.gz").touch()
        results = find_bids_files(
            tmp_path, ".nii.gz", bids_filters=["sub-01", "sub-02", "task-rest"]
        )
        assert len(results) == 2
        assert all(f.task == "rest" for f in results)

    def test_skips_non_parseable(self, tmp_path: Path):
        (tmp_path / "README.txt").touch()
        (tmp_path / "sub-01_bold.nii.gz").touch()
        results = find_bids_files(tmp_path, ".nii.gz")
        assert len(results) == 1

    def test_recursive(self, tmp_path: Path):
        dir = tmp_path / "sub-01" / "func"
        dir.mkdir(parents=True)
        (dir / "sub-01_bold.nii.gz").touch()
        assert find_bids_files(tmp_path, ".nii.gz") == []
        results = find_bids_files(tmp_path, ".nii.gz", recursive=True)
        assert len(results) == 1

    def test_invalid_extension_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="Invalid extension"):
            find_bids_files(tmp_path, "nii.gz")

    def test_returns_sorted(self, tmp_path: Path):
        (tmp_path / "sub-02_bold.nii.gz").touch()
        (tmp_path / "sub-01_bold.nii.gz").touch()
        results = find_bids_files(tmp_path, ".nii.gz")
        assert len(results) == 2
        assert results == sorted(results)


class TestValidateBidsEntities:
    def test_valid(self):
        validate_bids_entities("sub-01", "ses-1", "task-rest")

    def test_invalid(self):
        with pytest.raises(ValueError, match="Invalid BIDS entity"):
            validate_bids_entities("sub-01", "BAD")
