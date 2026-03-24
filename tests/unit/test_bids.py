from pathlib import Path

import pytest

from hypline.bids import BIDSPath, validate_bids_entities


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


class TestValidateBidsEntities:
    def test_valid(self):
        validate_bids_entities("sub-01", "ses-1", "task-rest")

    def test_invalid(self):
        with pytest.raises(ValueError, match="Invalid BIDS entity"):
            validate_bids_entities("sub-01", "BAD")
