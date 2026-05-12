import os
from pathlib import Path
from typing import Literal

from hypline.bids import (
    BIDSPath,
    find_bids_files,
    normalize_bids_filters,
    validate_extension,
)

_Area = Literal["stimuli", "features", "fmriprep"]


def _area_root(root: Path, area: _Area) -> Path:
    if area == "fmriprep":
        return root / "derivatives" / "fmriprep"
    return root / area


def _kind_dirs(sub_dir: Path, kind: str, ses_values: list[str] | None) -> list[Path]:
    """Return existing <kind>/ dirs under sub_dir for the given sessions.

    When `ses_values` is None, returns both `ses-*/<kind>/` dirs and the
    sub-level `<kind>/` dir (BIDS allows session to be omitted).
    """
    if ses_values is None:
        ses_dirs = sorted(p for p in sub_dir.glob("ses-*") if p.is_dir())
        candidates = [d / kind for d in ses_dirs] + [sub_dir / kind]
    else:
        candidates = [sub_dir / f"ses-{ses}" / kind for ses in ses_values]
    return [d for d in candidates if d.is_dir()]


class _Find:
    def __init__(self, root: Path):
        self._root = root

    def stimuli(
        self,
        *,
        sub: str,
        kind: str,
        ext: str,
        bids_filters: list[str] | None = None,
    ) -> list["BIDSPath"]:
        """Find stimulus files.

        `kind` maps to the stim-<kind> entity and the per-session subdirectory name.
        """
        filters = normalize_bids_filters(bids_filters, reserved={"sub", "stim"})
        ses_values = [f[4:] for f in filters if f.startswith("ses-")] or None
        filters = [f for f in filters if not f.startswith("ses-")]
        filters.extend([f"sub-{sub}", f"stim-{kind}"])

        sub_dir = _area_root(self._root, "stimuli") / f"sub-{sub}"
        if not sub_dir.exists():
            return []

        results: list[BIDSPath] = []
        for dir in _kind_dirs(sub_dir, kind, ses_values):
            results.extend(find_bids_files(dir, ext, bids_filters=filters))
        return sorted(results)

    def features(
        self,
        *,
        sub: str,
        kind: str,
        bids_filters: list[str] | None = None,
    ) -> list["BIDSPath"]:
        """Find feature files.

        `kind` maps to the feat-<kind> entity and the per-session subdirectory
        name. Extension is `.parquet`.
        """
        filters = normalize_bids_filters(bids_filters, reserved={"sub", "feat"})
        ses_values = [f[4:] for f in filters if f.startswith("ses-")] or None
        filters = [f for f in filters if not f.startswith("ses-")]
        filters.extend([f"sub-{sub}", f"feat-{kind}"])

        sub_dir = _area_root(self._root, "features") / f"sub-{sub}"
        if not sub_dir.exists():
            return []

        results: list[BIDSPath] = []
        for dir in _kind_dirs(sub_dir, kind, ses_values):
            results.extend(find_bids_files(dir, ".parquet", bids_filters=filters))
        return sorted(results)

    def fmriprep(
        self,
        *,
        sub: str,
        suffix: str,
        ext: str,
        bids_filters: list[str] | None = None,
    ) -> list["BIDSPath"]:
        filters = normalize_bids_filters(bids_filters, reserved={"sub"})
        ses_values = [f[4:] for f in filters if f.startswith("ses-")] or None
        filters = [f for f in filters if not f.startswith("ses-")]
        filters.extend([f"sub-{sub}"])

        sub_dir = _area_root(self._root, "fmriprep") / f"sub-{sub}"
        if not sub_dir.exists():
            return []

        results: list[BIDSPath] = []
        for dir in _kind_dirs(sub_dir, "func", ses_values):
            results.extend(
                find_bids_files(dir, ext, suffix=suffix, bids_filters=filters)
            )
        return sorted(results)


class _Build:
    def __init__(self, root: Path):
        self._root = root

    def stimulus(
        self,
        *,
        source: BIDSPath,
        kind: str,
        ext: str,
        **entity_overrides: str,
    ) -> BIDSPath:
        """Derive a stimulus output path from `source`.

        Sets stim-<kind>, applies `entity_overrides`, and places the result
        under stimuli/sub-XX/[ses-YY/]<kind>/.
        """
        validate_extension(ext)
        bp = source.with_entity("stim", kind)
        for key, value in entity_overrides.items():
            bp = bp.with_entity(key, value)

        entities = bp.entities
        sub = entities.get("sub")
        ses = entities.get("ses")
        if sub is None:
            raise ValueError(f"source has no 'sub' entity: {source!r}")

        sub_dir = _area_root(self._root, "stimuli") / f"sub-{sub}"
        out_dir = sub_dir / f"ses-{ses}" / kind if ses is not None else sub_dir / kind

        stem = "_".join(f"{k}-{v}" for k, v in entities.items())
        name = f"{stem}{ext}"

        return BIDSPath(out_dir / name)

    def feature(
        self,
        *,
        source: BIDSPath,
        kind: str,
        **entity_overrides: str,
    ) -> BIDSPath:
        """Derive a feature output path from `source`.

        Sets feat-<kind>, applies `entity_overrides`, and places the result
        under features/sub-XX/[ses-YY/]<kind>/ with `.parquet` extension.
        """
        bp = source.with_entity("feat", kind)
        for key, value in entity_overrides.items():
            bp = bp.with_entity(key, value)

        entities = bp.entities
        sub = entities.get("sub")
        ses = entities.get("ses")
        if sub is None:
            raise ValueError(f"source has no 'sub' entity: {source!r}")

        sub_dir = _area_root(self._root, "features") / f"sub-{sub}"
        out_dir = sub_dir / f"ses-{ses}" / kind if ses is not None else sub_dir / kind

        stem = "_".join(f"{k}-{v}" for k, v in entities.items())
        name = f"{stem}.parquet"
        return BIDSPath(out_dir / name)


class _List:
    def __init__(self, root: Path):
        self._root = root

    def subjects(self, *, area: _Area) -> list[str]:
        """Return sorted unique subject IDs present in the given area."""
        area_dir = _area_root(self._root, area)
        if not area_dir.exists():
            return []

        ids: set[str] = set()
        for p in area_dir.iterdir():
            if p.is_dir() and p.name.startswith("sub-"):
                ids.add(p.name[4:])

        return sorted(ids)

    def sessions(self, *, sub: str, area: _Area) -> list[str]:
        """Return sorted unique session IDs for a subject in the given area."""
        sub_dir = _area_root(self._root, area) / f"sub-{sub}"
        if not sub_dir.exists():
            return []

        ids: set[str] = set()
        for p in sub_dir.iterdir():
            if p.is_dir() and p.name.startswith("ses-"):
                ids.add(p.name[4:])

        return sorted(ids)


class BIDSLayout:
    """Single authority on path discovery and derivation for a hypline BIDS tree.

    Validates root_dir exists on construction; does not validate per-area subdirs
    (features/ may be absent before featuregen runs).
    """

    def __init__(self, root_dir: str | os.PathLike[str]):
        self._root = Path(root_dir)
        if not self._root.exists():
            raise FileNotFoundError(f"root_dir does not exist: {self._root}")
        self.find = _Find(self._root)
        self.build = _Build(self._root)
        self.list = _List(self._root)
