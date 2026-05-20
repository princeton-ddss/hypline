import os
from pathlib import Path
from typing import Literal

from hypline.bids import (
    BOLD_IDENTITY_ENTITIES,
    BIDSPath,
    find_bids_files,
    normalize_bids_filters,
    validate_extension,
    validate_suffix,
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


def _diagnose_lookup(
    *,
    area_root: Path,
    area: _Area,
    sub: str,
    kind_dir_name: str,
    ses_values: list[str] | None,
    ext: str,
    required_entity: tuple[str, str] | None,
    suffix: str | None,
    user_filters: list[str],
) -> str:
    """Return a tier-specific message explaining why a lookup returned no files.

    Called only after a `_Find` query yielded zero results; walks the directory
    hierarchy top-down (area → subject → session → kind subdir → files →
    extension → required entity → user filters) and stops at the first failing
    tier. The user-filter tier is a fallback: it reports that filters were
    present without pinpointing which one failed.
    """
    if not area_root.exists():
        return f"BIDS area {area!r} not found at {area_root}"

    sub_dir = area_root / f"sub-{sub}"
    if not sub_dir.exists():
        present = sorted(
            p.name[4:]
            for p in area_root.iterdir()
            if p.is_dir() and p.name.startswith("sub-")
        )
        hint = f" (available: {present})" if present else ""
        return f"Subject 'sub-{sub}' not found under {area}/{hint}"

    if ses_values is not None:
        existing_ses = sorted(
            p.name[4:]
            for p in sub_dir.iterdir()
            if p.is_dir() and p.name.startswith("ses-")
        )
        missing = [s for s in ses_values if s not in existing_ses]
        if missing and existing_ses:
            return (
                f"Sessions {missing} not found for sub-{sub} under {area}/ "
                f"(available: {existing_ses})"
            )

    kind_dirs = _kind_dirs(sub_dir, kind_dir_name, ses_values)
    if not kind_dirs:
        ses_dirs = [sub_dir, *(p for p in sub_dir.glob("ses-*") if p.is_dir())]
        siblings = {c.name for d in ses_dirs for c in d.iterdir() if c.is_dir()}
        siblings.discard(kind_dir_name)
        hint = f" (found instead: {sorted(siblings)})" if siblings else ""
        return (
            f"No {kind_dir_name!r} subdirectory found under "
            f"{area}/sub-{sub}/[ses-*/]{hint}"
        )

    all_files = [f for d in kind_dirs for f in d.iterdir() if f.is_file()]
    if not all_files:
        return f"Directory {kind_dir_name!r} is empty for sub-{sub} under {area}/"

    ends_with = f"_{suffix}{ext}" if suffix is not None else ext
    ext_matches = [f for f in all_files if f.name.endswith(ends_with)]
    if not ext_matches:
        present_exts = sorted({"".join(f.suffixes) for f in all_files if f.suffixes})
        return (
            f"No files matching suffix={suffix!r}, ext={ext!r} found under "
            f"{area}/sub-{sub}/[ses-*/]{kind_dir_name}/ "
            f"(present extensions: {present_exts})"
        )

    parsed: list[BIDSPath] = []
    for f in ext_matches:
        try:
            parsed.append(BIDSPath(f))
        except ValueError:
            continue
    if not parsed:
        return (
            f"Files found under {area}/sub-{sub}/[ses-*/]{kind_dir_name}/ "
            f"but none parsed as valid BIDS (example: {ext_matches[0].name!r})"
        )

    if required_entity is not None:
        key, expected = required_entity
        if not any(bp.entities.get(key) == expected for bp in parsed):
            return (
                f"Files found but none carry the required entity "
                f"{key}-{expected!r} (example: {parsed[0].path.name!r})"
            )

    if user_filters:
        return (
            f"Files found but none matched user filters {user_filters} "
            f"under {area}/sub-{sub}/[ses-*/]{kind_dir_name}/ "
            f"(check for typos in filter keys/values)"
        )

    return (
        f"No files found under {area}/sub-{sub}/[ses-*/]{kind_dir_name}/ "
        f"(unknown cause)"
    )


def _require_task(results: list[BIDSPath]) -> None:
    """Raise if any file in `results` lacks a `task` entity.

    Enforced at discovery, not in `BIDSPath`, since non-func paths (anat, xfm)
    legitimately omit `task`.
    """
    missing = [bp for bp in results if "task" not in bp.entities]
    if missing:
        raise ValueError(
            f"Files in this area must carry a 'task' entity; "
            f"{len(missing)} file(s) lack it "
            f"(example: {missing[0].path.name!r})"
        )


class _Find:
    def __init__(self, root: Path):
        self._root = root

    def _find(
        self,
        *,
        area: _Area,
        sub: str,
        kind_dir_name: str,
        required_entity: tuple[str, str] | None,
        ext: str,
        suffix: str | None,
        ses_values: list[str] | None,
        match_filters: list[str],
        user_filters: list[str],
    ) -> list[BIDSPath]:
        """Core file lookup with tiered diagnostics on empty results.

        `match_filters` is the full set passed to `find_bids_files`, including
        structural entities (`sub-*`, `stim-*`, etc.) appended by the caller.
        `user_filters` is the caller-supplied subset, used only for error attribution.
        """
        area_root = _area_root(self._root, area)
        sub_dir = area_root / f"sub-{sub}"

        results: list[BIDSPath] = []
        for d in _kind_dirs(sub_dir, kind_dir_name, ses_values):
            results.extend(
                find_bids_files(d, ext, suffix=suffix, bids_filters=match_filters)
            )

        if not results:
            raise FileNotFoundError(
                _diagnose_lookup(
                    area_root=area_root,
                    area=area,
                    sub=sub,
                    kind_dir_name=kind_dir_name,
                    ses_values=ses_values,
                    ext=ext,
                    required_entity=required_entity,
                    suffix=suffix,
                    user_filters=user_filters,
                )
            )

        return sorted(results)

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
        user_filters = [f for f in filters if not f.startswith("ses-")]
        match_filters = user_filters + [f"sub-{sub}", f"stim-{kind}"]
        results = self._find(
            area="stimuli",
            sub=sub,
            kind_dir_name=kind,
            required_entity=("stim", kind),
            ext=ext,
            suffix=None,
            ses_values=ses_values,
            match_filters=match_filters,
            user_filters=user_filters,
        )
        _require_task(results)
        return results

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
        user_filters = [f for f in filters if not f.startswith("ses-")]
        match_filters = user_filters + [f"sub-{sub}", f"feat-{kind}"]
        results = self._find(
            area="features",
            sub=sub,
            kind_dir_name=kind,
            required_entity=("feat", kind),
            ext=".parquet",
            suffix=None,
            ses_values=ses_values,
            match_filters=match_filters,
            user_filters=user_filters,
        )
        _require_task(results)
        return results

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
        user_filters = [f for f in filters if not f.startswith("ses-")]
        match_filters = user_filters + [f"sub-{sub}"]
        return self._find(
            area="fmriprep",
            sub=sub,
            kind_dir_name="func",
            required_entity=None,
            ext=ext,
            suffix=suffix,
            ses_values=ses_values,
            match_filters=match_filters,
            user_filters=user_filters,
        )


class _Path:
    def __init__(self, root: Path):
        self._root = root

    def raw(
        self,
        *,
        source: BIDSPath,
        suffix: str,
        ext: str,
    ) -> BIDSPath:
        """Derive a raw-BIDS run path from `source`.

        Filters `source` to run identity entities only, then places the result
        under `bids_root/sub-XX/[ses-YY/]func/` with the given suffix and ext.
        Use for raw func-datatype run artifacts (events.tsv, events.json,
        `*_bold.json`, physio.tsv.gz, etc.) whose canonical names carry
        identity entities only. `source` may be a fmriprep-derivatives path
        or any path carrying the run's identity entities — resolution is
        independent of where it lives. Output entities preserve `source`'s
        order, filtered to identity entities only. May not exist on disk;
        callers check.
        """
        validate_suffix(suffix)
        validate_extension(ext)
        entities = {
            k: v for k, v in source.entities.items() if k in BOLD_IDENTITY_ENTITIES
        }

        sub = entities.get("sub")
        ses = entities.get("ses")
        if sub is None:
            raise ValueError(f"source has no 'sub' entity: {source!r}")

        sub_dir = self._root / f"sub-{sub}"
        run_dir = (
            sub_dir / f"ses-{ses}" / "func" if ses is not None else sub_dir / "func"
        )

        stem = "_".join(f"{k}-{v}" for k, v in entities.items())

        return BIDSPath(run_dir / f"{stem}_{suffix}{ext}")

    def _derive_path(
        self,
        *,
        area: _Area,
        entity_key: str,
        source: BIDSPath,
        kind: str,
        ext: str,
        entity_overrides: dict[str, str],
    ) -> BIDSPath:
        validate_extension(ext)
        bp = source.with_entity(entity_key, kind)
        for key, value in entity_overrides.items():
            bp = bp.with_entity(key, value)

        entities = bp.entities
        sub = entities.get("sub")
        ses = entities.get("ses")
        if sub is None:
            raise ValueError(f"source has no 'sub' entity: {source!r}")

        sub_dir = _area_root(self._root, area) / f"sub-{sub}"
        out_dir = sub_dir / f"ses-{ses}" / kind if ses is not None else sub_dir / kind

        stem = "_".join(f"{k}-{v}" for k, v in entities.items())

        return BIDSPath(out_dir / f"{stem}{ext}")

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
        return self._derive_path(
            area="stimuli",
            entity_key="stim",
            source=source,
            kind=kind,
            ext=ext,
            entity_overrides=entity_overrides,
        )

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

        Pass `desc=<label>` as a variant tag to distinguish features generated
        from the same source under different settings (e.g. model version).
        """
        return self._derive_path(
            area="features",
            entity_key="feat",
            source=source,
            kind=kind,
            ext=".parquet",
            entity_overrides=entity_overrides,
        )


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

    Validates bids_root exists on construction; does not validate per-area subdirs
    (features/ may be absent before featuregen runs).
    """

    def __init__(self, bids_root: str | os.PathLike[str]):
        self._root = Path(bids_root)
        if not self._root.exists():
            raise FileNotFoundError(f"bids_root does not exist: {self._root}")
        self.find = _Find(self._root)
        self.path = _Path(self._root)
        self.list = _List(self._root)
