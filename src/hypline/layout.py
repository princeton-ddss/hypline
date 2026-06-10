import json
import os
from pathlib import Path
from typing import Any, Literal

import polars as pl

from hypline._version import __version__
from hypline.bids import (
    BOLD_IDENTITY_ENTITIES,
    CATEGORY_ENTITIES,
    STRUCTURAL_ENTITIES,
    BIDSPath,
    Identity,
    find_bids_files,
    normalize_bids_filters,
    parse_filter_groups,
    validate_extension,
    validate_suffix,
)
from hypline.events import resolve_entities

Area = Literal[
    "stimuli",
    "features",
    "confounds",
    "nuisance",
    "fmriprep",
    "hypline",
    "results",
]

DERIVATIVE_BIDS_VERSION = "1.9.0"


PARTICIPANTS_TSV = "participants.tsv"


def area_root(root: Path, area: Area) -> Path:
    if area in ("fmriprep", "hypline"):
        return root / "derivatives" / area
    return root / area


def read_participants(root: Path) -> dict[str, str]:
    """Read `participants.tsv` into a bare `sub -> dyad` mapping.

    `participant_id` carries the BIDS-required `sub-` prefix; the custom
    `dyad_id` column carries a `dyad-` prefix. Both are stripped to bare IDs so
    the mapping composes with `BIDSPath.with_identity` and matches `list.*`.

    Raises FileNotFoundError if the file is absent (the mapping is required infra
    for dyad-keyed areas), and ValueError on a missing column or a duplicate
    `participant_id` (a subject paired into two dyads is a mis-keyed mapping, not
    a last-write-wins).
    """
    path = root / PARTICIPANTS_TSV
    if not path.exists():
        raise FileNotFoundError(f"{PARTICIPANTS_TSV} not found at bids_root: {path}")

    df = pl.read_csv(path, separator="\t")
    for column in ("participant_id", "dyad_id"):
        if column not in df.columns:
            raise ValueError(f"{PARTICIPANTS_TSV} missing {column!r} column: {path}")

    subs = [s.removeprefix("sub-") for s in df["participant_id"]]
    if len(set(subs)) != len(subs):
        raise ValueError(f"Duplicate participant_id in {PARTICIPANTS_TSV}: {path}")
    dyads = [d.removeprefix("dyad-") for d in df["dyad_id"]]

    return dict(zip(subs, dyads))


def kind_subdir(
    root: Path,
    area: Area,
    *,
    identity: tuple[Identity, str],
    ses: str | None,
    kind: str,
    desc: str | None,
) -> Path:
    """Resolve `<root>/<area>/<key>-XX/[ses-YY/]<kind>[-<desc>]/`.

    `identity` is the leading `(key, value)` pair — `("dyad", …)` for the
    shared-conversation areas (stimuli/features/confounds), `("sub", …)` for
    `results`.
    """
    key, value = identity
    id_dir = area_root(root, area) / f"{key}-{value}"
    subdir = f"{kind}-{desc}" if desc else kind
    return id_dir / f"ses-{ses}" / subdir if ses is not None else id_dir / subdir


def _split_filters_by_structurality(
    user_filters: list[str],
) -> tuple[list[str], list[str]]:
    """Partition `entity-value` filters into (structural, descriptive).

    Structural filters address entities that live on filenames by hypline
    convention (BIDS identity, category, image-variant descriptors); these
    can be pre-filtered on disk via `find_bids_files`. Descriptive filters
    address `events.json` `Levels.metadata` entries and must be matched
    post-resolve.
    """
    structural: list[str] = []
    descriptive: list[str] = []
    for f in user_filters:
        key = f.partition("-")[0]
        (structural if key in STRUCTURAL_ENTITIES else descriptive).append(f)
    return structural, descriptive


def _kind_parents(sub_dir: Path, ses_values: list[str] | None) -> list[Path]:
    """Return candidate parent dirs for kind subdirectories under sub_dir.

    With `ses_values` None, includes both `ses-*/` and the sub-level (BIDS
    allows session to be omitted). Returned paths are not existence-checked;
    callers that iterate a parent guard with `.is_dir()`.
    """
    if ses_values is None:
        parents = sorted(p for p in sub_dir.glob("ses-*") if p.is_dir())
        parents.append(sub_dir)
        return parents
    return [sub_dir / f"ses-{ses}" for ses in ses_values]


def _list_variant_subdirs(
    sub_dir: Path, kind: str, ses_values: list[str] | None
) -> list[str]:
    """Return sorted-unique names of `<kind>/` and `<kind>-*/` dirs under sub_dir.

    Sole globber of the desc-variant convention (the `desc="*"` path). Returns
    names, not paths, so a variant spanning several sessions collapses to one.
    """
    prefix = f"{kind}-"
    names = {
        d.name
        for parent in _kind_parents(sub_dir, ses_values)
        if parent.is_dir()
        for d in parent.iterdir()
        if d.is_dir() and (d.name == kind or d.name.startswith(prefix))
    }
    return sorted(names)


def _diagnose_lookup(
    *,
    area_root: Path,
    area: Area,
    identity: tuple[Identity, str],
    kind: str,
    ses_values: list[str] | None,
    ext: str,
    required_entity: tuple[str, str] | None,
    suffix: str | None,
    user_filters: list[str],
) -> str:
    """Return a tier-specific message explaining why a lookup returned no files.

    Called only after a `_Find` query yielded zero results; walks the directory
    hierarchy top-down (area → identity → session → kind subdir → files →
    extension → required entity → user filters) and stops at the first failing
    tier. The user-filter tier is a fallback: it reports that filters were
    present without pinpointing which one failed.

    `identity` is the leading `(key, value)` pair the area is keyed by (`sub`
    or `dyad`); it shapes both the directory walk and the message text.
    """
    id_key, id_value = identity
    id_prefix = f"{id_key}-"
    id_label = f"{id_key}-{id_value}"
    if not area_root.exists():
        return f"BIDS area {area!r} not found at {area_root}"

    id_dir = area_root / id_label
    if not id_dir.exists():
        present = sorted(
            p.name[len(id_prefix) :]
            for p in area_root.iterdir()
            if p.is_dir() and p.name.startswith(id_prefix)
        )
        hint = f" (available: {present})" if present else ""
        return f"{id_key} {id_label!r} not found under {area}/{hint}"

    if ses_values is not None:
        existing_ses = sorted(
            p.name[4:]
            for p in id_dir.iterdir()
            if p.is_dir() and p.name.startswith("ses-")
        )
        missing = [s for s in ses_values if s not in existing_ses]
        if missing and existing_ses:
            return (
                f"Sessions {missing} not found for {id_label} under {area}/ "
                f"(available: {existing_ses})"
            )

    prefix = f"{kind}-"
    kind_dirs = [
        d
        for parent in _kind_parents(id_dir, ses_values)
        if parent.is_dir()
        for d in sorted(parent.iterdir())
        if d.is_dir() and (d.name == kind or d.name.startswith(prefix))
    ]
    if not kind_dirs:
        ses_dirs = [id_dir, *(p for p in id_dir.glob("ses-*") if p.is_dir())]
        siblings = {c.name for d in ses_dirs for c in d.iterdir() if c.is_dir()}
        variant_prefix = f"{kind}-"
        siblings = {
            s for s in siblings if s != kind and not s.startswith(variant_prefix)
        }
        hint = f" (found instead: {sorted(siblings)})" if siblings else ""
        return f"No {kind!r} subdirectory found under {area}/{id_label}/[ses-*/]{hint}"

    all_files = [f for d in kind_dirs for f in d.iterdir() if f.is_file()]
    if not all_files:
        return f"Directory {kind!r} is empty for {id_label} under {area}/"

    ends_with = f"_{suffix}{ext}" if suffix is not None else ext
    ext_matches = [f for f in all_files if f.name.endswith(ends_with)]
    if not ext_matches:
        present_exts = sorted({"".join(f.suffixes) for f in all_files if f.suffixes})
        return (
            f"No files matching suffix={suffix!r}, ext={ext!r} found under "
            f"{area}/{id_label}/[ses-*/]{kind}[-*]/ "
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
            f"Files found under {area}/{id_label}/[ses-*/]{kind}[-*]/ "
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
            f"under {area}/{id_label}/[ses-*/]{kind}[-*]/ "
            f"(check for typos in filter keys/values)"
        )

    return f"No files found under {area}/{id_label}/[ses-*/]{kind}[-*]/ (unknown cause)"


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
    """File discovery within a BIDS area.

    Each finder accepts `bids_filters` split into two tiers: structural filters
    (BIDS identity, category, image-variant descriptors) match against filenames
    on disk; descriptive filters (e.g. `cond-R`) match against each file's
    resolved entities — filename merged with `events.json` `Levels.metadata` for
    the matching segment. See `events.resolve_entities` for the merge contract.
    """

    def __init__(self, layout: "BIDSLayout"):
        self._layout = layout

    def _find(
        self,
        *,
        area: Area,
        identity: tuple[Identity, str],
        kind: str,
        desc: str | None,
        required_entity: tuple[str, str] | None,
        ext: str,
        suffix: str | None,
        ses_values: list[str] | None,
        match_filters: list[str],
        user_filters: list[str],
    ) -> list[BIDSPath]:
        """Core file lookup with tiered diagnostics on empty results.

        `identity` is the leading `(key, value)` pair the area is keyed by —
        `("dyad", …)` for stimuli/features/confounds, `("sub", …)` for the
        per-brain areas — and selects the `<key>-<value>/` directory to scan.

        `desc` selects which subdirectory variant(s) to read: `None` -> bare
        `<kind>/`; `"<label>"` -> `<kind>-<label>/`; `"*"` -> all variant folders
        gathered together under one aggregate empty-check. `kind` also feeds the
        family-aware diagnostic message.

        `match_filters` is the full set passed to `find_bids_files`, including
        structural entities (`sub-*`/`dyad-*`, `stim-*`, etc.) appended by the
        caller. `user_filters` is the caller-supplied subset, used only for
        error attribution.
        """
        root = area_root(self._layout.root, area)
        id_key, id_value = identity
        id_dir = root / f"{id_key}-{id_value}"

        if desc is None:
            kind_dir_names = [kind]
        elif desc == "*":
            kind_dir_names = _list_variant_subdirs(id_dir, kind, ses_values)
        else:
            kind_dir_names = [f"{kind}-{desc}"]

        results: list[BIDSPath] = []
        for parent in _kind_parents(id_dir, ses_values):
            for name in kind_dir_names:
                d = parent / name
                if not d.is_dir():
                    continue
                results.extend(
                    find_bids_files(d, ext, suffix=suffix, bids_filters=match_filters)
                )

        if not results:
            raise FileNotFoundError(
                _diagnose_lookup(
                    area_root=root,
                    area=area,
                    identity=identity,
                    kind=kind,
                    ses_values=ses_values,
                    ext=ext,
                    required_entity=required_entity,
                    suffix=suffix,
                    user_filters=user_filters,
                )
            )

        return sorted(results)

    def _apply_metadata_filters(
        self,
        candidates: list[BIDSPath],
        *,
        filters: list[str],
        where: str,
    ) -> list[BIDSPath]:
        """Narrow candidates by descriptive filters resolved against events.json.

        Same-key filter values OR-match; different keys AND-match. Raises
        `FileNotFoundError` if every candidate is filtered out; `where` is the
        human-readable path fragment for the error message. Resolve failures
        are re-raised as `ValueError` with filename context.
        """
        if not filters:
            return candidates

        groups = parse_filter_groups(filters)
        results: list[BIDSPath] = []
        for bids in candidates:
            try:
                entities = resolve_entities(self._layout, bids)
            except ValueError as err:
                raise ValueError(f"{err} (at {bids.path.name})") from None
            if all(entities.get(k) in vals for k, vals in groups.items()):
                results.append(bids)

        if not results:
            raise FileNotFoundError(
                f"Files found but none matched descriptive filters "
                f"{filters} under {where} "
                f"(filters apply against filename entities merged with "
                f"events.json Levels metadata; check for typos in filter keys/values)"
            )
        return results

    def stimuli(
        self,
        *,
        dyad: str,
        kind: str,
        ext: str,
        bids_filters: list[str] | None = None,
    ) -> list["BIDSPath"]:
        """Find stimulus files for a dyad.

        Stimuli describe the shared conversation, so they are dyad-keyed. `kind`
        maps to the stim-<kind> entity and the per-session subdirectory name.
        """
        filters = normalize_bids_filters(bids_filters, reserved={"dyad", "stim"})
        ses_values = [f[4:] for f in filters if f.startswith("ses-")] or None
        user_filters = [f for f in filters if not f.startswith("ses-")]
        structural, descriptive = _split_filters_by_structurality(user_filters)
        candidates = self._find(
            area="stimuli",
            identity=("dyad", dyad),
            kind=kind,
            desc=None,
            required_entity=("stim", kind),
            ext=ext,
            suffix=None,
            ses_values=ses_values,
            match_filters=structural + [f"dyad-{dyad}", f"stim-{kind}"],
            user_filters=structural,
        )
        _require_task(candidates)
        return self._apply_metadata_filters(
            candidates,
            filters=descriptive,
            where=f"stimuli/dyad-{dyad}/[ses-*/]{kind}[-*]/",
        )

    def features(
        self,
        *,
        dyad: str,
        kind: str,
        desc: str | None = None,
        bids_filters: list[str] | None = None,
    ) -> list["BIDSPath"]:
        """Find feature files for a dyad.

        Features are derived from the shared conversation, so they are
        dyad-keyed. `kind` maps to the feat-<kind> entity and the per-session
        subdirectory name. `desc` selects which variant folder(s) to read:
        `None` -> bare `<kind>/` only; `"<label>"` -> `<kind>-<label>/` only;
        `"*"` -> all variant folders gathered together. Extension is `.parquet`.
        """
        filters = normalize_bids_filters(
            bids_filters, reserved={"dyad", "feat", "desc"}
        )
        ses_values = [f[4:] for f in filters if f.startswith("ses-")] or None
        user_filters = [f for f in filters if not f.startswith("ses-")]
        structural, descriptive = _split_filters_by_structurality(user_filters)
        candidates = self._find(
            area="features",
            identity=("dyad", dyad),
            kind=kind,
            desc=desc,
            required_entity=("feat", kind),
            ext=".parquet",
            suffix=None,
            ses_values=ses_values,
            match_filters=structural + [f"dyad-{dyad}", f"feat-{kind}"],
            user_filters=structural,
        )
        _require_task(candidates)
        return self._apply_metadata_filters(
            candidates,
            filters=descriptive,
            where=f"features/dyad-{dyad}/[ses-*/]{kind}[-*]/",
        )

    def confounds(
        self,
        *,
        dyad: str,
        kind: str,
        desc: str | None = None,
        bids_filters: list[str] | None = None,
    ) -> list["BIDSPath"]:
        """Find confound files for a dyad.

        Confounds are derived from the shared conversation, so they are
        dyad-keyed. `kind` maps to the conf-<kind> entity and the per-session
        subdirectory name. `desc` selects which variant folder(s) to read:
        `None` -> bare `<kind>/` only; `"<label>"` -> `<kind>-<label>/` only;
        `"*"` -> all variant folders gathered together. Extension is `.parquet`.
        """
        filters = normalize_bids_filters(
            bids_filters, reserved={"dyad", "conf", "desc"}
        )
        ses_values = [f[4:] for f in filters if f.startswith("ses-")] or None
        user_filters = [f for f in filters if not f.startswith("ses-")]
        structural, descriptive = _split_filters_by_structurality(user_filters)
        candidates = self._find(
            area="confounds",
            identity=("dyad", dyad),
            kind=kind,
            desc=desc,
            required_entity=("conf", kind),
            ext=".parquet",
            suffix=None,
            ses_values=ses_values,
            match_filters=structural + [f"dyad-{dyad}", f"conf-{kind}"],
            user_filters=structural,
        )
        _require_task(candidates)
        return self._apply_metadata_filters(
            candidates,
            filters=descriptive,
            where=f"confounds/dyad-{dyad}/[ses-*/]{kind}[-*]/",
        )

    def nuisance(
        self,
        *,
        sub: str,
        kind: str,
        desc: str | None = None,
        bids_filters: list[str] | None = None,
    ) -> list["BIDSPath"]:
        """Find nuisance files.

        `kind` maps to the nuis-<kind> entity and the per-session subdirectory
        name. `desc` selects which variant folder(s) to read: `None` -> bare
        `<kind>/` only; `"<label>"` -> `<kind>-<label>/` only; `"*"` -> all
        variant folders gathered together. Files are wide TSVs carrying the
        `_timeseries` suffix and `.tsv` extension (one scalar column per
        run-level regressor), unlike the parquet feature/confound tiers.
        """
        filters = normalize_bids_filters(bids_filters, reserved={"sub", "nuis", "desc"})
        ses_values = [f[4:] for f in filters if f.startswith("ses-")] or None
        user_filters = [f for f in filters if not f.startswith("ses-")]
        structural, descriptive = _split_filters_by_structurality(user_filters)
        candidates = self._find(
            area="nuisance",
            identity=("sub", sub),
            kind=kind,
            desc=desc,
            required_entity=("nuis", kind),
            ext=".tsv",
            suffix="timeseries",
            ses_values=ses_values,
            match_filters=structural + [f"sub-{sub}", f"nuis-{kind}"],
            user_filters=structural,
        )
        _require_task(candidates)
        return self._apply_metadata_filters(
            candidates,
            filters=descriptive,
            where=f"nuisance/sub-{sub}/[ses-*/]{kind}[-*]/",
        )

    def fmriprep(
        self,
        *,
        sub: str,
        suffix: str,
        ext: str,
        bids_filters: list[str] | None = None,
    ) -> list["BIDSPath"]:
        return self._find_func(
            area="fmriprep",
            sub=sub,
            suffix=suffix,
            ext=ext,
            bids_filters=bids_filters,
        )

    def hypline(
        self,
        *,
        sub: str,
        suffix: str,
        ext: str,
        bids_filters: list[str] | None = None,
    ) -> list["BIDSPath"]:
        return self._find_func(
            area="hypline",
            sub=sub,
            suffix=suffix,
            ext=ext,
            bids_filters=bids_filters,
        )

    def _find_func(
        self,
        *,
        area: Area,
        sub: str,
        suffix: str,
        ext: str,
        bids_filters: list[str] | None,
    ) -> list["BIDSPath"]:
        filters = normalize_bids_filters(bids_filters, reserved={"sub"})
        ses_values = [f[4:] for f in filters if f.startswith("ses-")] or None
        user_filters = [f for f in filters if not f.startswith("ses-")]
        structural, descriptive = _split_filters_by_structurality(user_filters)
        candidates = self._find(
            area=area,
            identity=("sub", sub),
            kind="func",
            desc=None,
            required_entity=None,
            ext=ext,
            suffix=suffix,
            ses_values=ses_values,
            match_filters=structural + [f"sub-{sub}"],
            user_filters=structural,
        )
        return self._apply_metadata_filters(
            candidates,
            filters=descriptive,
            where=f"derivatives/{area}/sub-{sub}/[ses-*/]func/",
        )


class _Path:
    def __init__(self, layout: "BIDSLayout"):
        self._layout = layout

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

        sub_dir = self._layout.root / f"sub-{sub}"
        run_dir = (
            sub_dir / f"ses-{ses}" / "func" if ses is not None else sub_dir / "func"
        )

        stem = "_".join(f"{k}-{v}" for k, v in entities.items())

        return BIDSPath(run_dir / f"{stem}_{suffix}{ext}")

    def _derive_path(
        self,
        *,
        area: Area,
        entity_key: str,
        source: BIDSPath,
        kind: str,
        ext: str,
        desc: str | None,
    ) -> BIDSPath:
        validate_extension(ext)
        bp = source.with_entity(entity_key, kind)
        for key in CATEGORY_ENTITIES - {entity_key}:
            bp = bp.without_entity(key)
        if desc is not None:
            bp = bp.with_entity("desc", desc)

        entities = bp.entities
        dyad = entities.get("dyad")
        ses = entities.get("ses")
        if dyad is None:
            raise ValueError(f"source has no 'dyad' entity: {source!r}")

        desc = entities.get("desc")
        out_dir = kind_subdir(
            self._layout.root,
            area,
            identity=("dyad", dyad),
            ses=ses,
            kind=kind,
            desc=desc,
        )

        stem = "_".join(f"{k}-{v}" for k, v in entities.items())

        return BIDSPath(out_dir / f"{stem}{ext}")

    def _derive_derivative_path(
        self,
        *,
        area: Area,
        source: BIDSPath,
        desc: str,
    ) -> BIDSPath:
        """Derive a func-datatype BIDS-derivative path from `source`.

        Mirrors fmriprep's tree shape: `derivatives/<area>/sub-XX/[ses-YY/]func/`,
        preserving the source's full BOLD identity (every entity, including
        non-identity ones like `space`/`hemi`) and only swapping `desc`. Suffix
        and extension carry over from `source`, so one helper serves both volume
        (`.nii.gz`) and per-hemisphere surface (`.func.gii`) runs.

        Unlike `_derive_path` (kind-foldered root areas), there is no category
        entity and no `<kind>[-<desc>]/` subfolder; preserving all entities is
        what keeps L/R surface runs on distinct paths. May not exist on disk;
        callers check.
        """
        bp = source.with_entity("desc", desc)

        entities = bp.entities
        sub = entities.get("sub")
        ses = entities.get("ses")
        if sub is None:
            raise ValueError(f"source has no 'sub' entity: {source!r}")

        sub_dir = area_root(self._layout.root, area) / f"sub-{sub}"
        out_dir = (
            sub_dir / f"ses-{ses}" / "func" if ses is not None else sub_dir / "func"
        )

        stem = "_".join(f"{k}-{v}" for k, v in entities.items())
        # TODO: drop guard once BIDSPath requires suffix
        suffix = f"_{source.suffix}" if source.suffix else ""

        return BIDSPath(out_dir / f"{stem}{suffix}{source.ext}")

    def denoised(self, *, source: BIDSPath) -> BIDSPath:
        """Derive a denoised-BOLD output path from `source`.

        Sets `desc-denoised`. `source` is the fmriprep `desc-preproc` BOLD;
        relocating to the hypline derivatives tree gives the denoised output
        honest provenance instead of inheriting fmriprep's `GeneratedBy`.
        """
        return self._derive_derivative_path(
            area="hypline", source=source, desc="denoised"
        )

    def stimulus(
        self,
        *,
        source: BIDSPath,
        kind: str,
        ext: str,
    ) -> BIDSPath:
        """Derive a stimulus output path from `source`.

        Sets stim-<kind> and places the result under
        stimuli/dyad-XX/[ses-YY/]<kind>/. Stimuli have no `desc` variants: a
        stimulus is the experimental record (the audio, transcript, movie),
        with one ground truth. An artifact that needs variants is a feature,
        not a stimulus.

        Raises ValueError if `source` carries a `desc` entity.
        """
        if "desc" in source.entities:
            raise ValueError(
                f"source carries 'desc' but stimuli have no variants: {source!r}"
            )
        return self._derive_path(
            area="stimuli",
            entity_key="stim",
            source=source,
            kind=kind,
            ext=ext,
            desc=None,
        )

    def feature(
        self,
        *,
        source: BIDSPath,
        kind: str,
        desc: str | None = None,
    ) -> BIDSPath:
        """Derive a feature output path from `source`.

        Sets feat-<kind> and places the result under
        features/dyad-XX/[ses-YY/]<kind>[-<desc>]/ with `.parquet` extension.

        Pass `desc=<label>` as a variant tag to distinguish features generated
        from the same source under different settings (e.g. model version);
        the subdirectory becomes `<kind>-<desc>` so variants live in separate
        folders.
        """
        return self._derive_path(
            area="features",
            entity_key="feat",
            source=source,
            kind=kind,
            ext=".parquet",
            desc=desc,
        )

    def confound(
        self,
        *,
        source: BIDSPath,
        kind: str,
        desc: str | None = None,
    ) -> BIDSPath:
        """Derive a confound output path from `source`.

        Sets conf-<kind> and places the result under
        confounds/dyad-XX/[ses-YY/]<kind>[-<desc>]/ with `.parquet` extension.

        Pass `desc=<label>` to name which derivation of the kind's source this
        is (e.g. `onset` vs `rate` for phonemic); the subdirectory becomes
        `<kind>-<desc>` so derivations live in separate folders.
        """
        return self._derive_path(
            area="confounds",
            entity_key="conf",
            source=source,
            kind=kind,
            ext=".parquet",
            desc=desc,
        )

    def result(
        self,
        *,
        sub: str,
        kind: str,
        desc: str,
        ext: str = ".joblib",
    ) -> BIDSPath:
        """Build a results output path, anchored to `sub` only.

        Sets result-<kind> and places the file under
        results/sub-XX/<kind>-<desc>/ — no source run and no `ses` segment,
        since one analysis output (an encoding model) consumes many runs across
        sessions. `(sub, kind, desc)` fully determines the path, so the loader
        needs only those three. `desc` is required: an untagged result is hard
        to disambiguate (diverges from features/confounds, where it is optional).
        """
        bp = BIDSPath.from_entities(
            ext=ext,
            sub=sub,
            result=kind,
            desc=desc,
        )
        out_dir = kind_subdir(
            self._layout.root,
            "results",
            identity=("sub", sub),
            ses=None,
            kind=kind,
            desc=desc,
        )
        return BIDSPath(out_dir / bp.path.name)


class _List:
    def __init__(self, layout: "BIDSLayout"):
        self._layout = layout

    def subjects(self, *, area: Area) -> list[str]:
        """Return sorted unique subject IDs present in the given area."""
        return self._identity_dirs(area, "sub-")

    def dyads(self, *, area: Area) -> list[str]:
        """Return sorted unique dyad IDs present in the given area.

        Mirrors `subjects` for the dyad-keyed areas (stimuli/features/confounds).
        Scans on-disk subdirs, so it answers "which dyads have artifacts here" —
        distinct from the `participants.tsv` dyad<->sub mapping.
        """
        return self._identity_dirs(area, "dyad-")

    def _identity_dirs(self, area: Area, prefix: str) -> list[str]:
        area_dir = area_root(self._layout.root, area)
        if not area_dir.exists():
            return []

        ids: set[str] = set()
        for p in area_dir.iterdir():
            if p.is_dir() and p.name.startswith(prefix):
                ids.add(p.name[len(prefix) :])

        return sorted(ids)


class BIDSLayout:
    """Single authority on path discovery and derivation for a hypline BIDS tree.

    Validates bids_root exists on construction; does not validate per-area subdirs
    (features/ may be absent before featuregen runs).
    """

    def __init__(self, bids_root: str | Path):
        self._root = Path(bids_root)
        if not self._root.exists():
            raise FileNotFoundError(f"bids_root does not exist: {self._root}")
        self.find = _Find(self)
        self.path = _Path(self)
        self.list = _List(self)
        self._sub_to_dyad: dict[str, str] | None = None

    @property
    def root(self) -> Path:
        return self._root

    def _participants(self) -> dict[str, str]:
        """Read-once bare `sub -> dyad` mapping from `participants.tsv`."""
        if self._sub_to_dyad is None:
            self._sub_to_dyad = read_participants(self._root)
        return self._sub_to_dyad

    def dyad_of(self, sub: str) -> str:
        """Return the dyad a subject belongs to, via `participants.tsv`.

        Raises KeyError if the subject has no row (a missing entry is a
        mis-keyed mapping).
        """
        mapping = self._participants()
        if sub not in mapping:
            raise KeyError(f"sub-{sub} not in {PARTICIPANTS_TSV}")
        return mapping[sub]

    def subjects_of(self, dyad: str) -> list[str]:
        """Return the sorted subjects making up a dyad, via `participants.tsv`.

        Raises KeyError if no subject maps to the dyad.
        """
        subs = sorted(s for s, d in self._participants().items() if d == dyad)
        if not subs:
            raise KeyError(f"dyad-{dyad} not in {PARTICIPANTS_TSV}")
        return subs

    def bids_uri(self, bold: BIDSPath, *, area: Area) -> str:
        """Render `bold` (living under `area`) as a `bids:<area>:<rel path>` URI.

        The cross-dataset BIDS URI a derivative's `Sources` uses to reference an
        input in a sibling dataset; resolved by the consumer via the matching
        `DatasetLinks.<area>` entry. Path is relative to the area root.
        """
        rel = bold.path.relative_to(area_root(self._root, area))
        return f"bids:{area}:{rel.as_posix()}"

    def stamp_dataset_description(self, *, area: Area, sources: list[Area]) -> None:
        """Write `<area>/dataset_description.json` if absent; leave it if present.

        Stamps the minimal compliant derivative header. `sources` are sibling
        areas this derivative draws from; each lands in `DatasetLinks` as a path
        relative to `area`'s root, resolving the `bids:<source>:` URIs that
        per-file `Sources` use (see `bids_uri`).

        Write-if-absent, not re-stamp: the file describes the pipeline, not a run,
        and re-stamping races under parallel `--sub-ids`. Per-run version truth
        lives in each sidecar.
        """
        area_dir = area_root(self._root, area)
        path = area_dir / "dataset_description.json"
        if path.exists():
            return
        description: dict[str, Any] = {
            "Name": area,
            "BIDSVersion": DERIVATIVE_BIDS_VERSION,
            "DatasetType": "derivative",
            "GeneratedBy": [{"Name": "hypline", "Version": __version__}],
        }
        if sources:
            description["DatasetLinks"] = {
                src: os.path.relpath(area_root(self._root, src), area_dir)
                for src in sources
            }
        area_dir.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(description, f, indent=2)
