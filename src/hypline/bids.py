import re
from collections.abc import Iterable, Sequence
from pathlib import Path

BIDS_ENTITY_KEY_RE = re.compile(r"^[a-z]+$")
BIDS_ENTITY_VALUE_RE = re.compile(r"^[a-zA-Z0-9]+$")
BIDS_ENTITY_RE = re.compile(r"^[a-z]+-[a-zA-Z0-9]+$")
BIDS_SUFFIX_RE = re.compile(r"^[a-zA-Z0-9]+$")
EXTENSION_RE = re.compile(r"^\.[a-zA-Z0-9]+(\.[a-zA-Z0-9]+)*$")

# Entities identifying a single BOLD run
BOLD_IDENTITY_ENTITIES = frozenset(("sub", "ses", "task", "run"))

# BIDS entities hypline rejects at construction. Hyperscanning fixes a single
# acquisition protocol, so methodological-variation entities are disallowed.
UNSUPPORTED_ENTITIES = frozenset(("acq", "ce", "rec", "dir", "echo", "part", "chunk"))

# Identity entities plus unsupported ones. Used to bar `events.tsv` segment
# rows from colliding with BIDS-reserved entity names.
RESERVED_BIDS_ENTITIES = BOLD_IDENTITY_ENTITIES | UNSUPPORTED_ENTITIES

# Hypline derivative-category tags; a derived output carries exactly one.
# `result` tags analysis outputs (encoding models, correlations) under results/;
# unlike the others it anchors to `sub` only, not a source run.
CATEGORY_ENTITIES = frozenset({"stim", "feat", "conf", "nuis", "result"})

# Entities distinguishing processing/image variants of the same logical run
VARIANT_DESCRIPTORS = frozenset({"desc", "space", "res", "den"})

# Filename entities allowed without an events.json sidecar counterpart.
# Anything outside this set is descriptive metadata and must be declared
# in events.json `Levels`.
STRUCTURAL_ENTITIES = BOLD_IDENTITY_ENTITIES | CATEGORY_ENTITIES | VARIANT_DESCRIPTORS

# Fixed slots for `BIDSPath.from_entities`; non-fixed keys fill the middle
# alphabetically, keeping `desc` adjacent to the category it modifies
_LEADING_ENTITY_ORDER = ("sub", "ses", "task", "run")
_TRAILING_ENTITY_ORDER = (*sorted(CATEGORY_ENTITIES), "desc")
_FIXED_ENTITIES = frozenset(_LEADING_ENTITY_ORDER) | frozenset(_TRAILING_ENTITY_ORDER)


def validate_bids_entities(*entities: str) -> None:
    for entity in entities:
        if not BIDS_ENTITY_RE.match(entity):
            raise ValueError(f"Invalid BIDS entity: {entity!r}")


def validate_suffix(suffix: str) -> None:
    if not BIDS_SUFFIX_RE.match(suffix):
        raise ValueError(f"Invalid BIDS suffix: {suffix!r}")


def validate_extension(ext: str) -> None:
    if not EXTENSION_RE.match(ext):
        raise ValueError(f"Invalid extension: {ext!r}")


class BIDSPath:
    _entities: dict[str, str]
    _suffix: str | None
    _ext: str
    _path: Path

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._entities = {}
        self._suffix = None

        name = self._path.name
        head, sep, tail = name.rpartition("_")
        stem_tail, dot, ext = tail.partition(".")
        self._ext = f".{ext}" if dot else ""
        stem = f"{head}{sep}{stem_tail}"

        segments = stem.split("_")
        for i, segment in enumerate(segments):
            if "-" in segment:
                key, _, value = segment.partition("-")
                validate_bids_entities(segment)
                if key in self._entities:
                    raise ValueError(f"Duplicate BIDS entity key: {key!r}")
                if key in UNSUPPORTED_ENTITIES:
                    raise ValueError(
                        f"BIDS entity {key!r} is not supported by hypline "
                        f"(got {name!r}); unsupported: "
                        f"{sorted(UNSUPPORTED_ENTITIES)}"
                    )
                self._entities[key] = value
            elif i == len(segments) - 1:
                validate_suffix(segment)
                self._suffix = segment
            else:
                raise ValueError(
                    f"Non-entity segment {segment!r} must be the last segment (suffix)"
                )

        if not self._entities:
            raise ValueError(f"No BIDS entities found in: {name!r}")
        if next(iter(self._entities)) != "sub":
            raise ValueError(f"BIDS filename must start with 'sub-': {name!r}")
        validate_extension(self._ext)

    @classmethod
    def from_entities(
        cls,
        *,
        ext: str,
        suffix: str | None = None,
        parent: str | Path = ".",
        **entities: str,
    ) -> "BIDSPath":
        """Build a BIDSPath from entity kwargs in canonical order.

        Order: identity (`sub`/`ses`/`task`/`run`), then non-fixed keys
        alphabetically, then category (`stim`/`feat`/`conf`/`nuis`/`result`)
        and `desc`. Requires `sub`; rejects unsupported entities and
        more than one category entity.
        """
        if "sub" not in entities:
            raise ValueError("`sub` is required")

        categories = CATEGORY_ENTITIES & entities.keys()
        if len(categories) > 1:
            raise ValueError(
                f"At most one category entity allowed, got {sorted(categories)}"
            )

        for key, value in entities.items():
            validate_bids_entities(f"{key}-{value}")
            if key in UNSUPPORTED_ENTITIES:
                raise ValueError(
                    f"BIDS entity {key!r} is not supported by hypline; "
                    f"unsupported: {sorted(UNSUPPORTED_ENTITIES)}"
                )

        custom = sorted(k for k in entities if k not in _FIXED_ENTITIES)
        ordered_keys = [
            *(k for k in _LEADING_ENTITY_ORDER if k in entities),
            *custom,
            *(k for k in _TRAILING_ENTITY_ORDER if k in entities),
        ]
        parts = [f"{k}-{entities[k]}" for k in ordered_keys]
        if suffix is not None:
            validate_suffix(suffix)
            parts.append(suffix)
        validate_extension(ext)
        name = "_".join(parts) + ext

        return cls(Path(parent) / name)

    @property
    def entities(self) -> dict[str, str]:
        return dict(self._entities)

    @property
    def suffix(self) -> str | None:
        return self._suffix

    @property
    def ext(self) -> str:
        return self._ext

    @property
    def path(self) -> Path:
        return self._path

    def __getattr__(self, name: str) -> str:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._entities[name]
        except KeyError:
            raise AttributeError(
                f"No BIDS entity {name!r} in {self._path.name!r}"
            ) from None

    def with_entity(self, key: str, value: str) -> "BIDSPath":
        """Return a new BIDSPath with `key` set to `value`.

        Existing keys keep their position; new keys are appended. Order is
        preserved in the filename stem of derived paths.
        """
        validate_bids_entities(f"{key}-{value}")
        entities = dict(self._entities)
        entities[key] = value
        return self._rebuild(entities)

    def without_entity(self, key: str) -> "BIDSPath":
        """Return a new BIDSPath with `key` removed.

        No-op if `key` is absent. Refuses to remove `sub`, which is required.
        """
        if key == "sub":
            raise ValueError("Cannot remove required 'sub' entity")
        if key not in self._entities:
            return self
        entities = {k: v for k, v in self._entities.items() if k != key}
        return self._rebuild(entities)

    def with_ext(self, ext: str) -> Path:
        """Return the sibling path with the data extension swapped for `ext`.

        Strips the full (possibly multi-part, e.g. `.func.gii`) extension and
        appends `ext`. Returns a plain `Path`: the result need not be a valid
        BIDSPath (a `.json` sidecar carries no recognized suffix).
        """
        stem = self._path.name[: -len(self._ext)] if self._ext else self._path.name
        return self._path.with_name(f"{stem}{ext}")

    def _rebuild(self, entities: dict[str, str]) -> "BIDSPath":
        parts = [f"{k}-{v}" for k, v in entities.items()]
        if self._suffix:
            parts.append(self._suffix)
        name = "_".join(parts) + self._ext
        return BIDSPath(self._path.with_name(name))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BIDSPath):
            return NotImplemented
        return self._path == other._path

    def __hash__(self) -> int:
        return hash(self._path)

    def __lt__(self, other: "BIDSPath") -> bool:
        if not isinstance(other, BIDSPath):
            return NotImplemented
        return self._path < other._path

    def __repr__(self) -> str:
        return f"BIDSPath({str(self._path)!r})"


def validate_entity_invariance(
    paths: Sequence[BIDSPath], entities: Iterable[str]
) -> None:
    """Raise if any entity has more than one distinct value across paths."""
    for entity in entities:
        values = {p.entities.get(entity) for p in paths}
        if len(values) > 1:
            display = sorted(v if v is not None else "(none)" for v in values)
            raise ValueError(f"Inconsistent {entity!r} across files: found {display}.")


def normalize_bids_filters(
    filters: list[str] | None,
    *,
    reserved: Iterable[str] = (),
) -> list[str]:
    filters = list(filters or [])
    validate_bids_entities(*filters)
    reserved_set = frozenset(reserved)
    for entity in filters:
        key = entity.split("-", 1)[0]
        if key in reserved_set:
            raise ValueError(
                f"bids_filters cannot contain {key!r} "
                "— use the dedicated argument instead"
            )
    return filters


def parse_filter_groups(filters: list[str]) -> dict[str, list[str]]:
    """Group `entity-value` filter strings by key.

    Same-key values are returned together (callers OR-match within a group);
    different keys yield distinct entries (callers AND-match across groups).
    """
    groups: dict[str, list[str]] = {}
    for entity in filters:
        key, _, value = entity.partition("-")
        groups.setdefault(key, []).append(value)
    return groups


def parse_kind_desc(entry: str) -> tuple[str, str | None]:
    """Parse a `<kind>` or `<kind>-<desc>` string into `(kind, desc)`.

    `<kind>` -> `(kind, None)`; `<kind>-<desc>` -> `(kind, desc)`. Both parts
    must match the BIDS entity-value rule, so `partition("-")` is unambiguous
    (entity values carry no `-`).
    """
    kind, _, desc = entry.partition("-")
    if not BIDS_ENTITY_VALUE_RE.match(kind):
        raise ValueError(f"Invalid kind in {entry!r}")
    if "-" in entry and not BIDS_ENTITY_VALUE_RE.match(desc):
        raise ValueError(f"Invalid desc in {entry!r}")
    return kind, (desc or None)


def format_kind_desc(kind: str, desc: str | None) -> str:
    """Render `(kind, desc)` back to `<kind>` or `<kind>-<desc>`.

    Inverse of `parse_kind_desc`; round-trips a parsed source ref to its string.
    """
    return kind if desc is None else f"{kind}-{desc}"


def find_bids_files(
    directory: str | Path,
    ext: str,
    *,
    suffix: str | None = None,
    bids_filters: list[str] | None = None,
    recursive: bool = False,
) -> list[BIDSPath]:
    """Find BIDS files under `directory` matching `suffix` and `ext`.

    Non-parseable filenames are silently skipped. Filters sharing a key
    are OR'd; filters with different keys are AND'd. `recursive=True`
    descends into subdirectories. Results are sorted by path.
    """
    validate_extension(ext)
    if bids_filters:
        validate_bids_entities(*bids_filters)
    directory = Path(directory)
    ends_with = f"_{suffix}{ext}" if suffix is not None else ext
    groups = parse_filter_groups(bids_filters or [])

    files = directory.rglob("*") if recursive else directory.iterdir()
    results: list[BIDSPath] = []
    for f in files:
        if not (f.is_file() and f.name.endswith(ends_with)):
            continue
        try:
            bp = BIDSPath(f)
        except ValueError:
            continue
        if all(bp._entities.get(k) in vals for k, vals in groups.items()):
            results.append(bp)

    return sorted(results)
