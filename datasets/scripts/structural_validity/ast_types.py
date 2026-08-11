"""Typed vocabulary for AST and structured-tree proximity evidence.

The primary observable is `name_matched_entity_co_touch`: two atomic commits
touched an entity carrying the same normalized qualified name, in the same file
path, on revisions that both parsed. That is a co-touch of a *name*, not a claim
that the entity is the same function across history, and it says nothing about
what the two edits do to each other.

Nothing here models def-use, symbol resolution, variable identity, data or
control flow, aliases, calls, or dependencies, and no field may be presented as
a proxy for any of them.

Every parse attempt carries its own report -- status, error-node ratio, bytes
covered, evidence kind -- so an unavailable grammar, a parse that degraded, and
a change that genuinely sits at top level stay distinguishable rather than
collapsing into one silent zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

MAX_ERROR_RATIO: Final = 0.10
"""Above this error-node ratio a parse is reported failed rather than trusted."""

MAX_DIFF_LINES: Final = 50_000
"""Line cap for interval extraction; larger files report `intervals_truncated`."""


class ParseStatus(str, Enum):
    PARSED = "parsed"
    UNSUPPORTED_SOURCE = "unsupported_source"
    NOT_ATTEMPTED = "not_attempted"
    PARSE_FAILED_QUALITY = "parse_failed_quality"
    SIDE_ABSENT = "side_absent"


class EvidenceKind(str, Enum):
    NAMED_ENTITY = "named_entity"
    TOP_LEVEL = "top_level"
    NO_ENTITY_MAP = "no_entity_map"
    SCHEMA_PATH = "schema_path"
    NONE = "none"


class EntityRole(str, Enum):
    FUNCTION = "function"
    METHOD = "method"
    TYPE = "type"
    MODULE = "module"
    BLOCK = "block"


class ComparisonStatus(str, Enum):
    COMPARED = "compared"
    PATH_UNMATCHED = "path_unmatched"
    ROLE_UNMATCHED = "role_unmatched"
    EVIDENCE_UNAVAILABLE = "evidence_unavailable"


@dataclass(frozen=True, slots=True)
class ParseReport:
    """One parser attempt, or one honest reason no parser ran."""

    status: ParseStatus
    grammar: str | None
    evidence_kind: EvidenceKind
    error_node_ratio: float
    bytes_covered: int
    bytes_total: int
    intervals_truncated: bool = False

    @property
    def attempted(self) -> bool:
        return self.status in {ParseStatus.PARSED, ParseStatus.PARSE_FAILED_QUALITY}


@dataclass(frozen=True, slots=True)
class Entity:
    """A named node enclosing a changed interval.

    `arity` is the parameter count when the grammar exposes one. It joins the
    match key so overloads stay distinct, while surviving parameter renames --
    a signature string would not, and would silently split a matched entity in
    two whenever one side renamed an argument.
    """

    qualified_name: str
    role: EntityRole
    arity: int | None
    start_byte: int
    end_byte: int

    @property
    def key(self) -> tuple[str, str, int | None]:
        return (self.qualified_name, self.role.value, self.arity)


@dataclass(frozen=True, slots=True)
class SchemaPath:
    """A key/block path in a structured file; never an entity."""

    path: str
    start_byte: int
    end_byte: int


@dataclass(frozen=True, slots=True)
class SideEvidence:
    report: ParseReport
    entities: tuple[Entity, ...] = ()
    schema_paths: tuple[SchemaPath, ...] = ()
    identifiers: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FileEvidence:
    """Both revisions of one changed path, each independently reported."""

    path: str
    language: str | None
    grammar: str | None
    before: SideEvidence
    after: SideEvidence

    @property
    def entity_keys(self) -> frozenset[tuple[str, str, int | None]]:
        return frozenset(
            entity.key for side in (self.before, self.after) for entity in side.entities
        )

    @property
    def schema_keys(self) -> frozenset[str]:
        return frozenset(
            item.path for side in (self.before, self.after) for item in side.schema_paths
        )

    @property
    def identifier_set(self) -> frozenset[str]:
        return frozenset(
            name for side in (self.before, self.after) for name in side.identifiers
        )


@dataclass(frozen=True, slots=True)
class PairFileEvidence:
    """One path touched by both atomic commits of a pair.

    `name_matched_entity_co_touch` is None -- not False -- whenever the file is
    not source-classified or either side lacks usable evidence, so a structured
    file and a genuinely unmatched source file never share a cell.
    """

    path: str
    status: ComparisonStatus
    name_matched_entity_co_touch: bool | None
    schema_path_co_touch: bool | None
    matched_entities: tuple[str, ...]
    syntax_identifier_string_overlap: float | None
    entity_span_overlap: float | None


@dataclass(frozen=True, slots=True)
class ForbiddenClaimError(Exception):
    """A rendered claim used vocabulary this study cannot support."""

    phrase: str
    text: str
