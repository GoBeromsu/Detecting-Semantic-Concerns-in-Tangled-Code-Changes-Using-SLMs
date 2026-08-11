"""Typed vocabulary for file roles, detection methods, and grammar manifests.

Roles and dispositions stay in one module so the detection, manifest, and
inventory lanes cannot drift into private copies of the same enum.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol


class FileRole(str, Enum):
    SOURCE_AST = "source_ast"
    STRUCTURED_PARSE_TREE = "structured_parse_tree"
    TEXT_ONLY = "text_only"
    BINARY_GENERATED = "binary_generated"
    UNRESOLVED_SOURCE = "unresolved_source"


class DetectionMethod(str, Enum):
    EXACT_FILENAME = "exact_filename"
    EXTENSION = "extension"
    SHEBANG = "shebang"
    CONTENT_SIGNATURE = "content_signature"
    PARSER_SCORE = "parser_score"
    BINARY_CONTENT = "binary_content"
    AMBIGUOUS = "ambiguous"


class ParserDisposition(str, Enum):
    AVAILABLE = "available"
    UNSUPPORTED_SOURCE = "unsupported_source"
    UNAVAILABLE_STRUCTURED = "unavailable_structured"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True, slots=True)
class ParseScore:
    error_ratio: float
    named_nodes: int


@dataclass(frozen=True, slots=True)
class RuntimeMetadata:
    package_version: str
    runtime_version: str
    language_abi_max: int
    language_abi_min: int


class GrammarRuntime(Protocol):
    def metadata(self) -> RuntimeMetadata: ...

    def grammar_abi(self, grammar: str) -> int | None: ...

    def score(self, grammar: str, source: bytes) -> ParseScore | None: ...


@dataclass(frozen=True, slots=True)
class GrammarRule:
    language: str
    grammar: str | None
    role: FileRole
    extensions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FileClassification:
    path: str
    language: str | None
    grammar: str | None
    role: FileRole
    method: DetectionMethod
    parser_disposition: ParserDisposition


@dataclass(frozen=True, slots=True)
class DistributionManifest:
    distribution: str
    version: str


@dataclass(frozen=True, slots=True)
class RuntimeManifest:
    distribution: str
    version: str
    language_abi_min: int
    language_abi_max: int


@dataclass(frozen=True, slots=True)
class GrammarManifestEntry:
    language: str
    grammar: str
    role: FileRole
    extensions: tuple[str, ...]
    detector_order: tuple[str, ...]
    grammar_abi: int | None
    parser_disposition: ParserDisposition


@dataclass(frozen=True, slots=True)
class GrammarManifest:
    schema_version: int
    package: DistributionManifest
    runtime: RuntimeManifest
    grammars: tuple[GrammarManifestEntry, ...]


@dataclass(frozen=True, slots=True)
class ManifestConflictError(Exception):
    path: Path


@dataclass(frozen=True, slots=True)
class GrammarLoadError(Exception):
    """A grammar the distribution registers could not be materialized.

    Raised instead of degrading to ``unsupported_source`` so a cold cache or a
    failed download can never masquerade as an upstream coverage gap.
    """

    grammar: str
    reason: str


@dataclass(frozen=True, slots=True)
class UnknownGrammarError(Exception):
    """A rule names a grammar the distribution does not register at all."""

    grammar: str
    language: str


@dataclass(slots=True)
class CliArguments:
    output: Path


@dataclass(frozen=True, slots=True)
class DispositionInventoryEntry:
    extension: str
    count: int
    role: FileRole


@dataclass(frozen=True, slots=True)
class CommittedSourceInventory:
    name: str
    records: int
    diff_fragments: int
    paths: int


@dataclass(frozen=True, slots=True)
class RepositorySplitInventory:
    train_repositories: int
    test_repositories: int
    train_type_supply: int
    test_type_supply: int


@dataclass(frozen=True, slots=True)
class ContainerInventory:
    name: str
    role: FileRole


@dataclass(frozen=True, slots=True)
class CommittedDataInventory:
    sources: tuple[CommittedSourceInventory, ...]
    split: RepositorySplitInventory
    containers: tuple[ContainerInventory, ...]
    dispositions: tuple[DispositionInventoryEntry, ...]
    path_count: int
    disposed_path_count: int
    extension_count: int
    container_count: int
    sha256: str


@dataclass(frozen=True, slots=True)
class CommittedSourceSpec:
    name: str
    diff_column: str
    nested_diffs: bool
