"""Stable import surface for grammar roles, detection, manifests, and inventory.

The implementation lives in focused modules (`grammar_types`, `grammar_rules`,
`grammar_runtime`, `grammar_detect`, `grammar_manifest`, `grammar_inventory`).
This module re-exports the contract so downstream snapshot and AST lanes keep a
single import site, and remains the `-m structural_validity.grammars` entrypoint.
"""

from __future__ import annotations

from .grammar_detect import DETECTOR_ORDER, classify_file, parser_disposition
from .grammar_inventory import (
    COMMITTED_SOURCE_SPECS,
    inventory_committed_data,
    inventory_dispositions,
    inventory_role,
)
from .grammar_manifest import DEFAULT_MANIFEST_PATH, build_grammar_manifest, main, write_grammar_manifest
from .grammar_rules import (
    BINARY_EXTENSIONS,
    DIFF_PATH_PATTERN,
    EXACT_TEXT_NAMES,
    GENERATED_EXTENSIONS,
    KNOWN_ABSENT_GRAMMARS,
    RULE_BY_EXTENSION,
    RULE_BY_LANGUAGE,
    RULES,
    SCORE_LIMIT,
    SCORE_MARGIN,
    TEXT_EXTENSIONS,
    exact_rule,
)
from .grammar_runtime import TreeSitterRuntime, registered_grammars, validate_rule_grammars
from .grammar_types import (
    CliArguments,
    CommittedDataInventory,
    CommittedSourceInventory,
    CommittedSourceSpec,
    ContainerInventory,
    DetectionMethod,
    DispositionInventoryEntry,
    DistributionManifest,
    FileClassification,
    FileRole,
    GrammarLoadError,
    GrammarManifest,
    GrammarManifestEntry,
    GrammarRule,
    GrammarRuntime,
    ManifestConflictError,
    ParserDisposition,
    ParseScore,
    RepositorySplitInventory,
    RuntimeManifest,
    RuntimeMetadata,
    UnknownGrammarError,
)

__all__ = (
    "BINARY_EXTENSIONS",
    "COMMITTED_SOURCE_SPECS",
    "DEFAULT_MANIFEST_PATH",
    "DETECTOR_ORDER",
    "DIFF_PATH_PATTERN",
    "EXACT_TEXT_NAMES",
    "GENERATED_EXTENSIONS",
    "KNOWN_ABSENT_GRAMMARS",
    "RULES",
    "RULE_BY_EXTENSION",
    "RULE_BY_LANGUAGE",
    "SCORE_LIMIT",
    "SCORE_MARGIN",
    "TEXT_EXTENSIONS",
    "CliArguments",
    "CommittedDataInventory",
    "CommittedSourceInventory",
    "CommittedSourceSpec",
    "ContainerInventory",
    "DetectionMethod",
    "DispositionInventoryEntry",
    "DistributionManifest",
    "FileClassification",
    "FileRole",
    "GrammarLoadError",
    "GrammarManifest",
    "GrammarManifestEntry",
    "GrammarRule",
    "GrammarRuntime",
    "ManifestConflictError",
    "ParseScore",
    "ParserDisposition",
    "RepositorySplitInventory",
    "RuntimeManifest",
    "RuntimeMetadata",
    "TreeSitterRuntime",
    "UnknownGrammarError",
    "build_grammar_manifest",
    "classify_file",
    "exact_rule",
    "inventory_committed_data",
    "inventory_dispositions",
    "inventory_role",
    "main",
    "parser_disposition",
    "registered_grammars",
    "validate_rule_grammars",
    "write_grammar_manifest",
)


if __name__ == "__main__":
    raise SystemExit(main())
