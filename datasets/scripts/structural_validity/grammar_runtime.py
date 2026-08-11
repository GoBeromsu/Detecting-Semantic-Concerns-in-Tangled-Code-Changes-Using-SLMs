"""Tree-sitter runtime access with registry-checked availability.

``get_language`` conflates three states: the distribution does not know the
name, the distribution knows it but the artifact is not materialized yet, and
the artifact loaded. Only the first is an honest ``unsupported_source``. The
registry is consulted first so a cold cache or a failed download raises instead
of silently shrinking AST coverage in a generated manifest.
"""

from __future__ import annotations

from importlib.metadata import version

import tree_sitter
import tree_sitter_language_pack as language_pack

from .grammar_rules import KNOWN_ABSENT_GRAMMARS, RULES
from .grammar_types import (
    GrammarLoadError,
    ParseScore,
    RuntimeMetadata,
    UnknownGrammarError,
)


def registered_grammars() -> frozenset[str]:
    """Names the pinned distribution can supply, independent of cache state.

    ``available_languages`` reports only what is materialized locally, so it
    shrinks on a cold cache. The distribution manifest is the fixed registry.
    """
    return frozenset(language_pack.manifest_languages())


class TreeSitterRuntime:
    def metadata(self) -> RuntimeMetadata:
        return RuntimeMetadata(
            package_version=version("tree-sitter-language-pack"),
            runtime_version=version("tree-sitter"),
            language_abi_max=tree_sitter.LANGUAGE_VERSION,
            language_abi_min=tree_sitter.MIN_COMPATIBLE_LANGUAGE_VERSION,
        )

    def grammar_abi(self, grammar: str) -> int | None:
        if not language_pack.has_language(grammar):
            return None
        try:
            return language_pack.get_language(grammar).abi_version
        except language_pack.Error as error:
            raise GrammarLoadError(grammar, type(error).__name__) from error

    def score(self, grammar: str, source: bytes) -> ParseScore | None:
        if not language_pack.has_language(grammar):
            return None
        try:
            root = language_pack.get_parser(grammar).parse(source).root_node
        except language_pack.Error as error:
            raise GrammarLoadError(grammar, type(error).__name__) from error
        nodes = [root]
        error_nodes = 0
        named_nodes = 0
        while nodes:
            node = nodes.pop()
            error_nodes += int(node.type == "ERROR" or node.is_missing)
            named_nodes += int(node.is_named)
            nodes.extend(node.children)
        return ParseScore(error_ratio=error_nodes / max(1, named_nodes), named_nodes=named_nodes)


def validate_rule_grammars(registry: frozenset[str] | None = None) -> tuple[str, ...]:
    """Return the grammars declared absent, rejecting names upstream never had.

    A misspelled grammar and an unavailable grammar both fail to load, but only
    the second is a real coverage limit. Requiring absent names to be declared
    keeps a typo from being reported as an upstream gap.
    """
    known = registry if registry is not None else registered_grammars()
    absent: list[str] = []
    for rule in RULES:
        if rule.grammar is None or rule.grammar in known:
            continue
        if rule.grammar not in KNOWN_ABSENT_GRAMMARS:
            raise UnknownGrammarError(rule.grammar, rule.language)
        absent.append(rule.grammar)
    return tuple(sorted(set(absent)))
