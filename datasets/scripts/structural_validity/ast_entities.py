"""Resolve changed intervals to the smallest enclosing named entity.

The walk is upward from the node covering a changed interval, so the entity
reported is the innermost one that contains the edit -- a nested function rather
than the class around it. The qualified name is the dot-joined chain of
enclosing entity names, which is what makes two edits inside the same nested
function compare equal while two same-named methods on different types do not.

A parse whose error-node ratio exceeds the pinned threshold is reported
`parse_failed_quality` and yields no entities. Malformed source therefore stays
in the attempted denominator instead of quietly leaving it.
"""

from __future__ import annotations

import tree_sitter
import tree_sitter_language_pack as language_pack

from .ast_intervals import ByteRange
from .ast_queries import (
    IDENTIFIER_NODES,
    STRUCTURED_KEY_NODES,
    EntityNode,
)
from .ast_schema import schema_paths
from .ast_types import (
    MAX_ERROR_RATIO,
    Entity,
    EntityRole,
    EvidenceKind,
    ParseReport,
    ParseStatus,
    SideEvidence,
)
from .ast_walk import covering_node
from .grammar_types import GrammarLoadError


def parse_tree(grammar: str, source: bytes) -> tree_sitter.Tree:
    try:
        return language_pack.get_parser(grammar).parse(source)
    except language_pack.Error as error:
        raise GrammarLoadError(grammar, type(error).__name__) from error


def tree_quality(root: tree_sitter.Node) -> tuple[float, int]:
    """Return (error-node ratio, bytes covered by non-error nodes).

    Error spans are unioned by not descending into an error node: its whole
    subtree is already inside it, and counting the children again would subtract
    the same bytes twice and report a clean file as uncovered.
    """
    stack = [root]
    errors = 0
    named = 0
    error_bytes = 0
    while stack:
        node = stack.pop()
        if node.is_named:
            named += 1
        if node.type == "ERROR" or node.is_missing:
            errors += 1
            error_bytes += node.end_byte - node.start_byte
            continue
        stack.extend(node.children)
    covered = max(0, (root.end_byte - root.start_byte) - error_bytes)
    return (errors / max(1, named), covered)


def degraded(ratio: float, covered: int, total: int) -> bool:
    """Whether a parse is too damaged to carry entity evidence.

    Node counting alone is not enough: Tree-sitter recovers from a file of pure
    noise with a handful of ERROR nodes wrapping hundreds of named ones, which
    reads as a 2% error ratio. How many bytes the error nodes swallowed is the
    signal that actually separates a recovered typo from an unparseable file.
    """
    if ratio > MAX_ERROR_RATIO:
        return True
    return total > 0 and covered < (1.0 - MAX_ERROR_RATIO) * total


def _node_name(node: tree_sitter.Node, spec: EntityNode) -> str | None:
    child = node.child_by_field_name(spec.name_field)
    if child is None:
        return None
    if spec.descend_declarator:
        inner = child
        while inner is not None and inner.type in {"function_declarator", "pointer_declarator", "parenthesized_declarator"}:
            inner = inner.child_by_field_name("declarator")
        child = inner if inner is not None else child
    text = child.text
    return text.decode("utf-8", errors="replace").strip() if text is not None else None


def _arity(node: tree_sitter.Node, spec: EntityNode) -> int | None:
    if spec.parameters_field is None:
        return None
    parameters = node.child_by_field_name(spec.parameters_field)
    if parameters is None:
        return None
    return sum(1 for child in parameters.children if child.is_named)


def enclosing_entity(node: tree_sitter.Node, entities: dict[str, EntityNode]) -> Entity | None:
    """Walk up to the innermost entity, qualifying it by its enclosing chain."""
    chain: list[tuple[str, EntityNode, tree_sitter.Node]] = []
    current: tree_sitter.Node | None = node
    while current is not None:
        spec = entities.get(current.type)
        if spec is not None:
            name = _node_name(current, spec)
            if name:
                chain.append((name, spec, current))
        current = current.parent
    if not chain:
        return None
    name, spec, innermost = chain[0]
    qualified = ".".join(part for part, _, _ in reversed(chain))
    return Entity(qualified, spec.role, _arity(innermost, spec), innermost.start_byte, innermost.end_byte)


def _identifiers(root: tree_sitter.Node, spans: tuple[ByteRange, ...], source: bytes) -> tuple[str, ...]:
    found: set[str] = set()
    for span in spans:
        stack = [covering_node(root, span, source)]
        while stack:
            node = stack.pop()
            if node.type in IDENTIFIER_NODES and node.text is not None:
                found.add(node.text.decode("utf-8", errors="replace"))
            stack.extend(node.children)
    return tuple(sorted(found))


def side_evidence(
    grammar: str | None,
    language: str | None,
    source: bytes | None,
    spans: tuple[ByteRange, ...],
    entities: dict[str, EntityNode] | None,
    *,
    structured: bool = False,
    truncated: bool = False,
) -> SideEvidence:
    """One revision's evidence, or the reason no parser ran on it."""
    if source is None:
        return SideEvidence(ParseReport(ParseStatus.SIDE_ABSENT, grammar, EvidenceKind.NONE, 0.0, 0, 0))
    total = len(source)
    if grammar is None or not language_pack.has_language(grammar):
        status = ParseStatus.UNSUPPORTED_SOURCE if grammar is not None else ParseStatus.NOT_ATTEMPTED
        return SideEvidence(ParseReport(status, grammar, EvidenceKind.NONE, 0.0, 0, total))
    root = parse_tree(grammar, source).root_node
    ratio, covered = tree_quality(root)
    if degraded(ratio, covered, total):
        report = ParseReport(ParseStatus.PARSE_FAILED_QUALITY, grammar, EvidenceKind.NONE, ratio, covered, total, truncated)
        return SideEvidence(report)
    if structured:
        key_spec = STRUCTURED_KEY_NODES.get(language or "")
        paths = schema_paths(root, spans, source, *key_spec) if key_spec is not None else ()
        kind = EvidenceKind.SCHEMA_PATH if paths else EvidenceKind.NO_ENTITY_MAP
        return SideEvidence(ParseReport(ParseStatus.PARSED, grammar, kind, ratio, covered, total, truncated), schema_paths=paths)
    if entities is None:
        report = ParseReport(ParseStatus.PARSED, grammar, EvidenceKind.NO_ENTITY_MAP, ratio, covered, total, truncated)
        return SideEvidence(report, identifiers=_identifiers(root, spans, source))
    found: dict[tuple[str, str, int | None], Entity] = {}
    for span in spans:
        entity = enclosing_entity(covering_node(root, span, source), entities)
        if entity is not None:
            _ = found.setdefault(entity.key, entity)
    kind = EvidenceKind.NAMED_ENTITY if found else EvidenceKind.TOP_LEVEL
    report = ParseReport(ParseStatus.PARSED, grammar, kind, ratio, covered, total, truncated)
    return SideEvidence(
        report, entities=tuple(found[key] for key in sorted(found)), identifiers=_identifiers(root, spans, source)
    )


__all__ = [
    "Entity",
    "EntityRole",
    "enclosing_entity",
    "parse_tree",
    "side_evidence",
    "tree_quality",
]
