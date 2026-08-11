"""Key paths in structured files, which are never entities.

A structured file carries no functions to co-touch, so this lane reports only
which key or block paths a change landed on. The paths are compared as strings
between two commits and mean proximity in the document tree -- nothing about
what either change does to the values.
"""

from __future__ import annotations

import tree_sitter

from .ast_intervals import ByteRange
from .ast_queries import NAME_LIKE_NODES
from .ast_types import SchemaPath
from .ast_walk import covering_node


def _text(node: tree_sitter.Node) -> str:
    """Node text with surrounding quotes removed.

    Stripping quotes lets a quoted JSON key and a bare TOML key of the same name
    compare as one path instead of two.
    """
    return node.text.decode("utf-8", errors="replace").strip().strip("\"'") if node.text is not None else ""


def _key_text(node: tree_sitter.Node, key_field: str | None) -> str | None:
    """The key a mapping node carries, by field where the grammar assigns one.

    Where it does not, the leading name-like children are read instead, which
    keeps an HCL block's labels (`resource.aws_s3.logs`) in its path rather than
    collapsing every resource of one type onto a single key.
    """
    if key_field is not None:
        key = node.child_by_field_name(key_field)
        if key is not None:
            return _text(key) or None
    parts: list[str] = []
    for child in node.children:
        if not child.is_named:
            continue
        if child.type not in NAME_LIKE_NODES:
            break
        parts.append(_text(child))
    return ".".join(part for part in parts if part) or None


def _key_path(node: tree_sitter.Node, key_nodes: tuple[str, ...], key_field: str | None) -> str | None:
    parts: list[str] = []
    current: tree_sitter.Node | None = node
    while current is not None:
        if current.type in key_nodes:
            text = _key_text(current, key_field)
            if text:
                parts.append(text)
        current = current.parent
    return ".".join(reversed(parts)) if parts else None


def schema_paths(
    root: tree_sitter.Node,
    spans: tuple[ByteRange, ...],
    source: bytes,
    key_nodes: tuple[str, ...],
    key_field: str | None,
) -> tuple[SchemaPath, ...]:
    """Key paths a changed span touches, from inside it and from around it.

    An upward walk alone finds nothing when the span covers a whole object, and
    a downward walk alone finds nothing when the span sits on one scalar value,
    so both directions are collected.
    """
    found: dict[str, SchemaPath] = {}

    def record(node: tree_sitter.Node) -> None:
        path = _key_path(node, key_nodes, key_field)
        if path is not None:
            _ = found.setdefault(path, SchemaPath(path, node.start_byte, node.end_byte))

    for span in spans:
        covering = covering_node(root, span, source)
        stack = [covering]
        while stack:
            current = stack.pop()
            if current.type in key_nodes:
                record(current)
            stack.extend(current.children)
        ancestor = covering.parent
        while ancestor is not None:
            if ancestor.type in key_nodes:
                record(ancestor)
            ancestor = ancestor.parent
    return tuple(found[key] for key in sorted(found))


__all__ = ["schema_paths"]
