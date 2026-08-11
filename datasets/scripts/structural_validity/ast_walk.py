"""Mapping a changed byte interval onto the node that contains it."""

from __future__ import annotations

import tree_sitter

from .ast_intervals import ByteRange


def covering_node(root: tree_sitter.Node, span: ByteRange, source: bytes) -> tree_sitter.Node:
    """Smallest node containing the span, after trimming its surrounding blank.

    Line-granular intervals start at the indentation and end past the newline.
    Left untrimmed, a one-line edit on a declaration line covers whitespace
    outside the declaration, so the smallest containing node becomes the
    enclosing block and the entity itself is lost.
    """
    start, end = span.start, span.end
    while start < end and source[start : start + 1].isspace():
        start += 1
    while end > start and source[end - 1 : end].isspace():
        end -= 1
    if start >= end:
        start, end = span.start, span.end
    start = min(start, max(0, root.end_byte - 1))
    end = min(max(end, start + 1), root.end_byte)
    return root.descendant_for_byte_range(start, end) or root


__all__ = ["covering_node"]
