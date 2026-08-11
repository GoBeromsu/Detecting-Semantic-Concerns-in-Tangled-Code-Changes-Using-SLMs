"""Pinned entity node types per language, verified against the real grammars.

Node type names and name-carrying fields are not guessable -- `impl_item` names
its type through `type`, C and C++ bury the identifier one level down inside
`function_declarator`, and Go's `type_declaration` has no name at all while its
`type_spec` child does. Every entry below was read off the pinned distribution
by parsing a fixture, and `__test__` re-derives the same facts so a grammar
upgrade that renames a node fails the suite instead of silently emptying the
entity denominator.

Languages the dataset contains but this map does not cover resolve to
`no_entity_map`, which is reported separately from `top_level`. Conflating them
would let missing coverage read as "the change was outside any function".

Coverage of the committed dataset's 35,960 source-classified paths:
typescript 29.4%, rust 14.2%, python 12.5%, java 11.4%, tsx 6.9%,
javascript 6.1%, go 5.5%, scss 3.4%, bash 1.8%, cpp 0.7%, c 0.2% -- 92.1%.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from .ast_types import EntityRole


@dataclass(frozen=True, slots=True)
class EntityNode:
    """How one node type yields a name, a role, and (optionally) an arity."""

    role: EntityRole
    name_field: str = "name"
    descend_declarator: bool = False
    parameters_field: str | None = None


def _function(parameters: str | None = "parameters") -> EntityNode:
    return EntityNode(EntityRole.FUNCTION, parameters_field=parameters)


def _method(parameters: str | None = "parameters") -> EntityNode:
    return EntityNode(EntityRole.METHOD, parameters_field=parameters)


_TYPE: Final = EntityNode(EntityRole.TYPE)
_MODULE: Final = EntityNode(EntityRole.MODULE)

_TYPESCRIPT: Final[dict[str, EntityNode]] = {
    "class_declaration": _TYPE,
    "abstract_class_declaration": _TYPE,
    "interface_declaration": _TYPE,
    "type_alias_declaration": _TYPE,
    "enum_declaration": _TYPE,
    "internal_module": _MODULE,
    "function_declaration": _function(),
    "generator_function_declaration": _function(),
    "method_definition": _method(),
    "method_signature": _method(),
}

_JAVASCRIPT: Final[dict[str, EntityNode]] = {
    "class_declaration": _TYPE,
    "function_declaration": _function(),
    "generator_function_declaration": _function(),
    "method_definition": _method(),
}

_RUST: Final[dict[str, EntityNode]] = {
    "struct_item": _TYPE,
    "enum_item": _TYPE,
    "trait_item": _TYPE,
    "union_item": _TYPE,
    "type_item": _TYPE,
    "mod_item": _MODULE,
    # `impl S` and `impl Trait for S` both name S through `type`, which keeps a
    # method qualified under the type it is implemented on.
    "impl_item": EntityNode(EntityRole.TYPE, name_field="type"),
    "function_item": _function(),
    "function_signature_item": _function(),
}

_PYTHON: Final[dict[str, EntityNode]] = {
    # `decorated_definition` carries no name; its inner definition does, and
    # being the smaller node it wins the enclosing-entity search anyway.
    "class_definition": _TYPE,
    "function_definition": _function(),
}

_JAVA: Final[dict[str, EntityNode]] = {
    "class_declaration": _TYPE,
    "interface_declaration": _TYPE,
    "enum_declaration": _TYPE,
    "record_declaration": _TYPE,
    "annotation_type_declaration": _TYPE,
    "method_declaration": _method(),
    "constructor_declaration": _method(),
}

_GO: Final[dict[str, EntityNode]] = {
    # `type_declaration` is the unnamed wrapper; `type_spec` holds the name.
    "type_spec": _TYPE,
    "function_declaration": _function(),
    "method_declaration": _method(),
}

_C_FAMILY: Final[dict[str, EntityNode]] = {
    "struct_specifier": _TYPE,
    "union_specifier": _TYPE,
    "enum_specifier": _TYPE,
    # The identifier sits under function_declarator, so `declarator` must be
    # followed rather than read as text: reading it yields "free_fn(int a)".
    "function_definition": EntityNode(
        EntityRole.FUNCTION, name_field="declarator", descend_declarator=True, parameters_field="parameters"
    ),
}

_CPP: Final[dict[str, EntityNode]] = {
    **_C_FAMILY,
    "class_specifier": _TYPE,
    "namespace_definition": _MODULE,
}

_BASH: Final[dict[str, EntityNode]] = {"function_definition": _function(parameters=None)}

_SCSS: Final[dict[str, EntityNode]] = {
    "mixin_statement": _function(parameters=None),
    "function_statement": _function(parameters=None),
}

ENTITY_QUERIES: Final[dict[str, dict[str, EntityNode]]] = {
    "typescript": _TYPESCRIPT,
    "tsx": _TYPESCRIPT,
    "javascript": _JAVASCRIPT,
    "jsx": _JAVASCRIPT,
    "rust": _RUST,
    "python": _PYTHON,
    "java": _JAVA,
    "go": _GO,
    "c": _C_FAMILY,
    "cpp": _CPP,
    "objective_c": _C_FAMILY,
    "bash": _BASH,
    # Plain CSS is deliberately absent: the `css` grammar has no mixin or
    # function statement, so it resolves to `no_entity_map` rather than
    # reporting every rule-set edit as top level.
    "scss": _SCSS,
}

IDENTIFIER_NODES: Final = frozenset(
    {"identifier", "type_identifier", "field_identifier", "property_identifier", "shorthand_property_identifier"}
)
"""Node types read for the robustness-only identifier overlap field.

This is a string overlap between two edits' visible identifier tokens. It is not
symbol resolution and carries no def-use, alias, or dependency meaning, which is
why it stays out of X/Y/Z and out of the primary figure.
"""

STRUCTURED_KEY_NODES: Final[dict[str, tuple[tuple[str, ...], str | None]]] = {
    "json": (("pair",), "key"),
    "yaml": (("block_mapping_pair",), "key"),
    # TOML's `pair` declares no fields at all, and HCL declares `key` globally
    # without assigning it on `attribute`, so both name themselves through their
    # leading name-like children instead, which `None` selects.
    #
    # The table and block types are listed because neither format nests its
    # pairs inside the thing that qualifies them: a TOML pair is a sibling of
    # its `[table]` header, and an HCL attribute sits under a `body`. Without
    # them, `target` under two different tables would read as one shared key.
    "toml": (("table", "table_array_element", "pair"), None),
    "hcl": (("block", "attribute"), None),
}
"""Key/block node types plus the name field, for structured key-path proximity only."""

NAME_LIKE_NODES: Final = frozenset({"identifier", "string_lit", "bare_key", "quoted_key", "dotted_key"})
"""Child types read as a node's own name when the grammar declares no key field.

Scanning stops at the first child outside this set, which is what keeps a TOML
pair's value string and an HCL block's body out of the key it is naming.
"""


def entity_map(language: str | None) -> dict[str, EntityNode] | None:
    return ENTITY_QUERIES.get(language) if language is not None else None
