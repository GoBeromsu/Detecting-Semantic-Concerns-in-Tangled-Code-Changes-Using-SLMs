"""AST and structured-tree proximity evidence.

The regression block at the end re-derives every pinned node type and name field
against the real distribution, so a grammar upgrade that renames a node fails
here instead of silently emptying the entity denominator and reporting every
change as `top_level`.
"""

from __future__ import annotations

import pytest
import tree_sitter_language_pack as language_pack
from structural_validity.ast_entities import side_evidence
from structural_validity.ast_intervals import ByteRange, changed_intervals
from structural_validity.ast_metrics import (
    FORBIDDEN_CLAIM_PHRASES,
    file_evidence,
    pair_file_evidence,
    validate_claim,
)
from structural_validity.ast_queries import (
    ENTITY_QUERIES,
    STRUCTURED_KEY_NODES,
    entity_map,
)
from structural_validity.ast_types import (
    ComparisonStatus,
    EvidenceKind,
    ForbiddenClaimError,
    ParseStatus,
)
from structural_validity.grammar_rules import RULES
from structural_validity.grammar_types import FileRole

_GRAMMAR_FOR: dict[str, str] = {rule.language: rule.grammar for rule in RULES if rule.grammar is not None}

PYTHON_BASE = b'''\
class Service:
    def handler(self):
        def inner():
            value = 1
            other = 2
            return value + other
        return inner()
'''

PYTHON_EDIT_VALUE = PYTHON_BASE.replace(b"value = 1", b"value = 11")
PYTHON_EDIT_OTHER = PYTHON_BASE.replace(b"other = 2", b"other = 22")
PYTHON_EDIT_ELSEWHERE = PYTHON_BASE.replace(b"return inner()", b"return inner() or None")

JAVA_OVERLOADS = b'''\
class Calc {
    int add(int a) { return a; }
    int add(int a, int b) { return a + b; }
}
'''


def _source(path: str, language: str, before: bytes | None, after: bytes | None):
    return file_evidence(
        path,
        language=language,
        grammar=_GRAMMAR_FOR[language],
        role=FileRole.SOURCE_AST,
        before=before,
        after=after,
    )


def _structured(path: str, language: str, before: bytes | None, after: bytes | None):
    return file_evidence(
        path,
        language=language,
        grammar=_GRAMMAR_FOR[language],
        role=FileRole.STRUCTURED_PARSE_TREE,
        before=before,
        after=after,
    )


def test_two_edits_in_the_same_nested_function_resolve_to_one_qualified_entity() -> None:
    left = _source("svc.py", "python", PYTHON_BASE, PYTHON_EDIT_VALUE)
    right = _source("svc.py", "python", PYTHON_BASE, PYTHON_EDIT_OTHER)

    pair = pair_file_evidence(left, right, FileRole.SOURCE_AST)

    assert pair.status is ComparisonStatus.COMPARED
    assert pair.name_matched_entity_co_touch is True
    assert pair.matched_entities == ("Service.handler.inner",)


def test_edits_in_different_entities_do_not_match_on_the_enclosing_class() -> None:
    """The innermost entity decides, so a shared outer class is not a match."""
    left = _source("svc.py", "python", PYTHON_BASE, PYTHON_EDIT_VALUE)
    right = _source("svc.py", "python", PYTHON_BASE, PYTHON_EDIT_ELSEWHERE)

    pair = pair_file_evidence(left, right, FileRole.SOURCE_AST)

    assert pair.name_matched_entity_co_touch is False
    assert pair.matched_entities == ()


def test_overloads_stay_distinct_because_arity_joins_the_match_key() -> None:
    one_argument = JAVA_OVERLOADS.replace(b"{ return a; }", b"{ return a + 0; }")
    two_arguments = JAVA_OVERLOADS.replace(b"{ return a + b; }", b"{ return b + a; }")
    left = _source("Calc.java", "java", JAVA_OVERLOADS, one_argument)
    right = _source("Calc.java", "java", JAVA_OVERLOADS, two_arguments)

    assert left.entity_keys == {("Calc.add", "method", 1)}
    assert right.entity_keys == {("Calc.add", "method", 2)}
    assert pair_file_evidence(left, right, FileRole.SOURCE_AST).name_matched_entity_co_touch is False


def test_arity_survives_a_parameter_rename_that_a_signature_string_would_split() -> None:
    renamed = JAVA_OVERLOADS.replace(b"int add(int a) { return a; }", b"int add(int first) { return first; }")
    left = _source("Calc.java", "java", JAVA_OVERLOADS, JAVA_OVERLOADS.replace(b"return a; }", b"return a + 0; }"))
    right = _source("Calc.java", "java", JAVA_OVERLOADS, renamed)

    assert pair_file_evidence(left, right, FileRole.SOURCE_AST).name_matched_entity_co_touch is True


def test_paths_that_do_not_match_are_never_compared() -> None:
    left = _source("a.py", "python", PYTHON_BASE, PYTHON_EDIT_VALUE)
    right = _source("b.py", "python", PYTHON_BASE, PYTHON_EDIT_VALUE)

    pair = pair_file_evidence(left, right, FileRole.SOURCE_AST)

    assert pair.status is ComparisonStatus.PATH_UNMATCHED
    assert pair.name_matched_entity_co_touch is None


# --- dispositions -----------------------------------------------------------


def test_a_source_file_with_a_pinned_grammar_receives_exactly_one_parse_disposition() -> None:
    evidence = _source("svc.py", "python", PYTHON_BASE, PYTHON_EDIT_VALUE)

    assert evidence.before.report.status is ParseStatus.PARSED
    assert evidence.after.report.status is ParseStatus.PARSED
    assert evidence.after.report.evidence_kind is EvidenceKind.NAMED_ENTITY
    assert evidence.after.report.bytes_total == len(PYTHON_EDIT_VALUE)


def test_a_language_without_a_pinned_grammar_is_not_attempted_and_fakes_no_invocation() -> None:
    evidence = file_evidence(
        "a.zzz", language="unknown", grammar=None, role=FileRole.SOURCE_AST, before=b"x\n", after=b"y\n"
    )

    assert evidence.after.report.status is ParseStatus.NOT_ATTEMPTED
    assert evidence.after.report.evidence_kind is EvidenceKind.NONE
    assert evidence.after.entities == ()


def test_a_grammar_missing_from_the_distribution_reports_unsupported_source() -> None:
    evidence = file_evidence(
        "a.zzz", language="zzz", grammar="not_a_real_grammar", role=FileRole.SOURCE_AST, before=b"x\n", after=b"y\n"
    )

    assert evidence.after.report.status is ParseStatus.UNSUPPORTED_SOURCE


def test_an_absent_side_is_reported_rather_than_dropped() -> None:
    evidence = _source("svc.py", "python", None, PYTHON_BASE)

    assert evidence.before.report.status is ParseStatus.SIDE_ABSENT
    assert evidence.after.report.status is ParseStatus.PARSED


def test_a_supported_language_with_no_entity_map_is_distinguished_from_top_level() -> None:
    evidence = file_evidence(
        "app.vue",
        language="vue",
        grammar=_GRAMMAR_FOR["vue"],
        role=FileRole.SOURCE_AST,
        before=b"<template><p>a</p></template>\n",
        after=b"<template><p>b</p></template>\n",
    )

    assert evidence.after.report.status is ParseStatus.PARSED
    assert evidence.after.report.evidence_kind is EvidenceKind.NO_ENTITY_MAP
    assert evidence.after.entities == ()


def test_a_change_genuinely_outside_any_entity_reports_top_level() -> None:
    before = b"import os\nCONST = 1\n"
    after = b"import os\nimport sys\nCONST = 1\n"

    evidence = _source("m.py", "python", before, after)

    assert evidence.after.report.status is ParseStatus.PARSED
    assert evidence.after.report.evidence_kind is EvidenceKind.TOP_LEVEL


def test_a_non_source_role_yields_no_parse_attempt_at_all() -> None:
    evidence = file_evidence(
        "README.md", language="markdown", grammar="markdown", role=FileRole.TEXT_ONLY, before=b"a\n", after=b"b\n"
    )

    assert evidence.before.report.status is ParseStatus.NOT_ATTEMPTED
    assert evidence.after.report.status is ParseStatus.NOT_ATTEMPTED
    assert evidence.after.report.grammar is None
    assert evidence.after.report.bytes_total == 2


def test_a_text_only_file_absent_on_one_revision_keeps_the_two_facts_apart() -> None:
    """`not_attempted` means no parser ran; `side_absent` means no revision."""
    evidence = file_evidence(
        "README.md", language="markdown", grammar="markdown", role=FileRole.TEXT_ONLY, before=None, after=b"b\n"
    )

    assert evidence.before.report.status is ParseStatus.SIDE_ABSENT
    assert evidence.after.report.status is ParseStatus.NOT_ATTEMPTED


# --- parse quality ----------------------------------------------------------


def test_malformed_source_reports_parse_failed_quality_and_no_entities() -> None:
    garbage = b"".join(b"def ((( %d ]]] :::\n" % index for index in range(40))

    evidence = _source("broken.py", "python", PYTHON_BASE, garbage)
    report = evidence.after.report

    assert report.status is ParseStatus.PARSE_FAILED_QUALITY
    assert report.error_node_ratio > 0.10
    assert evidence.after.entities == ()
    assert evidence.after.identifiers == ()


def test_malformed_source_stays_in_the_attempted_denominator() -> None:
    garbage = b"".join(b"def ((( %d ]]] :::\n" % index for index in range(40))

    evidence = _source("broken.py", "python", PYTHON_BASE, garbage)

    assert evidence.after.report.attempted is True
    assert evidence.after.report.bytes_total == len(garbage)


def test_a_pair_whose_only_evidence_failed_quality_is_unavailable_not_unmatched() -> None:
    garbage = b"".join(b"def ((( %d ]]] :::\n" % index for index in range(40))
    left = _source("broken.py", "python", garbage, garbage + b"def &&&\n")
    right = _source("broken.py", "python", PYTHON_BASE, PYTHON_EDIT_VALUE)

    pair = pair_file_evidence(left, right, FileRole.SOURCE_AST)

    assert pair.status is ComparisonStatus.EVIDENCE_UNAVAILABLE
    assert pair.name_matched_entity_co_touch is None
    assert pair.syntax_identifier_string_overlap is None
    assert pair.entity_span_overlap is None


# --- structured files -------------------------------------------------------


JSON_BASE = b'''\
{
  "build": {
    "target": "es5",
    "strict": false
  },
  "test": {
    "runner": "vitest"
  }
}
'''


def test_structured_files_report_schema_paths_and_never_entity_fields() -> None:
    edited = JSON_BASE.replace(b'"es5"', b'"es2020"')
    left = _structured("tsconfig.json", "json", JSON_BASE, edited)
    right = _structured("tsconfig.json", "json", JSON_BASE, edited)

    pair = pair_file_evidence(left, right, FileRole.STRUCTURED_PARSE_TREE)

    assert left.after.entities == ()
    assert left.after.report.evidence_kind is EvidenceKind.SCHEMA_PATH
    assert pair.name_matched_entity_co_touch is None
    assert pair.schema_path_co_touch is True
    assert "build.target" in pair.matched_entities


def test_structured_files_touching_different_keys_do_not_co_touch() -> None:
    left = _structured("c.json", "json", JSON_BASE, JSON_BASE.replace(b'"es5"', b'"es2020"'))
    right = _structured("c.json", "json", JSON_BASE, JSON_BASE.replace(b'"vitest"', b'"jest"'))

    pair = pair_file_evidence(left, right, FileRole.STRUCTURED_PARSE_TREE)

    assert pair.schema_path_co_touch is False
    assert pair.name_matched_entity_co_touch is None


@pytest.mark.parametrize(
    ("language", "path", "source", "expected"),
    [
        ("toml", "pyproject.toml", b'[build]\ntarget = "es5"\n', "build.target"),
        ("hcl", "main.tf", b'resource "a" "b" {\n  count = 1\n}\n', "resource.a.b.count"),
        ("yaml", "ci.yml", b"build:\n  target: es5\n", "build.target"),
    ],
)
def test_formats_whose_key_is_not_a_named_field_still_yield_a_path(
    language: str, path: str, source: bytes, expected: str
) -> None:
    """TOML exposes no fields and HCL never assigns its declared `key` field."""
    evidence = _structured(path, language, source, source.replace(b"es5", b"es2020").replace(b"1", b"2"))

    assert expected in evidence.schema_keys


def test_a_structured_language_without_a_key_map_reports_no_entity_map() -> None:
    evidence = file_evidence(
        "a.xml", language="xml", grammar=_GRAMMAR_FOR["xml"], role=FileRole.STRUCTURED_PARSE_TREE,
        before=b"<r><a>1</a></r>\n", after=b"<r><a>2</a></r>\n",
    )

    assert evidence.after.report.evidence_kind is EvidenceKind.NO_ENTITY_MAP
    assert evidence.after.schema_paths == ()


# --- robustness-only fields -------------------------------------------------


def test_robustness_fields_are_reported_but_carry_no_dependency_status() -> None:
    left = _source("svc.py", "python", PYTHON_BASE, PYTHON_EDIT_VALUE)
    right = _source("svc.py", "python", PYTHON_BASE, PYTHON_EDIT_OTHER)

    pair = pair_file_evidence(left, right, FileRole.SOURCE_AST)

    assert pair.syntax_identifier_string_overlap is not None
    assert 0.0 <= pair.syntax_identifier_string_overlap <= 1.0
    assert pair.entity_span_overlap is not None
    assert 0.0 <= pair.entity_span_overlap <= 1.0


def test_span_overlap_is_none_when_neither_side_resolved_an_entity() -> None:
    before = b"import os\nCONST = 1\n"
    left = _source("m.py", "python", before, b"import os\nimport sys\nCONST = 1\n")
    right = _source("m.py", "python", before, b"import os\nimport io\nCONST = 1\n")

    pair = pair_file_evidence(left, right, FileRole.SOURCE_AST)

    assert pair.entity_span_overlap is None


# --- claim validation -------------------------------------------------------


@pytest.mark.parametrize(
    "phrase",
    ["shared variable", "data flow", "control flow", "call graph", "dependency entanglement", "alias", "def-use"],
)
def test_a_positive_claim_using_forbidden_vocabulary_is_rejected(phrase: str) -> None:
    with pytest.raises(ForbiddenClaimError) as raised:
        _ = validate_claim(f"Tangled pairs exhibit {phrase} between their concerns.")

    assert raised.value.phrase in FORBIDDEN_CLAIM_PHRASES


def test_a_disclaimer_of_the_same_vocabulary_is_allowed() -> None:
    text = "No data flow, control flow, call graph, alias, or def-use analysis was performed."

    assert validate_claim(text) == text


def test_a_permitted_claim_passes_unchanged() -> None:
    text = "In 41% of pairs both commits touched an entity with the same qualified name."

    assert validate_claim(text) == text


def test_validation_inspects_every_sentence_not_only_the_first() -> None:
    with pytest.raises(ForbiddenClaimError):
        _ = validate_claim("We report name co-touch only. This demonstrates data flow between concerns.")


# --- intervals --------------------------------------------------------------


def test_an_added_file_contributes_its_whole_surviving_side() -> None:
    intervals = changed_intervals(None, b"a\nb\n")

    assert intervals.before == ()
    assert intervals.after == (ByteRange(0, 4),)


def test_a_pure_insertion_keeps_the_deleting_side_attributable() -> None:
    intervals = changed_intervals(b"a\nc\n", b"a\nb\nc\n")

    assert len(intervals.before) == 1
    assert intervals.before[0].start == intervals.before[0].end == 2


def test_a_file_over_the_line_cap_is_reported_truncated_rather_than_dropped() -> None:
    before = b"x\n" * 50_001
    after = before + b"y\n"

    intervals = changed_intervals(before, after)

    assert intervals.truncated is True
    assert intervals.before == (ByteRange(0, len(before)),)


def test_truncation_is_carried_into_the_parse_report() -> None:
    before = b"x = 0\n" * 50_001
    evidence = _source("big.py", "python", before, before + b"y = 1\n")

    assert evidence.after.report.intervals_truncated is True


# --- grammar regression -----------------------------------------------------


@pytest.mark.parametrize("language", sorted(ENTITY_QUERIES))
def test_every_pinned_entity_node_and_field_still_exists_in_the_grammar(language: str) -> None:
    """A renamed node must fail here, not silently empty the denominator."""
    grammar = _GRAMMAR_FOR[language]
    parser_language = language_pack.get_language(grammar)
    specs = ENTITY_QUERIES[language]

    missing_nodes = [node for node in specs if parser_language.id_for_node_kind(node, True) is None]
    missing_fields = sorted(
        {
            field
            for spec in specs.values()
            for field in (spec.name_field, spec.parameters_field)
            if field is not None and parser_language.field_id_for_name(field) is None
        }
    )

    assert missing_nodes == []
    assert missing_fields == []


@pytest.mark.parametrize("language", sorted(STRUCTURED_KEY_NODES))
def test_every_pinned_structured_key_node_still_exists_in_the_grammar(language: str) -> None:
    parser_language = language_pack.get_language(_GRAMMAR_FOR[language])
    nodes, field = STRUCTURED_KEY_NODES[language]

    assert [node for node in nodes if parser_language.id_for_node_kind(node, True) is None] == []
    if field is not None:
        assert parser_language.field_id_for_name(field) is not None


def test_every_queried_language_is_one_the_detector_can_actually_emit() -> None:
    """A key the classifier never produces is a query that can never run."""
    known = {rule.language for rule in RULES}

    assert set(ENTITY_QUERIES) <= known
    assert set(STRUCTURED_KEY_NODES) <= known


@pytest.mark.parametrize(
    ("language", "source", "marker", "expected"),
    [
        ("rust", b"impl Store {\n    fn get(&self, k: u8) -> u8 {\n        k\n    }\n}\n", b"        k\n", "Store.get"),
        (
            "go",
            b"type Store struct {\n\tn int\n}\n\nfunc (s *Store) Get() int {\n\treturn s.n\n}\n",
            b"\treturn s.n\n",
            "Get",
        ),
        ("cpp", b"namespace ns {\nint free_fn(int a) {\n    return a;\n}\n}\n", b"    return a;\n", "ns.free_fn"),
        ("c", b"int free_fn(int a) {\n    return a;\n}\n", b"    return a;\n", "free_fn"),
        (
            "typescript",
            b"class S {\n    run(a: number) {\n        return a;\n    }\n}\n",
            b"        return a;\n",
            "S.run",
        ),
        ("bash", b"greet() {\n    echo hi\n}\n", b"    echo hi\n", "greet"),
        ("scss", b"@mixin theme($c) {\n    color: $c;\n}\n", b"    color: $c;\n", "theme"),
    ],
)
def test_the_non_obvious_name_paths_resolve_against_the_real_grammars(
    language: str, source: bytes, marker: bytes, expected: str
) -> None:
    """rust names impls via `type`, C buries the name under `function_declarator`."""
    start = source.index(marker)
    spans = (ByteRange(start, start + len(marker)),)

    evidence = side_evidence(_GRAMMAR_FOR[language], language, source, spans, entity_map(language))

    assert [entity.qualified_name for entity in evidence.entities] == [expected]
