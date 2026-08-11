"""Maximum-coverage AST and structured-tree proximity evidence.

The primary observable is `name_matched_entity_co_touch`: within one file path
touched by both atomic commits of a pair, did each commit touch an entity whose
normalized qualified name, role, and arity agree. It reports that two edits
landed on the same *name*, nothing more. It is not function identity across
history, and it is not evidence that either edit depends on the other.

`syntax_identifier_string_overlap` and `entity_span_overlap` are robustness
fields only. They are string and interval comparisons; they resolve no symbols.
Both are excluded from X/Y/Z and from the primary figure, and `validate_claim`
refuses any rendered sentence that would dress either of them up as data flow,
control flow, a call graph, an alias, a dependency, or def-use.
"""

from __future__ import annotations

import re
from typing import Final

from .ast_entities import side_evidence
from .ast_intervals import changed_intervals
from .ast_queries import entity_map
from .ast_types import (
    ComparisonStatus,
    EvidenceKind,
    FileEvidence,
    ForbiddenClaimError,
    PairFileEvidence,
    ParseStatus,
)
from .grammar_types import FileRole

FORBIDDEN_CLAIM_PHRASES: Final = (
    "shared variable",
    "data flow",
    "dataflow",
    "control flow",
    "controlflow",
    "call graph",
    "callgraph",
    "dependency entanglement",
    "alias",
    "def-use",
    "def use",
)
_NEGATION: Final = re.compile(r"\b(no|not|never|without|cannot|does not|do not|excludes?|neither|nor)\b")


def validate_claim(text: str) -> str:
    """Reject a claim that asserts a relationship this evidence cannot support.

    Only positive claims are rejected. A sentence is allowed to *disclaim* the
    vocabulary -- the manuscript has to be able to say that no data-flow
    analysis was performed -- so a negation in the same sentence clears it.
    """
    for sentence in re.split(r"(?<=[.;:])\s+|\n", text):
        lowered = sentence.lower()
        if _NEGATION.search(lowered):
            continue
        for phrase in FORBIDDEN_CLAIM_PHRASES:
            if phrase in lowered:
                raise ForbiddenClaimError(phrase, sentence.strip())
    return text


def file_evidence(
    path: str,
    *,
    language: str | None,
    grammar: str | None,
    role: FileRole,
    before: bytes | None,
    after: bytes | None,
) -> FileEvidence:
    """Evidence for both revisions of one changed path.

    Exactly one disposition per side: a parser runs when the pinned grammar is
    available, otherwise the side carries `unsupported_source` or
    `not_attempted` and no invocation is faked.
    """
    structured = role is FileRole.STRUCTURED_PARSE_TREE
    if role not in {FileRole.SOURCE_AST, FileRole.STRUCTURED_PARSE_TREE}:
        # Each side still sees its own payload, so a text file that exists on a
        # revision reports `not_attempted` rather than `side_absent`: the role
        # carries no parser, which is not the same fact as a missing revision.
        return FileEvidence(
            path,
            language,
            grammar,
            side_evidence(None, language, before, (), None),
            side_evidence(None, language, after, (), None),
        )
    intervals = changed_intervals(before, after)
    entities = None if structured else entity_map(language)
    return FileEvidence(
        path,
        language,
        grammar,
        side_evidence(
            grammar, language, before, intervals.before, entities,
            structured=structured, truncated=intervals.truncated,
        ),
        side_evidence(
            grammar, language, after, intervals.after, entities,
            structured=structured, truncated=intervals.truncated,
        ),
    )


def _usable(evidence: FileEvidence) -> bool:
    return any(side.report.status is ParseStatus.PARSED for side in (evidence.before, evidence.after))


def _span_overlap(left: FileEvidence, right: FileEvidence) -> float | None:
    """Fraction of the smaller entity-span union the two sides share.

    Interval arithmetic only. Two edits overlapping in byte space says nothing
    about whether either reads what the other writes.
    """
    left_spans = [(e.start_byte, e.end_byte) for side in (left.before, left.after) for e in side.entities]
    right_spans = [(e.start_byte, e.end_byte) for side in (right.before, right.after) for e in side.entities]
    if not left_spans or not right_spans:
        return None
    shared = sum(
        max(0, min(a_end, b_end) - max(a_start, b_start))
        for a_start, a_end in left_spans
        for b_start, b_end in right_spans
    )
    smaller = min(sum(end - start for start, end in left_spans), sum(end - start for start, end in right_spans))
    return round(min(1.0, shared / smaller), 6) if smaller else None


def _identifier_overlap(left: FileEvidence, right: FileEvidence) -> float | None:
    first, second = left.identifier_set, right.identifier_set
    if not first or not second:
        return None
    return round(len(first & second) / len(first | second), 6)


def pair_file_evidence(left: FileEvidence, right: FileEvidence, role: FileRole) -> PairFileEvidence:
    """Compare two atomic commits' evidence for one shared path.

    Comparison happens only within path-matched files, and only source-classified
    files may populate the entity fields. A structured file carries schema-path
    proximity and leaves `name_matched_entity_co_touch` as None, so it can never
    be counted as a matched entity or as an unmatched one.
    """
    if left.path != right.path:
        return PairFileEvidence(left.path, ComparisonStatus.PATH_UNMATCHED, None, None, (), None, None)
    if role is FileRole.STRUCTURED_PARSE_TREE:
        if not (_usable(left) and _usable(right)):
            return PairFileEvidence(left.path, ComparisonStatus.EVIDENCE_UNAVAILABLE, None, None, (), None, None)
        shared_keys = left.schema_keys & right.schema_keys
        return PairFileEvidence(
            left.path, ComparisonStatus.COMPARED, None, bool(shared_keys), tuple(sorted(shared_keys)), None, None
        )
    if role is not FileRole.SOURCE_AST:
        return PairFileEvidence(left.path, ComparisonStatus.ROLE_UNMATCHED, None, None, (), None, None)
    if not (_usable(left) and _usable(right)):
        return PairFileEvidence(left.path, ComparisonStatus.EVIDENCE_UNAVAILABLE, None, None, (), None, None)
    matched = left.entity_keys & right.entity_keys
    return PairFileEvidence(
        left.path,
        ComparisonStatus.COMPARED,
        bool(matched),
        None,
        tuple(sorted(name for name, _, _ in matched)),
        _identifier_overlap(left, right),
        _span_overlap(left, right),
    )


__all__ = [
    "FORBIDDEN_CLAIM_PHRASES",
    "ComparisonStatus",
    "EvidenceKind",
    "FileEvidence",
    "ForbiddenClaimError",
    "PairFileEvidence",
    "ParseStatus",
    "file_evidence",
    "pair_file_evidence",
    "validate_claim",
]
