"""AST and structured-tree proximity evidence for one path-matched file.

The observable is `shares_function`: within one file path touched by both atomic
commits of a pair, did each commit touch an entity whose normalized qualified
name, role, and arity agree. It reports that two edits landed on the same
*name*, nothing more. It is not function identity across history, and it is not
evidence that either edit depends on the other.

`shared_identifier_ratio` is a robustness field only -- a string comparison that
resolves no symbols. It is excluded from X/Y/Z and from the primary figure.
`claim_guard.validate_claim` polices the vocabulary of anything rendered from
these numbers.
"""

from __future__ import annotations

from .ast_entities import side_evidence
from .ast_intervals import changed_intervals
from .ast_queries import entity_map
from .ast_types import (
    ComparisonStatus,
    EvidenceKind,
    FileEvidence,
    PairFileEvidence,
    ParseStatus,
)
from .grammar_types import FileRole


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


def _identifier_overlap(left: FileEvidence, right: FileEvidence) -> float | None:
    first, second = left.identifier_set, right.identifier_set
    if not first or not second:
        return None
    return round(len(first & second) / len(first | second), 6)


def _inconclusive(path: str, status: ComparisonStatus) -> PairFileEvidence:
    """No comparison was possible, and the reason is carried rather than dropped."""
    return PairFileEvidence(path, status, None, None, (), None)


def pair_file_evidence(left: FileEvidence, right: FileEvidence, role: FileRole) -> PairFileEvidence:
    """Compare two atomic commits' evidence for one shared path.

    Comparison happens only within path-matched files, and only source-classified
    files may populate `shares_function`. A structured file carries schema-key
    proximity and leaves `shares_function` as None, so it can never be counted as
    a matched entity or as an unmatched one.
    """
    if left.path != right.path:
        return _inconclusive(left.path, ComparisonStatus.PATH_UNMATCHED)
    if role not in {FileRole.SOURCE_AST, FileRole.STRUCTURED_PARSE_TREE}:
        return _inconclusive(left.path, ComparisonStatus.ROLE_UNMATCHED)
    if not (_usable(left) and _usable(right)):
        return _inconclusive(left.path, ComparisonStatus.EVIDENCE_UNAVAILABLE)
    if role is FileRole.STRUCTURED_PARSE_TREE:
        shared_keys = left.schema_keys & right.schema_keys
        return PairFileEvidence(
            left.path, ComparisonStatus.COMPARED, None, bool(shared_keys), tuple(sorted(shared_keys)), None
        )
    matched = left.entity_keys & right.entity_keys
    return PairFileEvidence(
        left.path,
        ComparisonStatus.COMPARED,
        bool(matched),
        None,
        tuple(sorted(name for name, _, _ in matched)),
        _identifier_overlap(left, right),
    )


__all__ = [
    "ComparisonStatus",
    "EvidenceKind",
    "FileEvidence",
    "PairFileEvidence",
    "ParseStatus",
    "file_evidence",
    "pair_file_evidence",
]
