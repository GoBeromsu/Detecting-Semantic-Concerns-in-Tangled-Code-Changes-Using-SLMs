"""Deterministic language detection over filename, content, and parser score.

Detector precedence is exact filename, extension, shebang, content signature,
then bounded parser scoring. Scoring only ever breaks a tie between candidates
a content signature already proposed, so a permissive parser accepting ordinary
prose can never promote an ambiguous file into a language.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Final

from .grammar_rules import (
    BINARY_EXTENSIONS,
    CONTENT_SIGNATURES,
    EXACT_TEXT_NAMES,
    GENERATED_EXTENSIONS,
    RULE_BY_EXTENSION,
    RULE_BY_LANGUAGE,
    SCORE_LIMIT,
    SCORE_MARGIN,
    TEXT_EXTENSIONS,
    exact_rule,
)
from .grammar_runtime import TreeSitterRuntime
from .grammar_types import (
    DetectionMethod,
    FileClassification,
    FileRole,
    GrammarRule,
    GrammarRuntime,
    ParserDisposition,
)

DETECTOR_ORDER: Final = tuple(
    method.value for method in DetectionMethod if method is not DetectionMethod.BINARY_CONTENT
)
SHEBANG_PATTERN: Final = re.compile(r"^#!.*\b(python(?:\d+(?:\.\d+)*)?|node|bash|sh|ruby)\b")
INTERPRETER_LANGUAGES: Final = {"node": "javascript", "sh": "bash"}


def _is_binary(source: bytes) -> bool:
    sample = source[:8192]
    return b"\x00" in sample or sample.startswith((b"\x89PNG", b"GIF8", b"\x7fELF"))


def _content_candidates(source: bytes) -> tuple[tuple[str, ...], DetectionMethod] | None:
    first_line = source.splitlines()[0].decode("utf-8", errors="ignore") if source else ""
    shebang = SHEBANG_PATTERN.match(first_line)
    if shebang:
        interpreter = re.sub(r"\d+(?:\.\d+)*$", "", shebang.group(1))
        return ((INTERPRETER_LANGUAGES.get(interpreter, interpreter),), DetectionMethod.SHEBANG)
    text = source[:32768].decode("utf-8", errors="ignore")
    for pattern, candidates in CONTENT_SIGNATURES:
        if re.search(pattern, text):
            return (candidates, DetectionMethod.CONTENT_SIGNATURE)
    return None


def parser_disposition(rule: GrammarRule, runtime: GrammarRuntime) -> ParserDisposition:
    if rule.grammar is None:
        return ParserDisposition.NOT_APPLICABLE
    if runtime.grammar_abi(rule.grammar) is not None:
        return ParserDisposition.AVAILABLE
    if rule.role is FileRole.STRUCTURED_PARSE_TREE:
        return ParserDisposition.UNAVAILABLE_STRUCTURED
    return ParserDisposition.UNSUPPORTED_SOURCE


def _classified(path: str, rule: GrammarRule, method: DetectionMethod, runtime: GrammarRuntime) -> FileClassification:
    return FileClassification(path, rule.language, rule.grammar, rule.role, method, parser_disposition(rule, runtime))


def _scored_rule(candidates: tuple[str, ...], source: bytes, runtime: GrammarRuntime) -> GrammarRule | None:
    registered = tuple(
        rule
        for candidate in sorted(set(candidates))
        if (rule := RULE_BY_LANGUAGE.get(candidate)) is not None and rule.grammar is not None
    )[:SCORE_LIMIT]
    scores = [(rule, runtime.score(rule.grammar, source)) for rule in registered if rule.grammar is not None]
    scored = sorted(
        ((rule, score) for rule, score in scores if score is not None),
        key=lambda item: (item[1].error_ratio, -item[1].named_nodes, item[0].language),
    )
    if scored and (len(scored) == 1 or scored[1][1].error_ratio - scored[0][1].error_ratio >= SCORE_MARGIN):
        return scored[0][0]
    return None


def classify_file(path: str, source: bytes, *, runtime: GrammarRuntime | None = None) -> FileClassification:
    active_runtime = runtime or TreeSitterRuntime()
    name = PurePosixPath(path).name
    extension = PurePosixPath(name).suffix.lower()
    if _is_binary(source) or extension in BINARY_EXTENSIONS or extension in GENERATED_EXTENSIONS or name.endswith(".min.js"):
        return FileClassification(path, None, None, FileRole.BINARY_GENERATED, DetectionMethod.BINARY_CONTENT, ParserDisposition.NOT_APPLICABLE)
    exact = exact_rule(name)
    if exact is not None:
        return _classified(path, exact, DetectionMethod.EXACT_FILENAME, active_runtime)
    extension_rule = RULE_BY_EXTENSION.get(extension)
    if extension_rule is not None:
        return _classified(path, extension_rule, DetectionMethod.EXTENSION, active_runtime)
    content_match = _content_candidates(source)
    if content_match is not None:
        candidates, method = content_match
        if len(candidates) == 1 and (rule := RULE_BY_LANGUAGE.get(candidates[0])) is not None:
            return _classified(path, rule, method, active_runtime)
        winner = _scored_rule(candidates, source, active_runtime)
        if winner is not None:
            return _classified(path, winner, DetectionMethod.PARSER_SCORE, active_runtime)
    if extension in TEXT_EXTENSIONS or name in EXACT_TEXT_NAMES:
        return FileClassification(path, None, None, FileRole.TEXT_ONLY, DetectionMethod.EXTENSION, ParserDisposition.NOT_APPLICABLE)
    return FileClassification(path, None, None, FileRole.UNRESOLVED_SOURCE, DetectionMethod.AMBIGUOUS, ParserDisposition.NOT_APPLICABLE)
