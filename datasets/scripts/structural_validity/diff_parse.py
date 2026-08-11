"""Per-file unified-diff block parsing into typed path/hunk/line evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from collections.abc import Sequence
from typing import Final, TypeAlias

from .diff_path import (
    DEV_NULL,
    ExtendedHeaders,
    parse_diff_git_paths,
    paths_from_minus_plus_headers,
    scan_extended_headers,
)

DIFF_FILE_SPLIT_RE: Final[re.Pattern[str]] = re.compile(r"(?=^diff --git )", re.MULTILINE)
HUNK_HEADER_RE: Final[re.Pattern[str]] = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?:[ \t](.*))?$"
)

class ChangeStatus(str, Enum):
    ORDINARY = "ordinary"
    ADD = "add"
    DELETE = "delete"
    RENAME = "rename"
    COPY = "copy"
    BINARY = "binary"
    AMBIGUOUS_PATH = "ambiguous_path"

class DiffReasonCode(str, Enum):
    AMBIGUOUS_PATH = "ambiguous_path"
    BINARY = "binary"
    NO_NEW_SIDE_LINES = "no_new_side_lines"
    EMPTY_DIFF = "empty_diff"

# Public alias kept for callers that imported ReasonCode from diff_metrics.
ReasonCode = DiffReasonCode

LegacyValue: TypeAlias = (
    set[str]
    | frozenset[str]
    | list[str]
    | tuple[str, ...]
    | set[int]
    | frozenset[int]
    | list[int]
    | tuple[int, ...]
)
LegacyMap: TypeAlias = dict[str, dict[str, LegacyValue]]

@dataclass(frozen=True, slots=True)
class PathIdentity:
    path: str | None
    old_path: str | None
    new_path: str | None
    status: ChangeStatus
    reason_codes: tuple[DiffReasonCode, ...] = ()

@dataclass(frozen=True, slots=True)
class FileDiffEvidence:
    identity: PathIdentity
    contexts: frozenset[str]
    hunk_starts: tuple[int, ...]
    new_side_lines: frozenset[int]
    is_binary: bool
    max_new_line: int | None

@dataclass(frozen=True, slots=True)
class AtomicDiffEvidence:
    files: tuple[FileDiffEvidence, ...]
    reason_codes: tuple[DiffReasonCode, ...] = ()

    def headline_paths(self) -> frozenset[str]:
        return frozenset(
            item.identity.path for item in self.files if item.identity.path is not None
        )

    def by_headline_path(self) -> dict[str, FileDiffEvidence]:
        mapping: dict[str, FileDiffEvidence] = {}
        for item in self.files:
            path = item.identity.path
            if path is None:
                continue
            prior = mapping.get(path)
            mapping[path] = item if prior is None else merge_file_evidence(prior, item)
        return mapping

    def as_legacy_map(self) -> LegacyMap:
        return {
            path: {"contexts": set(item.contexts), "starts": list(item.hunk_starts)}
            for path, item in self.by_headline_path().items()
        }

def split_diff_into_files(diff_text: str) -> list[str]:
    """Split one combined unified diff into per-file blocks."""
    return [
        block
        for block in DIFF_FILE_SPLIT_RE.split(diff_text)
        if block.lstrip().startswith("diff --git ")
    ]

def file_path_from_block(block: str) -> str | None:
    """Compatibility helper returning the headline path for one file block."""
    return parse_file_block(block).identity.path

def parse_committed_diff(diff_text: str) -> AtomicDiffEvidence:
    """Parse committed unified-diff text into typed path/hunk/line evidence."""
    if not diff_text or not diff_text.strip():
        return AtomicDiffEvidence(files=(), reason_codes=(DiffReasonCode.EMPTY_DIFF,))
    files = tuple(parse_file_block(block) for block in split_diff_into_files(diff_text))
    if not files:
        return AtomicDiffEvidence(files=(), reason_codes=(DiffReasonCode.EMPTY_DIFF,))
    return AtomicDiffEvidence(files=files)

def parse_file_block(block: str) -> FileDiffEvidence:
    """Parse one ``diff --git`` file block."""
    lines = block.splitlines()
    if not lines:
        return ambiguous_file()
    old_path, new_path = parse_diff_git_paths(lines[0])
    extended = scan_extended_headers(lines[1:])
    is_binary = extended.is_binary or any(
        line.startswith("Binary files ") or line.startswith("GIT binary patch") for line in lines
    )
    old_path = extended.rename_from or extended.copy_from or old_path
    new_path = extended.rename_to or extended.copy_to or new_path
    old_path, new_path = paths_from_minus_plus_headers(lines, old_path, new_path)
    status = classify_status(old_path, new_path, extended, is_binary)
    if status is ChangeStatus.AMBIGUOUS_PATH:
        return ambiguous_file(is_binary=is_binary)
    headline = headline_path(status, old_path, new_path)
    if headline is None:
        return ambiguous_file(is_binary=is_binary)
    contexts, hunk_starts, parsed_lines = parse_hunks(lines)
    reasons: tuple[DiffReasonCode, ...] = (DiffReasonCode.BINARY,) if is_binary else ()
    if is_binary:
        status = ChangeStatus.BINARY
        new_side_lines: frozenset[int] = frozenset()
    else:
        new_side_lines = parsed_lines
    return FileDiffEvidence(
        identity=PathIdentity(headline, old_path, new_path, status, reasons),
        contexts=contexts,
        hunk_starts=hunk_starts,
        new_side_lines=new_side_lines,
        is_binary=is_binary,
        max_new_line=None if is_binary or not new_side_lines else max(new_side_lines),
    )

def classify_status(
    old_path: str | None,
    new_path: str | None,
    extended: ExtendedHeaders,
    is_binary: bool,
) -> ChangeStatus:
    if old_path is None and new_path is None:
        return ChangeStatus.AMBIGUOUS_PATH
    if extended.is_rename:
        return ChangeStatus.RENAME
    if extended.is_copy:
        return ChangeStatus.COPY
    if extended.is_deleted_file or (
        new_path in {None, DEV_NULL} and old_path not in {None, DEV_NULL}
    ):
        return ChangeStatus.DELETE
    if extended.is_new_file or (
        old_path in {None, DEV_NULL} and new_path not in {None, DEV_NULL}
    ):
        return ChangeStatus.ADD
    if is_binary:
        return ChangeStatus.BINARY
    if old_path is None or new_path is None:
        return ChangeStatus.AMBIGUOUS_PATH
    return ChangeStatus.ORDINARY

def headline_path(
    status: ChangeStatus, old_path: str | None, new_path: str | None
) -> str | None:
    if status is ChangeStatus.DELETE:
        return None if old_path in {None, DEV_NULL} else old_path
    candidate = new_path if new_path not in {None, DEV_NULL} else old_path
    return None if candidate in {None, DEV_NULL} else candidate

def parse_hunks(lines: Sequence[str]) -> tuple[frozenset[str], tuple[int, ...], frozenset[int]]:
    contexts: set[str] = set()
    hunk_starts: list[int] = []
    new_side_lines: set[int] = set()
    index = 0
    while index < len(lines):
        match = HUNK_HEADER_RE.match(lines[index])
        if match is None:
            index += 1
            continue
        new_start = int(match.group(3))
        new_count = int(match.group(4) or "1")
        context = (match.group(5) or "").strip()
        hunk_starts.append(new_start)
        if context:
            contexts.add(context)
        index += 1
        new_line = new_start
        while index < len(lines):
            line = lines[index]
            if line.startswith("diff --git ") or HUNK_HEADER_RE.match(line):
                break
            if line.startswith("\\"):
                index += 1
                continue
            if line.startswith("+") and not line.startswith("+++"):
                new_side_lines.add(new_line)
                new_line += 1
            elif line.startswith("-") and not line.startswith("---"):
                pass
            else:
                new_line += 1
            index += 1
            if new_count == 0 and not (line.startswith("-") or line.startswith("\\")):
                break
    return frozenset(contexts), tuple(hunk_starts), frozenset(new_side_lines)

def ambiguous_file(*, is_binary: bool = False) -> FileDiffEvidence:
    return FileDiffEvidence(
        identity=PathIdentity(
            None, None, None, ChangeStatus.AMBIGUOUS_PATH, (DiffReasonCode.AMBIGUOUS_PATH,)
        ),
        contexts=frozenset(),
        hunk_starts=(),
        new_side_lines=frozenset(),
        is_binary=is_binary,
        max_new_line=None,
    )

def merge_file_evidence(left: FileDiffEvidence, right: FileDiffEvidence) -> FileDiffEvidence:
    new_side = left.new_side_lines | right.new_side_lines
    maxima = [value for value in (left.max_new_line, right.max_new_line) if value is not None]
    if new_side:
        maxima.append(max(new_side))
    return FileDiffEvidence(
        identity=left.identity,
        contexts=left.contexts | right.contexts,
        hunk_starts=left.hunk_starts + right.hunk_starts,
        new_side_lines=new_side,
        is_binary=left.is_binary or right.is_binary,
        max_new_line=max(maxima) if maxima else None,
    )
