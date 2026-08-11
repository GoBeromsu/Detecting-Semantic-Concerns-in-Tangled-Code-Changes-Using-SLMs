"""Public path/hunk/line metrics API for structural validity (Todo 3)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias

from .diff_parse import (
    AtomicDiffEvidence,
    ChangeStatus,
    DiffReasonCode,
    FileDiffEvidence,
    LegacyMap,
    LegacyValue,
    PathIdentity,
    ReasonCode,
    file_path_from_block,
    parse_committed_diff,
    split_diff_into_files,
)
from .diff_path import dirname_of

__all__ = (
    "AtomicDiffEvidence",
    "ChangeStatus",
    "FileDiffEvidence",
    "PairDiffMetrics",
    "PathIdentity",
    "ReasonCode",
    "dirname_of",
    "file_path_from_block",
    "pair_colocation",
    "pair_diff_metrics",
    "parse_atomic_diff",
    "parse_committed_diff",
    "split_diff_into_files",
)


@dataclass(frozen=True, slots=True)
class PairDiffMetrics:
    same_dir: bool
    same_file: bool
    same_function: bool
    min_line_gap: int | None
    diff_local_normalized_gap: float | None
    legacy_hunk_start_gap: int | None
    same_line: bool
    reason_codes: tuple[DiffReasonCode, ...]


LegacyInput: TypeAlias = Mapping[str, Mapping[str, LegacyValue]]


def parse_atomic_diff(diff_text: str) -> LegacyMap:
    """Backward-compatible parser returning ``contexts``/``starts`` per path."""
    return parse_committed_diff(diff_text).as_legacy_map()


def pair_diff_metrics(atomic_a: AtomicDiffEvidence, atomic_b: AtomicDiffEvidence) -> PairDiffMetrics:
    """Compute path/hunk/line co-location for one concern pair."""
    files_a = atomic_a.by_headline_path()
    files_b = atomic_b.by_headline_path()
    shared = frozenset(files_a) & frozenset(files_b)
    same_dir = bool({dirname_of(path) for path in files_a} & {dirname_of(path) for path in files_b})
    same_file = bool(shared)
    same_function = False
    same_line = False
    line_gaps: list[int] = []
    legacy_gaps: list[int] = []
    max_new_lines: list[int] = []
    reasons: list[DiffReasonCode] = []

    for path in shared:
        left, right = files_a[path], files_b[path]
        same_function = same_function or bool(left.contexts & right.contexts)
        same_line = same_line or bool(left.new_side_lines & right.new_side_lines)
        if left.new_side_lines and right.new_side_lines:
            line_gaps.extend(abs(a - b) for a in left.new_side_lines for b in right.new_side_lines)
            if left.max_new_line is not None:
                max_new_lines.append(left.max_new_line)
            if right.max_new_line is not None:
                max_new_lines.append(right.max_new_line)
        else:
            reasons.append(DiffReasonCode.NO_NEW_SIDE_LINES)
        if left.hunk_starts and right.hunk_starts:
            legacy_gaps.extend(abs(a - b) for a in left.hunk_starts for b in right.hunk_starts)
        if left.is_binary or right.is_binary:
            reasons.append(DiffReasonCode.BINARY)

    min_gap = min(line_gaps) if line_gaps else None
    if min_gap is None:
        normalized: float | None = None
    else:
        normalized = min_gap / max(1, max(max_new_lines) if max_new_lines else 1)
    return PairDiffMetrics(
        same_dir=same_dir,
        same_file=same_file,
        same_function=same_function,
        min_line_gap=min_gap,
        diff_local_normalized_gap=normalized,
        legacy_hunk_start_gap=min(legacy_gaps) if legacy_gaps else None,
        same_line=same_line,
        reason_codes=tuple(dict.fromkeys(reasons)),
    )


def pair_colocation(
    atomic_a: LegacyInput | AtomicDiffEvidence,
    atomic_b: LegacyInput | AtomicDiffEvidence,
) -> dict[str, bool | int | None]:
    """Backward-compatible pair schema used by ``analyze_colocation``."""
    metrics = pair_diff_metrics(coerce_evidence(atomic_a), coerce_evidence(atomic_b))
    return {
        "same_dir": metrics.same_dir,
        "same_file": metrics.same_file,
        "same_function": metrics.same_function,
        "min_line_gap": metrics.min_line_gap,
    }


def coerce_evidence(
    value: LegacyInput | AtomicDiffEvidence,
) -> AtomicDiffEvidence:
    """Accept rich evidence or the legacy ``{path: {contexts, starts}}`` map."""
    if isinstance(value, AtomicDiffEvidence):
        return value
    files: list[FileDiffEvidence] = []
    for path, payload in value.items():
        contexts = _as_str_set(payload.get("contexts"))
        starts = _as_int_tuple(payload.get("starts"))
        explicit_lines = _as_int_frozenset(payload.get("new_side_lines"))
        new_side = explicit_lines if explicit_lines else frozenset(starts)
        files.append(
            FileDiffEvidence(
                identity=PathIdentity(str(path), str(path), str(path), ChangeStatus.ORDINARY),
                contexts=contexts,
                hunk_starts=starts,
                new_side_lines=new_side,
                is_binary=False,
                max_new_line=max(new_side) if new_side else None,
            )
        )
    return AtomicDiffEvidence(files=tuple(files))


def _as_str_set(value: LegacyValue | None) -> frozenset[str]:
    if not isinstance(value, (set, frozenset, list, tuple)):
        return frozenset()
    return frozenset(str(item) for item in value)


def _as_int_tuple(value: LegacyValue | None) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(_as_int(item) for item in value)


def _as_int_frozenset(value: LegacyValue | None) -> frozenset[int]:
    if not isinstance(value, (set, frozenset, list, tuple)) or not value:
        return frozenset()
    return frozenset(_as_int(item) for item in value)


def _as_int(value: str | int) -> int:
    if isinstance(value, int):
        return value
    return int(value)
