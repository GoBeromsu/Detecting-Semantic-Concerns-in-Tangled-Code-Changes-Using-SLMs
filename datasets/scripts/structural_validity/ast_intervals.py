"""Changed byte intervals between two revisions of one file.

Intervals are derived from the two whole blobs rather than from unified diff
text, so they stay correct for files whose committed diff was truncated, and so
an added or deleted file yields the whole surviving side rather than nothing.

Line-oriented matching is deliberate: byte-level opcodes would split a single
edited statement into several unrelated intervals and inflate the number of
entities a change appears to touch.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher

from .ast_types import MAX_DIFF_LINES


@dataclass(frozen=True, slots=True)
class ByteRange:
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class ChangedIntervals:
    before: tuple[ByteRange, ...]
    after: tuple[ByteRange, ...]
    truncated: bool


def _line_offsets(payload: bytes) -> tuple[list[bytes], list[int]]:
    lines = payload.splitlines(keepends=True)
    offsets: list[int] = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    return lines, offsets


def _span(offsets: list[int], start: int, end: int) -> ByteRange:
    """Byte range of lines [start, end), widened to a caret when the run is empty.

    A pure insertion has an empty range on the deleting side; collapsing it to a
    zero-width point at the junction keeps that side attributable to the
    enclosing entity instead of dropping it.
    """
    last = len(offsets) - 1
    low = offsets[min(start, last)]
    high = offsets[min(max(end, start), last)]
    return ByteRange(low, high)


def changed_intervals(before: bytes | None, after: bytes | None) -> ChangedIntervals:
    """Byte ranges touched on each side; a missing side contributes nothing."""
    if before is None and after is None:
        return ChangedIntervals((), (), False)
    if before is None:
        return ChangedIntervals((), (ByteRange(0, len(after or b"")),), False)
    if after is None:
        return ChangedIntervals((ByteRange(0, len(before)),), (), False)
    before_lines, before_offsets = _line_offsets(before)
    after_lines, after_offsets = _line_offsets(after)
    if max(len(before_lines), len(after_lines)) > MAX_DIFF_LINES:
        return ChangedIntervals(
            (ByteRange(0, len(before)),),
            (ByteRange(0, len(after)),),
            True,
        )
    matcher = SequenceMatcher(None, before_lines, after_lines, autojunk=False)
    before_ranges: list[ByteRange] = []
    after_ranges: list[ByteRange] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        before_ranges.append(_span(before_offsets, i1, i2))
        after_ranges.append(_span(after_offsets, j1, j2))
    return ChangedIntervals(tuple(before_ranges), tuple(after_ranges), False)
