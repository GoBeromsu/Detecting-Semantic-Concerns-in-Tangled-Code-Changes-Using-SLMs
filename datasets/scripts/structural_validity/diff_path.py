"""Git path tokenization and C-style quoted-path decoding."""

from __future__ import annotations

import re
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Final

DEV_NULL: Final[str] = "/dev/null"
SPACED_AB_RE: Final[re.Pattern[str]] = re.compile(r"^(a/.+) (b/.+)$")
_OCTAL_DIGITS: Final[str] = "01234567"
_ESCAPE_BYTES: Final[dict[str, int]] = {
    "a": 0x07,
    "b": 0x08,
    "t": 0x09,
    "n": 0x0A,
    "v": 0x0B,
    "f": 0x0C,
    "r": 0x0D,
}


def dirname_of(path: str) -> str:
    """Full dirname of a git path (forward-slash-separated)."""
    index = path.rfind("/")
    return path[:index] if index >= 0 else ""


def unquote_git_path(raw: str) -> str:
    """Decode a git path token, including C-style quoted octal UTF-8 bytes."""
    value = raw.strip()
    if len(value) < 2 or value[0] != '"' or value[-1] != '"':
        return value
    return _decode_c_style_bytes(value[1:-1]).decode("utf-8")


def strip_ab_prefix(path: str) -> str | None:
    """Strip a leading ``a/`` or ``b/`` marker after unquoting."""
    cleaned = unquote_git_path(path)
    if cleaned == DEV_NULL:
        return DEV_NULL
    if cleaned.startswith("a/") or cleaned.startswith("b/"):
        return cleaned[2:]
    return cleaned


def parse_diff_git_paths(header_line: str) -> tuple[str | None, str | None]:
    """Parse ``diff --git`` old/new path tokens."""
    if not header_line.startswith("diff --git "):
        return None, None
    payload = header_line[len("diff --git ") :]
    tokens = tokenize_diff_paths(payload)
    if len(tokens) == 2:
        return strip_ab_prefix(tokens[0]), strip_ab_prefix(tokens[1])
    match = SPACED_AB_RE.match(payload)
    if match is None:
        return None, None
    return strip_ab_prefix(match.group(1)), strip_ab_prefix(match.group(2))


def paths_from_minus_plus_headers(
    lines: list[str], old_path: str | None, new_path: str | None
) -> tuple[str | None, str | None]:
    """Fill missing sides from ``---`` / ``+++`` headers when present."""
    resolved_old, resolved_new = old_path, new_path
    for line in lines[1:]:
        if line.startswith("@@") or line.startswith("diff --git "):
            break
        if line.startswith("--- "):
            resolved_old = strip_ab_prefix(line[4:].strip()) or resolved_old
        elif line.startswith("+++ "):
            resolved_new = strip_ab_prefix(line[4:].strip()) or resolved_new
    return resolved_old, resolved_new


def tokenize_diff_paths(payload: str) -> list[str]:
    """Split a ``diff --git`` payload into at most two path tokens."""
    tokens: list[str] = []
    index = 0
    length = len(payload)
    while index < length:
        while index < length and payload[index].isspace():
            index += 1
        if index >= length:
            break
        if payload[index] == '"':
            end = _find_closing_quote(payload, index + 1)
            tokens.append(payload[index : end + 1] if end >= 0 else payload[index:])
            index = end + 1 if end >= 0 else length
            continue
        start = index
        while index < length and not payload[index].isspace():
            index += 1
        tokens.append(payload[start:index])
    return tokens


def _find_closing_quote(payload: str, start: int) -> int:
    index = start
    length = len(payload)
    while index < length:
        char = payload[index]
        if char == "\\" and index + 1 < length:
            index += 2
            continue
        if char == '"':
            return index
        index += 1
    return -1


@dataclass(frozen=True, slots=True)
class ExtendedHeaders:
    is_new_file: bool = False
    is_deleted_file: bool = False
    is_rename: bool = False
    is_copy: bool = False
    is_binary: bool = False
    rename_from: str | None = None
    rename_to: str | None = None
    copy_from: str | None = None
    copy_to: str | None = None


def scan_extended_headers(lines: Sequence[str]) -> ExtendedHeaders:
    """Read rename/copy/new/delete/binary markers before the first hunk."""
    is_new_file = False
    is_deleted_file = False
    is_rename = False
    is_copy = False
    is_binary = False
    rename_from: str | None = None
    rename_to: str | None = None
    copy_from: str | None = None
    copy_to: str | None = None
    for line in lines:
        if line.startswith("@@") or line.startswith("diff --git "):
            break
        if line.startswith("new file mode"):
            is_new_file = True
        elif line.startswith("deleted file mode"):
            is_deleted_file = True
        elif line.startswith("rename from "):
            is_rename = True
            rename_from = unquote_git_path(line[len("rename from ") :])
        elif line.startswith("rename to "):
            is_rename = True
            rename_to = unquote_git_path(line[len("rename to ") :])
        elif line.startswith("copy from "):
            is_copy = True
            copy_from = unquote_git_path(line[len("copy from ") :])
        elif line.startswith("copy to "):
            is_copy = True
            copy_to = unquote_git_path(line[len("copy to ") :])
        elif line.startswith("Binary files ") or line.startswith("GIT binary patch"):
            is_binary = True
    return ExtendedHeaders(
        is_new_file=is_new_file,
        is_deleted_file=is_deleted_file,
        is_rename=is_rename,
        is_copy=is_copy,
        is_binary=is_binary,
        rename_from=rename_from,
        rename_to=rename_to,
        copy_from=copy_from,
        copy_to=copy_to,
    )


def _decode_c_style_bytes(payload: str) -> bytes:
    """Git quote.c unquote: octal escapes are raw bytes, then UTF-8 text."""
    out = bytearray()
    index = 0
    length = len(payload)
    while index < length:
        char = payload[index]
        if char != "\\":
            out.extend(char.encode("utf-8"))
            index += 1
            continue
        index += 1
        if index >= length:
            break
        esc = payload[index]
        if esc in _OCTAL_DIGITS:
            value = 0
            digits = 0
            while index < length and payload[index] in _OCTAL_DIGITS and digits < 3:
                value = (value << 3) + int(payload[index], 8)
                index += 1
                digits += 1
            out.append(value & 0xFF)
            continue
        if esc in _ESCAPE_BYTES:
            out.append(_ESCAPE_BYTES[esc])
        else:
            out.extend(esc.encode("utf-8"))
        index += 1
    return bytes(out)
