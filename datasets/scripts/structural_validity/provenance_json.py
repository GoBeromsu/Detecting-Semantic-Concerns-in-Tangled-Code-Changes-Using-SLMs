from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from structural_validity.provenance_types import MetadataReason, MetadataValidationError


def _fail(location: str, detail: str) -> MetadataValidationError:
    return MetadataValidationError(MetadataReason.MALFORMED_JSON, location, detail)


def _parse_json_string(text: str, index: int, location: str) -> tuple[str, int]:
    if index >= len(text) or text[index] != '"':
        raise _fail(location, "expected JSON string")
    index += 1
    chars: list[str] = []
    while index < len(text):
        char = text[index]
        if char == '"':
            return "".join(chars), index + 1
        if char == "\\":
            index += 1
            if index >= len(text):
                raise _fail(location, "unterminated escape")
            escaped = text[index]
            mapping = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}
            if escaped not in mapping:
                raise _fail(location, "unsupported escape")
            chars.append(mapping[escaped])
            index += 1
            continue
        chars.append(char)
        index += 1
    raise _fail(location, "unterminated string")


def _skip_ws(text: str, index: int) -> int:
    while index < len(text) and text[index] in " \t\r\n":
        index += 1
    return index


def parse_string_list(raw: str, location: str) -> tuple[str, ...]:
    text = raw.strip()
    if not text.startswith("["):
        raise _fail(location, "expected JSON string array")
    index = _skip_ws(text, 1)
    if index < len(text) and text[index] == "]":
        return ()
    values: list[str] = []
    while True:
        value, index = _parse_json_string(text, index, location)
        values.append(value)
        index = _skip_ws(text, index)
        if index >= len(text):
            raise _fail(location, "unterminated array")
        if text[index] == "]":
            if _skip_ws(text, index + 1) != len(text):
                raise _fail(location, "trailing array content")
            return tuple(values)
        if text[index] != ",":
            raise _fail(location, "expected comma in array")
        index = _skip_ws(text, index + 1)


def _parse_string_array_field(text: str, index: int, location: str) -> tuple[tuple[str, ...], int]:
    if index >= len(text) or text[index] != "[":
        raise _fail(location, "expected array")
    index = _skip_ws(text, index + 1)
    if index < len(text) and text[index] == "]":
        return (), index + 1
    values: list[str] = []
    while True:
        value, index = _parse_json_string(text, index, location)
        values.append(value)
        index = _skip_ws(text, index)
        if index >= len(text):
            raise _fail(location, "unterminated array")
        if text[index] == "]":
            return tuple(values), index + 1
        if text[index] != ",":
            raise _fail(location, "expected comma in array")
        index = _skip_ws(text, index + 1)


def _skip_json_value(text: str, index: int, location: str) -> int:
    index = _skip_ws(text, index)
    if index >= len(text):
        raise _fail(location, "expected JSON value")
    char = text[index]
    if char == '"':
        _value, index = _parse_json_string(text, index, location)
        return index
    if char == "{":
        index = _skip_ws(text, index + 1)
        if index < len(text) and text[index] == "}":
            return index + 1
        while True:
            _key, index = _parse_json_string(text, index, location)
            index = _skip_ws(text, index)
            if index >= len(text) or text[index] != ":":
                raise _fail(location, "expected colon")
            index = _skip_json_value(text, index + 1, location)
            index = _skip_ws(text, index)
            if index >= len(text):
                raise _fail(location, "unterminated object")
            if text[index] == "}":
                return index + 1
            if text[index] != ",":
                raise _fail(location, "expected comma in object")
            index = _skip_ws(text, index + 1)
    if char == "[":
        index = _skip_ws(text, index + 1)
        if index < len(text) and text[index] == "]":
            return index + 1
        while True:
            index = _skip_json_value(text, index, location)
            index = _skip_ws(text, index)
            if index >= len(text):
                raise _fail(location, "unterminated array")
            if text[index] == "]":
                return index + 1
            if text[index] != ",":
                raise _fail(location, "expected comma in array")
            index = _skip_ws(text, index + 1)
    if text.startswith("true", index):
        return index + 4
    if text.startswith("false", index):
        return index + 5
    if text.startswith("null", index):
        return index + 4
    if char in "-0123456789":
        index += 1
        while index < len(text) and text[index] in "0123456789.eE+-":
            index += 1
        return index
    raise _fail(location, "unsupported JSON value")


def parse_repo_splits(path: Path) -> Mapping[str, frozenset[str]]:
    text = path.read_text(encoding="utf-8")
    location = str(path)
    index = _skip_ws(text, 0)
    if index >= len(text) or text[index] != "{":
        raise _fail(location, "expected JSON object")
    index = _skip_ws(text, index + 1)
    result: dict[str, frozenset[str]] = {}
    if index < len(text) and text[index] == "}":
        raise _fail(location, "missing train/test repos")
    while True:
        key, index = _parse_json_string(text, index, location)
        index = _skip_ws(text, index)
        if index >= len(text) or text[index] != ":":
            raise _fail(location, "expected colon")
        index = _skip_ws(text, index + 1)
        if key in {"train", "test"}:
            values, index = _parse_string_array_field(text, index, location)
            result[key] = frozenset(values)
        else:
            index = _skip_json_value(text, index, location)
        index = _skip_ws(text, index)
        if index >= len(text):
            raise _fail(location, "unterminated object")
        if text[index] == "}":
            if _skip_ws(text, index + 1) != len(text):
                raise _fail(location, "trailing object content")
            break
        if text[index] != ",":
            raise _fail(location, "expected comma in object")
        index = _skip_ws(text, index + 1)
    if "train" not in result or "test" not in result:
        raise _fail(location, "invalid train/test repos")
    return result
