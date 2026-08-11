"""Role inventory over the committed seed-43 CSV artifacts.

Paths are recovered from committed diff text only, so this lane never needs a
repository mirror. Every observed extension receives a role, and the returned
counts are asserted against the raw path total by callers so no extension can
quietly leave the denominator.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Final

from .grammar_rules import (
    BINARY_EXTENSIONS,
    DIFF_PATH_PATTERN,
    EXACT_TEXT_NAMES,
    GENERATED_EXTENSIONS,
    RULE_BY_EXTENSION,
    TEXT_EXTENSIONS,
    exact_rule,
)
from .grammar_types import (
    CommittedDataInventory,
    CommittedSourceInventory,
    CommittedSourceSpec,
    ContainerInventory,
    DispositionInventoryEntry,
    FileRole,
    RepositorySplitInventory,
)

COMMITTED_SOURCE_SPECS: Final = (
    CommittedSourceSpec("CCS Dataset.csv", "git_diff", False),
    CommittedSourceSpec("repo_grouped_pool.csv", "git_diff", False),
    CommittedSourceSpec("tangled_ccs_dataset_test.csv", "diff", True),
    CommittedSourceSpec("tangled_ccs_dataset_train.csv", "diff", True),
)
CSV_FIELD_LIMIT: Final = 10_000_000


def inventory_role(path: str) -> FileRole:
    name = PurePosixPath(path).name
    suffix = PurePosixPath(name).suffix.lower()
    if suffix in BINARY_EXTENSIONS or suffix in GENERATED_EXTENSIONS or name.endswith(".min.js"):
        return FileRole.BINARY_GENERATED
    rule = exact_rule(name) or RULE_BY_EXTENSION.get(suffix)
    if rule is not None:
        return rule.role
    if suffix in TEXT_EXTENSIONS or name in EXACT_TEXT_NAMES:
        return FileRole.TEXT_ONLY
    return FileRole.UNRESOLVED_SOURCE


def inventory_dispositions(paths: Iterable[str]) -> tuple[DispositionInventoryEntry, ...]:
    materialized_paths = tuple(paths)
    counts = Counter(PurePosixPath(path).suffix.lower() or "<none>" for path in materialized_paths)
    roles: dict[str, FileRole] = {}
    for path in materialized_paths:
        extension = PurePosixPath(path).suffix.lower() or "<none>"
        if extension not in roles:
            roles[extension] = inventory_role(path)
    return tuple(DispositionInventoryEntry(extension, count, roles[extension]) for extension, count in sorted(counts.items()))


def _scan_json_string(raw: str, index: int) -> tuple[str, int] | None:
    values: list[str] = []
    escapes = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}
    while index < len(raw):
        character = raw[index]
        index += 1
        if character == '"':
            return "".join(values), index
        if character != "\\":
            values.append(character)
            continue
        if index >= len(raw):
            return None
        escape = raw[index]
        index += 1
        if escape == "u":
            codepoint = raw[index : index + 4]
            if len(codepoint) != 4 or not all(char in "0123456789abcdefABCDEF" for char in codepoint):
                return None
            values.append(chr(int(codepoint, 16)))
            index += 4
            continue
        decoded = escapes.get(escape)
        if decoded is None:
            return None
        values.append(decoded)
    return None


def _json_string_array(raw: str) -> tuple[str, ...]:
    index = 0
    while index < len(raw) and raw[index].isspace():
        index += 1
    if index >= len(raw) or raw[index] != "[":
        return ()
    index += 1
    values: list[str] = []
    while index < len(raw):
        while index < len(raw) and raw[index].isspace():
            index += 1
        if index < len(raw) and raw[index] == "]":
            return tuple(values)
        if index >= len(raw) or raw[index] != '"':
            return ()
        scanned = _scan_json_string(raw, index + 1)
        if scanned is None:
            return ()
        value, index = scanned
        values.append(value)
        while index < len(raw) and raw[index].isspace():
            index += 1
        if index < len(raw) and raw[index] == ",":
            index += 1
            continue
        if index < len(raw) and raw[index] == "]":
            return tuple(values)
        return ()
    return ()


def _diff_fragments(raw: str, nested: bool) -> tuple[str, ...]:
    return _json_string_array(raw) if nested else (raw,)


def _split_metadata(raw: str) -> RepositorySplitInventory:
    train_match = re.search(r'"train"\s*:\s*(\[[^]]*\])', raw)
    test_match = re.search(r'"test"\s*:\s*(\[[^]]*\])', raw)
    train_supply_match = re.search(r'"train_type_supply"\s*:\s*(\{[^}]*\})', raw)
    test_supply_match = re.search(r'"test_type_supply"\s*:\s*(\{[^}]*\})', raw)
    if train_match is None or test_match is None or train_supply_match is None or test_supply_match is None:
        return RepositorySplitInventory(0, 0, 0, 0)
    train_values = _json_string_array(train_match.group(1))
    test_values = _json_string_array(test_match.group(1))
    train_supply = sum(int(match.group(1)) for match in re.finditer(r'"[^"\\]+"\s*:\s*(\d+)', train_supply_match.group(1)))
    test_supply = sum(int(match.group(1)) for match in re.finditer(r'"[^"\\]+"\s*:\s*(\d+)', test_supply_match.group(1)))
    return RepositorySplitInventory(len(train_values), len(test_values), train_supply, test_supply)


def _read_committed_source(data_dir: Path, spec: CommittedSourceSpec) -> tuple[CommittedSourceInventory, tuple[str, ...]]:
    records = 0
    fragment_count = 0
    paths: list[str] = []
    with (data_dir / spec.name).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            records += 1
            fragments = _diff_fragments(row[spec.diff_column], spec.nested_diffs)
            fragment_count += len(fragments)
            paths.extend(
                match.group(1)
                for fragment in fragments
                for line in fragment.splitlines()
                if (match := DIFF_PATH_PATTERN.match(line))
            )
    return CommittedSourceInventory(spec.name, records, fragment_count, len(paths)), tuple(paths)


def inventory_committed_data(data_dir: Path) -> CommittedDataInventory:
    previous_limit = csv.field_size_limit()
    _ = csv.field_size_limit(CSV_FIELD_LIMIT)
    try:
        source_results = tuple(_read_committed_source(data_dir, spec) for spec in COMMITTED_SOURCE_SPECS)
    finally:
        _ = csv.field_size_limit(previous_limit)
    sources = tuple(summary for summary, _ in source_results)
    paths = tuple(path for _, source_paths in source_results for path in source_paths)
    dispositions = inventory_dispositions(paths)
    split = _split_metadata((data_dir / "repo_split.json").read_text(encoding="utf-8"))
    container_names = tuple(spec.name for spec in COMMITTED_SOURCE_SPECS) + ("repo_split.json",)
    containers = tuple(ContainerInventory(name, inventory_role(name)) for name in container_names)
    digest_payload = {
        "containers": [asdict(container) for container in containers],
        "dispositions": [asdict(disposition) for disposition in dispositions],
        "sources": [asdict(source) for source in sources],
        "split": asdict(split),
    }
    digest = hashlib.sha256(json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return CommittedDataInventory(
        sources=sources,
        split=split,
        containers=containers,
        dispositions=dispositions,
        path_count=len(paths),
        disposed_path_count=sum(entry.count for entry in dispositions),
        extension_count=len(dispositions),
        container_count=len(containers),
        sha256=digest,
    )
