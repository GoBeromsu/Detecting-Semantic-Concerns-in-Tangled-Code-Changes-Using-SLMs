from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath

import pytest
import tree_sitter_language_pack as language_pack
from structural_validity import grammar_detect, grammars
from structural_validity.grammars import (
    RULES,
    DetectionMethod,
    FileRole,
    GrammarLoadError,
    ManifestConflictError,
    ParserDisposition,
    ParseScore,
    RuntimeMetadata,
    TreeSitterRuntime,
    UnknownGrammarError,
    build_grammar_manifest,
    classify_file,
    inventory_dispositions,
    registered_grammars,
    validate_rule_grammars,
    write_grammar_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "datasets" / "data"


@dataclass(slots=True)
class FakeRuntime:
    available: set[str]
    scores: dict[str, ParseScore] = field(default_factory=dict)
    probes: list[str] = field(default_factory=list)

    def metadata(self) -> RuntimeMetadata:
        return RuntimeMetadata(
            package_version="test-pack",
            runtime_version="test-runtime",
            language_abi_max=15,
            language_abi_min=13,
        )

    def grammar_abi(self, grammar: str) -> int | None:
        return 14 if grammar in self.available else None

    def score(self, grammar: str, source: bytes) -> ParseScore | None:
        del source
        self.probes.append(grammar)
        if grammar not in self.available:
            return None
        return self.scores.get(grammar, ParseScore(error_ratio=0.0, named_nodes=5))


def test_classify_file_when_representative_formats_uses_honest_roles() -> None:
    runtime = FakeRuntime(
        available={
            "python",
            "tsx",
            "vue",
            "elixir",
            "nix",
            "hcl",
            "proto",
            "yaml",
            "markdown",
        },
    )
    fixtures = (
        ("module.py", b"def f():\n    return 1\n", "python", FileRole.SOURCE_AST),
        ("view.tsx", b"export const A = () => <div />;", "tsx", FileRole.SOURCE_AST),
        ("view.vue", b"<template><p /></template>", "vue", FileRole.SOURCE_AST),
        ("room.ex", b"defmodule Room do\nend\n", "elixir", FileRole.SOURCE_AST),
        ("flake.nix", b"{ pkgs }: pkgs.hello", "nix", FileRole.SOURCE_AST),
        ("main.tf", b'resource "x" "y" {}', "hcl", FileRole.STRUCTURED_PARSE_TREE),
        ("event.proto", b"syntax = \"proto3\";", "proto", FileRole.STRUCTURED_PARSE_TREE),
        ("config.yaml", b"enabled: true\n", "yaml", FileRole.STRUCTURED_PARSE_TREE),
        ("README.md", b"# Heading\n", "markdown", FileRole.STRUCTURED_PARSE_TREE),
    )

    results = tuple(classify_file(path, content, runtime=runtime) for path, content, _, _ in fixtures)

    assert tuple((item.language, item.role) for item in results) == tuple(
        (language, role) for _, _, language, role in fixtures
    )
    assert results[5].role is FileRole.STRUCTURED_PARSE_TREE
    assert results[5].parser_disposition is ParserDisposition.AVAILABLE


def test_classify_file_when_exact_name_conflicts_with_extension_prefers_name() -> None:
    runtime = FakeRuntime(available={"dockerfile"})

    result = classify_file("Dockerfile.ci", b"FROM python:3.12\n", runtime=runtime)

    assert result.language == "dockerfile"
    assert result.method is DetectionMethod.EXACT_FILENAME


def test_classify_file_when_extensionless_shebang_detects_language() -> None:
    runtime = FakeRuntime(available={"python"})

    result = classify_file("tool", b"#!/usr/bin/env python3\nprint('ok')\n", runtime=runtime)

    assert result.language == "python"
    assert result.method is DetectionMethod.SHEBANG


def test_classify_file_when_success_is_misleading_stays_ambiguous() -> None:
    runtime = FakeRuntime(available={"python", "javascript"})

    result = classify_file("NOTICE", b"ordinary words accepted by permissive parsers\n", runtime=runtime)

    assert result.role is FileRole.UNRESOLVED_SOURCE
    assert result.method is DetectionMethod.AMBIGUOUS
    assert runtime.probes == []


def test_classify_file_when_binary_bytes_never_invokes_parser() -> None:
    runtime = FakeRuntime(available={"python"})

    result = classify_file("payload.py", b"\x89PNG\r\n\x1a\n\x00", runtime=runtime)

    assert result.role is FileRole.BINARY_GENERATED
    assert result.method is DetectionMethod.BINARY_CONTENT
    assert runtime.probes == []


def test_classify_file_when_candidates_need_scoring_is_bounded_and_decisive() -> None:
    runtime = FakeRuntime(
        available={"java", "csharp", "javascript", "typescript", "cpp"},
        scores={
            "java": ParseScore(error_ratio=0.0, named_nodes=8),
            "csharp": ParseScore(error_ratio=0.5, named_nodes=3),
        },
    )

    result = classify_file("Service", b"public class Service { public void run() {} }", runtime=runtime)

    assert result.language == "java"
    assert result.method is DetectionMethod.PARSER_SCORE
    # Probes carry the distribution's grammar name, not the study-internal key.
    assert runtime.probes == ["csharp", "java"]
    assert len(runtime.probes) <= 4


def test_build_manifest_when_runtime_changes_does_not_reuse_stale_availability() -> None:
    runtime = FakeRuntime(available={"python"})
    first = build_grammar_manifest(runtime=runtime, languages=("python",))
    runtime.available.clear()

    second = build_grammar_manifest(runtime=runtime, languages=("python",))

    assert first.grammars[0].parser_disposition is ParserDisposition.AVAILABLE
    assert second.grammars[0].parser_disposition is ParserDisposition.UNSUPPORTED_SOURCE


def test_build_manifest_when_structured_parser_is_missing_keeps_role_separate() -> None:
    runtime = FakeRuntime(available=set())

    manifest = build_grammar_manifest(runtime=runtime, languages=("csv",))

    assert manifest.grammars[0].role is FileRole.STRUCTURED_PARSE_TREE
    assert manifest.grammars[0].parser_disposition is ParserDisposition.UNAVAILABLE_STRUCTURED


def test_write_manifest_when_language_order_changes_is_byte_identical(tmp_path: Path) -> None:
    runtime = FakeRuntime(available={"json", "python", "yaml"})
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"

    _ = write_grammar_manifest(
        first_path,
        runtime=runtime,
        languages=("yaml", "python", "json", "python"),
    )
    _ = write_grammar_manifest(
        second_path,
        runtime=runtime,
        languages=("json", "yaml", "python"),
    )

    assert first_path.read_bytes() == second_path.read_bytes()
    assert hashlib.sha256(first_path.read_bytes()).hexdigest() == hashlib.sha256(
        second_path.read_bytes()
    ).hexdigest()


def test_classify_file_when_best_parser_is_unregistered_stays_ambiguous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = FakeRuntime(
        available={"invented"},
        scores={"invented": ParseScore(error_ratio=0.0, named_nodes=100)},
    )

    def unregistered_candidates(
        source: bytes,
    ) -> tuple[tuple[str, ...], DetectionMethod] | None:
        del source
        return (("invented",), DetectionMethod.CONTENT_SIGNATURE)

    monkeypatch.setattr(grammar_detect, "_content_candidates", unregistered_candidates)

    result = classify_file("mystery", b"accepted by invented parser", runtime=runtime)

    assert result.role is FileRole.UNRESOLVED_SOURCE
    assert result.method is DetectionMethod.AMBIGUOUS
    assert runtime.probes == []


def test_write_manifest_when_real_package_is_imported_records_runtime(tmp_path: Path) -> None:
    output = tmp_path / "grammar_manifest.json"

    manifest = write_grammar_manifest(output, languages=("python", "json"))

    assert output.is_file()
    assert manifest.package.distribution == "tree-sitter-language-pack"
    assert manifest.package.version
    assert manifest.runtime.distribution == "tree-sitter"
    assert manifest.runtime.language_abi_min <= manifest.runtime.language_abi_max
    assert {entry.parser_disposition for entry in manifest.grammars} == {ParserDisposition.AVAILABLE}
    assert {entry.role for entry in manifest.grammars} == {
        FileRole.SOURCE_AST,
        FileRole.STRUCTURED_PARSE_TREE,
    }
    assert output.read_text(encoding="utf-8") == json.dumps(
        asdict(manifest), indent=2, sort_keys=True
    ) + "\n"


def test_write_manifest_when_existing_bytes_differ_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "grammar_manifest.json"
    original = b'{"different":true}\n'
    _ = output.write_bytes(original)

    with pytest.raises(ManifestConflictError):
        _ = write_grammar_manifest(
            output,
            runtime=FakeRuntime(available={"python"}),
            languages=("python",),
        )

    assert output.read_bytes() == original
    assert not tuple(tmp_path.glob(".grammar_manifest.json.*"))


def test_cli_when_output_is_requested_writes_manifest_atomically(tmp_path: Path) -> None:
    output = tmp_path / "grammar_manifest.json"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO_ROOT / "datasets" / "scripts")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "structural_validity.grammars",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    assert output.is_file()
    assert not tuple(tmp_path.glob(".grammar_manifest.json.*"))


def test_inventory_dispositions_when_given_committed_paths_covers_every_extension() -> None:
    _ = csv.field_size_limit(10_000_000)
    paths: list[str] = []
    for data_path in (DATA_DIR / "CCS Dataset.csv", DATA_DIR / "repo_grouped_pool.csv"):
        with data_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                paths.extend(
                    match.group(1)
                    for line in row["git_diff"].splitlines()
                    if (match := re.match(r"^(?:---|\+\+\+) \"?[ab]/(.+?)\"?$", line))
                )
    expected_extensions = {
        PurePosixPath(path).suffix.lower() or "<none>"
        for path in paths
    }

    inventory = inventory_dispositions(path for path in paths)

    assert {item.extension for item in inventory} == expected_extensions
    assert len(inventory) >= 90
    assert all(item.count > 0 for item in inventory)
    assert all(item.role in FileRole for item in inventory)


def test_committed_inventory_when_seed43_artifacts_are_loaded_is_frozen() -> None:
    inventory = grammars.inventory_committed_data(DATA_DIR)

    assert tuple(
        (source.name, source.records, source.diff_fragments, source.paths)
        for source in inventory.sources
    ) == (
        ("CCS Dataset.csv", 2000, 2000, 18069),
        ("repo_grouped_pool.csv", 1309, 1309, 7741),
        ("tangled_ccs_dataset_test.csv", 350, 1050, 5161),
        ("tangled_ccs_dataset_train.csv", 1400, 4200, 18898),
    )
    assert inventory.path_count == 49869
    assert inventory.extension_count == 100
    assert inventory.disposed_path_count == inventory.path_count
    assert inventory.split.train_repositories == 21
    assert inventory.split.test_repositories == 15
    assert inventory.split.train_type_supply == 1005
    assert inventory.split.test_type_supply == 304
    assert inventory.container_count == 5
    assert all(container.role is FileRole.STRUCTURED_PARSE_TREE for container in inventory.containers)
    assert inventory.sha256 == "7484087c5a273c7fb4192862fe143105a7a18f3f2094e2ad487877987b0ce2b7"


def test_rule_grammars_when_checked_against_distribution_are_all_resolvable() -> None:
    """A misspelled grammar name is indistinguishable from an absent one at load
    time, so every rule name is checked against the distribution registry."""
    registry = registered_grammars()

    absent = validate_rule_grammars(registry)

    assert absent == ("erg",)
    unresolved = tuple(
        rule.language
        for rule in RULES
        if rule.grammar is not None and rule.grammar not in registry and rule.grammar not in absent
    )
    assert unresolved == ()


def test_rule_grammars_when_name_is_misspelled_is_rejected_not_reported_absent() -> None:
    registry = registered_grammars()

    with pytest.raises(UnknownGrammarError) as caught:
        _ = validate_rule_grammars(registry - {"csharp"})

    assert caught.value.grammar == "csharp"
    assert caught.value.language == "c_sharp"


def test_registered_grammars_when_cache_is_cold_still_reports_full_registry() -> None:
    """`available_languages` shrinks to what is materialized locally; the
    manifest registry must not, or a cold cache understates AST coverage."""
    registry = registered_grammars()

    assert {"csharp", "embeddedtemplate"} <= registry
    assert len(registry) > len(language_pack.available_languages())


def test_grammar_abi_when_registered_grammar_fails_to_load_raises_not_degrades(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed download must not be recorded as `unsupported_source`."""

    def failing_get_language(name: str) -> object:
        raise language_pack.DownloadError(f"cold cache for {name}")

    def always_registered(name: str) -> bool:
        del name
        return True

    monkeypatch.setattr(language_pack, "has_language", always_registered)
    monkeypatch.setattr(language_pack, "get_language", failing_get_language)

    with pytest.raises(GrammarLoadError) as caught:
        _ = TreeSitterRuntime().grammar_abi("python")

    assert caught.value.grammar == "python"
    assert caught.value.reason == "DownloadError"


def test_grammar_abi_when_grammar_is_absent_upstream_reports_unsupported() -> None:
    assert TreeSitterRuntime().grammar_abi("erg") is None


def test_manifest_when_built_from_real_distribution_covers_dataset_source_roles() -> None:
    """`.cs` and `.erb` resolve to real grammars; only upstream-absent `erg` may
    remain unsupported."""
    manifest = build_grammar_manifest(languages=("c_sharp", "erb", "erg"))
    dispositions = {entry.language: entry.parser_disposition for entry in manifest.grammars}

    assert dispositions["c_sharp"] is ParserDisposition.AVAILABLE
    assert dispositions["erb"] is ParserDisposition.AVAILABLE
    assert dispositions["erg"] is ParserDisposition.UNSUPPORTED_SOURCE
