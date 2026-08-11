"""Build and atomically publish the pinned grammar manifest.

The manifest records distribution version, Tree-sitter ABI bounds, grammar
name, role, detector order, and parser availability so a later AST lane can
prove which dispositions its coverage numbers were computed under.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Final

from .grammar_detect import DETECTOR_ORDER
from .grammar_rules import RULE_BY_LANGUAGE
from .grammar_runtime import TreeSitterRuntime, validate_rule_grammars
from .grammar_types import (
    CliArguments,
    DistributionManifest,
    FileRole,
    GrammarManifest,
    GrammarManifestEntry,
    GrammarRuntime,
    ManifestConflictError,
    ParserDisposition,
    RuntimeManifest,
)

DEFAULT_MANIFEST_PATH: Final = (
    Path(__file__).resolve().parents[3]
    / "results"
    / "analysis"
    / "seed43_structural_validity"
    / "grammar_manifest.json"
)


def _entry_disposition(abi: int | None, role: FileRole) -> ParserDisposition:
    if abi is not None:
        return ParserDisposition.AVAILABLE
    if role is FileRole.STRUCTURED_PARSE_TREE:
        return ParserDisposition.UNAVAILABLE_STRUCTURED
    return ParserDisposition.UNSUPPORTED_SOURCE


def build_grammar_manifest(
    *,
    runtime: GrammarRuntime | None = None,
    languages: Sequence[str] | None = None,
) -> GrammarManifest:
    active_runtime = runtime or TreeSitterRuntime()
    metadata = active_runtime.metadata()
    selected_names = sorted(set(languages)) if languages is not None else sorted(RULE_BY_LANGUAGE)
    selected = tuple(RULE_BY_LANGUAGE[name] for name in selected_names)
    entries = tuple(
        GrammarManifestEntry(
            rule.language,
            rule.grammar,
            rule.role,
            rule.extensions,
            DETECTOR_ORDER,
            abi,
            _entry_disposition(abi, rule.role),
        )
        for rule in selected
        if rule.grammar is not None
        for abi in (active_runtime.grammar_abi(rule.grammar),)
    )
    return GrammarManifest(
        schema_version=1,
        package=DistributionManifest("tree-sitter-language-pack", metadata.package_version),
        runtime=RuntimeManifest("tree-sitter", metadata.runtime_version, metadata.language_abi_min, metadata.language_abi_max),
        grammars=entries,
    )


def _publish(path: Path, payload: bytes) -> None:
    with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        _ = handle.write(payload)
        handle.flush()
        _ = os.fsync(handle.fileno())
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            if not path.is_file() or path.read_bytes() != payload:
                raise ManifestConflictError(path) from None
    finally:
        temporary.unlink(missing_ok=True)


def write_grammar_manifest(
    path: Path,
    *,
    runtime: GrammarRuntime | None = None,
    languages: Sequence[str] | None = None,
) -> GrammarManifest:
    manifest = build_grammar_manifest(runtime=runtime, languages=languages)
    payload = (json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_file() and path.read_bytes() == payload:
            return manifest
        raise ManifestConflictError(path)
    _publish(path, payload)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the deterministic Tree-sitter grammar manifest.",
        epilog=(
            "Canonical invocation: PYTHONPATH=datasets/scripts uv run python -m "
            "structural_validity.grammars --output "
            "results/analysis/seed43_structural_validity/grammar_manifest.json"
        ),
    )
    _ = parser.add_argument("--output", type=Path, default=DEFAULT_MANIFEST_PATH)
    arguments = parser.parse_args(argv, namespace=CliArguments(DEFAULT_MANIFEST_PATH))
    absent = validate_rule_grammars()
    output = arguments.output
    manifest = write_grammar_manifest(output)
    summary = f"wrote {output} with {len(manifest.grammars)} grammar dispositions"
    print(f"{summary} for tree-sitter-language-pack {manifest.package.version}")
    if absent:
        print(f"declared unsupported upstream: {', '.join(absent)}")
    return 0
