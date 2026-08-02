from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from __test__.unsloth.adapter_fixtures import (
    BASE,
    CONFIG_PATH,
    EXCLUDED_HASH,
    MODEL_REVISION,
    load_json,
    safetensors_payload,
    write_adapter_files,
    write_contract,
)
from RQ.SLM.unsloth._types import ContractError
from RQ.SLM.unsloth.adapter import validate_adapter


def test_validate_adapter_when_smoke_contract_is_complete_returns_frozen_report(
    tmp_path: Path,
) -> None:
    # Given: a dirty-worktree smoke adapter with captured standalone evidence.
    _ = write_contract(tmp_path)

    # When: the adapter boundary recomputes its content addresses.
    report = validate_adapter(tmp_path, CONFIG_PATH)

    # Then: smoke evidence validates without claiming publication-clean provenance.
    assert report.base_model == BASE
    assert report.model_revision == MODEL_REVISION
    assert report.training_rows == 1399
    field_name = "training_rows"
    with pytest.raises(FrozenInstanceError):
        setattr(report, field_name, 1)


def test_validate_adapter_when_manifest_uses_format_only_hashes_rejects(
    tmp_path: Path,
) -> None:
    # Given: a valid schema whose captured config hash was replaced by a dummy digest.
    _ = write_contract(tmp_path)
    manifest_path = tmp_path / "run_manifest.json"
    manifest = load_json(manifest_path)
    hashes = manifest["hashes"]
    assert isinstance(hashes, dict)
    artifacts = hashes["artifacts"]
    assert isinstance(artifacts, dict)
    artifacts["evidence/config.yml"] = "a" * 64
    _ = manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    # When/Then: a hash-shaped string that does not name the actual bytes is rejected.
    with pytest.raises(ContractError, match="hashes.artifacts"):
        _ = validate_adapter(tmp_path)


def test_validate_adapter_when_manifest_git_sha_changes_rejects(tmp_path: Path) -> None:
    # Given: a one-byte provenance mutation after the evidence bundle was captured.
    _ = write_contract(tmp_path)
    manifest_path = tmp_path / "run_manifest.json"
    manifest = load_json(manifest_path)
    git = manifest["git"]
    assert isinstance(git, dict)
    git["sha"] = "b" * 40
    _ = manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    # When/Then: the manifest cannot silently diverge from its captured provenance bytes.
    with pytest.raises(ContractError, match="git"):
        _ = validate_adapter(tmp_path)


def test_validate_adapter_when_adapter_payload_is_arbitrary_bytes_rejects(
    tmp_path: Path,
) -> None:
    # Given: an attacker updated the payload hash but supplied arbitrary non-safetensors bytes.
    _ = write_contract(tmp_path)
    payload = tmp_path / "adapter_model.safetensors"
    _ = payload.write_bytes(b"safetensors")
    manifest_path = tmp_path / "run_manifest.json"
    manifest = load_json(manifest_path)
    hashes = manifest["hashes"]
    assert isinstance(hashes, dict)
    payload_hashes = hashes["artifacts"]
    assert isinstance(payload_hashes, dict)
    payload_hashes[payload.name] = hashlib.sha256(payload.read_bytes()).hexdigest()
    _ = manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    # When/Then: the payload parser refuses bytes that only have a safetensors filename.
    with pytest.raises(ContractError, match="adapter_model.safetensors"):
        _ = validate_adapter(tmp_path)


@pytest.mark.parametrize("evidence_file", ("config.yml", "template.txt", "training_identity.json"))
def test_validate_adapter_when_evidence_byte_changes_rejects(
    tmp_path: Path, evidence_file: str
) -> None:
    # Given: one captured evidence byte was changed after the manifest was written.
    _ = write_contract(tmp_path)
    path = tmp_path / "evidence" / evidence_file
    _ = path.write_bytes(path.read_bytes() + b"\n")

    # When/Then: recomputed content addresses refuse the modified evidence.
    with pytest.raises(ContractError, match="hashes.artifacts"):
        _ = validate_adapter(tmp_path)


def test_validate_adapter_when_selected_config_bytes_differ_rejects(tmp_path: Path) -> None:
    # Given: a config that parses successfully but differs by one non-semantic byte.
    _ = write_contract(tmp_path)
    changed_config = tmp_path.parent / f"{tmp_path.name}-selected.yml"
    _ = changed_config.write_bytes(CONFIG_PATH.read_bytes() + b"\n")

    # When/Then: the project loader parses it, then exact byte provenance rejects it.
    with pytest.raises(ContractError, match="config_sha256"):
        _ = validate_adapter(tmp_path, changed_config)


def test_validate_adapter_when_full_model_shard_is_present_rejects(tmp_path: Path) -> None:
    # Given: a complete adapter directory contaminated by a full-model shard.
    _ = write_contract(tmp_path)
    _ = (tmp_path / "model-00001-of-00002.safetensors").write_bytes(safetensors_payload())

    # When/Then: the adapter-only boundary refuses merged or full model output.
    with pytest.raises(ContractError, match="full-model"):
        _ = validate_adapter(tmp_path)


def test_build_manifest_when_full_run_is_dirty_rejects(tmp_path: Path) -> None:
    # Given: a saved adapter whose full-run provenance reports a dirty worktree.
    from RQ.SLM.unsloth.train import build_manifest
    from RQ.SLM.unsloth._types import ExcludedRow, ManifestEvidence, RunProvenance
    from RQ.SLM.unsloth.config import load_unsloth_config

    write_adapter_files(tmp_path)
    evidence = ManifestEvidence(
        tmp_path, CONFIG_PATH, "{{ messages }}", None, None,
        (ExcludedRow(1326, EXCLUDED_HASH, 16816),), 1399,
    )

    # When/Then: full publication refuses dirty Git even though smoke validation permits it.
    with pytest.raises(ContractError, match="git.clean"):
        _ = build_manifest(
            load_unsloth_config(CONFIG_PATH), RunProvenance("a" * 40, False, "full"), evidence
        )


def test_cli_when_valid_adapter_writes_json_report_atomically(tmp_path: Path) -> None:
    # Given: a content-addressed smoke adapter and the original source config.
    _ = write_contract(tmp_path)
    output_path = tmp_path / "report.json"

    # When: the import-safe CLI validates and emits the report.
    completed = subprocess.run(
        [sys.executable, "-m", "RQ.SLM.unsloth.adapter", "--adapter-dir", str(tmp_path),
         "--config", str(CONFIG_PATH), "--output", str(output_path)],
        check=False, capture_output=True, text=True, cwd=CONFIG_PATH.parents[3],
    )

    # Then: it succeeds without ML imports and leaves no partial output behind.
    assert completed.returncode == 0, completed.stderr
    assert load_json(output_path)["training_rows"] == 1399
    assert not tuple(tmp_path.glob(".report.json.*"))


def test_cli_help_when_requested_exits_zero_without_input_files() -> None:
    # Given: no adapter, config, or output files are available.
    # When: the contract script is asked for help.
    completed = subprocess.run(
        [sys.executable, "RQ/SLM/unsloth/adapter.py", "--help"],
        check=False, capture_output=True, text=True, cwd=CONFIG_PATH.parents[4],
    )

    # Then: help is printed successfully before validation or file access.
    assert completed.returncode == 0, completed.stderr
    assert "--adapter-dir" in completed.stdout
    assert "--config" in completed.stdout
    assert "--output" in completed.stdout
