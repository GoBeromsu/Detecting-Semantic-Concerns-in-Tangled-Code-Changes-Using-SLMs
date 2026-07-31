from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Literal

import pytest

from __test__.unsloth.adapter_fixtures import (
    BASE,
    CONFIG_PATH,
    DATASET_REVISION,
    EXCLUDED_HASH,
    MODEL_REVISION,
    TARGETS,
    artifact_hashes,
    contract_manifest,
    load_json,
    write_contract,
)
from RQ.SLM.unsloth._types import ContractError, JsonValue
from RQ.SLM.unsloth.adapter import validate_adapter


def test_validate_adapter_when_contract_is_complete_returns_frozen_report(
    tmp_path: Path,
) -> None:
    # Given: a complete content-addressed smoke adapter.
    _ = write_contract(tmp_path)

    # When: validation recomputes the saved evidence hashes.
    report = validate_adapter(tmp_path)

    # Then: the pinned adapter report remains immutable.
    assert report.base_model == BASE
    assert report.model_revision == MODEL_REVISION
    assert report.training_rows == 1399
    field_name = "training_rows"
    with pytest.raises(FrozenInstanceError):
        setattr(report, field_name, 1)


def test_validate_adapter_when_manifest_is_training_output_accepts(tmp_path: Path) -> None:
    # Given: the training writer's real safetensors and copied evidence output.
    manifest = write_contract(tmp_path)

    # When: standalone validation uses only the saved directory.
    report = validate_adapter(tmp_path, CONFIG_PATH)

    # Then: the copied config and actual tokenizer template are content-addressed.
    assert report.training_rows == 1399
    assert manifest["run_mode"] == "smoke"
    hashes = manifest["hashes"]
    assert isinstance(hashes, dict)
    artifacts = hashes["artifacts"]
    assert isinstance(artifacts, dict)
    assert artifacts["evidence/config.yml"] == artifact_hashes(tmp_path)["evidence/config.yml"]
    assert (tmp_path / "evidence" / "config.yml").read_bytes() == CONFIG_PATH.read_bytes()


@pytest.mark.parametrize(
    ("field", "value"),
    (("base_model_name_or_path", "Qwen/Qwen3-14B"), ("revision", "wrong"),
     ("r", 16), ("target_modules", TARGETS[:-1])),
)
def test_validate_adapter_when_adapter_config_differs_rejects(
    tmp_path: Path, field: str, value: JsonValue
) -> None:
    # Given: a changed adapter field with its file hash recomputed in the manifest.
    _ = write_contract(tmp_path)
    config_path = tmp_path / "adapter_config.json"
    config = load_json(config_path)
    config[field] = value
    _ = config_path.write_text(json.dumps(config), encoding="utf-8")
    _ = contract_manifest(tmp_path, artifact_hashes(tmp_path))

    # When/Then: the effective training identity rejects the changed adapter field.
    with pytest.raises(ContractError, match=field):
        _ = validate_adapter(tmp_path)


def test_validate_adapter_when_bpe_tokenizer_form_is_complete_accepts(tmp_path: Path) -> None:
    # Given: a saved adapter with the vocab.json plus merges.txt tokenizer form.
    _ = write_contract(tmp_path, tokenizer_form="bpe")

    # When: the alternate tokenizer form is validated.
    report = validate_adapter(tmp_path)

    # Then: the report records the BPE evidence files.
    assert report.tokenizer_files == ("vocab.json", "merges.txt", "tokenizer_config.json")


def test_validate_adapter_when_tokenizer_is_incomplete_rejects(tmp_path: Path) -> None:
    # Given: a BPE adapter with its merge evidence removed after publication.
    _ = write_contract(tmp_path, tokenizer_form="bpe")
    _ = (tmp_path / "merges.txt").unlink()
    _ = contract_manifest(tmp_path, artifact_hashes(tmp_path))

    # When/Then: neither accepted tokenizer form remains complete.
    with pytest.raises(ContractError, match="tokenizer"):
        _ = validate_adapter(tmp_path)


@pytest.mark.parametrize(
    "name", ("adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json", "run_manifest.json"),
)
def test_validate_adapter_when_required_file_is_missing_rejects(tmp_path: Path, name: str) -> None:
    # Given: a complete adapter with one mandatory contract file removed.
    _ = write_contract(tmp_path)
    _ = (tmp_path / name).unlink()

    # When/Then: validation fails at the missing-file boundary.
    with pytest.raises(ContractError):
        _ = validate_adapter(tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    (("merged", False), ("training_rows", 1399), ("model_revision", MODEL_REVISION),
     ("dataset_revision", DATASET_REVISION), ("excluded_sha256", EXCLUDED_HASH)),
)
def test_validate_adapter_when_manifest_contains_legacy_flat_field_rejects(
    tmp_path: Path, field: str, value: str | int | bool
) -> None:
    # Given: a valid manifest polluted with a legacy flat field.
    _ = write_contract(tmp_path)
    manifest = load_json(tmp_path / "run_manifest.json")
    manifest[field] = value
    _ = (tmp_path / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    # When/Then: the closed schema rejects legacy manifest shape.
    with pytest.raises(ContractError, match="run_manifest"):
        _ = validate_adapter(tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    (("git", {"sha": "", "clean": True}),
     ("model", {"id": BASE, "revision": "wrong"}),
     ("dataset", {"id": "wrong", "revision": DATASET_REVISION, "final_row_count": 1399}),
     ("hashes", {"config_sha256": "x", "template_sha256": "b" * 64,
                  "host_profile_sha256": None, "excluded_rows": []}),
     ("seed", 41), ("objective", "wrong"), ("packing", True),
     ("excluded_rows", []), ("merged_model", True)),
)
def test_validate_adapter_when_nested_manifest_mismatches_rejects(
    tmp_path: Path, field: str, value: JsonValue
) -> None:
    # Given: a legacy or nested manifest value changed from the writer output.
    _ = write_contract(tmp_path)
    manifest = load_json(tmp_path / "run_manifest.json")
    manifest[field] = value
    _ = (tmp_path / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    # When/Then: no noncanonical manifest representation reaches publication.
    with pytest.raises(ContractError):
        _ = validate_adapter(tmp_path)


ManifestSection = Literal["git", "model", "identity", "hashes"]


@pytest.mark.parametrize(
    ("section", "field", "value"),
    (("git", "sha", ""), ("git", "clean", "yes"),
     ("model", "id", "wrong"), ("model", "revision", "wrong"),
     ("model", "dataset_id", "wrong"), ("model", "dataset_revision", "wrong"),
     ("identity", "final_row_count", 1400), ("identity", "packing", True),
     ("identity", "merged_model", True), ("hashes", "artifacts", {}),
     ("hashes", "template_sha256", "x" * 64)),
)
def test_validate_adapter_when_nested_manifest_field_mismatches_rejects(
    tmp_path: Path, section: ManifestSection, field: str, value: JsonValue
) -> None:
    # Given: one field changed inside a current closed-schema manifest section.
    _ = write_contract(tmp_path)
    manifest = load_json(tmp_path / "run_manifest.json")
    match section:
        case "git" | "identity" | "hashes":
            nested = manifest[section]
        case "model":
            identity = manifest["identity"]
            assert isinstance(identity, dict)
            effective = identity["effective_config"]
            assert isinstance(effective, dict)
            nested = effective["model"]
    assert isinstance(nested, dict)
    nested[field] = value
    _ = (tmp_path / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    # When/Then: the changed nested identity is never accepted.
    with pytest.raises(ContractError):
        _ = validate_adapter(tmp_path)


def test_validate_adapter_when_selected_config_is_invalid_rejects(tmp_path: Path) -> None:
    # Given: a selected config that the project loader cannot parse as a valid run.
    _ = write_contract(tmp_path)
    config_path = tmp_path.parent / f"{tmp_path.name}-invalid.yml"
    _ = config_path.write_text("model: {}\n", encoding="utf-8")

    # When/Then: loader validation rejects it before publication succeeds.
    with pytest.raises(ContractError, match="config"):
        _ = validate_adapter(tmp_path, config_path)
