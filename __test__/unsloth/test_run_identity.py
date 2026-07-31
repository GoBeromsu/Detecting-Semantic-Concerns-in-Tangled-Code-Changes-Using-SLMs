from __future__ import annotations

from pathlib import Path

import pytest

from RQ.SLM.unsloth import infer as inference_contract


def _identity_input(
    adapter_path: Path, config_path: Path, *, shas: tuple[tuple[str, ...], ...]
):
    return inference_contract.RunIdentityInput(
        adapter_path=adapter_path,
        config_path=config_path,
        model_id="Qwen/Qwen3.6-27B",
        model_revision="model-revision",
        dataset_id="Berom0227/tangled-ccs-commits",
        dataset_revision="dataset-revision",
        ordered_test_shas=shas,
        seed=42,
        temperature=0.3,
        max_new_tokens=128,
        contexts=(12288, 8192),
        message_conditions=(False, True),
    )


def test_build_run_identity_when_ordered_shas_are_reordered_changes_digest(tmp_path: Path) -> None:
    # Given: identical run inputs except for the ordered test SHA sequence.
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    _ = (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    config = tmp_path / "config.yml"
    _ = config.write_text("config", encoding="utf-8")

    # When: content-addressed identities are calculated for each ordered source.
    first = inference_contract.build_run_identity(
        _identity_input(adapter, config, shas=(("a",), ("b",)))
    )
    reordered = inference_contract.build_run_identity(
        _identity_input(adapter, config, shas=(("b",), ("a",)))
    )

    # Then: order drift has a different immutable identity.
    assert first.digest != reordered.digest


def test_establish_run_when_resuming_identity_drift_rejects(tmp_path: Path) -> None:
    # Given: a persisted run identity and a different requested identity.
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    _ = (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    config = tmp_path / "config.yml"
    _ = config.write_text("config", encoding="utf-8")
    run_directory = tmp_path / "run"
    initial = inference_contract.build_run_identity(
        _identity_input(adapter, config, shas=(("a",),))
    )
    drifted = inference_contract.build_run_identity(
        _identity_input(adapter, config, shas=(("b",),))
    )
    _ = inference_contract.establish_run(run_directory, initial, resume=False)

    # When/Then: resume refuses a different content-addressed identity.
    with pytest.raises(inference_contract.RunIdentityError, match="identity"):
        _ = inference_contract.establish_run(run_directory, drifted, resume=True)


def test_establish_run_when_resuming_missing_directory_rejects(tmp_path: Path) -> None:
    # Given: an identity with no explicit persisted run directory.
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    _ = (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    config = tmp_path / "config.yml"
    _ = config.write_text("config", encoding="utf-8")
    identity = inference_contract.build_run_identity(
        _identity_input(adapter, config, shas=(("a",),))
    )

    # When/Then: resume never creates a fresh timestamped replacement directory.
    with pytest.raises(inference_contract.RunIdentityError, match="existing"):
        _ = inference_contract.establish_run(tmp_path / "missing", identity, resume=True)
