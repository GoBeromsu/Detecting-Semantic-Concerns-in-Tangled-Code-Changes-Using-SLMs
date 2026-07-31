from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Final

import pytest

from RQ.SLM.unsloth.config import load_config, load_host_profile
from RQ.SLM.unsloth._types import (
    GIB_BYTES,
    MODEL_STORAGE_BYTES,
    PROTECTED_CACHE_NAMES,
    JsonObject,
    JsonValue,
    PreflightError,
    PreflightInputs,
    ProbeFacts,
)
from RQ.SLM.unsloth.preflight import (
    missing_model_bytes,
    parse_probe_json,
    validate_preflight,
)


REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
CONFIG_PATH: Final[Path] = REPO_ROOT / "RQ/SLM/unsloth/configs/qwen3_6_27b.yml"
HOST_PATH: Final[Path] = (
    REPO_ROOT / "RQ/SLM/configs/hosts/blackwell-rtx-pro-6000.yml"
)
SCRIPT_PATH: Final[Path] = REPO_ROOT / "RQ/SLM/unsloth/preflight.py"
MIB_BYTES: Final[int] = 1024**2


def _probe_document() -> JsonObject:
    return {
        "valid": True,
        "hostname": "dcs33979",
        "memory": {
            "ram": {"total_bytes": 31 * GIB_BYTES, "free_bytes": 12 * GIB_BYTES},
            "swap": {"total_bytes": 980 * MIB_BYTES, "free_bytes": 2 * MIB_BYTES},
        },
        "disks": [
            {
                "path": "/",
                "total_bytes": 500 * GIB_BYTES,
                "free_bytes": 108 * GIB_BYTES,
                "mount": "/",
            },
            {
                "path": "/home/beomsu",
                "total_bytes": 500 * GIB_BYTES,
                "free_bytes": 108 * GIB_BYTES,
                "mount": "/",
            },
            {
                "path": "/tmp",
                "total_bytes": 500 * GIB_BYTES,
                "free_bytes": 108 * GIB_BYTES,
                "mount": "/",
            },
        ],
        "gpus": [
            {
                "index": 0,
                "name": "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
                "uuid": "GPU-fixture",
                "memory": {
                    "total_mib": 97887,
                    "used_mib": 457,
                    "free_mib": 96793,
                },
                "driver_version": "595.71.05",
                "compute_capability": "12.0",
                "power_limit_w": 300.0,
                "ecc_mode": "Disabled",
                "persistence_mode": "Disabled",
            }
        ],
        "processes": {
            "compute": [],
            "graphics": [
                {"gpu_index": 0, "pid": 1482, "type": "G", "process_name": "/usr/lib/xorg/Xorg"},
                {"gpu_index": 0, "pid": 2064, "type": "G", "process_name": "/usr/bin/gnome-shell"},
                {"gpu_index": 0, "pid": 3179, "type": "G", "process_name": "/usr/bin/rustdesk"},
                {"gpu_index": 0, "pid": 4281, "type": "G", "process_name": "/usr/bin/obs"},
            ],
        },
        "torch": {
            "version": "2.11.0+cu128",
            "cuda": "12.8",
            "arch_list": ["sm_120"],
            "bf16_supported": True,
        },
    }


def _write_probe(path: Path, payload: Mapping[str, JsonValue]) -> None:
    _ = path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def probe(tmp_path: Path) -> ProbeFacts:
    path = tmp_path / "probe.json"
    _write_probe(path, _probe_document())
    return parse_probe_json(path)


def test_parse_probe_when_capture_is_complete_returns_frozen_typed_facts(
    probe: ProbeFacts,
) -> None:
    # Given/When: the probe script JSON plus supplied allocation facts is parsed.

    # Then: the boundary retains live resource facts without importing a GPU stack.
    assert probe.hostname == "dcs33979"
    assert probe.vram_free_mib == 96793
    assert probe.cuda_runtime == "12.8"
    assert probe.bf16_supported is True
    assert probe.graphics_processes == ("Xorg", "gnome-shell", "rustdesk", "obs")
    with pytest.raises(FrozenInstanceError):
        setattr(probe, "hostname", "other-host")


def test_parse_probe_when_bf16_fact_is_missing_fails_closed(tmp_path: Path) -> None:
    # Given: the captured probe has no separately supplied BF16 capability fact.
    payload = _probe_document()
    torch = payload["torch"]
    assert isinstance(torch, dict)
    del torch["bf16_supported"]
    path = tmp_path / "probe.json"
    _write_probe(path, payload)

    # When/Then: parsing refuses to infer BF16 support from the GPU model.
    with pytest.raises(PreflightError) as captured:
        _ = parse_probe_json(path)
    assert captured.value.field == "torch.bf16_supported"


def test_probe_parser_output_runs_happy_preflight(tmp_path: Path) -> None:
    # Given: the exact JSON shape emitted by the GPU probe parser.
    probe_path = tmp_path / "probe.json"
    _write_probe(probe_path, _probe_document())

    # When: the preflight consumes that parsed probe without allocation facts.
    probe = parse_probe_json(probe_path)
    result = validate_preflight(
        PreflightInputs(
            config=load_config(CONFIG_PATH),
            profile=load_host_profile(HOST_PATH),
            probe=probe,
            cached_bytes=4_000_000_000,
        )
    )

    # Then: the real probe contract passes all preserved resource gates.
    assert result.valid is True


def test_missing_model_bytes_uses_conservative_checkpoint_size() -> None:
    # Given: part, all, or more than all checkpoint bytes are already cached.

    # When/Then: only absent bytes are budgeted and the result never becomes negative.
    assert MODEL_STORAGE_BYTES == 55_562_855_904
    assert missing_model_bytes(4_000_000_000) == 51_562_855_904
    assert missing_model_bytes(MODEL_STORAGE_BYTES + 1) == 0


def test_validate_when_known_graphics_are_live_returns_warnings_and_protections(
    probe: ProbeFacts,
) -> None:
    # Given: the canonical host has four graphics clients but no compute process.
    inputs = PreflightInputs(
        config=load_config(CONFIG_PATH),
        profile=load_host_profile(HOST_PATH),
        probe=probe,
        cached_bytes=4_000_000_000,
    )

    # When: all allocation gates are evaluated from captured facts.
    result = validate_preflight(inputs)

    # Then: the desktop is accepted, risky host facts warn, and legacy caches are protected.
    assert result.valid is True
    assert result.missing_model_bytes == 51_562_855_904
    assert result.remaining_disk_bytes >= result.reserve_bytes
    assert result.require_peak_rss_measurement is True
    assert frozenset(result.warnings) == frozenset(
        ("limited_ram", "swap_nearly_full", "ecc_disabled", "persistence_disabled")
    )
    assert result.protected_cache_names == PROTECTED_CACHE_NAMES


UNSAFE_CASES: Final[
    tuple[tuple[Callable[[ProbeFacts], ProbeFacts], str], ...]
] = (
    (lambda value: replace(value, hostname="wrong-host"), "hostname"),
    (lambda value: replace(value, gpu_count=2), "gpus"),
    (lambda value: replace(value, compute_capability="11.0"), "gpus[0].compute_capability"),
    (lambda value: replace(value, cuda_runtime="12.6"), "torch.cuda"),
    (lambda value: replace(value, bf16_supported=False), "bf16_supported"),
    (lambda value: replace(value, vram_free_mib=96792), "gpus[0].memory.free_mib"),
    (lambda value: replace(value, disk_free_bytes=0), "disks.home.free_bytes"),
    (lambda value: replace(value, compute_processes=("python",)), "processes.compute"),
)


@pytest.mark.parametrize(
    ("mutate", "expected_field"),
    UNSAFE_CASES,
    ids=(
        "wrong-host",
        "multiple-gpus",
        "wrong-compute-capability",
        "wrong-cuda",
        "missing-bf16",
        "low-vram",
        "low-disk",
        "compute-process",
    ),
)
def test_validate_when_allocation_fact_is_unsafe_fails_closed(
    probe: ProbeFacts,
    mutate: Callable[[ProbeFacts], ProbeFacts],
    expected_field: str,
) -> None:
    # Given: exactly one captured allocation fact violates the canonical profile.
    unsafe_probe = mutate(probe)
    inputs = PreflightInputs(
        config=load_config(CONFIG_PATH),
        profile=load_host_profile(HOST_PATH),
        probe=unsafe_probe,
        cached_bytes=0,
    )

    # When/Then: the typed gate identifies that fact and refuses allocation.
    with pytest.raises(PreflightError) as captured:
        _ = validate_preflight(inputs)
    assert captured.value.field == expected_field


def test_cli_writes_atomic_json_and_direct_script_help_is_available(
    tmp_path: Path,
) -> None:
    # Given: complete captured facts and an output path in an existing directory.
    probe_path = tmp_path / "probe.json"
    output_path = tmp_path / "preflight.json"
    _write_probe(probe_path, _probe_document())

    # When: the CPU-only preflight is invoked through its real script surface.
    result = subprocess.run(
        (
            sys.executable,
            str(SCRIPT_PATH),
            "--config",
            str(CONFIG_PATH),
            "--host-profile",
            str(HOST_PATH),
            "--probe-json",
            str(probe_path),
            "--cached-bytes",
            "0",
            "--output",
            str(output_path),
        ),
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    # Then: direct execution succeeds and publishes one complete JSON document.
    assert result.returncode == 0, result.stderr
    assert json.loads(output_path.read_text(encoding="utf-8"))["result"]["valid"] is True


def test_preflight_evidence_when_result_is_forged_rejects(tmp_path: Path) -> None:
    # Given: a successful preflight document whose reported result is altered.
    from RQ.SLM.unsloth.preflight import main, verify_preflight_evidence

    probe_path = tmp_path / "probe.json"
    output_path = tmp_path / "preflight.json"
    _write_probe(probe_path, _probe_document())
    assert main((
        "--config", str(CONFIG_PATH),
        "--host-profile", str(HOST_PATH),
        "--probe-json", str(probe_path),
        "--cached-bytes", "0",
        "--output", str(output_path),
    )) == 0
    before, marker, after = output_path.read_text(encoding="utf-8").rpartition('"valid": true')
    assert marker == '"valid": true'
    _ = output_path.write_text(f'{before}"valid": false{after}', encoding="utf-8")

    # When/Then: validation recomputes the result instead of trusting the artifact.
    with pytest.raises(PreflightError):
        _ = verify_preflight_evidence(output_path, CONFIG_PATH, HOST_PATH)


def test_preflight_source_has_no_cache_deletion_api() -> None:
    # Given/When: the complete preflight source is inspected as a safety boundary.
    source = SCRIPT_PATH.read_text(encoding="utf-8")

    # Then: it cannot remove either protected cache or any other filesystem path.
    assert all(
        deletion_api not in source
        for deletion_api in (".unlink(", "os.remove(", "shutil.rmtree(", ".rmdir(")
    )
