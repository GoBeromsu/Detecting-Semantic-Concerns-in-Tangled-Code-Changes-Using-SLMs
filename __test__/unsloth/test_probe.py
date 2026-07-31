from __future__ import annotations

from pathlib import Path

import pytest

from RQ.SLM.unsloth._types import PreflightError
from RQ.SLM.unsloth.preflight import parse_probe_json
from RQ.SLM.unsloth.probe import (
    CapturedFacts,
    GpuFacts,
    ProbeCaptureError,
    build_probe_document,
    main,
)

GIB: int = 1024**3
MIB: int = 1024**2


def _facts() -> CapturedFacts:
    return CapturedFacts(
        hostname="dcs33979",
        gpu=GpuFacts(
            name="NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
            compute_capability="12.0",
            total_mib=97887,
            free_mib=96793,
            power_limit_w=300.0,
            ecc_mode="Disabled",
            persistence_mode="Disabled",
        ),
        ram_total_bytes=31 * GIB,
        ram_free_bytes=12 * GIB,
        swap_total_bytes=980 * MIB,
        swap_free_bytes=2 * MIB,
        root_free_bytes=108 * GIB,
        home_free_bytes=205 * GIB,
        compute_process_names=(),
        graphics_process_names=("Xorg", "gnome-shell"),
        cuda_runtime="12.8",
        bf16_supported=True,
    )


def test_build_probe_document_round_trips_through_preflight_parser(tmp_path: Path) -> None:
    # Given: captured host facts assembled into the probe document.
    import json

    document = build_probe_document(_facts())
    probe_path = tmp_path / "gpu-profile.json"
    _ = probe_path.write_text(json.dumps(document), encoding="utf-8")

    # When: preflight parses the written document.
    probe = parse_probe_json(probe_path)

    # Then: every captured fact survives the round-trip.
    assert probe.hostname == "dcs33979"
    assert probe.gpu_name == "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"
    assert probe.gpu_count == 1
    assert probe.compute_capability == "12.0"
    assert probe.vram_total_mib == 97887
    assert probe.vram_free_mib == 96793
    assert probe.power_limit_w == 300.0
    assert probe.ecc_mode == "Disabled"
    assert probe.persistence_mode == "Disabled"
    assert probe.cuda_runtime == "12.8"
    assert probe.bf16_supported is True
    assert probe.compute_processes == ()
    assert probe.graphics_processes == ("Xorg", "gnome-shell")
    assert probe.disk_free_bytes == 205 * GIB


def test_build_probe_document_places_home_at_disks_index_one() -> None:
    # Given/When: the document is built from known root/home free bytes.
    document = build_probe_document(_facts())

    # Then: disks[1] is the home filesystem preflight reads.
    disks = document["disks"]
    assert isinstance(disks, list)
    home = disks[1]
    assert isinstance(home, dict)
    assert home["free_bytes"] == 205 * GIB


def test_main_writes_invalid_document_and_returns_one_on_capture_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: collection fails on the host.
    from RQ.SLM.unsloth import probe

    def broken_collect() -> CapturedFacts:
        raise ProbeCaptureError("nvidia-smi reported no GPU")

    monkeypatch.setattr(probe, "_collect", broken_collect)
    output = tmp_path / "gpu-profile.json"

    # When: the CLI runs.
    exit_code = main(("--output", str(output)))

    # Then: it fails closed with a recorded reason, and preflight rejects it.
    assert exit_code == 1
    with pytest.raises(PreflightError) as captured:
        _ = parse_probe_json(output)
    assert captured.value.field == "valid"
