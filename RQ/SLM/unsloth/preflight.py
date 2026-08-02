from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, Never

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "RQ.SLM.unsloth"

from ._types import (
    GIB_BYTES,
    MODEL_STORAGE_BYTES,
    PROTECTED_CACHE_NAMES,
    JsonDecoder,
    JsonValue,
    PreflightArguments,
    PreflightError,
    PreflightEvidence,
    PreflightInputs,
    PreflightResult,
    ProbeFacts,
)

EXPECTED_HOSTNAME, EXPECTED_COMPUTE_CAPABILITY, EXPECTED_CUDA_RUNTIME, EXPECTED_GPU_COUNT, EXPECTED_DISK_RESERVE_GIB = ("dcs33979", "12.0", "12.8", 1, 15)


def _json_decoder() -> JsonDecoder:
    return json.JSONDecoder()


JSON_DECODER: Final[JsonDecoder] = _json_decoder()


def _fail(field: str, reason: str) -> Never:
    raise PreflightError(field=field, reason=reason)


def _mapping(value: JsonValue, field: str) -> Mapping[str, JsonValue]:
    if isinstance(value, dict):
        return value
    _fail(field, "must be a mapping")


def _array(value: JsonValue, field: str) -> list[JsonValue]:
    if isinstance(value, list):
        return value
    _fail(field, "must be an array")


def _required(values: Mapping[str, JsonValue], key: str, field: str) -> JsonValue:
    if key in values:
        return values[key]
    _fail(field, "is required")


def _text(value: JsonValue, field: str) -> str:
    if isinstance(value, str):
        return value
    _fail(field, "must be a string")


def _integer(value: JsonValue, field: str) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    _fail(field, "must be an integer")


def _number(value: JsonValue, field: str) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    _fail(field, "must be numeric")


def _boolean(value: JsonValue, field: str) -> bool:
    if isinstance(value, bool):
        return value
    _fail(field, "must be a boolean")


def _process_names(values: Mapping[str, JsonValue], key: str) -> tuple[str | None, ...]:
    rows = _array(_required(values, key, f"processes.{key}"), f"processes.{key}")
    names: list[str | None] = []
    for index, row in enumerate(rows):
        field = f"processes.{key}[{index}].process_name"
        record = _mapping(row, field)
        name = _required(record, "process_name", field)
        if name is not None and not isinstance(name, str):
            _fail(field, "must be a string or null")
        names.append(None if name is None else Path(name).name)
    return tuple(names)


def _parse_probe(root: Mapping[str, JsonValue]) -> ProbeFacts:
    if not _boolean(_required(root, "valid", "valid"), "valid"):
        _fail("valid", "probe reported invalid facts")
    gpu_rows = _array(_required(root, "gpus", "gpus"), "gpus")
    if not gpu_rows:
        _fail("gpus", "must contain a selected GPU")
    gpu = _mapping(gpu_rows[0], "gpus[0]")
    gpu_memory = _mapping(_required(gpu, "memory", "memory"), "memory")
    memory = _mapping(_required(root, "memory", "memory"), "memory")
    ram = _mapping(_required(memory, "ram", "ram"), "ram")
    swap = _mapping(_required(memory, "swap", "swap"), "swap")
    disks = _array(_required(root, "disks", "disks"), "disks")
    if len(disks) < 2:
        _fail("disks", "must contain the home filesystem")
    home_disk = _mapping(disks[1], "disks[1]")
    processes = _mapping(_required(root, "processes", "processes"), "processes")
    torch = _mapping(_required(root, "torch", "torch"), "torch")
    return ProbeFacts(
        hostname=_text(_required(root, "hostname", "hostname"), "hostname"),
        gpu_name=_text(_required(gpu, "name", "gpus[0].name"), "gpus[0].name"),
        gpu_count=len(gpu_rows),
        compute_capability=_text(_required(gpu, "compute_capability", "gpus[0].compute_capability"), "gpus[0].compute_capability"),
        vram_total_mib=_integer(_required(gpu_memory, "total_mib", "gpus[0].memory.total_mib"), "gpus[0].memory.total_mib"),
        vram_free_mib=_integer(_required(gpu_memory, "free_mib", "gpus[0].memory.free_mib"), "gpus[0].memory.free_mib"),
        power_limit_w=_number(_required(gpu, "power_limit_w", "gpus[0].power_limit_w"), "gpus[0].power_limit_w"),
        ecc_mode=_text(_required(gpu, "ecc_mode", "gpus[0].ecc_mode"), "gpus[0].ecc_mode"),
        persistence_mode=_text(_required(gpu, "persistence_mode", "gpus[0].persistence_mode"), "gpus[0].persistence_mode"),
        cuda_runtime=_text(_required(torch, "cuda", "torch.cuda"), "torch.cuda"),
        bf16_supported=_boolean(_required(torch, "bf16_supported", "torch.bf16_supported"), "torch.bf16_supported"),
        compute_processes=_process_names(processes, "compute"),
        graphics_processes=_process_names(processes, "graphics"),
        ram_total_bytes=_integer(_required(ram, "total_bytes", "memory.ram.total_bytes"), "memory.ram.total_bytes"),
        ram_free_bytes=_integer(_required(ram, "free_bytes", "memory.ram.free_bytes"), "memory.ram.free_bytes"),
        swap_total_bytes=_integer(_required(swap, "total_bytes", "memory.swap.total_bytes"), "memory.swap.total_bytes"),
        swap_free_bytes=_integer(_required(swap, "free_bytes", "memory.swap.free_bytes"), "memory.swap.free_bytes"),
        disk_free_bytes=_integer(_required(home_disk, "free_bytes", "disks.home.free_bytes"), "disks.home.free_bytes"),
    )


def parse_probe_json(path: Path) -> ProbeFacts:
    return _parse_probe(_mapping(_read_json(path), "root"))


def _read_json(path: Path) -> JsonValue:
    try:
        return JSON_DECODER.decode(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PreflightError(str(path), str(error)) from error


def verify_preflight_evidence(
    path: Path, config_path: Path, host_profile_path: Path
) -> PreflightEvidence:
    root = _mapping(_read_json(path), "root")
    expected_keys = {
        "config_sha256",
        "host_profile_sha256",
        "hostname",
        "cached_bytes",
        "probe",
        "result",
    }
    if set(root) != expected_keys:
        _fail("preflight", "has an unexpected evidence shape")
    evidence = PreflightEvidence(
        config_sha256=_text(_required(root, "config_sha256", "config_sha256"), "config_sha256"),
        host_profile_sha256=_text(_required(root, "host_profile_sha256", "host_profile_sha256"), "host_profile_sha256"),
        hostname=_text(_required(root, "hostname", "hostname"), "hostname"),
        cached_bytes=_integer(_required(root, "cached_bytes", "cached_bytes"), "cached_bytes"),
        probe=dict(_mapping(_required(root, "probe", "probe"), "probe")),
        result=dict(_mapping(_required(root, "result", "result"), "result")),
    )
    if evidence.config_sha256 != _sha256_file(config_path):
        _fail("config_sha256", "does not match the current configuration")
    if evidence.host_profile_sha256 != _sha256_file(host_profile_path):
        _fail("host_profile_sha256", "does not match the current host profile")
    from RQ.SLM.unsloth.config import load_config, load_host_profile

    probe = _parse_probe(evidence.probe)
    if evidence.hostname != probe.hostname:
        _fail("hostname", "does not match the captured probe")
    result = validate_preflight(
        PreflightInputs(
            config=load_config(config_path),
            profile=load_host_profile(host_profile_path),
            probe=probe,
            cached_bytes=evidence.cached_bytes,
        )
    )
    if evidence.result != result.as_json():
        _fail("result", "does not match recomputed preflight validation")
    return evidence


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise PreflightError(str(path), str(error)) from error


def missing_model_bytes(cached_bytes: int) -> int:
    if cached_bytes < 0:
        _fail("cached_bytes", "must be non-negative")
    return max(MODEL_STORAGE_BYTES - cached_bytes, 0)


def _require(condition: bool, field: str, reason: str) -> None:
    if not condition:
        _fail(field, reason)


def validate_preflight(inputs: PreflightInputs) -> PreflightResult:
    config, profile, probe = inputs.config, inputs.profile, inputs.probe
    _require(profile.hostname == EXPECTED_HOSTNAME and probe.hostname == profile.hostname, "hostname", "does not match the selected Blackwell profile")
    _require(profile.gpu_count == EXPECTED_GPU_COUNT and probe.gpu_count == EXPECTED_GPU_COUNT, "gpus", "exactly one GPU is required")
    _require(probe.gpu_name == profile.gpu_name, "gpus[0].name", "does not match the host profile")
    _require(probe.compute_capability == EXPECTED_COMPUTE_CAPABILITY and profile.compute_capability == EXPECTED_COMPUTE_CAPABILITY, "gpus[0].compute_capability", "compute capability 12.0 is required")
    _require(probe.vram_total_mib == profile.vram_total_mib, "gpus[0].memory.total_mib", "does not match the host profile")
    _require(round(probe.power_limit_w) == profile.power_cap_w, "gpus[0].power_limit_w", "does not match the 300 W profile")
    _require(probe.cuda_runtime == EXPECTED_CUDA_RUNTIME and profile.cuda_toolkit == EXPECTED_CUDA_RUNTIME, "torch.cuda", "CUDA 12.8 is required")
    _require(probe.bf16_supported, "bf16_supported", "a supplied true BF16 capability fact is required")
    _require(not probe.compute_processes, "processes.compute", "an active compute process holds the GPU")
    _require(probe.vram_free_mib >= profile.vram_free_observed_mib, "gpus[0].memory.free_mib", "is below the observed allocation budget")
    _require(config.disk.min_free_reserve_gib == EXPECTED_DISK_RESERVE_GIB, "disk.min_free_reserve_gib", "must remain 15 GiB")
    missing_bytes = missing_model_bytes(inputs.cached_bytes)
    reserve_bytes = config.disk.min_free_reserve_gib * GIB_BYTES
    remaining_bytes = probe.disk_free_bytes - missing_bytes
    _require(remaining_bytes >= reserve_bytes, "disks.home.free_bytes", "cannot preserve the post-download disk reserve")
    warnings: list[str] = []
    if profile.ram_gib <= 31:
        warnings.append("limited_ram")
    if probe.swap_total_bytes > 0 and probe.swap_free_bytes * 20 <= probe.swap_total_bytes:
        warnings.append("swap_nearly_full")
    if probe.ecc_mode.casefold() == "disabled":
        warnings.append("ecc_disabled")
    if probe.persistence_mode.casefold() == "disabled":
        warnings.append("persistence_disabled")
    return PreflightResult(
        valid=True,
        observed_vram_free_mib=probe.vram_free_mib,
        required_vram_free_mib=profile.vram_free_observed_mib,
        disk_free_bytes=probe.disk_free_bytes,
        cached_bytes=inputs.cached_bytes,
        missing_model_bytes=missing_bytes,
        reserve_bytes=reserve_bytes,
        remaining_disk_bytes=remaining_bytes,
        warnings=tuple(warnings),
        require_peak_rss_measurement="limited_ram" in warnings or "swap_nearly_full" in warnings,
        protected_cache_names=PROTECTED_CACHE_NAMES,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate captured Blackwell facts before GPU allocation.")
    _ = parser.add_argument("--config", required=True, type=Path)
    _ = parser.add_argument("--host-profile", required=True, type=Path)
    _ = parser.add_argument("--probe-json", required=True, type=Path)
    _ = parser.add_argument("--cached-bytes", required=True, type=int)
    _ = parser.add_argument("--output", required=True, type=Path)
    return parser


def _write_atomic(path: Path, payload: Mapping[str, JsonValue]) -> None:
    _ = path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as temporary:
        json.dump(payload, temporary, indent=2, sort_keys=True)
        _ = temporary.write("\n")
        temporary.flush()
        os.fsync(temporary.fileno())
    os.replace(temporary.name, path)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = PreflightArguments()
    _ = _parser().parse_args(argv, namespace=arguments)
    output = arguments.output
    from RQ.SLM.unsloth.config import ConfigurationError, load_config, load_host_profile

    try:
        probe_document = _mapping(_read_json(arguments.probe_json), "root")
        inputs = PreflightInputs(
            config=load_config(arguments.config),
            profile=load_host_profile(arguments.host_profile),
            probe=_parse_probe(probe_document),
            cached_bytes=arguments.cached_bytes,
        )
        payload = PreflightEvidence(
            config_sha256=_sha256_file(arguments.config),
            host_profile_sha256=_sha256_file(arguments.host_profile),
            hostname=inputs.probe.hostname,
            cached_bytes=inputs.cached_bytes,
            probe=dict(probe_document),
            result=validate_preflight(inputs).as_json(),
        ).as_json()
    except (ConfigurationError, PreflightError) as error:
        payload = {"valid": False, "field": error.field, "error": str(error)}
        _write_atomic(output, payload)
        return 1
    _write_atomic(output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
