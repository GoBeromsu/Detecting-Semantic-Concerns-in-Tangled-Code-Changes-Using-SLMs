"""Read-only Blackwell host-fact probe.

Captures the facts `preflight.parse_probe_json` consumes and writes them as a
JSON document. Queries nvidia-smi, /proc, and disk usage only; it never
allocates GPU memory, loads a model, or mutates any device state.
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from shutil import disk_usage
from typing import Final, Protocol, runtime_checkable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "RQ.SLM.unsloth"

from ._types import JsonObject, JsonValue

_KIB_BYTES: Final[int] = 1024
_GPU_FIELDS: Final[str] = (
    "name,compute_cap,memory.total,memory.free,"
    "power.limit,ecc.mode.current,persistence_mode"
)


class _TorchVersion(Protocol):
    cuda: str | None


class _TorchCuda(Protocol):
    def is_bf16_supported(self) -> bool: ...


@runtime_checkable
class _ProbeTorch(Protocol):
    version: _TorchVersion
    cuda: _TorchCuda


@dataclass(frozen=True, slots=True)
class GpuFacts:
    name: str
    compute_capability: str
    total_mib: int
    free_mib: int
    power_limit_w: float
    ecc_mode: str
    persistence_mode: str


@dataclass(frozen=True, slots=True)
class CapturedFacts:
    hostname: str
    gpu: GpuFacts
    ram_total_bytes: int
    ram_free_bytes: int
    swap_total_bytes: int
    swap_free_bytes: int
    root_free_bytes: int
    home_free_bytes: int
    compute_process_names: tuple[str, ...]
    graphics_process_names: tuple[str, ...]
    cuda_runtime: str
    bf16_supported: bool


def build_probe_document(facts: CapturedFacts) -> JsonObject:
    """Assemble the preflight probe schema from captured facts. Pure: no I/O."""
    compute: list[JsonValue] = [{"process_name": name} for name in facts.compute_process_names]
    graphics: list[JsonValue] = [{"process_name": name} for name in facts.graphics_process_names]
    return {
        "valid": True,
        "hostname": facts.hostname,
        "gpus": [
            {
                "name": facts.gpu.name,
                "compute_capability": facts.gpu.compute_capability,
                "memory": {"total_mib": facts.gpu.total_mib, "free_mib": facts.gpu.free_mib},
                "power_limit_w": facts.gpu.power_limit_w,
                "ecc_mode": facts.gpu.ecc_mode,
                "persistence_mode": facts.gpu.persistence_mode,
            }
        ],
        "memory": {
            "ram": {"total_bytes": facts.ram_total_bytes, "free_bytes": facts.ram_free_bytes},
            "swap": {"total_bytes": facts.swap_total_bytes, "free_bytes": facts.swap_free_bytes},
        },
        "disks": [
            {"mount": "/", "free_bytes": facts.root_free_bytes},
            {"mount": "home", "free_bytes": facts.home_free_bytes},
        ],
        "processes": {"compute": compute, "graphics": graphics},
        "torch": {"cuda": facts.cuda_runtime, "bf16_supported": facts.bf16_supported},
    }


def _smi(query: str, extra: str) -> list[str]:
    completed = subprocess.run(
        ("nvidia-smi", query, f"--format=csv,noheader,{extra}"),
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def _collect_gpu() -> GpuFacts:
    rows = _smi(f"--query-gpu={_GPU_FIELDS}", "nounits")
    if not rows:
        raise ProbeCaptureError("nvidia-smi reported no GPU")
    fields = [cell.strip() for cell in rows[0].split(",")]
    if len(fields) != 7:
        raise ProbeCaptureError("unexpected nvidia-smi GPU field count")
    name, capability, total, free, power, ecc, persistence = fields
    return GpuFacts(
        name=name,
        compute_capability=capability,
        total_mib=int(total),
        free_mib=int(free),
        power_limit_w=float(power),
        ecc_mode=ecc,
        persistence_mode=persistence,
    )


def _collect_process_names() -> tuple[str, ...]:
    names: list[str] = []
    for row in _smi("--query-compute-apps=process_name", ""):
        candidate = row.strip()
        if candidate and candidate.casefold() != "[not found]":
            names.append(Path(candidate).name)
    return tuple(names)


def _meminfo_bytes(text: str, key: str) -> int:
    for line in text.splitlines():
        label, _, remainder = line.partition(":")
        if label.strip() == key:
            return int(remainder.strip().split()[0]) * _KIB_BYTES
    raise ProbeCaptureError(f"/proc/meminfo missing {key}")


def _collect_memory() -> tuple[int, int, int, int]:
    text = Path("/proc/meminfo").read_text(encoding="utf-8")
    return (
        _meminfo_bytes(text, "MemTotal"),
        _meminfo_bytes(text, "MemAvailable"),
        _meminfo_bytes(text, "SwapTotal"),
        _meminfo_bytes(text, "SwapFree"),
    )


def _collect_torch() -> tuple[str, bool]:
    torch = importlib.import_module("torch")
    if not isinstance(torch, _ProbeTorch):
        raise ProbeCaptureError("torch does not expose version/cuda facts")
    cuda_runtime = torch.version.cuda
    return (cuda_runtime if cuda_runtime is not None else "", torch.cuda.is_bf16_supported())


def _collect() -> CapturedFacts:
    import socket

    ram_total, ram_free, swap_total, swap_free = _collect_memory()
    cuda_runtime, bf16_supported = _collect_torch()
    return CapturedFacts(
        hostname=socket.gethostname(),
        gpu=_collect_gpu(),
        ram_total_bytes=ram_total,
        ram_free_bytes=ram_free,
        swap_total_bytes=swap_total,
        swap_free_bytes=swap_free,
        root_free_bytes=disk_usage("/").free,
        home_free_bytes=disk_usage(str(Path.home())).free,
        compute_process_names=_collect_process_names(),
        graphics_process_names=(),
        cuda_runtime=cuda_runtime,
        bf16_supported=bf16_supported,
    )


class ProbeCaptureError(Exception):
    pass


class ProbeArguments(argparse.Namespace):
    output: Path = Path()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture read-only Blackwell host facts.")
    _ = parser.add_argument("--output", required=True, type=Path)
    return parser


def _write(path: Path, document: JsonObject) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    arguments = ProbeArguments()
    _ = _parser().parse_args(argv, namespace=arguments)
    output = arguments.output
    try:
        document = build_probe_document(_collect())
    except (OSError, ValueError, subprocess.SubprocessError, ProbeCaptureError) as error:
        _write(output, {"valid": False, "error": str(error)})
        return 1
    _write(output, document)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
