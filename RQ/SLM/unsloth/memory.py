from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final, assert_never

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "RQ.SLM.unsloth"

from RQ.SLM.unsloth._memory_types import (
    DEFAULT_LENGTHS,
    ChildRequest,
    Measurement,
    ParentOptions,
    ProbeError,
    Qualification,
    has_full_qualification_ladder,
    json_bytes,
    measurement_qualifies,
    read_measurement,
    sha256_file,
    write_bytes_atomic,
    write_json_atomic,
)

DEFAULT_CONFIG_PATH: Final[Path] = Path("RQ/SLM/unsloth/configs/qwen3_6_27b.yml")
DEFAULT_HOST_PROFILE_PATH: Final[Path] = Path("RQ/SLM/configs/hosts/blackwell-rtx-pro-6000.yml")
DEFAULT_OUTPUT_DIRECTORY: Final[Path] = Path(".omo/evidence/unsloth")
REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
_CHILD_ERROR_LINES: Final[int] = 12
ChildRunner = Callable[[ChildRequest], Measurement]


@dataclass(frozen=True, slots=True)
class ChildOptions:
    parent_options: ParentOptions
    request: ChildRequest


@dataclass(slots=True)
class _Arguments:
    config: Path = DEFAULT_CONFIG_PATH
    host_profile: Path = DEFAULT_HOST_PROFILE_PATH
    lengths: list[int] | tuple[int, ...] = DEFAULT_LENGTHS
    output: Path = DEFAULT_OUTPUT_DIRECTORY
    preflight: Path | None = None
    child_length: int | None = None
    child_result: Path | None = None


def parse_command(argv: Sequence[str] | None = None) -> ParentOptions | ChildOptions:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    _ = parser.add_argument("--host-profile", type=Path, default=DEFAULT_HOST_PROFILE_PATH)
    _ = parser.add_argument("--lengths", type=int, nargs="+", default=DEFAULT_LENGTHS)
    _ = parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    _ = parser.add_argument("--preflight", type=Path, help=argparse.SUPPRESS)
    _ = parser.add_argument("--child-length", type=int, help=argparse.SUPPRESS)
    _ = parser.add_argument("--child-result", type=Path, help=argparse.SUPPRESS)
    namespace = _Arguments()
    _ = parser.parse_args(argv, namespace=namespace)
    lengths = tuple(namespace.lengths)
    _validate_lengths(parser, lengths)
    if namespace.child_length is None:
        if namespace.child_result is not None:
            parser.error("--child-result requires --child-length")
        return ParentOptions(namespace.config, namespace.host_profile, lengths, namespace.output, namespace.preflight or namespace.output / "preflight.json")
    if namespace.child_result is None:
        parser.error("--child-length requires --child-result")
    parent = ParentOptions(namespace.config, namespace.host_profile, lengths, namespace.output, namespace.preflight or namespace.output / "preflight.json")
    return ChildOptions(parent, ChildRequest(namespace.child_length, namespace.child_result))


def _validate_lengths(parser: argparse.ArgumentParser, lengths: tuple[int, ...]) -> None:
    if not lengths or any(length <= 0 for length in lengths):
        parser.error("--lengths must contain positive sequence lengths")
    if tuple(sorted(set(lengths))) != lengths:
        parser.error("--lengths must be unique and strictly ascending")


def run_parent(options: ParentOptions, child_runner: ChildRunner | None = None) -> Qualification:
    preflight = None
    if options.preflight_path is not None:
        from RQ.SLM.unsloth.preflight import verify_preflight_evidence

        preflight = verify_preflight_evidence(
            options.preflight_path, options.config_path, options.host_profile_path
        )
    runner = child_runner or _runner_for(options)
    measurements: list[Measurement] = []
    stopped = False
    for length in options.lengths:
        if stopped:
            measurements.append(Measurement(length, "not_attempted_after_boundary", "boundary_reached"))
            continue
        request = ChildRequest(length, options.output_directory / ".children" / f"{length}.json")
        measurement = runner(request)
        if measurement.requested_length != length:
            measurement = Measurement(length, "terminal_failure", "child_length_mismatch")
        measurements.append(measurement)
        if measurement.status == "terminal_failure":
            stopped = True
    measurements_payload = b"".join(json_bytes(measurement.as_json()) for measurement in measurements)
    write_bytes_atomic(options.output_directory / "measurements.jsonl", measurements_payload)
    qualification = _qualify(
        options,
        tuple(measurements),
        measurements_payload,
        preflight_hostname=None if preflight is None else preflight.hostname,
        preflight_sha256=None if options.preflight_path is None else sha256_file(options.preflight_path),
    )
    write_json_atomic(options.output_directory / "qualification.json", qualification.as_json())
    return qualification


def _runner_for(options: ParentOptions) -> ChildRunner:
    return lambda request: run_child_process(options, request)


def run_child_process(options: ParentOptions, request: ChildRequest) -> Measurement:
    # Every path is resolved before it crosses the process boundary. The child runs with its
    # own working directory, so a relative path taken from the parent's command line resolves
    # against the wrong root inside it — which is how the ladder died on its first rung with a
    # plain missing-file error, reported as nothing more informative than "child_exit_1".
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        str(options.config_path.resolve()),
        "--host-profile",
        str(options.host_profile_path.resolve()),
        "--output",
        str(options.output_directory.resolve()),
        "--child-length",
        str(request.length),
        "--child-result",
        str(request.result_path.resolve()),
    ]
    _ = request.result_path.unlink(missing_ok=True)
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return Measurement(request.length, "terminal_failure", "child_launch_error")
    _preserve_child_output(request, completed)
    measurement = _child_result(request, completed.returncode)
    if completed.returncode != 0 and measurement.status == "completed":
        return replace(measurement, status="terminal_failure", failure_class="child_exit_nonzero")
    return measurement


def _preserve_child_output(
    request: ChildRequest, completed: subprocess.CompletedProcess[str]
) -> None:
    """Surface a failed child's output instead of discarding it.

    The child's streams are captured so they cannot interleave with the parent's, but a
    captured stream nobody reads is worse than no capture at all: a rung that died on a
    missing file was recorded as `child_exit_1` and the traceback explaining it was dropped
    on the floor. A ladder rung is expensive enough that its failure must say why.
    """
    if completed.returncode == 0:
        return
    captured = "".join(part for part in (completed.stdout, completed.stderr) if part)
    if not captured:
        return
    log_path = request.result_path.parent / f"{request.length}.stderr"
    saved = True
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        _ = log_path.write_text(captured, encoding="utf-8")
    except OSError:
        saved = False
    print(
        f"[memory] child for length {request.length} exited {completed.returncode}:",
        file=sys.stderr,
    )
    for line in captured.strip().splitlines()[-_CHILD_ERROR_LINES:]:
        print(f"[memory]   {line}", file=sys.stderr)
    if saved:
        print(f"[memory]   full child output: {log_path}", file=sys.stderr)


def _child_result(request: ChildRequest, return_code: int) -> Measurement:
    if not request.result_path.is_file():
        return Measurement(request.length, "terminal_failure", f"child_exit_{return_code}")
    try:
        measurement = read_measurement(request.result_path)
    except ProbeError:
        return Measurement(request.length, "terminal_failure", "invalid_child_evidence")
    if measurement.requested_length != request.length:
        return Measurement(request.length, "terminal_failure", "child_length_mismatch")
    return measurement


def _qualify(
    options: ParentOptions,
    measurements: tuple[Measurement, ...],
    evidence_payload: bytes,
    preflight_hostname: str | None,
    preflight_sha256: str | None,
) -> Qualification:
    approved: int | None = None
    boundary: int | None = None
    for measurement in measurements:
        if _measurement_qualifies(measurement) and boundary is None:
            approved = measurement.requested_length
        elif boundary is None:
            boundary = measurement.requested_length
    full_ladder = options.lengths == DEFAULT_LENGTHS and has_full_qualification_ladder(measurements)
    status = "approved_16384" if full_ladder and preflight_hostname is not None else "requires_owner_decision"
    return Qualification(
        status=status,
        approved_max_seq_length=16384 if full_ladder and preflight_hostname is not None else approved,
        first_failure_boundary=boundary,
        lengths=options.lengths,
        config_sha256=sha256_file(options.config_path),
        host_profile_sha256=sha256_file(options.host_profile_path),
        evidence_sha256=hashlib.sha256(evidence_payload).hexdigest(),
        preflight_hostname=preflight_hostname,
        preflight_sha256=preflight_sha256,
    )


def _measurement_qualifies(measurement: Measurement) -> bool:
    return measurement_qualifies(measurement)


def run_child(options: ChildOptions) -> int:
    from RQ.SLM.unsloth._memory_worker import run_probe_child

    try:
        measurement = run_probe_child(options.parent_options, options.request)
    except RuntimeError as error:
        failure_class = cuda_failure_class(error)
        if failure_class is None:
            raise
        measurement = Measurement(options.request.length, "terminal_failure", failure_class)
    write_json_atomic(options.request.result_path, measurement.as_json())
    return 0 if measurement.status == "completed" else 2


def cuda_failure_class(error: RuntimeError) -> str | None:
    message = str(error).casefold()
    if "out of memory" in message:
        return "cuda_oom"
    if "cuda" in message:
        return "cuda_runtime"
    return None


_cuda_failure_class = cuda_failure_class


def main(argv: Sequence[str] | None = None) -> None:
    command = parse_command(argv)
    match command:
        case ParentOptions():
            _ = run_parent(command)
            return
        case ChildOptions():
            raise SystemExit(run_child(command))
    assert_never(command)


if __name__ == "__main__":
    main()
