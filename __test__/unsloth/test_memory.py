from __future__ import annotations

import hashlib
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from RQ.SLM.unsloth._memory_types import (
    DEFAULT_LENGTHS,
    JSON_DECODER,
    ChildRequest,
    JsonObject,
    Measurement,
    ParentOptions,
    ProbeBatch,
    json_bytes,
)
from RQ.SLM.unsloth._memory_worker import measure_step
from RQ.SLM.unsloth.memory import (
    cuda_failure_class,
    parse_command,
    run_child_process,
    run_parent,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class _PadlessTokenizer:
    pad_token_id: int | None = None


def _json_object(payload: str) -> JsonObject:
    decoded = JSON_DECODER.decode(payload)
    assert isinstance(decoded, dict)
    return decoded


def _options(output: Path) -> ParentOptions:
    return ParentOptions(
        config_path=Path("RQ/SLM/unsloth/configs/qwen3_6_27b.yml"),
        host_profile_path=Path("RQ/SLM/configs/hosts/blackwell-rtx-pro-6000.yml"),
        lengths=DEFAULT_LENGTHS,
        output_directory=output,
    )


def _success(length: int) -> Measurement:
    return Measurement(
        requested_length=length,
        status="completed",
        failure_class=None,
        input_shape=(1, length),
        unpadded_length=length - 2,
        supervised_token_count=2,
        loss_is_finite=True,
        global_grad_norm=1.0,
        in_proj_a_grad_norm=0.5,
        in_proj_a_grad_is_finite=True,
        in_proj_b_grad_norm=0.5,
        in_proj_b_grad_is_finite=True,
        optimizer_state_allocated=True,
        optimizer_step_succeeded=True,
        packing_enabled=False,
        max_memory_allocated_bytes=10,
        max_memory_reserved_bytes=12,
        pre_step_free_vram_bytes=100,
        post_step_free_vram_bytes=20,
        host_peak_rss_bytes=30,
        wall_time_seconds=0.1,
    )


def test_parse_command_when_output_is_omitted_uses_evidence_directory() -> None:
    # Given: the operator supplies no output override.

    # When: the parent command is parsed.
    command = parse_command(())

    # Then: generated qualification evidence stays outside research results.
    assert isinstance(command, ParentOptions)
    assert command.output_directory == Path(".omo/evidence/unsloth")


def test_measure_step_when_pad_token_is_missing_records_terminal_evidence() -> None:
    # Given: a tokenizer has neither a pad token nor permission to reuse EOS.
    tokenizer = _PadlessTokenizer()
    batch = ProbeBatch(2048, (11, 12), (-100, 12))

    # When: the measurement boundary validates padding before touching the GPU.
    measurement = measure_step(None, tokenizer, 1e-5, batch)

    # Then: the missing token is terminal evidence rather than token 0 or EOS padding.
    assert measurement.status == "terminal_failure"
    assert measurement.failure_class == "pad_token_unavailable"


def test_run_parent_when_all_measurements_qualify_approves_16384(tmp_path: Path) -> None:
    # Given: one isolated successful child result for every required length.
    requests: list[ChildRequest] = []

    def runner(request: ChildRequest) -> Measurement:
        requests.append(request)
        return _success(request.length)

    # When: the ordered ladder is measured.
    options = _options(tmp_path)
    qualification = run_parent(options, child_runner=runner)

    # Then: unbound probe output remains diagnostic until preflight binds it.
    assert tuple(request.length for request in requests) == DEFAULT_LENGTHS
    assert len({request.result_path for request in requests}) == len(DEFAULT_LENGTHS)
    assert qualification.status == "requires_owner_decision"
    assert qualification.approved_max_seq_length == 16384
    measurements_payload = (tmp_path / "measurements.jsonl").read_bytes()
    measurements = [
        _json_object(line)
        for line in measurements_payload.decode("utf-8").splitlines()
    ]
    payload = _json_object((tmp_path / "qualification.json").read_text(encoding="utf-8"))
    assert [measurement["requested_length"] for measurement in measurements] == list(DEFAULT_LENGTHS)
    assert all(measurement["packing_enabled"] is False for measurement in measurements)
    assert payload["status"] == "requires_owner_decision"
    expected_hashes = {
        "config_sha256": hashlib.sha256(options.config_path.read_bytes()).hexdigest(),
        "host_profile_sha256": hashlib.sha256(options.host_profile_path.read_bytes()).hexdigest(),
        "evidence_sha256": hashlib.sha256(measurements_payload).hexdigest(),
    }
    assert {key: payload[key] for key in expected_hashes} == expected_hashes


def test_run_parent_when_custom_singleton_ladder_reaches_16384_remains_diagnostic(
    tmp_path: Path,
) -> None:
    # Given: a successful 16384 measurement without the required ordered ladder.
    options = replace(_options(tmp_path), lengths=(16384,))

    # When: the diagnostic custom ladder is measured.
    qualification = run_parent(options, child_runner=lambda request: _success(request.length))

    # Then: a custom ladder cannot self-authorize full training.
    assert qualification.status == "requires_owner_decision"


def test_cuda_failure_class_when_runtime_reports_oom_is_terminal() -> None:
    # Given: the isolated child raises the CUDA allocator's OOM message.
    error = RuntimeError("CUDA out of memory while allocating a tensor")

    # When: the child boundary classifies the runtime failure.
    failure_class = cuda_failure_class(error)

    # Then: the parent receives the terminal OOM evidence class.
    assert failure_class == "cuda_oom"


def test_run_parent_when_oom_stops_larger_children_and_marks_boundary(tmp_path: Path) -> None:
    # Given: the third child reports the fatal CUDA OOM boundary.
    requests: list[int] = []

    def runner(request: ChildRequest) -> Measurement:
        requests.append(request.length)
        if request.length == 8192:
            return replace(_success(request.length), status="terminal_failure", failure_class="cuda_oom")
        return _success(request.length)

    # When: the parent reaches the fatal boundary.
    qualification = run_parent(_options(tmp_path), child_runner=runner)

    # Then: later lengths have no process reuse and are explicitly not attempted.
    assert requests == [2048, 4096, 8192]
    assert qualification.status == "requires_owner_decision"
    assert qualification.first_failure_boundary == 8192
    evidence = [
        _json_object(line)
        for line in (tmp_path / "measurements.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [measurement["status"] for measurement in evidence] == [
        "completed",
        "completed",
        "terminal_failure",
        "not_attempted_after_boundary",
        "not_attempted_after_boundary",
    ]
    assert [measurement["failure_class"] for measurement in evidence[2:]] == [
        "cuda_oom",
        "boundary_reached",
        "boundary_reached",
    ]


def test_run_child_process_when_stale_result_exists_discards_it_before_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: a prior child result that a new child process must not reuse.
    import json
    import subprocess

    from RQ.SLM.unsloth import memory as probe_unsloth_memory

    request = ChildRequest(2048, tmp_path / ".children" / "2048.json")
    _ = request.result_path.parent.mkdir(parents=True)
    _ = request.result_path.write_text(json.dumps(_success(2048).as_json()), encoding="utf-8")
    def child_exits_without_result(*args: str, **kwargs: str) -> subprocess.CompletedProcess[str]:
        _ = (args, kwargs)
        return subprocess.CompletedProcess((), 0)

    monkeypatch.setattr(
        "RQ.SLM.unsloth.memory.subprocess.run", child_exits_without_result
    )

    # When: the replacement child produces no new result.
    measurement = probe_unsloth_memory.run_child_process(_options(tmp_path), request)

    # Then: the stale completion cannot be accepted as this child run's evidence.
    assert measurement.status == "terminal_failure"
    assert measurement.failure_class == "child_exit_0"


def test_run_child_process_when_paths_are_relative_hands_the_child_absolute_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: a parent configured the way the CLI configures it — every path relative to the
    # repo root. The child runs under its own working directory, so any path still relative
    # when it crosses the boundary resolves against the wrong root and the rung dies before
    # it allocates anything.
    from RQ.SLM.unsloth import memory as probe_unsloth_memory

    recorded_command: list[str] = []
    recorded_cwd: list[Path | None] = []

    def record(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        recorded_command.extend(command)
        cwd = kwargs.get("cwd")
        recorded_cwd.append(cwd if isinstance(cwd, Path) else None)
        return subprocess.CompletedProcess((), 1, "", "")

    monkeypatch.setattr("RQ.SLM.unsloth.memory.subprocess.run", record)
    request = ChildRequest(2048, tmp_path / ".children" / "2048.json")

    # When: the parent spawns a rung.
    _ = probe_unsloth_memory.run_child_process(_options(tmp_path), request)

    # Then: nothing the child must open is left for its own cwd to interpret, and the two
    # inputs it reads immediately resolve to files that actually exist.
    for flag in ("--config", "--host-profile", "--output", "--child-result"):
        value = Path(recorded_command[recorded_command.index(flag) + 1])
        assert value.is_absolute(), f"{flag} crossed the process boundary as a relative path"
    for flag in ("--config", "--host-profile"):
        assert Path(recorded_command[recorded_command.index(flag) + 1]).is_file()
    assert recorded_cwd == [REPO_ROOT]


def test_run_child_process_when_the_child_fails_preserves_its_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Given: a child that dies with a traceback, which is the only account of why the rung
    # failed — the measurement itself records nothing beyond an exit code.
    from RQ.SLM.unsloth import memory as probe_unsloth_memory

    def failing_child(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        _ = (args, kwargs)
        return subprocess.CompletedProcess((), 1, "", "FileNotFoundError: missing host profile\n")

    monkeypatch.setattr("RQ.SLM.unsloth.memory.subprocess.run", failing_child)
    request = ChildRequest(2048, tmp_path / ".children" / "2048.json")

    # When: the rung fails.
    measurement = probe_unsloth_memory.run_child_process(_options(tmp_path), request)

    # Then: the reason survives both on stderr and on disk, beside the result it never wrote.
    assert measurement.status == "terminal_failure"
    assert "FileNotFoundError" in capsys.readouterr().err
    saved = (tmp_path / ".children" / "2048.stderr").read_text(encoding="utf-8")
    assert "FileNotFoundError" in saved


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("in_proj_a_grad_is_finite", False),
        ("post_step_free_vram_bytes", 9),
        ("supervised_token_count", 0),
    ),
)
def test_run_parent_when_qualification_evidence_is_invalid_rejects_target(
    tmp_path: Path, field: str, value: bool | int
) -> None:
    # Given: the first child has one disqualifying optimizer-step fact.
    def runner(request: ChildRequest) -> Measurement:
        result = _success(request.length)
        return replace(result, **{field: value}) if request.length == 2048 else result

    # When: all lengths are still isolated and measured.
    qualification = run_parent(_options(tmp_path), child_runner=runner)

    # Then: no unqualified length is approved.
    assert qualification.status == "requires_owner_decision"
    assert qualification.approved_max_seq_length is None
    assert qualification.first_failure_boundary == 2048


def test_run_parent_when_in_proj_a_norm_is_nan_rejects_target(tmp_path: Path) -> None:
    # Given: a known hybrid-projection failure with a non-finite in_proj_a norm.
    def runner(request: ChildRequest) -> Measurement:
        result = _success(request.length)
        if request.length == 2048:
            return replace(result, in_proj_a_grad_norm=float("nan"), in_proj_a_grad_is_finite=False)
        return result

    # When: the ordered probe records the full ladder.
    qualification = run_parent(_options(tmp_path), child_runner=runner)

    # Then: a NaN target gradient prevents every approval claim.
    assert qualification.status == "requires_owner_decision"
    assert qualification.approved_max_seq_length is None


def test_run_child_process_when_invoked_builds_a_fresh_self_subprocess(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Given: a child result written by a fake subprocess implementation.
    request = ChildRequest(2048, tmp_path / "child.json")
    _ = request.result_path.write_bytes(json_bytes(_success(2048).as_json()))
    calls: list[list[str]] = []

    def fake_run(
        command: list[str],
        *,
        cwd: Path,
        capture_output: bool,
        text: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        _ = cwd, capture_output, text, check
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    # When: the parent dispatches its child-only command.
    result = run_child_process(_options(tmp_path), request)

    # Then: the current interpreter re-enters this file exactly once with one length.
    assert result.requested_length == 2048
    # Paths are absolute: the child does not share this process's working directory, so a
    # relative path would resolve against a different root inside it.
    assert calls == [
        [
            sys.executable,
            str(REPO_ROOT / "RQ/SLM/unsloth/memory.py"),
            "--config",
            str(Path("RQ/SLM/unsloth/configs/qwen3_6_27b.yml").resolve()),
            "--host-profile",
            str(Path("RQ/SLM/configs/hosts/blackwell-rtx-pro-6000.yml").resolve()),
            "--output",
            str(tmp_path.resolve()),
            "--child-length",
            "2048",
            "--child-result",
            str(request.result_path.resolve()),
        ]
    ]


def test_cli_help_when_heavy_imports_are_blocked_succeeds() -> None:
    # Given: an interpreter where GPU and dataset packages cannot be imported.
    code = """
import builtins
real_import = builtins.__import__
blocked = {"torch", "transformers", "unsloth", "trl", "datasets"}
def guarded_import(name, *args, **kwargs):
    if name.split(".")[0] in blocked:
        raise RuntimeError(f"heavy import: {name}")
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from RQ.SLM.unsloth.memory import main
main(("--help",))
"""

    # When: help is rendered before the child workload is selected.
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )

    # Then: the operator can inspect the contract without a GPU stack.
    assert completed.returncode == 0, completed.stderr
    assert "--child-length" not in completed.stdout


def test_help_when_invoked_directly_is_available_without_unsloth() -> None:
    # Given: the directly executable memory-qualification path on this CPU-only host.
    command = (sys.executable, str(REPO_ROOT / "RQ/SLM/unsloth/memory.py"), "--help")

    # When: the file is launched as a bare script, not via -m.
    result = subprocess.run(command, capture_output=True, text=True, check=False)

    # Then: the package bootstrap resolves RQ.SLM.unsloth without a ModuleNotFoundError.
    assert result.returncode == 0, result.stderr
    assert "ModuleNotFoundError" not in result.stderr
    assert "--lengths" in result.stdout
