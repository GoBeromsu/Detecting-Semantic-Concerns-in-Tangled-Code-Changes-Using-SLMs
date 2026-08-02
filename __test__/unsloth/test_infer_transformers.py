from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import NoReturn

import pytest

from RQ.SLM.unsloth.results import expected_result_paths
from RQ.SLM.unsloth.data import DatasetRow
from RQ.SLM.unsloth.infer_options import (
    EvaluationOptions,
    VerifyOptions,
    parse_command,
)
from RQ.SLM.unsloth.infer import (
    main,
    run_evaluation,
)
from RQ.SLM.unsloth.generation import (
    GenerationRequest,
    PeftLoadRequest,
    GenerationError,
)
from utils.llms.constant import DEFAULT_DF_COLUMNS


REPO_ROOT = Path(__file__).resolve().parents[2]


class FakeBackend:
    def __init__(self) -> None:
        self.calls: list[int] = []
        self.fail_at: int | None = None
        self.load_count: int = 0
        self.validated: list[Path] = []

    def generate(self, request: GenerationRequest) -> tuple[str, ...]:
        self.calls.append(request.seed)
        if self.fail_at is not None and len(self.calls) == self.fail_at:
            raise GenerationError("generation", "simulated failure", "raw")
        return ("fix",)


class FakeModelConfig:
    id: str = "Qwen/Qwen3.6-27B"
    revision: str = "pinned-revision"


class FakeTrainingConfig:
    max_seq_length: int = 16384


class FakeConfig:
    model: FakeModelConfig = FakeModelConfig()
    training: FakeTrainingConfig = FakeTrainingConfig()


def _row(index: int) -> DatasetRow:
    return DatasetRow(
        commit_message=f"message {index}",
        diff='["diff"]',
        concern_count=1,
        shas=json.dumps([f"sha-{index}"]),
        types='["fix"]',
        repo="repo",
    )


def _options(output: Path, *, limit: int | None = 1, resume: bool = False) -> EvaluationOptions:
    return EvaluationOptions(
        config_path=Path("config.yml"),
        adapter_path=Path("adapter"),
        data_source="local",
        data_dir=Path("data"),
        contexts=(1024,),
        message_conditions=(False, True),
        seed=42,
        temperature=0.3,
        max_new_tokens=128,
        limit=limit,
        resume=resume,
        output_root=output,
        run_directory=None,
    )


def _patch_runtime(monkeypatch: pytest.MonkeyPatch, backend: FakeBackend) -> None:
    from RQ.SLM.unsloth import infer as infer_transformers

    def load_once(_: PeftLoadRequest) -> FakeBackend:
        backend.load_count += 1
        return backend

    def fake_load_config(_: Path) -> FakeConfig:
        return FakeConfig()

    def fake_load_split(
        _data_source: str, _split: str, _data_dir: Path
    ) -> tuple[DatasetRow, DatasetRow]:
        return (_row(0), _row(1))

    def fake_render_commits(
        rows: Sequence[DatasetRow], _context_len: int, _include_message: bool
    ) -> tuple[str, ...]:
        return tuple(f"commit-{index}" for index, _ in enumerate(rows))

    def validate_adapter(adapter_path: Path, _: Path) -> None:
        backend.validated.append(adapter_path)

    monkeypatch.setattr(infer_transformers, "load_config", fake_load_config)
    monkeypatch.setattr(infer_transformers, "load_split", fake_load_split)
    monkeypatch.setattr(infer_transformers, "load_backend", load_once)
    monkeypatch.setattr(infer_transformers, "_render_commits", fake_render_commits)
    monkeypatch.setattr(infer_transformers, "validate_adapter", validate_adapter, raising=False)


def test_parse_command_when_defaults_are_used_keeps_default_config_and_full_contract() -> None:
    # Given: the ordinary evaluation command with only its required adapter path.
    command = parse_command(("--adapter", "outputs/adapter"))

    # When/Then: defaults preserve the YAML path, sweep ordering, both message conditions, and limits.
    assert isinstance(command, EvaluationOptions)
    assert command.config_path == Path("RQ/SLM/unsloth/configs/qwen3_6_27b.yml")
    assert command.contexts == (12288, 8192, 4096, 2048, 1024)
    assert command.message_conditions == (False, True)
    assert command.max_new_tokens == 128


def test_run_evaluation_when_limited_loads_model_once_and_preserves_path_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Given: a no-GPU backend and the test split's first row only.
    backend = FakeBackend()
    _patch_runtime(monkeypatch, backend)

    # When: both message conditions are evaluated at one context.
    outcome = run_evaluation(_options(tmp_path), started_at=datetime(2026, 7, 31, 1, 2, 3))

    # Then: one loaded backend writes legacy-shaped rows in deterministic msg0/msg1 order.
    assert backend.load_count == 1
    assert len(backend.calls) == 2
    assert outcome.canonical is None
    paths = tuple(outcome.result_files)
    assert paths == (
        tmp_path / "Qwen3.6-27B-LoRA" / "20260731010203" / "msg0" / "1024_zs.csv",
        tmp_path / "Qwen3.6-27B-LoRA" / "20260731010203" / "msg1" / "1024_zs.csv",
    )
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            assert csv.DictReader(handle).fieldnames == DEFAULT_DF_COLUMNS
    assert backend.validated == [Path("adapter")]


def test_run_evaluation_when_generation_fails_writes_resumable_sidecar_without_corrupting_prefix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Given: a model failure after one successful first-condition row.
    backend = FakeBackend()
    backend.fail_at = 2
    _patch_runtime(monkeypatch, backend)

    # When: the run reaches the failing row.
    outcome = run_evaluation(_options(tmp_path, limit=2), started_at=datetime(2026, 7, 31))

    # Then: the successful prefix remains valid and the typed failure is externalized.
    failure_path = outcome.run_directory / "failures.jsonl"
    assert json.loads(failure_path.read_text(encoding="utf-8")) == {
        "row_index": 1,
        "shas": ["sha-1"],
        "context_len": 1024,
        "with_message": False,
        "error_type": "GenerationError",
        "error_message": "simulated failure",
        "raw_output": "raw",
    }
    with outcome.result_files[0].open(newline="", encoding="utf-8") as handle:
        assert len(tuple(csv.DictReader(handle))) == 1


def test_main_when_verify_only_never_loads_ml_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Given: a complete canonical result tree and a loader that would fail if called.
    from RQ.SLM.unsloth import infer as infer_transformers

    for path in expected_result_paths(tmp_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=DEFAULT_DF_COLUMNS)
            writer.writeheader()
            writer.writerows({
                "predicted_types": '{"types":["fix"]}', "actual_types": '["fix"]',
                "inference_time": "0.0", "shas": '["sha"]', "precision": "1.0",
                "recall": "1.0", "f1": "1.0", "exact_match": "True",
                "hamming_loss": "0.0", "context_len": path.stem.removesuffix("_zs"),
                "with_message": str(path.parent.name == "msg1"), "concern_count": "1",
            } for _ in range(350))
    _ = (tmp_path / "failures.jsonl").write_text("", encoding="utf-8")
    def fail_load_backend(_: PeftLoadRequest) -> NoReturn:
        pytest.fail("ML load")

    monkeypatch.setattr(infer_transformers, "load_backend", fail_load_backend)

    # When: verification is invoked through the public CLI path.
    main(("--verify-only", "--output", str(tmp_path)))

    # Then: certification completes without model construction.


def test_parse_command_when_verify_only_returns_no_adapter_command() -> None:
    # Given/When: a verification-only invocation with its run directory.
    command = parse_command(("--verify-only", "--output", "results/run"))

    # Then: it does not require an adapter/model configuration for certification.
    assert isinstance(command, VerifyOptions)
    assert command.run_directory == Path("results/run")


def test_parse_command_when_resume_has_no_explicit_run_directory_rejects() -> None:
    # Given: a resume request without an existing run directory to certify.
    arguments = ("--adapter", "outputs/adapter", "--resume")

    # When/Then: it cannot silently select a fresh timestamped directory.
    with pytest.raises(SystemExit):
        _ = parse_command(arguments)


def test_cli_help_when_optional_ml_imports_are_blocked_succeeds() -> None:
    # Given: the child interpreter rejects optional model/data packages.
    code = """
import builtins
real_import = builtins.__import__
blocked = {"torch", "transformers", "unsloth", "peft", "outlines", "datasets", "pandas", "tiktoken"}
def guarded_import(name, *args, **kwargs):
    if name.split(".")[0] in blocked:
        raise RuntimeError(f"heavy import: {name}")
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from RQ.SLM.unsloth.infer import main
main(("--help",))
"""

    # When: argparse renders help.
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )

    # Then: help is available before the GPU-only stack is imported.
    assert completed.returncode == 0, completed.stderr
    assert "--verify-only" in completed.stdout


def test_help_when_invoked_directly_is_available_without_unsloth() -> None:
    # Given: the directly executable inference path on this CPU-only host.
    command = (sys.executable, str(REPO_ROOT / "RQ/SLM/unsloth/infer.py"), "--help")

    # When: the file is launched as a bare script, not via -m.
    result = subprocess.run(command, capture_output=True, text=True, check=False)

    # Then: the package bootstrap resolves RQ.SLM.unsloth without a ModuleNotFoundError.
    assert result.returncode == 0, result.stderr
    assert "ModuleNotFoundError" not in result.stderr
    assert "--verify-only" in result.stdout
