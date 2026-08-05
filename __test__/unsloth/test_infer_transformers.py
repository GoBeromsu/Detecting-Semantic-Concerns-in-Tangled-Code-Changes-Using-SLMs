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
    InferenceRunError,
    VerifyOptions,
    parse_command,
)
from RQ.SLM.unsloth.infer import (
    EvaluationOutcome,
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
CONFIG_TEXT = "model: {id: Qwen/Qwen3.6-27B}\n"
ADAPTER_BYTES = b"checkpoint-500"


class FakeBackend:
    def __init__(self) -> None:
        self.calls: list[int] = []
        self.fail_calls: frozenset[int] = frozenset()
        self.load_count: int = 0
        self.validated: list[Path] = []
        self.loaded_adapters: list[Path | None] = []

    def generate(self, request: GenerationRequest) -> tuple[str, ...]:
        self.calls.append(request.seed)
        if len(self.calls) in self.fail_calls:
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


def _fixtures(output: Path, config_text: str, adapter_bytes: bytes) -> tuple[Path, Path]:
    """Materialize the config and adapter on disk, since the run identity hashes their bytes."""
    root = output / "fixtures"
    root.mkdir(exist_ok=True)
    config_path = root / "config.yml"
    _ = config_path.write_text(config_text, encoding="utf-8")
    adapter_path = root / "adapter"
    adapter_path.mkdir(exist_ok=True)
    _ = (adapter_path / "adapter_model.safetensors").write_bytes(adapter_bytes)
    return config_path, adapter_path


def _options(
    output: Path,
    *,
    limit: int | None = 1,
    resume: bool = False,
    base: bool = False,
    config_text: str = CONFIG_TEXT,
    adapter_bytes: bytes = ADAPTER_BYTES,
    run_directory: Path | None = None,
) -> EvaluationOptions:
    config_path, adapter_path = _fixtures(output, config_text, adapter_bytes)
    return EvaluationOptions(
        config_path=config_path,
        adapter_path=None if base else adapter_path,
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
        run_directory=run_directory,
    )


def _patch_runtime(
    monkeypatch: pytest.MonkeyPatch, backend: FakeBackend, row_count: int = 2
) -> None:
    from RQ.SLM.unsloth import infer as infer_transformers

    def load_once(request: PeftLoadRequest) -> FakeBackend:
        backend.load_count += 1
        backend.loaded_adapters.append(request.adapter_path)
        return backend

    def fake_load_config(_: Path) -> FakeConfig:
        return FakeConfig()

    def fake_load_split(
        _data_source: str, _split: str, _data_dir: Path
    ) -> tuple[DatasetRow, ...]:
        return tuple(_row(index) for index in range(row_count))

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
    assert backend.validated == [tmp_path / "fixtures" / "adapter"]


def test_run_evaluation_when_base_model_is_selected_skips_adapter_and_uses_its_own_tree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Given: the paired base-model arm of the ablation, with no adapter to attach.
    backend = FakeBackend()
    _patch_runtime(monkeypatch, backend)

    # When: the same sweep runs against the unadapted tower.
    outcome = run_evaluation(
        _options(tmp_path, base=True), started_at=datetime(2026, 7, 31, 1, 2, 3)
    )

    # Then: no adapter is validated or loaded, and results land beside — never inside —
    # the LoRA tree, so the two arms of the ablation can never overwrite each other.
    assert backend.validated == []
    assert backend.loaded_adapters == [None]
    assert outcome.result_files == (
        tmp_path / "Qwen3.6-27B" / "20260731010203" / "msg0" / "1024_zs.csv",
        tmp_path / "Qwen3.6-27B" / "20260731010203" / "msg1" / "1024_zs.csv",
    )


def test_run_evaluation_when_generation_fails_records_it_and_continues_the_sweep(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Given: a model failure after one successful first-condition row.
    backend = FakeBackend()
    backend.fail_calls = frozenset((2,))
    _patch_runtime(monkeypatch, backend)

    # When: the run reaches the failing row.
    outcome = run_evaluation(_options(tmp_path, limit=2), started_at=datetime(2026, 7, 31))

    # Then: the successful rows remain valid and the typed failure is externalized.
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

    # Then: one bad row costs one row — the msg1 condition still runs to completion, and
    # the residual failure is reported instead of being certified as canonical.
    with outcome.result_files[1].open(newline="", encoding="utf-8") as handle:
        assert len(tuple(csv.DictReader(handle))) == 2
    assert outcome.failure_count == 1
    assert outcome.canonical is None


def test_run_evaluation_when_resuming_fills_the_gap_in_the_same_run_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Given: a first pass whose second msg0 row failed and was skipped.
    backend = FakeBackend()
    backend.fail_calls = frozenset((2,))
    _patch_runtime(monkeypatch, backend)
    first = run_evaluation(_options(tmp_path, limit=2), started_at=datetime(2026, 7, 31))

    # When: the same run directory is resumed against a healthy backend.
    resumed_backend = FakeBackend()
    _patch_runtime(monkeypatch, resumed_backend)
    second = run_evaluation(
        _options(tmp_path, limit=2, resume=True, run_directory=first.run_directory)
    )

    # Then: resume reuses the directory rather than minting a new timestamp, and regenerates
    # only the missing row — completion is keyed by SHA, so the six already-written rows
    # are not re-inferred.
    assert second.run_directory == first.run_directory
    assert len(resumed_backend.calls) == 1
    assert second.failure_count == 0
    with second.result_files[0].open(newline="", encoding="utf-8") as handle:
        shas = [json.loads(row["shas"]) for row in csv.DictReader(handle)]
    assert shas == [["sha-0"], ["sha-1"]]


def test_run_evaluation_when_resumed_twice_still_finishes_the_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Given: a three-row sweep whose middle row fails in both message conditions, so each
    # CSV holds rows 0 and 2 and is missing row 1.
    first_backend = FakeBackend()
    first_backend.fail_calls = frozenset((2, 5))
    _patch_runtime(monkeypatch, first_backend, row_count=3)
    first = run_evaluation(_options(tmp_path, limit=3), started_at=datetime(2026, 7, 31))
    assert first.failure_count == 2

    # When: a first resume repairs msg0 but hits the failure again in msg1. The repaired row
    # was *appended*, so msg0 now reads rows 0, 2, 1 — out of source order.
    second_backend = FakeBackend()
    second_backend.fail_calls = frozenset((2,))
    _patch_runtime(monkeypatch, second_backend, row_count=3)
    second = run_evaluation(
        _options(tmp_path, limit=3, resume=True, run_directory=first.run_directory)
    )
    assert second.failure_count == 1

    # When: a second resume re-reads that repaired msg0 alongside the still-incomplete msg1.
    third_backend = FakeBackend()
    _patch_runtime(monkeypatch, third_backend, row_count=3)
    third = run_evaluation(
        _options(tmp_path, limit=3, resume=True, run_directory=first.run_directory)
    )

    # Then: the out-of-order msg0 it inherited is accepted rather than rejected, only the one
    # row msg1 still lacks is regenerated, and the run completes. Rejecting the appended row
    # would strand a 3,500-row sweep with no recovery short of hand-editing CSVs.
    assert len(third_backend.calls) == 1
    assert third.failure_count == 0
    assert third.run_directory == first.run_directory

    # Then: each repaired cell is left back in source order. Analysis pairs models by CSV row
    # position, so a permanently appended row would compare mismatched commits downstream.
    for path in third.result_files:
        with path.open(newline="", encoding="utf-8") as handle:
            shas = [json.loads(row["shas"])[0] for row in csv.DictReader(handle)]
        assert shas == ["sha-0", "sha-1", "sha-2"]


def test_run_evaluation_when_resuming_the_other_arms_directory_rejects(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Given: a completed base-model run directory.
    backend = FakeBackend()
    _patch_runtime(monkeypatch, backend)
    base = run_evaluation(_options(tmp_path, base=True), started_at=datetime(2026, 7, 31))

    # When/Then: resuming it as the LoRA arm is refused. Both arms share a SHA order, so
    # subsequence validation would not catch this — the blend would be silent.
    with pytest.raises(InferenceRunError, match="Qwen3.6-27B"):
        _ = run_evaluation(_options(tmp_path, resume=True, run_directory=base.run_directory))


@pytest.mark.parametrize(
    ("config_text", "adapter_bytes"),
    (
        # A later checkpoint of the same adapter — `save_total_limit: 5` keeps five of these
        # side by side, so pointing --adapter at the wrong one is an easy slip.
        (CONFIG_TEXT, b"checkpoint-1000"),
        # An edited config: a changed revision or max_seq_length silently changes the run.
        ("model: {id: Qwen/Qwen3.6-27B, revision: other}\n", ADAPTER_BYTES),
    ),
)
def test_run_evaluation_when_resuming_with_changed_inputs_rejects(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, config_text: str, adapter_bytes: bytes
) -> None:
    # Given: a run started with one adapter checkpoint and config.
    backend = FakeBackend()
    _patch_runtime(monkeypatch, backend)
    first = run_evaluation(_options(tmp_path), started_at=datetime(2026, 7, 31))

    # When/Then: resuming it with different inputs is refused. Every LoRA checkpoint maps to
    # the same tree name, so the arm check cannot see this — without the identity digest two
    # experiments would blend into one CSV and still finalize as canonical.
    with pytest.raises(InferenceRunError, match="do not match"):
        _ = run_evaluation(
            _options(
                tmp_path,
                resume=True,
                run_directory=first.run_directory,
                config_text=config_text,
                adapter_bytes=adapter_bytes,
            )
        )


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
                "inference_time": "0.0", "shas": json.dumps([f"sha-{index}"]), "precision": "1.0",
                "recall": "1.0", "f1": "1.0", "exact_match": "True",
                "hamming_loss": "0.0", "context_len": path.stem.removesuffix("_zs"),
                "with_message": str(path.parent.name == "msg1"), "concern_count": "1",
            } for index in range(350))
    _ = (tmp_path / "failures.jsonl").write_text("", encoding="utf-8")
    # Certification checks row order against the split the run was started on, so the run
    # identity is as much part of a certifiable tree as the CSVs themselves.
    _ = (tmp_path / "run_identity.json").write_text(
        json.dumps({"ordered_test_shas": [[f"sha-{index}"] for index in range(350)]}),
        encoding="utf-8",
    )
    def fail_load_backend(_: PeftLoadRequest) -> NoReturn:
        pytest.fail("ML load")

    monkeypatch.setattr(infer_transformers, "load_backend", fail_load_backend)

    # When: verification is invoked through the public CLI path.
    exit_code = main(("--verify-only", "--output", str(tmp_path)))

    # Then: certification completes without model construction and reports success.
    assert exit_code == 0


def test_run_evaluation_when_a_prior_run_left_an_empty_directory_reinitializes_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Given: the directory shell a process killed between mkdir and its first sidecar write
    # leaves behind — it exists, but holds no run identity and no results.
    backend = FakeBackend()
    _patch_runtime(monkeypatch, backend)
    stranded = tmp_path / "Qwen3.6-27B-LoRA" / "20260731010203"
    stranded.mkdir(parents=True)

    # When: the owner simply re-runs the same command.
    outcome = run_evaluation(_options(tmp_path), started_at=datetime(2026, 7, 31, 1, 2, 3))

    # Then: it starts cleanly rather than dead-ending — a plain retry would otherwise be
    # rejected as "already exists" while --resume rejects the same directory for having no
    # failure history, stranding a directory that never held a single result row.
    assert outcome.run_directory == stranded
    assert (stranded / "run_identity.json").is_file()
    assert len(backend.calls) == 2


def test_run_evaluation_when_a_prior_run_left_results_refuses_to_reuse_the_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Given: a directory carrying a real run's identity, i.e. data worth protecting.
    backend = FakeBackend()
    _patch_runtime(monkeypatch, backend)
    occupied = tmp_path / "Qwen3.6-27B-LoRA" / "20260731010203"
    occupied.mkdir(parents=True)
    _ = (occupied / "run_identity.json").write_text("{}", encoding="utf-8")

    # When/Then: a fresh run refuses it and points at --resume instead of overwriting.
    with pytest.raises(InferenceRunError, match="already exists"):
        _ = run_evaluation(_options(tmp_path), started_at=datetime(2026, 7, 31, 1, 2, 3))


def test_main_when_rows_failed_exits_nonzero_and_names_the_resume_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Given: a sweep driven through the real CLI in which one row's generation fails.
    backend = FakeBackend()
    backend.fail_calls = frozenset({1})
    _patch_runtime(monkeypatch, backend)
    config_path, adapter_path = _fixtures(tmp_path, CONFIG_TEXT, ADAPTER_BYTES)

    # When: the owner invokes the command exactly as the runbook prints it.
    exit_code = main(
        ("--adapter", str(adapter_path), "--config", str(config_path), "--output", str(tmp_path))
    )

    # Then: a nonzero exit makes the residual failure visible to a shell or scheduler, and the
    # message carries the run directory the owner must pass back to --resume. This is the only
    # place that directory is ever printed, so an interactive operator cannot recover without it.
    assert exit_code == 1
    stderr = capsys.readouterr().err
    assert "1 row(s) failed" in stderr
    run_directory = next((tmp_path / "Qwen3.6-27B-LoRA").iterdir())
    assert f"--resume --run-directory {run_directory}" in stderr


def test_main_when_every_row_succeeded_exits_zero_without_a_resume_hint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Given: an evaluation that completes with no residual failures.
    from RQ.SLM.unsloth import infer as infer_transformers

    def clean_run(options: EvaluationOptions) -> EvaluationOutcome:
        return EvaluationOutcome(options.output_root, (), 0, None)

    monkeypatch.setattr(infer_transformers, "run_evaluation", clean_run)

    # When: the same CLI path runs to completion.
    exit_code = main(("--base", "--output", str(tmp_path)))

    # Then: success is silent, so a resume hint never appears for a run with nothing to resume.
    assert exit_code == 0
    assert capsys.readouterr().err == ""


def test_parse_command_when_verify_only_returns_no_adapter_command() -> None:
    # Given/When: a verification-only invocation with its run directory.
    command = parse_command(("--verify-only", "--output", "results/run"))

    # Then: it does not require an adapter/model configuration for certification.
    assert isinstance(command, VerifyOptions)
    assert command.run_directory == Path("results/run")


def test_parse_command_when_base_is_requested_selects_the_unadapted_model() -> None:
    # Given/When: the base arm of the ablation is requested explicitly.
    command = parse_command(("--base",))

    # Then: the sweep contract is identical to the LoRA arm apart from the missing adapter.
    assert isinstance(command, EvaluationOptions)
    assert command.adapter_path is None
    assert command.contexts == (12288, 8192, 4096, 2048, 1024)


@pytest.mark.parametrize(
    "arguments",
    ((), ("--adapter", "outputs/adapter", "--base")),
)
def test_parse_command_when_the_model_mode_is_not_exactly_one_rejects(
    arguments: tuple[str, ...],
) -> None:
    # Given: neither mode, or both at once — a mistyped --adapter must never silently
    # degrade into a base run and quietly report base numbers as fine-tuned ones.

    # When/Then: the ambiguous invocation is refused before any GPU work.
    with pytest.raises(InferenceRunError, match="exactly one"):
        _ = parse_command(arguments)


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
