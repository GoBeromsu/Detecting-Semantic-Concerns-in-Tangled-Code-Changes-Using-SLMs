from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, Literal, Never

import pytest
from pydantic import TypeAdapter

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
TRAIN_PATH: Final[Path] = REPO_ROOT / "RQ/SLM/unsloth/train.py"


def _write_approved_qualification(
    root: Path, config_path: Path, host_profile_path: Path
) -> None:
    _ = (root / "template-inspection.json").write_text("{}", encoding="utf-8")
    _ = (root / "overflow-rows.json").write_text("{}", encoding="utf-8")
    measurements = root / "measurements.jsonl"
    _ = measurements.write_bytes(b'{"measurement":true}\n')
    qualification: dict[str, str | int | list[int] | None] = {
        "status": "approved_16384",
        "approved_max_seq_length": 16384,
        "first_failure_boundary": None,
        "lengths": [2048, 4096, 8192, 12288, 16384],
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "host_profile_sha256": hashlib.sha256(host_profile_path.read_bytes()).hexdigest(),
        "evidence_sha256": hashlib.sha256(measurements.read_bytes()).hexdigest(),
        "preflight_hostname": None,
        "preflight_sha256": None,
    }
    _ = (root / "qualification.json").write_text(
        json.dumps(qualification), encoding="utf-8"
    )


def test_cli_when_parsed_converts_paths_and_exposes_local_run_controls() -> None:
    # Given: a user request with every local-run control set.
    from RQ.SLM.unsloth.train import parse_args

    # When: the CLI input is parsed without importing Unsloth.
    arguments = parse_args(
        (
            "--config",
            "custom.yml",
            "--dataset-source",
            "hub",
            "--smoke-test",
            "--max-steps",
            "3",
            "--inspect-template",
            "--evidence-dir",
            "evidence",
            "--verify-adapter",
            "adapter",
            "--host-profile",
            "host.json",
            "--verify-adapter-path-file",
            "adapter-path.txt",
        )
    )

    # Then: filesystem options are parsed as Paths and all controls are preserved.
    assert arguments.config == Path("custom.yml")
    assert arguments.dataset_source == "hub"
    assert arguments.smoke is True
    assert arguments.max_steps == 3
    assert arguments.inspect_template is True
    assert arguments.evidence_dir == Path("evidence")
    assert arguments.verify_adapter == Path("adapter")
    assert arguments.host_profile == Path("host.json")
    assert arguments.verify_adapter_path_file == Path("adapter-path.txt")


def test_overflow_validation_when_default_requires_zero_exclusions() -> None:
    # Given: the observed default overflow — no row exceeds the 16,384-token budget.
    from RQ.SLM.unsloth._types import ExcludedRow
    from RQ.SLM.unsloth.train import TrainingDataError, validate_exclusions

    # When/Then: only a fully-retained 1,400-row split is admitted at the default limit.
    validate_exclusions((), final_count=1400, max_seq_length=16384)
    with pytest.raises(TrainingDataError):
        validate_exclusions(
            (ExcludedRow(index=1, row_hash="unexpected", token_length=17000),),
            final_count=1398,
            max_seq_length=16384,
        )


def test_run_config_when_parsed_uses_the_pinned_config_maximum() -> None:
    # Given: no CLI sequence-length override exists.
    from RQ.SLM.unsloth.train import load_run_config, parse_args

    # When: the default training command is parsed.
    config = load_run_config(parse_args(()))

    # Then: the config-owned 16,384-token limit is the only runtime value.
    assert config.training.max_seq_length == 16384


def test_checkpoint_dir_when_derived_from_a_run_timestamp_is_a_sibling_of_adapter_dir() -> None:
    # Given: the run-directory helpers keyed by the same timestamp.
    from RQ.SLM.unsloth.train import _adapter_dir, _checkpoint_dir, load_run_config, parse_args

    config = load_run_config(parse_args(()))
    timestamp = "20260101000000"

    # When: both directories for one run are derived.
    adapter_dir = _adapter_dir(config, timestamp)
    checkpoint_dir = _checkpoint_dir(config, timestamp)

    # Then: checkpoints never land inside the directory validate_adapter() walks.
    assert checkpoint_dir != adapter_dir
    assert checkpoint_dir.parent == adapter_dir.parent
    assert checkpoint_dir.name == "checkpoints"
    assert adapter_dir.name == "adapter"


@pytest.mark.parametrize(
    ("qualification", "raises"),
    (
        (None, True),
        (("requires_owner_decision", 16384), True),
        (("approved_16384", 8192), True),
        (("approved_16384-by-review", 16384), True),
        (("approved_16384", 16384), True),
    ),
)
def test_qualification_when_evidence_is_missing_rejected_or_mismatched_fails(
    tmp_path: Path,
    qualification: tuple[str, int] | None,
    raises: bool,
) -> None:
    # Given: complete, hash-bound evidence with one qualification condition varied.
    from RQ.SLM.unsloth.train import (
        TrainingDataError,
        parse_args,
        require_qualification_evidence,
    )

    config_path = tmp_path / "config.yml"
    host_profile_path = tmp_path / "host.yml"
    _ = config_path.write_bytes(b"config\n")
    _ = host_profile_path.write_bytes(b"host\n")
    _write_approved_qualification(tmp_path, config_path, host_profile_path)
    if qualification is None:
        _ = (tmp_path / "qualification.json").unlink()
    else:
        status, approved_max_seq_length = qualification
        payload = TypeAdapter(dict[str, str | int | list[int] | None]).validate_json(
            (tmp_path / "qualification.json").read_text(encoding="utf-8")
        )
        payload["status"] = status
        payload["approved_max_seq_length"] = approved_max_seq_length
        _ = (tmp_path / "qualification.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    arguments = parse_args((
        "--config", str(config_path),
        "--host-profile", str(host_profile_path),
        "--evidence-dir", str(tmp_path),
    ))
    # When/Then: only the exact canonical approval and matching length are admitted.
    if raises:
        with pytest.raises(TrainingDataError):
            _ = require_qualification_evidence(arguments, max_seq_length=16384)
    else:
        assert require_qualification_evidence(arguments, max_seq_length=16384) == tmp_path


def test_qualification_when_measurements_are_empty_but_hash_bound_rejects(
    tmp_path: Path,
) -> None:
    # Given: an otherwise hash-matching approval with no measurement rows.
    from RQ.SLM.unsloth.train import (
        TrainingDataError,
        parse_args,
        require_qualification_evidence,
    )

    config_path = tmp_path / "config.yml"
    host_profile_path = tmp_path / "host.yml"
    _ = config_path.write_bytes(b"config\n")
    _ = host_profile_path.write_bytes(b"host\n")
    _write_approved_qualification(tmp_path, config_path, host_profile_path)
    measurements = tmp_path / "measurements.jsonl"
    _ = measurements.write_bytes(b"")
    qualification = TypeAdapter(dict[str, str | int | list[int] | None]).validate_json(
        (tmp_path / "qualification.json").read_text(encoding="utf-8")
    )
    qualification["evidence_sha256"] = hashlib.sha256(measurements.read_bytes()).hexdigest()
    _ = (tmp_path / "qualification.json").write_text(
        json.dumps(qualification), encoding="utf-8"
    )
    arguments = parse_args((
        "--config", str(config_path),
        "--host-profile", str(host_profile_path),
        "--evidence-dir", str(tmp_path),
    ))

    # When/Then: an approval without structural measurement evidence is refused.
    with pytest.raises(TrainingDataError):
        _ = require_qualification_evidence(arguments, max_seq_length=16384)


@pytest.mark.parametrize(
    ("field", "has_host_profile", "error_field"),
    (
        ("config_sha256", True, "config_sha256"),
        ("host_profile_sha256", True, "host_profile_sha256"),
        ("evidence_sha256", True, "evidence_sha256"),
        (None, False, "--host-profile"),
    ),
)
def test_run_when_qualification_source_mismatches_rejects_before_runtime_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str | None,
    has_host_profile: bool,
    error_field: str,
) -> None:
    # Given: canonical approval whose selected source hash was changed after probing.
    from RQ.SLM.unsloth import train as train_unsloth
    from RQ.SLM.unsloth.config import UnslothConfig

    config_path = REPO_ROOT / "RQ/SLM/unsloth/configs/qwen3_6_27b.yml"
    host_profile_path = tmp_path / "host.yml"
    _ = host_profile_path.write_bytes(b"host\n")
    _write_approved_qualification(tmp_path, config_path, host_profile_path)
    qualification = TypeAdapter(dict[str, str | int | list[int] | None]).validate_json(
        (tmp_path / "qualification.json").read_text(encoding="utf-8")
    )
    if field is not None:
        qualification[field] = "0" * 64
    _ = (tmp_path / "qualification.json").write_text(
        json.dumps(qualification), encoding="utf-8"
    )
    def forbidden_runtime(config: UnslothConfig, device_index: int) -> Never:
        _ = (config, device_index)
        raise AssertionError("runtime created before qualification completed")

    monkeypatch.setattr(train_unsloth, "create_runtime", forbidden_runtime)
    monkeypatch.setenv("HF_HUB_TOKEN", "test-token")
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    argv = ["--config", str(config_path), "--evidence-dir", str(tmp_path)]
    if has_host_profile:
        argv.extend(("--host-profile", str(host_profile_path)))
    arguments = train_unsloth.parse_args(argv)

    # When/Then: qualification rejects before the full entrypoint creates a runtime.
    with pytest.raises(train_unsloth.TrainingDataError, match=error_field):
        train_unsloth.run(arguments)


def test_run_when_inspect_template_without_evidence_dir_rejects_before_runtime_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: --inspect-template requested but no --evidence-dir supplied.
    from RQ.SLM.unsloth import train as train_unsloth
    from RQ.SLM.unsloth.config import UnslothConfig

    config_path = REPO_ROOT / "RQ/SLM/unsloth/configs/qwen3_6_27b.yml"

    def forbidden_runtime(config: UnslothConfig, device_index: int) -> Never:
        _ = (config, device_index)
        raise AssertionError("runtime created before evidence-dir validation")

    monkeypatch.setattr(train_unsloth, "create_runtime", forbidden_runtime)
    arguments = train_unsloth.parse_args(
        ("--config", str(config_path), "--inspect-template")
    )

    # When/Then: the missing flag is rejected before the heavy runtime is built.
    with pytest.raises(
        train_unsloth.TrainingDataError, match="--inspect-template requires --evidence-dir"
    ):
        train_unsloth.run(arguments)


def test_verify_adapter_when_requested_delegates_selected_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: a verification request and the adapter-contract boundary.
    from RQ.SLM.unsloth import adapter
    from RQ.SLM.unsloth.train import verify_adapter

    received: list[tuple[Path, Path]] = []

    def validate(adapter_dir: Path, config_path: Path | None = None) -> None:
        assert config_path is not None
        received.append((adapter_dir, config_path))

    monkeypatch.setattr(adapter, "validate_adapter", validate)
    config_path = tmp_path / "config.yml"

    # When: verification is requested from the train entrypoint.
    verify_adapter(tmp_path / "adapter", config_path)

    # Then: the existing adapter contract receives the selected configuration.
    assert received == [(tmp_path / "adapter", config_path)]


@pytest.mark.parametrize(
    ("git_clean", "repo_reachable", "reason"),
    ((False, True, "clean Git worktree"), (True, False, "not reachable")),
)
def test_require_publishable_when_a_precondition_fails_rejects_before_any_gpu_time(
    monkeypatch: pytest.MonkeyPatch, git_clean: bool, repo_reachable: bool, reason: str
) -> None:
    # Given: a full run whose publication would fail only after hours of training — a dirty
    # worktree is caught by build_manifest(), a missing repo by the upload itself.
    from RQ.SLM.unsloth import train as train_unsloth
    from RQ.SLM.unsloth._types import RunProvenance
    from RQ.SLM.unsloth.config import load_config

    class FakeHubClient:
        def repo_info(self, *, repo_id: str, repo_type: str) -> object:
            _ = repo_id, repo_type
            if not repo_reachable:
                raise RuntimeError("404 repository not found")
            return object()

    def provenance(run_mode: Literal["full", "smoke"]) -> RunProvenance:
        return RunProvenance("a" * 40, git_clean, run_mode)

    monkeypatch.setattr(train_unsloth, "_provenance", provenance)
    monkeypatch.setattr(train_unsloth, "_hub_client", FakeHubClient)
    config = load_config(REPO_ROOT / "RQ/SLM/unsloth/configs/qwen3_6_27b.yml")

    # When/Then: the cheap precondition is refused up front, not after the GPU is spent.
    with pytest.raises(train_unsloth.TrainingDataError, match=reason):
        train_unsloth.require_publishable(config)


def test_hub_client_when_the_library_attaches_hfapi_lazily_still_builds_a_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: huggingface_hub as it actually ships — HfApi is not in the module's __dict__ but
    # is served by a module-level __getattr__. Every other test substitutes _hub_client
    # wholesale, so this is the only place the real function is exercised.
    import types

    from RQ.SLM.unsloth import train as train_unsloth

    built: list[str] = []

    class FakeHfApi:
        def __init__(self, *, token: str) -> None:
            built.append(token)

        def upload_folder(self, *, folder_path: str, repo_id: str, repo_type: str) -> None:
            _ = folder_path, repo_id, repo_type

        def repo_info(self, *, repo_id: str, repo_type: str) -> object:
            _ = repo_id, repo_type
            return object()

    def lazy_getattr(name: str) -> type[FakeHfApi]:
        if name != "HfApi":
            raise AttributeError(name)
        return FakeHfApi

    lazy = types.ModuleType("huggingface_hub")
    setattr(lazy, "__getattr__", lazy_getattr)

    def import_module(name: str) -> types.ModuleType:
        _ = name
        return lazy

    monkeypatch.setattr("RQ.SLM.unsloth.train.importlib.import_module", import_module)
    monkeypatch.setenv("HF_HUB_TOKEN", "token-value")

    # When: the publication gate builds its client.
    client = train_unsloth._hub_client()

    # Then: the lazy attribute resolves. A structural `isinstance` check against a
    # module-shaped Protocol would reject this, because Protocol checks resolve attributes via
    # inspect.getattr_static, which bypasses __getattr__ — failing the gate closed on every
    # real install and blocking a run it was written to protect.
    assert isinstance(client, FakeHfApi)
    assert built == ["token-value"]


def test_hub_client_when_the_library_lacks_hfapi_rejects(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given: a huggingface_hub that genuinely does not expose HfApi.
    import types

    from RQ.SLM.unsloth import train as train_unsloth

    def import_module(name: str) -> types.ModuleType:
        return types.ModuleType(name)

    monkeypatch.setattr("RQ.SLM.unsloth.train.importlib.import_module", import_module)
    monkeypatch.setenv("HF_HUB_TOKEN", "token-value")

    # When/Then: tolerating lazy attributes must not degrade into tolerating a missing one.
    with pytest.raises(train_unsloth.TrainingDataError, match="does not expose HfApi"):
        _ = train_unsloth._hub_client()


def test_run_when_full_training_saves_adapter_binds_qualification_evidence_after_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: a full run with completed qualification and a trainer that records its save.
    from RQ.SLM.unsloth import train as train_unsloth
    from RQ.SLM.unsloth._types import ManifestEvidence
    from RQ.SLM.unsloth.config import UnslothConfig
    from RQ.SLM.unsloth._types import RenderedExample, RunManifest, RunProvenance, TokenizerLike
    from RQ.SLM.unsloth.data import DatasetRow

    class FakeTokenizer:
        chat_template: str = "{{ messages }}"

        def save_pretrained(self, save_directory: str) -> None:
            _ = save_directory

    class FakeRuntime:
        tokenizer: FakeTokenizer

        def __init__(self) -> None:
            self.tokenizer = FakeTokenizer()

    class FakeTrainer:
        saved: bool

        def __init__(self) -> None:
            self.saved = False

        def train(self) -> None:
            return None

        def save_model(self, output_dir: str) -> None:
            self.saved = True
            _ = Path(output_dir).mkdir(parents=True, exist_ok=True)

    trainer = FakeTrainer()
    evidence_dir = tmp_path / "qualification"
    captured: list[tuple[Path, Path | None, str]] = []
    captured_checkpoint_dirs: list[Path] = []
    captured_report_to: list[str] = []
    captured_run_names: list[str | None] = []
    captured_manifest_report_to: list[str] = []

    def build_manifest_after_save(
        config: UnslothConfig,
        provenance: RunProvenance,
        evidence: ManifestEvidence,
        report_to: str = "wandb",
    ) -> RunManifest:
        _ = config
        assert trainer.saved is True
        captured.append((evidence.adapter_dir, evidence.qualification_dir, provenance.run_mode))
        captured_manifest_report_to.append(report_to)
        return {
            "schema_version": 2, "run_mode": "full", "git": {"sha": "a" * 40, "clean": True},
            "identity": {}, "hashes": {"artifacts": {}, "template_sha256": "a" * 64},
        }

    def qualification(arguments: train_unsloth.RunArguments, max_seq_length: int) -> Path:
        _ = arguments, max_seq_length
        return evidence_dir

    def credentials(environment: Mapping[str, str]) -> None:
        _ = environment

    def runtime(config: UnslothConfig, device_index: int) -> FakeRuntime:
        _ = config, device_index
        return FakeRuntime()

    def split(source: Literal["local", "hub"], name: str) -> tuple[DatasetRow, ...]:
        _ = source, name
        return ()

    def render(
        rows: Sequence[DatasetRow], tokenizer: TokenizerLike, maximum: int
    ) -> train_unsloth.PreparedTrainingData:
        _ = rows, tokenizer, maximum
        return train_unsloth.PreparedTrainingData((), ())

    def trainer_factory(
        runtime_value: FakeRuntime, examples: Sequence[RenderedExample], config: UnslothConfig,
        output_dir: Path, report_to: str, run_name: str | None, max_steps: int | None,
    ) -> FakeTrainer:
        _ = runtime_value, examples, config, max_steps
        captured_checkpoint_dirs.append(output_dir)
        captured_report_to.append(report_to)
        captured_run_names.append(run_name)
        return trainer

    def write_manifest(adapter_dir: Path, manifest: RunManifest) -> Path:
        _ = manifest
        return adapter_dir / "run_manifest.json"

    def verified(adapter_dir: Path, config: Path) -> None:
        _ = adapter_dir, config

    def uploaded(adapter_dir: Path, config: UnslothConfig) -> None:
        _ = adapter_dir, config

    def provenance(run_mode: Literal["full", "smoke"]) -> RunProvenance:
        return RunProvenance("a" * 40, True, run_mode)

    # `require_publishable` is left unpatched so the real precondition runs; only the
    # network client underneath it is faked, and it records what was probed.
    probed_repos: list[str] = []

    class FakeHubClient:
        def repo_info(self, *, repo_id: str, repo_type: str) -> object:
            _ = repo_type
            probed_repos.append(repo_id)
            return object()

        def upload_folder(self, *, folder_path: str, repo_id: str, repo_type: str) -> object:
            _ = folder_path, repo_id, repo_type
            return object()

    monkeypatch.setattr(train_unsloth, "_hub_client", FakeHubClient)
    monkeypatch.setattr(train_unsloth, "require_qualification_evidence", qualification)
    monkeypatch.setattr(train_unsloth, "require_full_credentials", credentials)
    monkeypatch.setattr(train_unsloth, "create_runtime", runtime)
    monkeypatch.setattr(train_unsloth, "load_split", split)
    monkeypatch.setattr(train_unsloth, "render_training_rows", render)
    monkeypatch.setattr(train_unsloth, "create_trainer", trainer_factory)
    monkeypatch.setattr(train_unsloth, "build_manifest", build_manifest_after_save)
    monkeypatch.setattr(train_unsloth, "write_manifest", write_manifest)
    monkeypatch.setattr(train_unsloth, "verify_adapter", verified)
    monkeypatch.setattr(train_unsloth, "_upload_adapter", uploaded)
    monkeypatch.setattr(train_unsloth, "_provenance", provenance)
    arguments = train_unsloth.parse_args((
        "--config", str(REPO_ROOT / "RQ/SLM/unsloth/configs/qwen3_6_27b.yml"), "--host-profile", str(tmp_path / "host.yml"),
        "--evidence-dir", str(evidence_dir),
    ))
    monkeypatch.delenv("WANDB_PROJECT", raising=False)

    # When: the trainer completes its adapter save in a full run.
    # Then (finally): WANDB_PROJECT is set directly by the code under test, not via
    # monkeypatch, so it cannot be auto-undone — pop it even if an assertion below fails,
    # or it leaks into later tests.
    try:
        train_unsloth.run(arguments)

        # Then: manifest capture receives the saved directory and full qualification evidence.
        assert captured and captured[0][0].is_dir()
        assert captured[0][1] == evidence_dir
        assert captured[0][2] == "full"

        # Then: the trainer's checkpoint output_dir is a sibling of adapter_dir, never
        # adapter_dir itself — build_manifest()/validate_adapter() would reject checkpoint-*
        # subdirectories found inside the adapter directory they walk.
        adapter_dir = captured[0][0]
        assert len(captured_checkpoint_dirs) == 1
        checkpoint_dir = captured_checkpoint_dirs[0]
        assert checkpoint_dir != adapter_dir
        assert checkpoint_dir.parent == adapter_dir.parent
        assert checkpoint_dir.name == "checkpoints"

        # Then: a full run reports to wandb under a timestamped run name, targets the
        # configured wandb project via the environment HF Trainer's WandbCallback reads, and
        # echoes report_to into the manifest identity next to logging_steps.
        assert captured_report_to == ["wandb"]
        assert captured_manifest_report_to == ["wandb"]
        assert len(captured_run_names) == 1
        run_name = captured_run_names[0]
        assert run_name is not None
        assert run_name.startswith("qwen3.6-27b-semantic-concern-slm-unsloth-lora-")
        assert os.environ["WANDB_PROJECT"] == (
            "Untangling-Multi-Concern-Commits-with-Small-Language-Models"
        )

        # Then: the upload destination was proven reachable, not discovered after training.
        assert probed_repos == ["Berom0227/Semantic-Concern-SLM-Qwen3.6-27B-adapter"]
    finally:
        os.environ.pop("WANDB_PROJECT", None)


def test_run_when_smoke_training_reports_to_none_without_wandb_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a smoke run, which must stay credential-free and never touch wandb.
    from RQ.SLM.unsloth import train as train_unsloth
    from RQ.SLM.unsloth._types import ManifestEvidence, RenderedExample, RunManifest, RunProvenance, TokenizerLike
    from RQ.SLM.unsloth.config import UnslothConfig
    from RQ.SLM.unsloth.data import DatasetRow

    class FakeTokenizer:
        chat_template: str = "{{ messages }}"

        def save_pretrained(self, save_directory: str) -> None:
            _ = save_directory

    class FakeRuntime:
        tokenizer: FakeTokenizer

        def __init__(self) -> None:
            self.tokenizer = FakeTokenizer()

    class FakeTrainer:
        def train(self) -> None:
            return None

        def save_model(self, output_dir: str) -> None:
            _ = Path(output_dir).mkdir(parents=True, exist_ok=True)

    trainer = FakeTrainer()
    captured_report_to: list[str] = []
    captured_run_names: list[str | None] = []

    def runtime(config: UnslothConfig, device_index: int) -> FakeRuntime:
        _ = config, device_index
        return FakeRuntime()

    def split(source: Literal["local", "hub"], name: str) -> tuple[DatasetRow, ...]:
        _ = source, name
        return ()

    def render(
        rows: Sequence[DatasetRow], tokenizer: TokenizerLike, maximum: int
    ) -> train_unsloth.PreparedTrainingData:
        _ = rows, tokenizer, maximum
        return train_unsloth.PreparedTrainingData((), ())

    def trainer_factory(
        runtime_value: FakeRuntime, examples: Sequence[RenderedExample], config: UnslothConfig,
        output_dir: Path, report_to: str, run_name: str | None, max_steps: int | None,
    ) -> FakeTrainer:
        _ = runtime_value, examples, config, output_dir, max_steps
        captured_report_to.append(report_to)
        captured_run_names.append(run_name)
        return trainer

    def build_manifest_smoke(
        config: UnslothConfig,
        provenance: RunProvenance,
        evidence: ManifestEvidence,
        report_to: str = "wandb",
    ) -> RunManifest:
        _ = config, provenance, evidence, report_to
        return {
            "schema_version": 2, "run_mode": "smoke", "git": {"sha": "a" * 40, "clean": True},
            "identity": {}, "hashes": {"artifacts": {}, "template_sha256": "a" * 64},
        }

    def write_manifest(adapter_dir: Path, manifest: RunManifest) -> Path:
        _ = manifest
        return adapter_dir / "run_manifest.json"

    def verified(adapter_dir: Path, config: Path) -> None:
        _ = adapter_dir, config

    def provenance(run_mode: Literal["full", "smoke"]) -> RunProvenance:
        return RunProvenance("a" * 40, True, run_mode)

    monkeypatch.setattr(train_unsloth, "create_runtime", runtime)
    monkeypatch.setattr(train_unsloth, "load_split", split)
    monkeypatch.setattr(train_unsloth, "render_training_rows", render)
    monkeypatch.setattr(train_unsloth, "create_trainer", trainer_factory)
    monkeypatch.setattr(train_unsloth, "build_manifest", build_manifest_smoke)
    monkeypatch.setattr(train_unsloth, "write_manifest", write_manifest)
    monkeypatch.setattr(train_unsloth, "verify_adapter", verified)
    monkeypatch.setattr(train_unsloth, "_provenance", provenance)
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    arguments = train_unsloth.parse_args((
        "--config", str(REPO_ROOT / "RQ/SLM/unsloth/configs/qwen3_6_27b.yml"), "--smoke",
    ))

    # When: the smoke entrypoint runs without qualification evidence or credentials.
    train_unsloth.run(arguments)

    # Then: the smoke run never asks HF Trainer's WandbCallback to activate and never
    # touches the wandb project environment variable.
    assert captured_report_to == ["none"]
    assert captured_run_names == [None]
    assert "WANDB_PROJECT" not in os.environ


def test_adapter_path_when_written_atomically_contains_the_validated_directory(
    tmp_path: Path,
) -> None:
    # Given: a completed adapter directory and an external verifier-path target.
    from RQ.SLM.unsloth.train import write_adapter_path

    adapter_dir = tmp_path / "adapter"
    path_file = tmp_path / "evidence" / "adapter-path.txt"

    # When: the verified adapter location is persisted.
    write_adapter_path(path_file, adapter_dir)

    # Then: consumers receive exactly the adapter path with no partial artifact.
    assert path_file.read_text(encoding="utf-8") == f"{adapter_dir}\n"
    assert not tuple(path_file.parent.glob(".adapter-path.txt.*"))


def test_train_module_when_examined_keeps_heavy_imports_outside_module_scope() -> None:
    # Given: the import-safe training entrypoint source.
    module = ast.parse(TRAIN_PATH.read_text(encoding="utf-8"))

    # When: its module-level imports are collected.
    imported = {
        alias.name.split(".")[0]
        for node in module.body
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".")[0]
        for node in module.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    # Then: macOS can import and ask for help without local-GPU dependencies.
    assert {"torch", "unsloth", "trl", "peft", "datasets"}.isdisjoint(imported)


def test_help_when_invoked_directly_is_available_without_unsloth() -> None:
    # Given: the directly executable training path on this CPU-only host.
    command = ("python", str(TRAIN_PATH), "--help")

    # When: argparse renders its help text.
    result = subprocess.run(command, capture_output=True, text=True, check=False)

    # Then: no ML runtime import was needed to explain the command.
    assert result.returncode == 0, result.stderr
    assert "--dataset-source" in result.stdout
    assert "--inspect-template" in result.stdout

