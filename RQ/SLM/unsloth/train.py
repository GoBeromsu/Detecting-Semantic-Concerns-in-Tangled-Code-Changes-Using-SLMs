from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal, Never, Protocol, TypeAlias, runtime_checkable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "RQ.SLM.unsloth"

from RQ.SLM.unsloth._memory_types import (
    JSON_DECODER,
    QualificationInputs,
    TrainingDataError,
)
from RQ.SLM.unsloth._memory_types import (
    require_qualification_evidence as _require_qualification_evidence,
)
from RQ.SLM.unsloth._types import (
    ContractError,
    ExcludedRow,
    ExcludedRowManifest,
    JsonValue,
    ManifestEvidence,
    RenderedExample,
    RunManifest,
    RunProvenance,
    TokenizerLike,
)
from RQ.SLM.unsloth.config import (
    UnslothConfig,
    load_unsloth_config,
)
from RQ.SLM.unsloth.data import (
    DatasetRow,
    build_chat_messages,
    canonical_row_hash,
    load_split,
    render_response_only,
)
from RQ.SLM.unsloth.runtime import create_runtime, create_trainer

__all__ = ["TrainingDataError", "require_qualification_evidence"]


DEFAULT_CONFIG_PATH: Final[Path] = Path("RQ/SLM/unsloth/configs/qwen3_6_27b.yml")
DEFAULT_MAX_SEQ_LENGTH: Final[int] = 16384
ADAPTER_SCHEMA_VERSION: Final[int] = 2
ADAPTER_EVIDENCE_DIR: Final[str] = "evidence"
COMMON_ADAPTER_EVIDENCE: Final[tuple[str, ...]] = (
    "config.yml", "template.txt", "training_identity.json", "provenance.json",
)
FULL_ADAPTER_EVIDENCE: Final[tuple[str, ...]] = (
    "host_profile.yml", "template-inspection.json", "overflow-rows.json",
    "measurements.jsonl", "qualification.json", "preflight.json",
)
EXPECTED_EXCLUDED_ROW: Final[tuple[int, str, int]] = (
    1326,
    "fc26be9a7f99e5f0d7db53cc53714dfd0c512a269fb3bd22ea3e7bdc12b742ba",
    16816,
)


@dataclass(frozen=True, slots=True)
class PreparedTrainingData:
    examples: tuple[RenderedExample, ...]
    excluded_rows: tuple[ExcludedRow, ...]


class RunArguments(argparse.Namespace):
    config: Path = DEFAULT_CONFIG_PATH; dataset_source: Literal["local", "hub"] = "local"
    smoke: bool = False; max_steps: int | None = None; inspect_template: bool = False
    evidence_dir: Path | None = None; verify_adapter: Path | None = None; host_profile: Path | None = None; verify_adapter_path_file: Path | None = None


class HubClient(Protocol):
    def upload_folder(self, *, folder_path: str, repo_id: str, repo_type: str) -> None: ...


class HubFactory(Protocol):
    def __call__(self, *, token: str) -> HubClient: ...


@runtime_checkable
class HubModule(Protocol):
    HfApi: HubFactory


def _adapter_fail(field: str, reason: str) -> Never:
    raise ContractError(f"{field}: {reason}")


def _adapter_write_atomic(path: Path, payload: bytes) -> None:
    _ = path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        _ = handle.write(payload)
        handle.flush()
        _ = os.fsync(handle.fileno())
    os.replace(temporary, path)


def _adapter_copy(source: Path, destination: Path, field: str) -> None:
    if source.is_symlink() or not source.is_file():
        _adapter_fail(field, "must be a regular file")
    _adapter_write_atomic(destination, source.read_bytes())


def _adapter_json(path: Path, field: str) -> Mapping[str, JsonValue]:
    try:
        value = JSON_DECODER.decode(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        _adapter_fail(field, f"invalid JSON ({error})")
    if not isinstance(value, dict):
        _adapter_fail(field, "must be a JSON object")
    return value


def _adapter_template(root: Path) -> str:
    template = _adapter_json(root / "tokenizer_config.json", "tokenizer_config").get("chat_template")
    if isinstance(template, str):
        return template
    path = root / "chat_template.jinja"
    if path.is_symlink() or not path.is_file():
        _adapter_fail("tokenizer", "embedded chat_template or chat_template.jinja is required")
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        _adapter_fail("chat_template.jinja", f"cannot read ({error})")


def _adapter_identity(config: UnslothConfig, evidence: ManifestEvidence) -> dict[str, JsonValue]:
    excluded: list[JsonValue] = [{"index": row.index, "hash": row.row_hash, "token_length": row.token_length} for row in evidence.excluded_rows]
    effective: dict[str, JsonValue] = {
        "model": {"id": config.model.id, "revision": config.model.revision, "text_only": config.model.text_only, "dataset_id": config.model.dataset_id, "dataset_revision": config.model.dataset_revision},
        "precision": {"load_in_16bit": config.precision.load_in_16bit, "load_in_4bit": config.precision.load_in_4bit, "load_in_8bit": config.precision.load_in_8bit},
        "lora": {"rank": config.lora.rank, "alpha": config.lora.alpha, "dropout": config.lora.dropout, "bias": config.lora.bias, "target_modules": list(config.lora.target_modules), "exclude_modules": config.lora.exclude_modules, "use_gradient_checkpointing": config.lora.use_gradient_checkpointing},
        "training": {"optim": config.training.optim, "learning_rate": config.training.learning_rate, "num_train_epochs": config.training.num_train_epochs, "per_device_train_batch_size": config.training.per_device_train_batch_size, "gradient_accumulation_steps": config.training.gradient_accumulation_steps, "warmup_ratio": config.training.warmup_ratio, "lr_scheduler_type": config.training.lr_scheduler_type, "seed": config.training.seed, "logging_steps": config.training.logging_steps, "save_strategy": config.training.save_strategy, "eval_strategy": config.training.eval_strategy, "max_seq_length": config.training.max_seq_length, "packing": config.training.packing, "padding_side": config.training.padding_side},
        "technical": {"attn_implementation": config.technical.attn_implementation, "device_map": config.technical.device_map}, "data": {"drop_rows_over_max_seq_length": config.data.drop_rows_over_max_seq_length},
        "masking": {"instruction_part": config.masking.instruction_part, "response_part": config.masking.response_part, "enable_thinking": config.masking.enable_thinking}, "output": {"adapter_dir_root": config.output.adapter_dir_root},
        "hub": {"adapter_repo": config.hub.adapter_repo, "merged_repo": config.hub.merged_repo, "gguf_repo": config.hub.gguf_repo}, "wandb": {"project": config.wandb.project, "experiment_name": config.wandb.experiment_name},
        "disk": {"min_free_reserve_gib": config.disk.min_free_reserve_gib}, "tags": list(config.tags), "labels": list(config.labels),
    }
    return {"effective_config": effective, "final_row_count": evidence.final_count, "excluded_rows": excluded, "objective": "response_only_json_eos", "packing": False, "merged_model": False}


def _adapter_artifacts(root: Path, provenance: RunProvenance) -> tuple[Path, ...]:
    if root.is_symlink() or not root.is_dir():
        _adapter_fail("adapter_dir", "must be an existing regular directory")
    paths: list[Path] = []
    for path in root.iterdir():
        if path.name in {"run_manifest.json", ADAPTER_EVIDENCE_DIR}:
            continue
        if path.is_symlink() or not path.is_file():
            _adapter_fail("adapter_dir", f"contains unsupported entry {path.name!r}")
        paths.append(path)
    names = COMMON_ADAPTER_EVIDENCE + (FULL_ADAPTER_EVIDENCE if provenance.run_mode == "full" else ())
    paths.extend(root / ADAPTER_EVIDENCE_DIR / name for name in names)
    return tuple(sorted(paths, key=lambda path: path.relative_to(root).as_posix()))


def build_manifest(config: UnslothConfig, provenance: RunProvenance, evidence: ManifestEvidence) -> RunManifest:
    root = evidence.adapter_dir
    if provenance.run_mode == "full" and not provenance.git_clean:
        _adapter_fail("git.clean", "full publication requires a clean Git worktree")
    if provenance.run_mode == "full" and (evidence.host_profile_path is None or evidence.qualification_dir is None):
        _adapter_fail("evidence", "full runs require host profile and qualification evidence")
    if provenance.run_mode == "smoke" and evidence.qualification_dir is not None:
        _adapter_fail("run_mode", "smoke runs cannot claim qualification evidence")
    template = _adapter_template(root)
    if template != evidence.chat_template:
        _adapter_fail("template", "saved tokenizer template differs from training template")
    destination = root / ADAPTER_EVIDENCE_DIR
    _adapter_copy(evidence.config_path, destination / "config.yml", "config")
    _adapter_write_atomic(destination / "template.txt", template.encode())
    if provenance.run_mode == "full":
        profile, qualification = evidence.host_profile_path, evidence.qualification_dir
        if profile is None or qualification is None:
            _adapter_fail("evidence", "full runs require host profile and qualification evidence")
        _adapter_copy(profile, destination / "host_profile.yml", "host_profile")
        for name in FULL_ADAPTER_EVIDENCE[1:]:
            _adapter_copy(qualification / name, destination / name, f"qualification.{name}")
    identity = _adapter_identity(config, evidence)
    _adapter_write_atomic(destination / "training_identity.json", (json.dumps(identity, sort_keys=True, separators=(",", ":")) + "\n").encode())
    captured: dict[str, JsonValue] = {"sha": provenance.git_sha, "clean": provenance.git_clean, "run_mode": provenance.run_mode}
    _adapter_write_atomic(destination / "provenance.json", (json.dumps(captured, sort_keys=True, separators=(",", ":")) + "\n").encode())
    hashes = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in _adapter_artifacts(root, provenance)}
    return {"schema_version": ADAPTER_SCHEMA_VERSION, "run_mode": provenance.run_mode, "git": {"sha": provenance.git_sha, "clean": provenance.git_clean}, "identity": identity, "hashes": {"artifacts": hashes, "template_sha256": hashlib.sha256(template.encode()).hexdigest()}}


def write_manifest(adapter_dir: Path, manifest: RunManifest) -> Path:
    path = adapter_dir / "run_manifest.json"
    _adapter_write_atomic(path, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
    return path


def parse_args(argv: Sequence[str] | None = None) -> RunArguments:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    _ = parser.add_argument("--dataset-source", choices=("local", "hub"), default="local")
    _ = parser.add_argument("--smoke", "--smoke-test", dest="smoke", action="store_true")
    _ = parser.add_argument("--max-steps", type=int)
    _ = parser.add_argument("--inspect-template", action="store_true")
    _ = parser.add_argument("--evidence-dir", type=Path)
    _ = parser.add_argument("--verify-adapter", type=Path)
    _ = parser.add_argument("--host-profile", type=Path)
    _ = parser.add_argument("--verify-adapter-path-file", type=Path)
    arguments = RunArguments()
    _ = parser.parse_args(argv, namespace=arguments)
    return arguments


def load_run_config(arguments: RunArguments) -> UnslothConfig:
    return load_unsloth_config(arguments.config)


def validate_exclusions(
    excluded_rows: Sequence[ExcludedRow], final_count: int, max_seq_length: int
) -> None:
    if max_seq_length != DEFAULT_MAX_SEQ_LENGTH:
        return
    expected = ExcludedRow(*EXPECTED_EXCLUDED_ROW)
    if tuple(excluded_rows) != (expected,) or final_count != 1399:
        raise TrainingDataError("default overflow evidence does not match the approved row")


def render_training_rows(
    rows: Sequence[DatasetRow], tokenizer: TokenizerLike, max_seq_length: int
) -> PreparedTrainingData:
    examples: list[RenderedExample] = []
    excluded_rows: list[ExcludedRow] = []
    for index, row in enumerate(rows):
        rendered = render_response_only(tokenizer, build_chat_messages(row)["messages"])
        token_length = len(rendered.input_ids)
        if token_length > max_seq_length:
            excluded_rows.append(ExcludedRow(index, canonical_row_hash(row), token_length))
        else:
            examples.append({"text": rendered.text})
    prepared = PreparedTrainingData(tuple(examples), tuple(excluded_rows))
    validate_exclusions(prepared.excluded_rows, len(prepared.examples), max_seq_length)
    return prepared


EvidenceValue: TypeAlias = str | int | list[ExcludedRowManifest]


def require_qualification_evidence(
    arguments: RunArguments, max_seq_length: int
) -> Path:
    return _require_qualification_evidence(QualificationInputs(
        arguments.evidence_dir, arguments.config, arguments.host_profile, max_seq_length
    ))


def verify_adapter(adapter_dir: Path, config_path: Path) -> None:
    from RQ.SLM.unsloth.adapter import validate_adapter

    _ = validate_adapter(adapter_dir, config_path)


def write_adapter_path(path_file: Path, adapter_dir: Path) -> None:
    _ = path_file.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path_file.parent, prefix=f".{path_file.name}.", encoding="utf-8", delete=False) as handle:
        temporary = Path(handle.name)
        _ = handle.write(f"{adapter_dir}\n")
        handle.flush()
        _ = os.fsync(handle.fileno())
    os.replace(temporary, path_file)


def _write_evidence(path: Path, name: str, payload: Mapping[str, EvidenceValue]) -> None:
    _ = path.mkdir(parents=True, exist_ok=True)
    destination = path / name
    with tempfile.NamedTemporaryFile(
        "w", dir=path, prefix=f".{name}.", encoding="utf-8", delete=False
    ) as handle:
        temporary = Path(handle.name)
        _ = handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        _ = os.fsync(handle.fileno())
    os.replace(temporary, destination)


def write_template_evidence(tokenizer: TokenizerLike, config: UnslothConfig, evidence_dir: Path) -> None:
    _write_evidence(evidence_dir, "template-inspection.json", {"template": str(getattr(tokenizer, "chat_template", "")), "instruction_part": config.masking.instruction_part, "response_part": config.masking.response_part})


def write_overflow_evidence(examples: Sequence[RenderedExample], excluded_rows: Sequence[ExcludedRow], evidence_dir: Path) -> None:
    excluded: list[ExcludedRowManifest] = [{"index": row.index, "hash": row.row_hash, "token_length": row.token_length} for row in excluded_rows]
    _write_evidence(evidence_dir, "overflow-rows.json", {"final_row_count": len(examples), "excluded_rows": excluded})


def require_full_credentials(environment: Mapping[str, str]) -> None:
    missing = tuple(name for name in ("HF_HUB_TOKEN", "WANDB_API_KEY") if not environment.get(name))
    if missing:
        raise TrainingDataError(f"missing credentials: {', '.join(missing)}")


def _provenance(run_mode: Literal["full", "smoke"]) -> RunProvenance:
    sha = subprocess.run(("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True)
    status = subprocess.run(("git", "status", "--porcelain"), check=True, capture_output=True, text=True)
    return RunProvenance(sha.stdout.strip(), not status.stdout.strip(), run_mode)


def _adapter_dir(config: UnslothConfig, timestamp: str) -> Path:
    return Path(config.output.adapter_dir_root) / timestamp / "adapter"


def _upload_adapter(adapter_dir: Path, config: UnslothConfig) -> None:
    hub = importlib.import_module("huggingface_hub")
    if not isinstance(hub, HubModule):
        raise TrainingDataError("huggingface_hub does not expose HfApi")
    _ = hub.HfApi(token=os.environ["HF_HUB_TOKEN"]).upload_folder(
        folder_path=str(adapter_dir), repo_id=config.hub.adapter_repo, repo_type="model"
    )


def run(arguments: RunArguments) -> None:
    config = load_run_config(arguments)
    if arguments.verify_adapter is not None:
        verify_adapter(arguments.verify_adapter, arguments.config)
        return
    evidence_dir = arguments.evidence_dir
    if arguments.inspect_template and evidence_dir is None:
        raise TrainingDataError("--inspect-template requires --evidence-dir")
    if not arguments.smoke and not arguments.inspect_template:
        evidence_dir = require_qualification_evidence(arguments, config.training.max_seq_length)
        require_full_credentials(os.environ)
        _ = _provenance("full")
    runtime = create_runtime(config, 0)
    if arguments.inspect_template:
        if evidence_dir is None:
            raise TrainingDataError("--inspect-template requires --evidence-dir")
        write_template_evidence(runtime.tokenizer, config, evidence_dir)
        return
    max_seq_length = config.training.max_seq_length
    train_rows = load_split(arguments.dataset_source, "train")
    prepared = render_training_rows(train_rows, runtime.tokenizer, max_seq_length)
    if evidence_dir is not None:
        write_overflow_evidence(prepared.examples, prepared.excluded_rows, evidence_dir)
    adapter_dir = _adapter_dir(config, datetime.now(UTC).strftime("%Y%m%d%H%M%S"))
    trainer = create_trainer(runtime, prepared.examples, config, adapter_dir, arguments.max_steps)
    trainer.train()
    trainer.save_model(str(adapter_dir))
    runtime.tokenizer.save_pretrained(str(adapter_dir))
    run_mode: Literal["full", "smoke"] = "smoke" if arguments.smoke else "full"
    _ = write_manifest(adapter_dir, build_manifest(
        config,
        _provenance(run_mode),
        ManifestEvidence(
            adapter_dir, arguments.config, runtime.tokenizer.chat_template,
            arguments.host_profile, None if arguments.smoke else evidence_dir,
            prepared.excluded_rows, len(prepared.examples),
        ),
    ))
    verify_adapter(adapter_dir, arguments.config)
    if arguments.verify_adapter_path_file is not None:
        write_adapter_path(arguments.verify_adapter_path_file, adapter_dir)
    if not arguments.smoke:
        _upload_adapter(adapter_dir, config)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
