"""GPU-lazy evaluator for the Qwen3.6 Transformers and PEFT inference path."""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import sys
from time import perf_counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol, override, runtime_checkable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RQ.SLM.unsloth.results import (
    CONTEXT_SWEEP,
    FAILURES_SIDECAR_NAME,
    JSON_DECODER,
    MESSAGE_CONDITIONS,
    CanonicalRun,
    FailureRecord,
    InferenceResult,
    ModelOutputError,
    build_result_path,
    finalize_run,
    format_run_timestamp,
    validate_source_order,
)
from RQ.SLM.unsloth.infer_options import (
    EvaluationOptions,
    InferenceRunError,
    VerifyOptions,
    parse_command,
)
from RQ.SLM.unsloth.config import load_config
from RQ.SLM.unsloth.data import (
    DatasetRow,
    JsonScalar,
    load_split,
)
from RQ.SLM.unsloth.adapter import validate_adapter
from RQ.SLM.unsloth.generation import (
    GenerationRequest,
    PeftLoadRequest,
    GenerationError,
    load_backend,
)
from utils.llms.constant import DEFAULT_DF_COLUMNS
from utils.prompt import get_prompt_by_type


MODEL_DISPLAY_NAME = "Qwen3.6-27B-LoRA"
IDENTITY_FILE_NAME = "run_identity.json"


@dataclass(frozen=True, slots=True)
class RunIdentityError(Exception):
    reason: str

    @override
    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class RunIdentityInput:
    adapter_path: Path
    config_path: Path
    model_id: str
    model_revision: str
    dataset_id: str
    dataset_revision: str
    ordered_test_shas: tuple[tuple[str, ...], ...]
    seed: int
    temperature: float
    max_new_tokens: int
    contexts: tuple[int, ...]
    message_conditions: tuple[bool, ...]


@dataclass(frozen=True, slots=True)
class RunIdentity:
    digest: str
    canonical_json: str


def _identity_file_hash(path: Path) -> str:
    if not path.is_file():
        raise RunIdentityError(f"identity input file is missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity_directory_hash(path: Path) -> str:
    if not path.is_dir():
        raise RunIdentityError(f"adapter directory is missing: {path}")
    records = tuple((candidate.relative_to(path).as_posix(), _identity_file_hash(candidate)) for candidate in sorted(path.rglob("*")) if candidate.is_file())
    if not records:
        raise RunIdentityError("adapter directory contains no files")
    return hashlib.sha256(json.dumps(records, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def build_run_identity(value: RunIdentityInput) -> RunIdentity:
    payload = {"adapter_sha256": _identity_directory_hash(value.adapter_path), "config_sha256": _identity_file_hash(value.config_path), "contexts": value.contexts, "dataset": {"id": value.dataset_id, "revision": value.dataset_revision}, "max_new_tokens": value.max_new_tokens, "message_conditions": value.message_conditions, "model": {"id": value.model_id, "revision": value.model_revision}, "ordered_test_shas": value.ordered_test_shas, "seed": value.seed, "temperature": value.temperature}
    canonical_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return RunIdentity(hashlib.sha256(canonical_json.encode("utf-8")).hexdigest(), canonical_json)


def establish_run(run_directory: Path, identity: RunIdentity, *, resume: bool) -> Path:
    identity_path = run_directory / IDENTITY_FILE_NAME
    if resume:
        if not run_directory.is_dir():
            raise RunIdentityError("resume requires an existing run directory")
        if not identity_path.is_file():
            raise RunIdentityError("resume requires persisted run identity")
        if identity_path.read_text(encoding="utf-8") != identity.canonical_json:
            raise RunIdentityError("resume identity does not exactly match existing run")
        return run_directory
    if run_directory.exists():
        raise RunIdentityError("new run directory already exists")
    run_directory.mkdir(parents=True)
    _ = identity_path.write_text(identity.canonical_json, encoding="utf-8")
    return run_directory


class LabelBackend(Protocol):
    def generate(self, request: GenerationRequest) -> tuple[str, ...]: ...


class TruncatedCommits(Protocol):
    def tolist(self) -> list[str]: ...


class TruncatedFrame(Protocol):
    def __getitem__(self, key: str) -> TruncatedCommits: ...


class DatasetFrame(Protocol):
    pass


class DataFrameFactory(Protocol):
    def __call__(self, rows: list[dict[str, JsonScalar]]) -> DatasetFrame: ...


@runtime_checkable
class PandasModule(Protocol):
    DataFrame: DataFrameFactory


@runtime_checkable
class RqMainModule(Protocol):
    def add_truncated_commits(
        self, tangled_df: DatasetFrame, context_window: int, include_message: bool
    ) -> TruncatedFrame: ...


@dataclass(frozen=True, slots=True)
class EvaluationOutcome:
    run_directory: Path
    result_files: tuple[Path, ...]
    failure_count: int
    canonical: CanonicalRun | None


def _string_tuple(raw: str) -> tuple[str, ...]:
    decoded = JSON_DECODER.decode(raw)
    if not isinstance(decoded, list):
        raise InferenceRunError("validated dataset JSON was not an array of strings")
    values: list[str] = []
    for value in decoded:
        if not isinstance(value, str):
            raise InferenceRunError("validated dataset JSON was not an array of strings")
        values.append(value)
    return tuple(values)


def _render_commits(
    rows: Sequence[DatasetRow], context_len: int, with_message: bool
) -> tuple[str, ...]:
    pandas = importlib.import_module("pandas")
    rq_main = importlib.import_module("RQ.main")
    if not isinstance(pandas, PandasModule) or not isinstance(rq_main, RqMainModule):
        raise InferenceRunError("pandas or RQ.main truncation API is unavailable")
    frame = pandas.DataFrame([dict(row.as_mapping()) for row in rows])
    truncated = rq_main.add_truncated_commits(
        frame, context_window=context_len, include_message=with_message
    )
    return tuple(str(value) for value in truncated["truncated_commit"].tolist())


def _prepare_csv(path: Path, resume: bool, expected_shas: Sequence[tuple[str, ...]]) -> int:
    if path.exists():
        if not resume:
            raise InferenceRunError(f"result already exists: {path}")
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != DEFAULT_DF_COLUMNS:
                raise InferenceRunError(f"columns do not match result contract: {path}")
            produced = tuple(reader)
        validate_source_order(expected_shas[: len(produced)], produced)
        return len(produced)
    _ = path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=DEFAULT_DF_COLUMNS).writeheader()
    return 0


def _append_result(path: Path, result: InferenceResult) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=DEFAULT_DF_COLUMNS).writerow(result.as_csv_row())


def _append_failure(path: Path, failure: FailureRecord) -> None:
    with path.open("a", encoding="utf-8") as handle:
        _ = handle.write(json.dumps(failure.as_json_record(), ensure_ascii=False) + "\n")


def _is_canonical(options: EvaluationOptions) -> bool:
    return (
        options.limit is None
        and options.contexts == tuple(CONTEXT_SWEEP)
        and options.message_conditions == MESSAGE_CONDITIONS
    )


def run_evaluation(
    options: EvaluationOptions, started_at: datetime | None = None
) -> EvaluationOutcome:
    """Evaluate only the canonical test split and persist resumable result prefixes."""
    configuration = load_config(options.config_path)
    _ = validate_adapter(options.adapter_path, options.config_path)
    backend: LabelBackend = load_backend(
        PeftLoadRequest(
            model_id=configuration.model.id,
            revision=configuration.model.revision,
            adapter_path=options.adapter_path,
            max_seq_length=configuration.training.max_seq_length,
        )
    )
    rows = load_split(options.data_source, "test", options.data_dir)
    selected_rows = rows if options.limit is None else rows[: options.limit]
    started = datetime.now() if started_at is None else started_at
    run_directory = options.run_directory if options.resume else options.output_root / MODEL_DISPLAY_NAME / format_run_timestamp(started)
    if run_directory is None:
        raise InferenceRunError("resume requires an explicit run directory")
    failure_path = run_directory / FAILURES_SIDECAR_NAME
    if options.resume and not failure_path.is_file():
        raise InferenceRunError("resume requires existing failure history")
    if not options.resume:
        run_directory.mkdir(parents=True, exist_ok=False)
        _ = failure_path.write_text("", encoding="utf-8")
    expected_shas = tuple(_string_tuple(row.shas) for row in selected_rows)
    result_files: list[Path] = []
    failure_count = 0
    for with_message in options.message_conditions:
        system_prompt = get_prompt_by_type("Zero-shot", include_message=with_message)
        for context_len in options.contexts:
            path = build_result_path(
                MODEL_DISPLAY_NAME, started, context_len, with_message, options.output_root
            )
            result_files.append(path)
            completed = _prepare_csv(path, options.resume, expected_shas)
            commits = _render_commits(selected_rows, context_len, with_message)
            for row_index in range(completed, len(selected_rows)):
                row = selected_rows[row_index]
                try:
                    started_row = perf_counter()
                    labels = backend.generate(
                        GenerationRequest(
                            system_prompt=system_prompt,
                            commit=commits[row_index],
                            seed=options.seed + row_index,
                            temperature=options.temperature,
                            max_new_tokens=options.max_new_tokens,
                        )
                    )
                    inference_time = perf_counter() - started_row
                except (GenerationError, ModelOutputError) as error:
                    _append_failure(
                        failure_path,
                        FailureRecord(
                            row_index=row_index,
                            shas=_string_tuple(row.shas),
                            context_len=context_len,
                            with_message=with_message,
                            error_type=type(error).__name__,
                            error_message=str(error),
                            raw_output=getattr(error, "raw_output", None),
                        ),
                    )
                    failure_count += 1
                    break
                _append_result(
                    path,
                    InferenceResult(
                        predicted_types=labels,
                        actual_types=_string_tuple(row.types),
                        inference_time=inference_time,
                        shas=_string_tuple(row.shas),
                        context_len=context_len,
                        with_message=with_message,
                    ),
                )
    canonical = finalize_run(run_directory, failure_count) if _is_canonical(options) else None
    return EvaluationOutcome(run_directory, tuple(result_files), failure_count, canonical)


def verify_run(options: VerifyOptions) -> CanonicalRun:
    """Certify an existing run without importing or allocating the ML stack."""
    failure_path = options.run_directory / FAILURES_SIDECAR_NAME
    failure_count = 0
    if failure_path.is_file():
        failure_count = sum(1 for line in failure_path.read_text(encoding="utf-8").splitlines() if line)
    return finalize_run(options.run_directory, failure_count)


def main(arguments: Sequence[str] | None = None) -> None:
    """Run evaluation or GPU-free result verification from the command line."""
    command = parse_command(arguments)
    match command:
        case EvaluationOptions():
            _ = run_evaluation(command)
        case VerifyOptions():
            _ = verify_run(command)


if __name__ == "__main__":
    main()
