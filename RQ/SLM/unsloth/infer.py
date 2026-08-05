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
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, runtime_checkable


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "RQ.SLM.unsloth"

from RQ.SLM.unsloth.results import (
    CONTEXT_SWEEP,
    FAILURES_SIDECAR_NAME,
    IDENTITY_FILE_NAME,
    JSON_DECODER,
    MESSAGE_CONDITIONS,
    CanonicalRun,
    FailureRecord,
    InferenceResult,
    ModelOutputError,
    completed_source_rows,
    expected_result_paths,
    finalize_run,
    format_run_timestamp,
    result_path_in_run,
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


BASE_DISPLAY_NAME = "Qwen3.6-27B"
LORA_DISPLAY_NAME = "Qwen3.6-27B-LoRA"


def model_display_name(adapter_path: Path | None) -> str:
    """Keep the base and LoRA sweeps in separate result trees."""
    return BASE_DISPLAY_NAME if adapter_path is None else LORA_DISPLAY_NAME


def _adapter_fingerprint(adapter_path: Path | None) -> str:
    """Hash the adapter's contents, so two different checkpoints never look interchangeable."""
    if adapter_path is None:
        return "base"
    digest = hashlib.sha256()
    files = sorted(path for path in adapter_path.rglob("*") if path.is_file())
    if not files:
        raise InferenceRunError(f"adapter directory contains no files: {adapter_path}")
    for path in files:
        digest.update(path.relative_to(adapter_path).as_posix().encode("utf-8"))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def run_identity(options: EvaluationOptions, expected_shas: Sequence[tuple[str, ...]]) -> str:
    """Pin every input that must not change between a run and its resume.

    `--resume` re-enters a directory that already holds results, so anything it appends is
    attributed to whatever produced the earlier rows. Without this, resuming with a different
    adapter checkpoint, an edited config, or a different split would blend two experiments into
    one CSV and still satisfy `finalize_run`, which validates row shape rather than provenance.

    `limit` and `data_dir` are covered transitively: both change `ordered_test_shas`, which is
    pinned here. What that does not cover is a source whose SHAs are identical but whose diffs
    differ — a stale local mirror. `data_source` is pinned to catch the reachable form of that;
    a same-source mirror going stale mid-run is out of scope.
    """
    return json.dumps(
        {
            "adapter_sha256": _adapter_fingerprint(options.adapter_path),
            "config_sha256": hashlib.sha256(options.config_path.read_bytes()).hexdigest(),
            "contexts": list(options.contexts),
            "data_source": options.data_source,
            "max_new_tokens": options.max_new_tokens,
            "message_conditions": list(options.message_conditions),
            "model": model_display_name(options.adapter_path),
            "ordered_test_shas": [list(shas) for shas in expected_shas],
            "seed": options.seed,
            "temperature": options.temperature,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


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


def _prepare_csv(
    path: Path, resume: bool, expected_shas: Sequence[tuple[str, ...]]
) -> frozenset[int]:
    """Open or create one result cell and report which source rows it already holds.

    Completion is resolved against the source SHA order rather than by row count, so a
    resumed sweep can fill gaps left by skipped failures instead of assuming an unbroken
    prefix.
    """
    if path.exists():
        if not resume:
            raise InferenceRunError(f"result already exists: {path}")
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != DEFAULT_DF_COLUMNS:
                raise InferenceRunError(f"columns do not match result contract: {path}")
            produced = tuple(reader)
        return frozenset(completed_source_rows(expected_shas, produced))
    _ = path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=DEFAULT_DF_COLUMNS).writeheader()
    return frozenset()


def _append_result(path: Path, result: InferenceResult) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=DEFAULT_DF_COLUMNS).writerow(result.as_csv_row())


def _sort_into_source_order(path: Path, expected_shas: Sequence[tuple[str, ...]]) -> None:
    """Restore source row order in a cell that a resume repaired out of order.

    Rows are appended, so a row regenerated by `--resume` lands at the end of the file rather
    than back in its slot. Downstream analysis pairs models by CSV row position
    (`RQ/analysis/compare_models.py` joins on `df.index`), so a permanently out-of-order file
    would silently compare one model's row against another model's different commit.

    Rewrites via a temporary file and `os.replace`, which is atomic on POSIX: a crash leaves
    either the old file or the new one, never a truncated cell. Reading still matches rows by
    SHA rather than by position, so a crash between the append and this rewrite is recoverable.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != DEFAULT_DF_COLUMNS:
            raise InferenceRunError(f"columns do not match result contract: {path}")
        rows = tuple(reader)
    # Drop any temp file a crashed rewrite left behind, whether or not one is needed now.
    temporary = path.with_name(f"{path.name}.reordering")
    temporary.unlink(missing_ok=True)
    indices = completed_source_rows(expected_shas, rows)
    if list(indices) == sorted(indices):
        return
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DEFAULT_DF_COLUMNS)
        writer.writeheader()
        writer.writerows(row for _, row in sorted(zip(indices, rows), key=lambda pair: pair[0]))
    _ = temporary.replace(path)


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
    if options.adapter_path is not None:
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
    started = datetime.now(UTC) if started_at is None else started_at
    run_directory = options.run_directory if options.resume else options.output_root / model_display_name(options.adapter_path) / format_run_timestamp(started)
    if run_directory is None:
        raise InferenceRunError("resume requires an explicit run directory")
    display_name = model_display_name(options.adapter_path)
    expected_shas = tuple(_string_tuple(row.shas) for row in selected_rows)
    identity = run_identity(options, expected_shas)
    failure_path = run_directory / FAILURES_SIDECAR_NAME
    identity_path = run_directory / IDENTITY_FILE_NAME
    if options.resume:
        if not failure_path.is_file():
            raise InferenceRunError("resume requires existing failure history")
        # Checked before the identity digest purely for the error message: the two arms share a
        # SHA order, so this is the mistake most likely to be made, and "wrong tree" is far more
        # actionable than "digest mismatch". The identity check below would also catch it.
        if run_directory.parent.name != display_name:
            raise InferenceRunError(
                f"run directory belongs to {run_directory.parent.name}, not {display_name}"
            )
        if not identity_path.is_file():
            raise InferenceRunError(f"resume requires a persisted run identity: {identity_path}")
        if identity_path.read_text(encoding="utf-8") != identity:
            raise InferenceRunError(
                "resume inputs do not match the run on disk (adapter, config, split, or sweep)"
            )
    else:
        # Refuse on evidence of a prior run rather than on the directory merely existing.
        # Creating the directory and writing its two sidecars are three separate syscalls, so a
        # process killed between them leaves a directory that `mkdir(exist_ok=False)` would
        # reject as "already exists" while `--resume` rejects it as missing its sidecars —
        # unrecoverable without a manual `rm -rf`, for a directory holding no results at all.
        # A run always writes its identity before generating a row, so "identity or any CSV
        # present" is the true test for data worth protecting; anything else is an empty shell.
        if identity_path.is_file() or any(path.exists() for path in expected_result_paths(run_directory)):
            raise InferenceRunError(
                f"run directory already exists: {run_directory} (use --resume to continue it)"
            )
        run_directory.mkdir(parents=True, exist_ok=True)
        _ = failure_path.write_text("", encoding="utf-8")
        _ = identity_path.write_text(identity, encoding="utf-8")
    result_files: list[Path] = []
    failure_count = 0
    for with_message in options.message_conditions:
        system_prompt = get_prompt_by_type("Zero-shot", include_message=with_message)
        for context_len in options.contexts:
            path = result_path_in_run(run_directory, context_len, with_message)
            result_files.append(path)
            completed = _prepare_csv(path, options.resume, expected_shas)
            commits = _render_commits(selected_rows, context_len, with_message)
            for row_index in range(len(selected_rows)):
                if row_index in completed:
                    continue
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
                    # One bad row costs one row: record it and keep the sweep moving.
                    # `--resume` re-attempts exactly the rows missing from this CSV.
                    failure_count += 1
                    continue
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
            if options.resume:
                _sort_into_source_order(path, expected_shas)
    # Certify only a gap-free canonical sweep. With residual failures the results already on
    # disk stay valid, so report them and let `--resume` fill the gaps before `--verify-only`.
    canonical = (
        finalize_run(run_directory)
        if _is_canonical(options) and failure_count == 0
        else None
    )
    return EvaluationOutcome(run_directory, tuple(result_files), failure_count, canonical)


def verify_run(options: VerifyOptions) -> CanonicalRun:
    """Certify an existing run without importing or allocating the ML stack."""
    return finalize_run(options.run_directory)


def main(arguments: Sequence[str] | None = None) -> int:
    """Run evaluation or GPU-free result verification from the command line."""
    command = parse_command(arguments)
    match command:
        case EvaluationOptions():
            outcome = run_evaluation(command)
            if outcome.failure_count:
                print(
                    f"{outcome.failure_count} row(s) failed; rerun with --resume --run-directory {outcome.run_directory}",
                    file=sys.stderr,
                )
                return 1
        case VerifyOptions():
            _ = verify_run(command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
