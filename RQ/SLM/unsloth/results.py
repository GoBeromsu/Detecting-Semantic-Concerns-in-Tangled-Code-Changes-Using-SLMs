"""Backend-neutral inference, result, and canonical-run contracts."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Protocol, TypeAlias

from utils.eval import calculate_metrics
from utils.llms.constant import COMMIT_TYPES, DEFAULT_DF_COLUMNS

from ._types import CsvResultRow, JsonValue
from .generation import STRICT_RESPONSE_SCHEMA

__all__ = ("STRICT_RESPONSE_SCHEMA",)

ProducedCell: TypeAlias = str | int | float | bool | Sequence[str] | None


RESULTS_ROOT, TIMESTAMP_FORMAT = Path("results"), "%Y%m%d%H%M%S"
CONTEXT_SWEEP: Final[list[int]] = [12288, 8192, 4096, 2048, 1024]
MESSAGE_CONDITIONS: Final[tuple[bool, bool]] = (False, True)
EXPECTED_SUCCESSFUL_ROWS: Final[int] = 350
FAILURES_SIDECAR_NAME: Final[str] = "failures.jsonl"


class _JsonDecoder(Protocol):
    def decode(self, s: str) -> JsonValue: ...

    def raw_decode(self, s: str, idx: int = 0) -> tuple[JsonValue, int]: ...


def _json_decoder() -> _JsonDecoder:
    return json.JSONDecoder()


JSON_DECODER: Final[_JsonDecoder] = _json_decoder()


@dataclass(frozen=True, slots=True)
class ModelOutputError(Exception):
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", (self.reason,))


@dataclass(frozen=True, slots=True)
class SourceOrderError(Exception):
    reason: str
    row_index: int | None = None

    def __post_init__(self) -> None:
        message = self.reason if self.row_index is None else f"index {self.row_index}: {self.reason}"
        object.__setattr__(self, "args", (message,))


@dataclass(frozen=True, slots=True)
class FinalizationError(Exception):
    reason: str
    path: Path | None = None

    def __post_init__(self) -> None:
        message = self.reason if self.path is None else f"{self.path}: {self.reason}"
        object.__setattr__(self, "args", (message,))


@dataclass(frozen=True, slots=True)
class InferenceResult:
    predicted_types: tuple[str, ...]
    actual_types: tuple[str, ...]
    inference_time: float
    shas: tuple[str, ...]
    context_len: int
    with_message: bool

    def as_csv_row(self) -> CsvResultRow:
        metrics = calculate_metrics(list(self.predicted_types), list(self.actual_types))
        return {
            "predicted_types": json.dumps(list(self.predicted_types), ensure_ascii=False),
            "actual_types": json.dumps(list(self.actual_types), ensure_ascii=False),
            "inference_time": self.inference_time,
            "shas": json.dumps(list(self.shas), ensure_ascii=False),
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "exact_match": bool(metrics["exact_match"]),
            "hamming_loss": metrics["hamming_loss"],
            "context_len": self.context_len,
            "with_message": self.with_message,
            "concern_count": len(self.actual_types),
        }


ProducedRow: TypeAlias = InferenceResult | Mapping[str, ProducedCell]


@dataclass(frozen=True, slots=True)
class FailureRecord:
    row_index: int
    shas: tuple[str, ...]
    context_len: int
    with_message: bool
    error_type: str
    error_message: str
    raw_output: str | None

    def as_json_record(self) -> Mapping[str, JsonValue]:
        return {
            "row_index": self.row_index,
            "shas": list(self.shas),
            "context_len": self.context_len,
            "with_message": self.with_message,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "raw_output": self.raw_output,
        }


@dataclass(frozen=True, slots=True)
class CanonicalRun:
    run_directory: Path
    result_files: tuple[Path, ...]
    successful_rows: int


def format_run_timestamp(started_at: datetime | None = None) -> str:
    """Format a supplied time, or compute the time only when called."""
    instant = datetime.now(UTC) if started_at is None else started_at
    return instant.strftime(TIMESTAMP_FORMAT)


def build_result_path(
    model_display_name: str,
    started_at: datetime,
    context_len: int,
    with_message: bool,
    results_root: Path = RESULTS_ROOT,
) -> Path:
    """Build the exact legacy result path without touching the filesystem."""
    message_directory = "msg1" if with_message else "msg0"
    return (
        results_root
        / model_display_name
        / format_run_timestamp(started_at)
        / message_directory
        / f"{context_len}_zs.csv"
    )


def _first_json_object(raw_text: str) -> dict[str, JsonValue]:
    try:
        decoded = JSON_DECODER.decode(raw_text)
    except json.JSONDecodeError as error:
        raise ModelOutputError("invalid JSON object") from error
    if not isinstance(decoded, dict):
        raise ModelOutputError("invalid JSON object")
    return decoded


def parse_model_output(raw_text: str) -> tuple[str, ...]:
    """Extract and strictly validate one ``{\"types\": [...]}`` object."""
    payload = _first_json_object(raw_text)
    keys = frozenset(payload)
    if "types" not in keys:
        raise ModelOutputError("missing required key 'types'")
    if keys != frozenset(("types",)):
        raise ModelOutputError("extra keys are not allowed")
    values = payload["types"]
    if not isinstance(values, list):
        raise ModelOutputError("types must be an array")
    labels: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise ModelOutputError("types items must be strings")
        labels.append(value)
    if not labels:
        raise ModelOutputError("types must contain at least one label")
    if len(labels) > len(COMMIT_TYPES):
        raise ModelOutputError("types exceeds the seven-label maximum")
    unknown = tuple(label for label in labels if label not in COMMIT_TYPES)
    if unknown:
        raise ModelOutputError(f"unknown label {unknown[0]!r}")
    if len(labels) != len(frozenset(labels)):
        raise ModelOutputError("duplicate label")
    return tuple(labels)


def _produced_shas(row: ProducedRow, row_index: int) -> tuple[str, ...]:
    if isinstance(row, InferenceResult):
        return row.shas
    if "shas" not in row:
        raise SourceOrderError("produced row has no shas", row_index)
    value = row["shas"]
    candidate: JsonValue | ProducedCell = value
    if isinstance(value, str):
        try:
            decoded = JSON_DECODER.decode(value)
        except json.JSONDecodeError as error:
            raise SourceOrderError("shas is invalid JSON", row_index) from error
        candidate = decoded
    if isinstance(candidate, (list, tuple)):
        shas: list[str] = []
        for sha in candidate:
            if not isinstance(sha, str):
                raise SourceOrderError("shas must be an array of strings", row_index)
            shas.append(sha)
        return tuple(shas)
    raise SourceOrderError("shas must be an array of strings", row_index)


def validate_source_order(
    expected_shas: Sequence[Sequence[str]], produced_rows: Sequence[ProducedRow]
) -> None:
    """Require produced rows to retain source length and exact SHA order."""
    if len(expected_shas) != len(produced_rows):
        raise SourceOrderError(
            f"length mismatch: expected {len(expected_shas)}, found {len(produced_rows)}"
        )
    for index, (expected, produced) in enumerate(
        zip(expected_shas, produced_rows, strict=True)
    ):
        if tuple(expected) != _produced_shas(produced, index):
            raise SourceOrderError("SHA order mismatch", index)


def expected_result_paths(run_directory: Path) -> tuple[Path, ...]:
    """Return all ten required result paths in deterministic sweep order."""
    return tuple(
        run_directory
        / ("msg1" if with_message else "msg0")
        / f"{context_len}_zs.csv"
        for with_message in MESSAGE_CONDITIONS
        for context_len in CONTEXT_SWEEP
    )


def _successful_row_count(path: Path) -> int:
    if not path.is_file():
        raise FinalizationError("missing result file", path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != DEFAULT_DF_COLUMNS:
            raise FinalizationError("columns do not match DEFAULT_DF_COLUMNS", path)
        count = 0
        for row in reader:
            try:
                _ = parse_model_output(row["predicted_types"])
                _ = parse_model_output(json.dumps({"types": json.loads(row["actual_types"])}))
                _ = _produced_shas(row, count)
                inference_time = float(row["inference_time"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError, ModelOutputError, SourceOrderError) as error:
                raise FinalizationError("semantic CSV row validation failed", path) from error
            if not math.isfinite(inference_time) or inference_time < 0:
                raise FinalizationError("semantic CSV row validation failed", path)
            count += 1
    if count != EXPECTED_SUCCESSFUL_ROWS:
        raise FinalizationError(
            f"expected {EXPECTED_SUCCESSFUL_ROWS} successful rows, found {count}", path
        )
    return count


def finalize_run(
    run_directory: Path, unresolved_failure_count: int
) -> CanonicalRun:
    """Certify a run only after every completeness gate succeeds."""
    if unresolved_failure_count != 0:
        raise FinalizationError(
            f"unresolved failure count is {unresolved_failure_count}"
        )
    failure_path = run_directory / FAILURES_SIDECAR_NAME
    if not failure_path.is_file():
        raise FinalizationError("failure history sidecar is missing", failure_path)
    if failure_path.read_text(encoding="utf-8"):
        raise FinalizationError("failure history contains unresolved failures", failure_path)
    result_files = expected_result_paths(run_directory)
    successful_rows = sum(_successful_row_count(path) for path in result_files)
    return CanonicalRun(run_directory, result_files, successful_rows)
