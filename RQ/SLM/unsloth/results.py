"""Backend-neutral inference, result, and canonical-run contracts."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict, deque
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
IDENTITY_FILE_NAME: Final[str] = "run_identity.json"


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


def result_path_in_run(run_directory: Path, context_len: int, with_message: bool) -> Path:
    """Locate one result cell inside an already-resolved run directory."""
    return run_directory / ("msg1" if with_message else "msg0") / f"{context_len}_zs.csv"


def build_result_path(
    model_display_name: str,
    started_at: datetime,
    context_len: int,
    with_message: bool,
    results_root: Path = RESULTS_ROOT,
) -> Path:
    """Build the exact legacy result path without touching the filesystem."""
    return result_path_in_run(
        results_root / model_display_name / format_run_timestamp(started_at),
        context_len,
        with_message,
    )


def _first_json_object(raw_text: str) -> dict[str, JsonValue]:
    try:
        decoded = JSON_DECODER.decode(raw_text)
    except json.JSONDecodeError as error:
        raise ModelOutputError("invalid JSON object") from error
    if not isinstance(decoded, dict):
        raise ModelOutputError("invalid JSON object")
    return decoded


def _validate_labels(values: JsonValue) -> tuple[str, ...]:
    """Apply the label-set rules, independent of how the labels were transported.

    A label set reaches this program in two shapes: wrapped in the model's ``{"types": ...}``
    envelope, and bare in a CSV column. The rules are the same either way, so they live here
    once and each transport unwraps its own shape before calling in.
    """
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


def parse_model_output(raw_text: str) -> tuple[str, ...]:
    """Extract and strictly validate one ``{\"types\": [...]}`` object."""
    payload = _first_json_object(raw_text)
    keys = frozenset(payload)
    if "types" not in keys:
        raise ModelOutputError("missing required key 'types'")
    if keys != frozenset(("types",)):
        raise ModelOutputError("extra keys are not allowed")
    return _validate_labels(payload["types"])


def parse_label_column(raw_text: str) -> tuple[str, ...]:
    """Validate one CSV label column, which stores the label array bare — no envelope.

    ``as_csv_row`` writes ``predicted_types`` and ``actual_types`` through the same
    ``json.dumps(list(...))`` call, so certification must read them the same way. Validating
    one column as an envelope and the other as an array is what made ``finalize_run`` reject
    every run regardless of data quality.
    """
    try:
        decoded = json.loads(raw_text)
    except json.JSONDecodeError as error:
        raise ModelOutputError("invalid JSON array") from error
    return _validate_labels(decoded)


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


def completed_source_rows(
    expected_shas: Sequence[Sequence[str]], produced_rows: Sequence[ProducedRow]
) -> tuple[int, ...]:
    """Report which source row each produced row is, in the order the file holds them.

    Gaps are legitimate: a row whose generation failed is externalized to the sidecar and
    skipped, so a partial file holds a subsequence rather than a prefix.

    Rows are matched by SHA membership, not by position. Result rows are appended, so a row
    that ``--resume`` regenerates lands at the *end* of the file rather than back in its
    source slot. Demanding source order would therefore reject a file this program itself
    wrote and strand the run: the second resume of a cell that a first resume repaired would
    raise instead of finishing it, with no recovery short of hand-sorting the CSV.

    Completion is reported as source *indices*, not as SHA values. Two source rows may carry
    the same SHA tuple; keying on the value would let one produced row mark both complete, so
    a resume would skip the un-produced twin forever and the cell could never reach 350. Each
    produced row therefore claims exactly one index, lowest first, and a row matching no
    unclaimed source row is rejected as foreign.

    Indices are returned in file order rather than as a set, so a caller can also tell whether
    the file is still sorted by source order — which resume needs in order to repair it.
    """
    if len(produced_rows) > len(expected_shas):
        raise SourceOrderError(
            f"length mismatch: expected at most {len(expected_shas)}, found {len(produced_rows)}"
        )
    unclaimed: dict[tuple[str, ...], deque[int]] = defaultdict(deque)
    for index, shas in enumerate(expected_shas):
        unclaimed[tuple(shas)].append(index)
    completed: list[int] = []
    for index, produced in enumerate(produced_rows):
        available = unclaimed[_produced_shas(produced, index)]
        if not available:
            raise SourceOrderError("row does not match any unclaimed source row", index)
        completed.append(available.popleft())
    return tuple(completed)


def expected_result_paths(run_directory: Path) -> tuple[Path, ...]:
    """Return all ten required result paths in deterministic sweep order."""
    return tuple(
        result_path_in_run(run_directory, context_len, with_message)
        for with_message in MESSAGE_CONDITIONS
        for context_len in CONTEXT_SWEEP
    )


def _expected_shas(run_directory: Path) -> tuple[tuple[str, ...], ...]:
    """Recover the source SHA order a run was started against."""
    path = run_directory / IDENTITY_FILE_NAME
    if not path.is_file():
        raise FinalizationError("run identity is missing", path)
    try:
        identity = JSON_DECODER.decode(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise FinalizationError("run identity is not valid JSON", path) from error
    if not isinstance(identity, dict):
        raise FinalizationError("run identity is not an object", path)
    rows = identity.get("ordered_test_shas")
    if not isinstance(rows, list):
        raise FinalizationError("run identity has no ordered_test_shas", path)
    expected: list[tuple[str, ...]] = []
    for shas in rows:
        if not isinstance(shas, list) or not all(isinstance(sha, str) for sha in shas):
            raise FinalizationError("ordered_test_shas is malformed", path)
        expected.append(tuple(sha for sha in shas if isinstance(sha, str)))
    return tuple(expected)


def _successful_row_count(path: Path, expected_shas: Sequence[Sequence[str]]) -> int:
    if not path.is_file():
        raise FinalizationError("missing result file", path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != DEFAULT_DF_COLUMNS:
            raise FinalizationError("columns do not match DEFAULT_DF_COLUMNS", path)
        rows = tuple(reader)
    try:
        claimed = completed_source_rows(expected_shas, rows)
    except SourceOrderError as error:
        raise FinalizationError("rows do not match the run's test split", path) from error
    # Certification must not depend on a resume having survived long enough to re-sort the
    # file. Downstream analysis pairs models by CSV row position, so a cell left in append
    # order after a repair would compare one model's row against another model's commit.
    if claimed != tuple(range(len(rows))):
        raise FinalizationError("rows are not in test-split order", path)
    for row in rows:
        try:
            _ = parse_label_column(row["predicted_types"])
            _ = parse_label_column(row["actual_types"])
            inference_time = float(row["inference_time"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, ModelOutputError) as error:
            raise FinalizationError("semantic CSV row validation failed", path) from error
        if not math.isfinite(inference_time) or inference_time < 0:
            raise FinalizationError("semantic CSV row validation failed", path)
    if len(rows) != EXPECTED_SUCCESSFUL_ROWS:
        raise FinalizationError(
            f"expected {EXPECTED_SUCCESSFUL_ROWS} successful rows, found {len(rows)}", path
        )
    return len(rows)


def finalize_run(run_directory: Path) -> CanonicalRun:
    """Certify a run from the data itself: ten complete, semantically valid result files.

    The failure sidecar is append-only provenance, not a gate. Gating on an empty sidecar
    would permanently disqualify a run in which a transient failure was later re-generated
    successfully on resume; ``_successful_row_count`` is the stronger, self-evident check.

    Row order is checked here rather than trusted: ``run_identity.json`` records the split the
    run was started against, so certification can prove each cell holds exactly that split in
    that order without depending on some earlier resume having survived to re-sort it.
    """
    failure_path = run_directory / FAILURES_SIDECAR_NAME
    if not failure_path.is_file():
        raise FinalizationError("failure history sidecar is missing", failure_path)
    expected_shas = _expected_shas(run_directory)
    result_files = expected_result_paths(run_directory)
    successful_rows = sum(_successful_row_count(path, expected_shas) for path in result_files)
    return CanonicalRun(run_directory, result_files, successful_rows)
