import csv
import json
import subprocess
import sys
from dataclasses import FrozenInstanceError
from datetime import datetime
from pathlib import Path
from typing import Protocol

import pytest

from RQ.SLM.unsloth.results import (
    CONTEXT_SWEEP,
    EXPECTED_SUCCESSFUL_ROWS,
    MESSAGE_CONDITIONS,
    FailureRecord,
    FinalizationError,
    InferenceResult,
    ModelOutputError,
    SourceOrderError,
    build_result_path,
    expected_result_paths,
    finalize_run,
    format_run_timestamp,
    parse_model_output,
    validate_source_order,
)
from RQ.SLM.unsloth.generation import STRICT_RESPONSE_SCHEMA
from utils.llms.constant import COMMIT_TYPES, DEFAULT_DF_COLUMNS


class TangledSplitLike(Protocol):
    sha_lists: tuple[tuple[str, ...], ...]


def _write_result_file(path: Path, row_count: int = EXPECTED_SUCCESSFUL_ROWS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DEFAULT_DF_COLUMNS)
        writer.writeheader()
        row = {
            "predicted_types": '{"types":["fix"]}',
            "actual_types": '["fix"]',
            "inference_time": "0.0",
            "shas": '["sha"]',
            "precision": "1.0",
            "recall": "1.0",
            "f1": "1.0",
            "exact_match": "True",
            "hamming_loss": "0.0",
            "context_len": path.stem.removesuffix("_zs"),
            "with_message": str(path.parent.name == "msg1"),
            "concern_count": "1",
        }
        for _ in range(row_count):
            writer.writerow(row)


def _write_complete_run(run_directory: Path) -> None:
    for path in expected_result_paths(run_directory):
        _write_result_file(path)
    _ = (run_directory / "failures.jsonl").write_text("", encoding="utf-8")


def _write_shape_only_run(run_directory: Path) -> None:
    for path in expected_result_paths(run_directory):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=DEFAULT_DF_COLUMNS)
            writer.writeheader()
            writer.writerows(dict.fromkeys(DEFAULT_DF_COLUMNS, "value") for _ in range(EXPECTED_SUCCESSFUL_ROWS))
    _ = (run_directory / "failures.jsonl").write_text("", encoding="utf-8")


@pytest.mark.parametrize(
    ("with_message", "message_directory"), ((False, "msg0"), (True, "msg1"))
)
def test_build_result_path_when_condition_changes_matches_legacy_shape(
    with_message: bool, message_directory: str
) -> None:
    # Given: a fixed run time and one legacy context window.
    started_at = datetime(2026, 7, 31, 12, 34, 56)

    # When: a backend-neutral result path is built.
    result = build_result_path(
        "Qwen3.6-27B-LoRA", started_at, 12288, with_message
    )

    # Then: the timestamp, message flag, and zero-shot filename are exact.
    assert result == (
        Path("results")
        / "Qwen3.6-27B-LoRA"
        / "20260731123456"
        / message_directory
        / "12288_zs.csv"
    )


def test_format_run_timestamp_when_time_is_passed_is_deterministic() -> None:
    # Given: a caller-supplied datetime.
    started_at = datetime(2025, 1, 2, 3, 4, 5)

    # When/Then: formatting uses the legacy fourteen-digit convention.
    assert format_run_timestamp(started_at) == "20250102030405"


def test_sweep_and_schema_constants_match_strict_contract() -> None:
    # Given/When: inference sweep and constrained-decoding schema constants.
    properties = STRICT_RESPONSE_SCHEMA["properties"]
    assert isinstance(properties, dict)
    types_schema = properties["types"]
    assert isinstance(types_schema, dict)
    items = types_schema["items"]
    assert isinstance(items, dict)

    # Then: ordering and every strict array/object constraint are pinned.
    assert CONTEXT_SWEEP == [12288, 8192, 4096, 2048, 1024]
    assert MESSAGE_CONDITIONS == (False, True)
    assert STRICT_RESPONSE_SCHEMA["required"] == ["types"]
    assert STRICT_RESPONSE_SCHEMA["additionalProperties"] is False
    assert items["enum"] is COMMIT_TYPES
    assert items["enum"] == COMMIT_TYPES
    assert types_schema["minItems"] == 1
    assert types_schema["maxItems"] == 7
    assert types_schema["uniqueItems"] is True


@pytest.mark.parametrize(
    "raw_text",
    (
        '{"types": ["fix", "test"]}',
    ),
)
def test_parse_model_output_when_json_object_is_valid_extracts_types(
    raw_text: str,
) -> None:
    # Given/When: valid raw constrained output.
    parsed = parse_model_output(raw_text)

    # Then: validated labels preserve source order.
    assert parsed == ("fix", "test")


def test_parse_model_output_when_constrained_response_contains_prose_rejects() -> None:
    # Given: a response that would require prose salvage around valid JSON.
    raw_text = 'model preface\n```json\n{"types": ["fix", "test"]}\n```'

    # When/Then: constrained generation accepts only the raw JSON response.
    with pytest.raises(ModelOutputError, match="invalid JSON"):
        _ = parse_model_output(raw_text)


@pytest.mark.parametrize(
    ("raw_text", "reason"),
    (
        ('{"types": [', "invalid JSON"),
        ('{"types": ["unknown"]}', "unknown label"),
        ('{"types": ["fix", "fix"]}', "duplicate label"),
        ('{"types": []}', "at least one"),
        ('{"types": ["fix"], "reason": "x"}', "extra keys"),
    ),
)
def test_parse_model_output_when_contract_is_violated_raises_typed_error(
    raw_text: str, reason: str
) -> None:
    # Given/When/Then: each malformed output class raises the parser's typed error.
    with pytest.raises(ModelOutputError, match=reason):
        _ = parse_model_output(raw_text)


def test_inference_result_when_serialized_uses_shared_column_order_and_metrics() -> None:
    # Given: one successful backend-neutral inference observation.
    result = InferenceResult(
        predicted_types=("fix", "feat"),
        actual_types=("fix", "test"),
        inference_time=1.25,
        shas=("abc", "def"),
        context_len=4096,
        with_message=True,
    )

    # When: it is converted to a CSV row.
    row = result.as_csv_row()

    # Then: shared columns and legacy seven-label metrics are preserved.
    assert list(row) == DEFAULT_DF_COLUMNS
    assert json.loads(row["predicted_types"]) == ["fix", "feat"]
    assert json.loads(row["shas"]) == ["abc", "def"]
    assert row["precision"] == 0.5
    assert row["recall"] == 0.5
    assert row["f1"] == 0.5
    assert row["exact_match"] is False
    assert row["hamming_loss"] == 2 / 7


def test_validate_source_order_when_rows_match_real_fixture_passes(
    test_split: TangledSplitLike,
) -> None:
    # Given: source SHA order and produced rows from the canonical test fixture.
    expected = test_split.sha_lists[:3]
    produced = tuple(
        {"shas": json.dumps(list(shas))} for shas in test_split.sha_lists[:3]
    )

    # When/Then: exact source order is accepted.
    validate_source_order(expected, produced)


def test_validate_source_order_when_rows_are_swapped_raises_typed_error(
    test_split: TangledSplitLike,
) -> None:
    # Given: two produced rows in reverse source order.
    expected = test_split.sha_lists[:2]
    produced = (
        {"shas": json.dumps(list(expected[1]))},
        {"shas": json.dumps(list(expected[0]))},
    )

    # When/Then: the first displaced row is rejected.
    with pytest.raises(SourceOrderError, match="index 0"):
        _ = validate_source_order(expected, produced)


def test_failure_record_is_frozen_and_has_jsonl_shape() -> None:
    # Given: one unresolved inference failure.
    failure = FailureRecord(
        row_index=7,
        shas=("abc",),
        context_len=2048,
        with_message=False,
        error_type="ModelOutputError",
        error_message="invalid JSON",
        raw_output="not-json",
    )

    # When: it is converted to a JSONL-ready record.
    record = failure.as_json_record()

    # Then: provenance and condition fields are explicit and immutable.
    assert record == {
        "row_index": 7,
        "shas": ["abc"],
        "context_len": 2048,
        "with_message": False,
        "error_type": "ModelOutputError",
        "error_message": "invalid JSON",
        "raw_output": "not-json",
    }
    with pytest.raises(FrozenInstanceError):
        setattr(failure, "row_index", 8)


def test_finalize_run_when_all_ten_files_are_complete_marks_canonical(
    tmp_path: Path,
) -> None:
    # Given: all context/message CSVs with 350 successful ordered rows.
    run_directory = tmp_path / "model" / "20260731123456"
    _write_complete_run(run_directory)

    # When: the finalization gate is evaluated with no unresolved failures.
    canonical = finalize_run(run_directory, unresolved_failure_count=0)

    # Then: exactly ten files and 3,500 rows are certified.
    assert canonical.run_directory == run_directory
    assert len(canonical.result_files) == 10
    assert canonical.successful_rows == 3500


@pytest.mark.parametrize(
    ("mutation", "reason"),
    (("missing", "missing"), ("short", "350"), ("columns", "columns")),
)
def test_finalize_run_when_a_file_contract_fails_rejects_run(
    tmp_path: Path, mutation: str, reason: str
) -> None:
    # Given: a complete run with one selected file contract broken.
    run_directory = tmp_path / mutation
    _write_shape_only_run(run_directory)
    target = expected_result_paths(run_directory)[0]
    if mutation == "missing":
        target.unlink()
    elif mutation == "short":
        _write_result_file(target, row_count=349)
    else:
        _ = target.write_text("wrong,column\n1,2\n", encoding="utf-8")

    # When/Then: the canonical gate rejects the run.
    with pytest.raises(FinalizationError, match=reason):
        _ = finalize_run(run_directory, unresolved_failure_count=0)


def test_finalize_run_when_failures_remain_rejects_run(tmp_path: Path) -> None:
    # Given: complete CSVs but one unresolved sidecar failure.
    run_directory = tmp_path / "run"
    _write_complete_run(run_directory)

    # When/Then: canonical status is denied before publication.
    with pytest.raises(FinalizationError, match="unresolved"):
        _ = finalize_run(run_directory, unresolved_failure_count=1)


def test_finalize_run_when_csv_rows_are_arbitrary_values_rejects(tmp_path: Path) -> None:
    # Given: all required files and counts but no semantically valid result rows.
    run_directory = tmp_path / "arbitrary-values"
    _write_shape_only_run(run_directory)

    # When/Then: shape-only CSV data cannot receive canonical certification.
    with pytest.raises(FinalizationError, match="semantic"):
        _ = finalize_run(run_directory, unresolved_failure_count=0)


def test_finalize_run_when_failure_history_sidecar_is_deleted_rejects(tmp_path: Path) -> None:
    # Given: complete-looking CSV files but no durable failure history sidecar.
    run_directory = tmp_path / "missing-sidecar"
    _write_complete_run(run_directory)
    (run_directory / "failures.jsonl").unlink()

    # When/Then: the run cannot be certified after deleting failure evidence.
    with pytest.raises(FinalizationError, match="failure history"):
        _ = finalize_run(run_directory, unresolved_failure_count=0)


def test_module_import_when_dependencies_are_blocked_stays_lightweight() -> None:
    # Given: a fresh interpreter that fails on every heavy ML/data package.
    code = """
import builtins
real_import = builtins.__import__
blocked = {"torch", "transformers", "unsloth", "peft", "trl", "outlines", "datasets", "pandas"}
def guarded_import(name, *args, **kwargs):
    if name.split(".")[0] in blocked:
        raise RuntimeError(f"heavy import: {name}")
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import RQ.SLM.unsloth.results
"""

    # When: the contract module alone is imported.
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    # Then: import succeeds without touching a blocked package.
    assert completed.returncode == 0, completed.stderr
