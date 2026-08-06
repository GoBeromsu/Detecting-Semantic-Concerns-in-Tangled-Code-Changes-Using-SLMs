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
    completed_source_rows,
    expected_result_paths,
    finalize_run,
    format_run_timestamp,
    parse_model_output,
)
from __test__.unsloth.result_fixtures import (
    write_complete_run as _write_complete_run,
    write_result_file as _write_result_file,
    write_run_identity as _write_run_identity,
)
from RQ.SLM.unsloth.generation import STRICT_RESPONSE_SCHEMA
from utils.llms.constant import COMMIT_TYPES, DEFAULT_DF_COLUMNS


class TangledSplitLike(Protocol):
    sha_lists: tuple[tuple[str, ...], ...]


def _write_shape_only_run(run_directory: Path) -> None:
    """Write files that satisfy every structural gate but hold no valid results.

    ``shas`` is real so the run still reaches the semantic gate; garbling it too would make
    the run fail as "not this split", which is a different rejection than the one under test.
    """
    for path in expected_result_paths(run_directory):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=DEFAULT_DF_COLUMNS)
            writer.writeheader()
            writer.writerows(
                dict.fromkeys(DEFAULT_DF_COLUMNS, "value") | {"shas": json.dumps([f"sha-{index}"])}
                for index in range(EXPECTED_SUCCESSFUL_ROWS)
            )
    _write_run_identity(run_directory)
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


def test_completed_source_rows_when_rows_match_real_fixture_reports_every_row(
    test_split: TangledSplitLike,
) -> None:
    # Given: source SHA order and produced rows from the canonical test fixture.
    expected = test_split.sha_lists[:3]
    produced = tuple(
        {"shas": json.dumps(list(shas))} for shas in test_split.sha_lists[:3]
    )

    # When: the produced rows are matched against source order.
    completed = completed_source_rows(expected, produced)

    # Then: exact source order is accepted and every row index is reported complete.
    assert completed == (0, 1, 2)


def test_completed_source_rows_when_a_failed_row_is_skipped_reports_the_gap(
    test_split: TangledSplitLike,
) -> None:
    # Given: three source rows whose middle row failed and was never written.
    expected = test_split.sha_lists[:3]
    produced = (
        {"shas": json.dumps(list(expected[0]))},
        {"shas": json.dumps(list(expected[2]))},
    )

    # When: the partial file is matched against source order.
    completed = completed_source_rows(expected, produced)

    # Then: the subsequence is accepted and only the skipped row stays outstanding.
    assert completed == (0, 2)
    assert 1 not in completed


def test_completed_source_rows_when_two_source_rows_share_shas_reports_only_one() -> None:
    # Given: two distinct source rows that happen to carry the same SHA tuple, of which
    # only the first has been produced.
    expected = (["sha-a"], ["sha-a"], ["sha-b"])
    produced = ({"shas": json.dumps(["sha-a"])},)

    # When: the partial file is matched against source order.
    completed = completed_source_rows(expected, produced)

    # Then: one produced row completes exactly one source row. Keying completion on the SHA
    # value instead would mark both twins done, so a resume would skip the un-produced one
    # forever and the cell could never reach its full row count.
    assert completed == (0,)


def test_completed_source_rows_when_a_resume_appended_a_row_out_of_order_accepts_it(
    test_split: TangledSplitLike,
) -> None:
    # Given: a cell whose middle row failed on the first pass and was regenerated by a
    # resume — result rows are appended, so the recovered row sits at the end of the file.
    expected = test_split.sha_lists[:3]
    produced = (
        {"shas": json.dumps(list(expected[0]))},
        {"shas": json.dumps(list(expected[2]))},
        {"shas": json.dumps(list(expected[1]))},
    )

    # When: a later resume re-reads the repaired file.
    completed = completed_source_rows(expected, produced)

    # Then: the file this program itself wrote is accepted as complete. Requiring source
    # order here would reject it and strand the run with no recovery short of hand-sorting.
    assert completed == (0, 2, 1)


def test_completed_source_rows_when_a_row_is_foreign_raises_typed_error(
    test_split: TangledSplitLike,
) -> None:
    # Given: a produced row whose SHAs belong to no source row in this cell.
    expected = test_split.sha_lists[:2]
    produced = (
        {"shas": json.dumps(list(expected[0]))},
        {"shas": json.dumps(["sha-from-another-split"])},
    )

    # When/Then: it is rejected rather than silently counted as progress.
    with pytest.raises(SourceOrderError, match="index 1"):
        _ = completed_source_rows(expected, produced)


def test_completed_source_rows_when_a_sha_repeats_beyond_its_source_count_raises(
    test_split: TangledSplitLike,
) -> None:
    # Given: one source row, produced twice — a duplicated append rather than progress.
    expected = test_split.sha_lists[:2]
    produced = (
        {"shas": json.dumps(list(expected[0]))},
        {"shas": json.dumps(list(expected[0]))},
    )

    # When/Then: the second copy claims no index and is refused, so a duplicated row can
    # never let a cell report itself complete while a real source row is still missing.
    with pytest.raises(SourceOrderError, match="index 1"):
        _ = completed_source_rows(expected, produced)


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

    # When: the finalization gate is evaluated.
    canonical = finalize_run(run_directory)

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
        _ = finalize_run(run_directory)


def test_finalize_run_when_past_failures_were_regenerated_still_certifies(
    tmp_path: Path,
) -> None:
    # Given: complete CSVs plus a sidecar recording a failure a resume later re-generated.
    run_directory = tmp_path / "recovered"
    _write_complete_run(run_directory)
    _ = (run_directory / "failures.jsonl").write_text(
        json.dumps({"row_index": 7, "error_type": "ModelOutputError"}) + "\n",
        encoding="utf-8",
    )

    # When/Then: the sidecar is provenance, so complete data still certifies.
    assert finalize_run(run_directory).successful_rows == 3500


def test_finalize_run_when_csv_rows_are_arbitrary_values_rejects(tmp_path: Path) -> None:
    # Given: all required files and counts but no semantically valid result rows.
    run_directory = tmp_path / "arbitrary-values"
    _write_shape_only_run(run_directory)

    # When/Then: shape-only CSV data cannot receive canonical certification.
    with pytest.raises(FinalizationError, match="semantic"):
        _ = finalize_run(run_directory)


def test_finalize_run_when_rows_are_complete_but_out_of_order_rejects(tmp_path: Path) -> None:
    # Given: a complete run whose first cell holds every row, but with two of them swapped —
    # what a crash between a resume's append and its re-sort leaves behind.
    run_directory = tmp_path / "unsorted"
    _write_complete_run(run_directory)
    order = list(range(EXPECTED_SUCCESSFUL_ROWS))
    order[1], order[-1] = order[-1], order[1]
    _write_result_file(expected_result_paths(run_directory)[0], order=order)

    # When/Then: certification proves order rather than trusting a resume to have restored it,
    # because downstream analysis pairs models by CSV row position.
    with pytest.raises(FinalizationError, match="order"):
        _ = finalize_run(run_directory)


def test_finalize_run_when_run_identity_is_missing_rejects(tmp_path: Path) -> None:
    # Given: a complete run whose recorded source split has been deleted.
    run_directory = tmp_path / "no-identity"
    _write_complete_run(run_directory)
    (run_directory / "run_identity.json").unlink()

    # When/Then: without the split there is nothing to check row order against, so the run
    # cannot be certified rather than being certified on shape alone.
    with pytest.raises(FinalizationError, match="run identity"):
        _ = finalize_run(run_directory)


def test_finalize_run_when_failure_history_sidecar_is_deleted_rejects(tmp_path: Path) -> None:
    # Given: complete-looking CSV files but no durable failure history sidecar.
    run_directory = tmp_path / "missing-sidecar"
    _write_complete_run(run_directory)
    (run_directory / "failures.jsonl").unlink()

    # When/Then: the run cannot be certified after deleting failure evidence.
    with pytest.raises(FinalizationError, match="failure history"):
        _ = finalize_run(run_directory)


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
