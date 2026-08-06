"""Builders for a certifiable result tree, derived from the production writer.

Every row here goes through ``InferenceResult.as_csv_row`` rather than being spelled out as
a literal dict. Certification reads back exactly what that method wrote, so a hand-authored
fixture lets the writer and ``finalize_run`` drift apart while both sides of the suite still
pass. They did drift: three separate hand-written fixtures claimed ``predicted_types`` held a
``{"types": ...}`` object while the writer emitted a bare array, so ``finalize_run`` rejected
every real run and no test noticed. Sharing one writer-derived builder makes that class of
disagreement unrepresentable.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path

from RQ.SLM.unsloth._types import CsvResultRow
from RQ.SLM.unsloth.results import (
    EXPECTED_SUCCESSFUL_ROWS,
    InferenceResult,
    expected_result_paths,
)
from utils.llms.constant import DEFAULT_DF_COLUMNS


def result_row(path: Path, index: int) -> CsvResultRow:
    """One successful row for the cell ``path`` identifies, as the writer would emit it."""
    return InferenceResult(
        predicted_types=("fix",),
        actual_types=("fix",),
        inference_time=0.0,
        shas=(f"sha-{index}",),
        context_len=int(path.stem.removesuffix("_zs")),
        with_message=path.parent.name == "msg1",
    ).as_csv_row()


def write_result_file(
    path: Path, row_count: int = EXPECTED_SUCCESSFUL_ROWS, order: Sequence[int] | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    indices = range(row_count) if order is None else order
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DEFAULT_DF_COLUMNS)
        writer.writeheader()
        writer.writerows(result_row(path, index) for index in indices)


def write_run_identity(run_directory: Path) -> None:
    """Certification recovers the run's source SHA order from here, so every run needs it."""
    run_directory.mkdir(parents=True, exist_ok=True)
    _ = (run_directory / "run_identity.json").write_text(
        json.dumps({"ordered_test_shas": [[f"sha-{i}"] for i in range(EXPECTED_SUCCESSFUL_ROWS)]}),
        encoding="utf-8",
    )


def write_complete_run(run_directory: Path) -> None:
    """A tree that must certify: ten full cells, a run identity, and an empty failure log."""
    for path in expected_result_paths(run_directory):
        write_result_file(path)
    write_run_identity(run_directory)
    _ = (run_directory / "failures.jsonl").write_text("", encoding="utf-8")
