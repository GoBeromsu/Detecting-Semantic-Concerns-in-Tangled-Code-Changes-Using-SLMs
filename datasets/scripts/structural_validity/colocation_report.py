"""Formatting for the backward-compatible co-location artifacts."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from .colocation_data import SummaryRow

SUMMARY_FIELDS: Final[tuple[str, ...]] = (
    "split",
    "k",
    "n_rows",
    "pct_rows_same_dir",
    "pct_rows_same_file",
    "pct_rows_same_function",
    "n_pairs",
    "pct_pairs_same_dir",
    "pct_pairs_same_file",
    "pct_pairs_same_function",
    "n_sharing_pairs",
    "median_min_line_gap",
    "q1_min_line_gap",
    "q3_min_line_gap",
    "pct_gap_le10",
    "pct_gap_le50",
)


def write_summary_csv(rows: Sequence[SummaryRow], path: Path) -> None:
    """Write the historical summary column order."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(SUMMARY_FIELDS)
        for row in rows:
            writer.writerow(_csv_values(row))


def format_markdown_table(rows: Sequence[SummaryRow]) -> str:
    header = (
        "split",
        "k",
        "n_rows",
        "%rows same-dir",
        "%rows same-file",
        "%rows same-func",
        "n_pairs",
        "%pairs same-dir",
        "%pairs same-file",
        "%pairs same-func",
        "median gap",
        "IQR gap",
        "%gap<=10",
        "%gap<=50",
    )
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        cells = (
            row.split,
            str(row.k),
            str(row.n_rows),
            _one_decimal(row.pct_rows_same_dir),
            _one_decimal(row.pct_rows_same_file),
            _one_decimal(row.pct_rows_same_function),
            str(row.n_pairs),
            _one_decimal(row.pct_pairs_same_dir),
            _one_decimal(row.pct_pairs_same_file),
            _one_decimal(row.pct_pairs_same_function),
            _optional_decimal(row.median_min_line_gap),
            _iqr(row),
            _optional_decimal(row.pct_gap_le10),
            _optional_decimal(row.pct_gap_le50),
        )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def format_latex_table(rows: Sequence[SummaryRow]) -> str:
    """Build the historical booktabs table without an ``[h]`` specifier."""
    lines = [
        r"\begin{table*}",
        r"\centering",
        r"\caption{Cross-concern co-location in the reconstructed tangled dataset, by split and concern count $k$.}",
        r"\label{tab:colocation}",
        r"\begin{tabular}{lrrrrrrrrrrrr}",
        r"\toprule",
        r"Split & $k$ & $n$ & RDir\% & RFile\% & RFunc\% & PDir\% & PFile\% & PFunc\% & MedGap & IQR & Gap$\leq$10\% & Gap$\leq$50\% \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            "".join(
                (
                    f"{row.split.capitalize()} & {row.k} & {row.n_rows} & ",
                    f"{_one_decimal(row.pct_rows_same_dir)} & {_one_decimal(row.pct_rows_same_file)} & ",
                    f"{_one_decimal(row.pct_rows_same_function)} & {_one_decimal(row.pct_pairs_same_dir)} & ",
                    f"{_one_decimal(row.pct_pairs_same_file)} & {_one_decimal(row.pct_pairs_same_function)} & ",
                    f"{_optional_decimal(row.median_min_line_gap)} & {_iqr(row)} & ",
                    f"{_optional_decimal(row.pct_gap_le10)} & {_optional_decimal(row.pct_gap_le50)} \\\\",
                )
            )
        )
    lines.extend((r"\bottomrule", r"\end{tabular}", r"\end{table*}"))
    return "\n".join(lines)


def build_factual_summary(rows: Sequence[SummaryRow]) -> str:
    """Build the legacy bounded factual description."""
    dir_range = _range(tuple(row.pct_pairs_same_dir for row in rows))
    file_range = _range(tuple(row.pct_pairs_same_file for row in rows))
    function_range = _range(tuple(row.pct_pairs_same_function for row in rows))
    gap_range = _range(
        tuple(row.pct_gap_le10 for row in rows if row.pct_gap_le10 is not None)
    )
    return " ".join(
        (
            f"Across k=2..5 and both splits, {dir_range[0]:.1f}-{dir_range[1]:.1f}% of cross-concern pairs touch a common directory, {file_range[0]:.1f}-{file_range[1]:.1f}% touch a common file, and {function_range[0]:.1f}-{function_range[1]:.1f}% touch a common function context.",
            "Row-level rates (whether at least one pair in the tangled commit co-locates) are higher than pair-level rates, as expected since a row's probability of containing at least one co-locating pair grows with the number of pairs it contains.",
            f"Among pairs that do share a file, the minimum edit-to-edit line gap has a median in the tens of lines, with {gap_range[0]:.1f}-{gap_range[1]:.1f}% of sharing pairs landing within 10 lines of each other across strata.",
            "Co-location rates at all three granularities (directory, file, function) generally do not increase monotonically with k, since a larger k spreads the same commit's diff across more constituent atomics without proportionally increasing shared-location edits per pair.",
        )
    )


def build_summary_markdown(rows: Sequence[SummaryRow]) -> str:
    table = format_markdown_table(rows)
    latex = format_latex_table(rows)
    factual = build_factual_summary(rows)
    return "\n\n".join(
        (
            "# Cross-Concern Co-Location Characterization (Reconstructed Tangled Dataset)",
            "Internal characterization of the reconstructed tangled dataset only. No externally-labelled tangled reference set is used.",
            "## Table",
            table,
            "## LaTeX (booktabs)",
            "```latex\n" + latex + "\n```",
            "## Summary",
            factual,
        )
    )


def _csv_values(row: SummaryRow) -> tuple[str | int | float, ...]:
    return (
        row.split,
        row.k,
        row.n_rows,
        row.pct_rows_same_dir,
        row.pct_rows_same_file,
        row.pct_rows_same_function,
        row.n_pairs,
        row.pct_pairs_same_dir,
        row.pct_pairs_same_file,
        row.pct_pairs_same_function,
        row.n_sharing_pairs,
        _csv_optional(row.median_min_line_gap),
        _csv_optional(row.q1_min_line_gap),
        _csv_optional(row.q3_min_line_gap),
        _csv_optional(row.pct_gap_le10),
        _csv_optional(row.pct_gap_le50),
    )


def _csv_optional(value: float | None) -> str | float:
    return "" if value is None else value


def _one_decimal(value: float) -> str:
    return f"{value:.1f}"


def _optional_decimal(value: float | None) -> str:
    return "n/a" if value is None else _one_decimal(value)


def _iqr(row: SummaryRow) -> str:
    if row.q1_min_line_gap is None or row.q3_min_line_gap is None:
        return "n/a"
    return f"[{row.q1_min_line_gap:.1f}, {row.q3_min_line_gap:.1f}]"


def _range(values: Sequence[float]) -> tuple[float, float]:
    return min(values), max(values)
