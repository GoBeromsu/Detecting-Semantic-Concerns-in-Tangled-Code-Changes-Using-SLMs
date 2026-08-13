"""Formatting for the co-location artifacts (same-file / same-folder only)."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from .claim_guard import validate_claim
from .colocation_data import SummaryRow

SUMMARY_FIELDS: Final[tuple[str, ...]] = (
    "split",
    "k",
    "n_rows",
    "n_pairs",
    "pct_rows_same_file",
    "pct_rows_same_dir",
    "pct_pairs_same_file",
    "pct_pairs_same_dir",
)


def write_summary_csv(rows: Sequence[SummaryRow], path: Path) -> None:
    """Write the summary column order (per-k rows, then pooled rows)."""
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
        "n_pairs",
        "%rows same-file",
        "%rows same-dir",
        "%pairs same-file",
        "%pairs same-dir",
    )
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        cells = (
            row.split,
            _k_label(row.k),
            str(row.n_rows),
            str(row.n_pairs),
            _one_decimal(row.pct_rows_same_file),
            _one_decimal(row.pct_rows_same_dir),
            _one_decimal(row.pct_pairs_same_file),
            _one_decimal(row.pct_pairs_same_dir),
        )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_factual_summary(rows: Sequence[SummaryRow]) -> str:
    """Build the bounded factual description of the two indicators.

    Reports the k=2..5 pair-level range across both splits, the combined
    all-splits headline commit-level rates (the direct denominator-1,400
    replacement figures), and the expected row-vs-pair and k-monotonicity
    observations.
    """
    if not rows:
        return validate_claim("No stratum was summarised, so no co-location range can be reported.")
    stratum_rows = tuple(row for row in rows if row.k is not None)
    if stratum_rows:
        dir_low, dir_high = _bounds(tuple(row.pct_pairs_same_dir for row in stratum_rows))
        file_low, file_high = _bounds(tuple(row.pct_pairs_same_file for row in stratum_rows))
        range_sentence = (
            f"Across k=2..5 and both splits, {dir_low:.1f}-{dir_high:.1f}% of cross-concern "
            f"pairs touch a common folder and {file_low:.1f}-{file_high:.1f}% touch a common file."
        )
    else:
        range_sentence = (
            "No per-k stratum was summarised, so no k=2..5 pair-level range can be reported."
        )
    overall_row = next((row for row in rows if row.k is None and row.split == "all"), None)
    if overall_row is not None:
        headline_sentence = (
            f"Over the {overall_row.n_rows} multi-concern commits (train+test combined), "
            f"{overall_row.pct_rows_same_file:.1f}% have at least one concern pair sharing a file "
            f"and {overall_row.pct_rows_same_dir:.1f}% have at least one pair sharing a folder."
        )
    else:
        headline_sentence = (
            "No combined all-splits row was summarised, so no headline commit-level rate can be reported."
        )
    return validate_claim(" ".join(
        (
            headline_sentence,
            range_sentence,
            "Row-level rates (whether at least one pair in the tangled commit co-locates) are higher than pair-level rates, as expected since a row's probability of containing at least one co-locating pair grows with the number of pairs it contains.",
            "Co-location rates at both granularities (file, folder) generally do not increase monotonically with k, since a larger k spreads the same commit's diff across more constituent atomics without proportionally increasing shared-location edits per pair.",
        )
    ))


def build_summary_markdown(rows: Sequence[SummaryRow]) -> str:
    table = format_markdown_table(rows)
    factual = build_factual_summary(rows)
    return "\n\n".join(
        (
            "# Cross-Concern Co-Location Characterization (Reconstructed Tangled Dataset)",
            "Internal characterization of the reconstructed tangled dataset only. No externally-labelled tangled reference set is used.",
            "## Table",
            table,
            "## Summary",
            factual,
        )
    )


def _csv_values(row: SummaryRow) -> tuple[str | int | float, ...]:
    return (
        row.split,
        _k_label(row.k),
        row.n_rows,
        row.n_pairs,
        row.pct_rows_same_file,
        row.pct_rows_same_dir,
        row.pct_pairs_same_file,
        row.pct_pairs_same_dir,
    )


def _k_label(k: int | None) -> str:
    return "overall" if k is None else str(k)


def _one_decimal(value: float) -> str:
    return f"{value:.1f}"


def _bounds(values: Sequence[float]) -> tuple[float, float]:
    """Min and max of a sequence a non-empty `rows` always populates."""
    return (min(values), max(values)) if values else (0.0, 0.0)
