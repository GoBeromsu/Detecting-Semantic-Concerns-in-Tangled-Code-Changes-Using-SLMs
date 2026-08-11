"""Reporting must survive strata that carry no measurable line gap.

`colocation_data` returns None for every gap statistic of a stratum with no
file-sharing pair, and `build_factual_summary` is the last step of the CLI's
`main()`, after the CSV and markdown are already on disk. A crash there loses
the run's conclusion while leaving partial output behind.
"""

from __future__ import annotations

from structural_validity.colocation_data import SummaryRow
from structural_validity.colocation_report import build_factual_summary


def _row(split: str, k: int, *, gap: float | None) -> SummaryRow:
    return SummaryRow(
        split=split,
        k=k,
        n_rows=10,
        pct_rows_same_dir=30.0,
        pct_rows_same_file=7.0,
        pct_rows_same_function=1.0,
        n_pairs=10 * (k * (k - 1) // 2),
        pct_pairs_same_dir=9.0,
        pct_pairs_same_file=1.5,
        pct_pairs_same_function=0.2,
        n_sharing_pairs=0 if gap is None else 3,
        median_min_line_gap=None if gap is None else 12.0,
        q1_min_line_gap=None if gap is None else 4.0,
        q3_min_line_gap=None if gap is None else 40.0,
        pct_gap_le10=gap,
        pct_gap_le50=gap,
    )


def test_summary_when_every_stratum_lacks_a_measurable_gap_reports_the_absence() -> None:
    summary = build_factual_summary(tuple(_row("train", k, gap=None) for k in (2, 3, 4, 5)))
    assert "no within-10-lines rate is reported" in summary
    assert "% of cross-concern pairs touch a common directory" in summary


def test_summary_when_some_stratum_has_a_gap_reports_the_range() -> None:
    rows = (_row("train", 2, gap=None), _row("train", 3, gap=25.0), _row("test", 4, gap=35.0))
    summary = build_factual_summary(rows)
    assert "25.0-35.0% of sharing pairs" in summary


def test_summary_when_there_are_no_strata_says_so_instead_of_raising() -> None:
    assert build_factual_summary(()) == (
        "No stratum was summarised, so no co-location range can be reported."
    )
