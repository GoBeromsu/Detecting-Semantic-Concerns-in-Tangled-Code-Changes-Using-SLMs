"""Co-location summary schema: same-file / same-folder rates + pooled rows.

`colocation_data.summarize_by_split_k` emits one row per (split, k) stratum
for k=2..5, one pooled `overall` row per split (k=None), and one combined
`all`-split pooled row (k=None) over the 1,400 multi-concern rows. There is
no function-context or line-gap column, and no union/either indicator: only
same-file and same-dir, at both the row (commit) level and the pair level.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from __test__.conftest import DATASET_DIR

from structural_validity.colocation_data import (
    PairRow,
    SummaryRow,
    load_diff_by_sha,
    load_pairs_by_split_k,
    summarize_by_split_k,
)
from structural_validity.colocation_report import (
    SUMMARY_FIELDS,
    build_factual_summary,
    write_summary_csv,
)

# Removed columns from the pre-simplification schema (function-context and
# line-gap granularities, plus the historical union/either column). None of
# these may reappear in the summary CSV header.
_REMOVED_COLUMNS: tuple[str, ...] = (
    "pct_rows_same_function",
    "pct_pairs_same_function",
    "n_sharing_pairs",
    "median_min_line_gap",
    "q1_min_line_gap",
    "q3_min_line_gap",
    "pct_gap_le10",
    "pct_gap_le50",
    "pct_rows_same_dir_or_file",
    "pct_pairs_same_dir_or_file",
)


@pytest.fixture(scope="module")
def pairs() -> tuple[PairRow, ...]:
    diffs = load_diff_by_sha(DATASET_DIR / "CCS Dataset.csv")
    split_paths = {
        "train": DATASET_DIR / "tangled_ccs_dataset_train.csv",
        "test": DATASET_DIR / "tangled_ccs_dataset_test.csv",
    }
    return load_pairs_by_split_k(diffs, split_paths)


@pytest.fixture(scope="module")
def summaries(pairs: tuple[PairRow, ...]) -> tuple[SummaryRow, ...]:
    return summarize_by_split_k(pairs)


def test_summary_row_has_no_removed_fields() -> None:
    # Then the dataclass carries only the eight simplified columns
    assert {f for f in SummaryRow.__dataclass_fields__} == {
        "split",
        "k",
        "n_rows",
        "n_pairs",
        "pct_rows_same_file",
        "pct_rows_same_dir",
        "pct_pairs_same_file",
        "pct_pairs_same_dir",
    }


def test_csv_header_has_no_removed_columns(
    summaries: tuple[SummaryRow, ...], tmp_path: Path
) -> None:
    # Given the real summary rows written to a scratch CSV
    csv_path = tmp_path / "colocation_by_k.csv"
    write_summary_csv(summaries, csv_path)

    # When the header row is read back
    with csv_path.open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))

    # Then it matches the simplified schema exactly, with no removed column
    assert header == list(SUMMARY_FIELDS)
    for removed in _REMOVED_COLUMNS:
        assert removed not in header


def test_format_latex_table_no_longer_exists() -> None:
    # Then importing the deleted LaTeX formatter fails
    with pytest.raises(ImportError):
        from structural_validity.colocation_report import (  # noqa: F401
            format_latex_table,
        )


def test_per_k_strata_row_rate_dominates_pair_rate_for_both_indicators(
    summaries: tuple[SummaryRow, ...],
) -> None:
    # Given every per-k stratum row (pooled rows excluded: k is not None)
    strata = tuple(row for row in summaries if row.k is not None)
    assert strata, "expected at least one k=2..5 stratum from the real dataset"

    # Then the row-level any-pair rate is >= the pair-level rate, for both
    # indicators, in every stratum (each row contributes C(k,2) >= 1 pairs,
    # so "at least one pair co-locates" can only be as rare as, or rarer
    # than, "this specific pair co-locates" on average).
    for row in strata:
        assert row.pct_rows_same_file >= row.pct_pairs_same_file
        assert row.pct_rows_same_dir >= row.pct_pairs_same_dir


def test_per_k_strata_cover_train_and_test_k2_to_k5(
    summaries: tuple[SummaryRow, ...],
) -> None:
    # Then every (split, k) combination for k=2..5 is present exactly once
    strata_keys = {(row.split, row.k) for row in summaries if row.k is not None}
    assert strata_keys == {
        (split, k) for split in ("train", "test") for k in (2, 3, 4, 5)
    }


def test_per_k_strata_n_rows_match_dataset_invariants(
    summaries: tuple[SummaryRow, ...],
) -> None:
    # Then each split has exactly 280 (train) / 70 (test) rows per k, and
    # n_pairs per stratum equals C(k,2) times n_rows
    by_key = {(row.split, row.k): row for row in summaries if row.k is not None}
    for k in (2, 3, 4, 5):
        pairs_per_row = k * (k - 1) // 2
        train_row = by_key[("train", k)]
        test_row = by_key[("test", k)]
        assert train_row.n_rows == 280
        assert test_row.n_rows == 70
        assert train_row.n_pairs == 280 * pairs_per_row
        assert test_row.n_pairs == 70 * pairs_per_row


def test_pooled_rows_aggregate_the_dataset_denominators(
    summaries: tuple[SummaryRow, ...],
) -> None:
    # Given the pooled rows (k is None)
    by_key = {(row.split, row.k): row for row in summaries if row.k is None}

    # Then the per-split overall rows and the combined all-splits row match
    # the multi-concern denominators: 1,120 train + 280 test = 1,400 all
    assert by_key[("train", None)].n_rows == 1120
    assert by_key[("test", None)].n_rows == 280
    assert by_key[("all", None)].n_rows == 1400
    assert (
        by_key[("all", None)].n_pairs
        == by_key[("train", None)].n_pairs + by_key[("test", None)].n_pairs
    )


def test_pooled_row_rates_fall_within_its_own_k_strata_bounds(
    summaries: tuple[SummaryRow, ...],
) -> None:
    # Given a split's per-k strata and its pooled overall row
    by_split_k = {
        (row.split, row.k): row for row in summaries if row.k is not None
    }
    pooled = {(row.split, row.k): row for row in summaries if row.k is None}

    # Then the pooled rate for each indicator is a weighted aggregate of its
    # own strata, so it must land within their min/max envelope
    for split in ("train", "test"):
        strata = [by_split_k[(split, k)] for k in (2, 3, 4, 5)]
        overall = pooled[(split, None)]
        for attr in (
            "pct_rows_same_file",
            "pct_rows_same_dir",
            "pct_pairs_same_file",
            "pct_pairs_same_dir",
        ):
            values = [getattr(row, attr) for row in strata]
            pooled_value = getattr(overall, attr)
            assert min(values) - 1e-9 <= pooled_value <= max(values) + 1e-9


def test_factual_summary_reports_file_and_folder_rates_only(
    summaries: tuple[SummaryRow, ...],
) -> None:
    # When the factual prose is rendered from the real summary rows
    text = build_factual_summary(summaries)

    # Then it names both indicators and stays clear of forbidden vocabulary
    # (claim_guard.validate_claim already raises on a violation; a clean
    # return means the guard passed)
    assert "sharing a file" in text
    assert "sharing a folder" in text
    assert "1400 multi-concern commits" in text
    lowered = text.lower()
    for forbidden in ("same_function", "line_gap", "shared variable", "call graph"):
        assert forbidden not in lowered


def test_factual_summary_reports_absence_when_no_strata() -> None:
    assert build_factual_summary(()) == (
        "No stratum was summarised, so no co-location range can be reported."
    )
