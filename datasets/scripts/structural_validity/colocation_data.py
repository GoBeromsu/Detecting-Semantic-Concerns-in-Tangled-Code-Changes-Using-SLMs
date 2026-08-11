"""Typed loading and aggregation for the compatibility co-location report."""

from __future__ import annotations

import csv
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from statistics import median
from typing import Final

from pydantic import TypeAdapter

from .diff_metrics import pair_diff_metrics, parse_committed_diff

CONCERN_COUNTS: Final[tuple[int, ...]] = (2, 3, 4, 5)
LINE_GAP_THRESHOLDS: Final[tuple[int, int]] = (10, 50)
SPLIT_NAMES: Final[tuple[str, str]] = ("train", "test")
_SHA_LIST_ADAPTER: Final[TypeAdapter[list[str]]] = TypeAdapter(list[str])
_DATA_DIR: Final[Path] = Path(__file__).resolve().parents[1] / "data"
_ = csv.field_size_limit(sys.maxsize)


@dataclass(frozen=True, slots=True)
class PairRow:
    split: str
    concern_count: int
    row_idx: int
    same_dir: bool
    same_file: bool
    same_function: bool
    min_line_gap: int | None


@dataclass(frozen=True, slots=True)
class SummaryRow:
    split: str
    k: int
    n_rows: int
    pct_rows_same_dir: float
    pct_rows_same_file: float
    pct_rows_same_function: float
    n_pairs: int
    pct_pairs_same_dir: float
    pct_pairs_same_file: float
    pct_pairs_same_function: float
    n_sharing_pairs: int
    median_min_line_gap: float | None
    q1_min_line_gap: float | None
    q3_min_line_gap: float | None
    pct_gap_le10: float | None
    pct_gap_le50: float | None


def load_diff_by_sha(path: Path) -> dict[str, str]:
    """Load committed SHA-to-diff evidence from the CCS CSV."""
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            _required(row, "sha"): _required(row, "git_diff")
            for row in csv.DictReader(handle)
        }


def load_pairs_by_split_k(
    ccs_diff_by_sha: Mapping[str, str], split_paths: Mapping[str, Path] | None = None
) -> tuple[PairRow, ...]:
    """Recover each atomic diff and emit every unordered concern pair."""
    records: list[PairRow] = []
    paths = (
        {
            "train": _DATA_DIR / "tangled_ccs_dataset_train.csv",
            "test": _DATA_DIR / "tangled_ccs_dataset_test.csv",
        }
        if split_paths is None
        else split_paths
    )
    for split_name in SPLIT_NAMES:
        with paths[split_name].open(encoding="utf-8", newline="") as handle:
            for row_idx, row in enumerate(csv.DictReader(handle)):
                concern_count = int(_required(row, "concern_count"))
                if concern_count < 2:
                    continue
                atomics = tuple(
                    parse_committed_diff(ccs_diff_by_sha[sha])
                    for sha in _parse_string_array(_required(row, "shas"))
                )
                for left, right in combinations(atomics, 2):
                    metrics = pair_diff_metrics(left, right)
                    records.append(
                        PairRow(
                            split=split_name,
                            concern_count=concern_count,
                            row_idx=row_idx,
                            same_dir=metrics.same_dir,
                            same_file=metrics.same_file,
                            same_function=metrics.same_function,
                            min_line_gap=metrics.min_line_gap,
                        )
                    )
    return tuple(records)


def summarize_by_split_k(pairs: Sequence[PairRow]) -> tuple[SummaryRow, ...]:
    """Aggregate pair evidence into the historical train/test by-k schema."""
    summaries: list[SummaryRow] = []
    for split_name in SPLIT_NAMES:
        for concern_count in CONCERN_COUNTS:
            stratum = tuple(
                row
                for row in pairs
                if row.split == split_name and row.concern_count == concern_count
            )
            if not stratum:
                continue
            gaps = tuple(
                row.min_line_gap
                for row in stratum
                if row.same_file and row.min_line_gap is not None
            )
            summaries.append(_summarize_stratum(split_name, concern_count, stratum, gaps))
    return tuple(summaries)


def _summarize_stratum(
    split_name: str,
    concern_count: int,
    rows: Sequence[PairRow],
    gaps: Sequence[int],
) -> SummaryRow:
    row_ids = frozenset(row.row_idx for row in rows)
    return SummaryRow(
        split=split_name,
        k=concern_count,
        n_rows=len(row_ids),
        pct_rows_same_dir=_row_any_rate(rows, row_ids, "dir"),
        pct_rows_same_file=_row_any_rate(rows, row_ids, "file"),
        pct_rows_same_function=_row_any_rate(rows, row_ids, "function"),
        n_pairs=len(rows),
        pct_pairs_same_dir=_bool_rate(tuple(row.same_dir for row in rows)),
        pct_pairs_same_file=_bool_rate(tuple(row.same_file for row in rows)),
        pct_pairs_same_function=_bool_rate(tuple(row.same_function for row in rows)),
        n_sharing_pairs=len(gaps),
        median_min_line_gap=float(median(gaps)) if gaps else None,
        q1_min_line_gap=_percentile(gaps, 0.25),
        q3_min_line_gap=_percentile(gaps, 0.75),
        pct_gap_le10=_threshold_rate(gaps, LINE_GAP_THRESHOLDS[0]),
        pct_gap_le50=_threshold_rate(gaps, LINE_GAP_THRESHOLDS[1]),
    )


def _row_any_rate(
    rows: Sequence[PairRow], row_ids: frozenset[int], metric: str
) -> float:
    positives = 0
    for row_idx in row_ids:
        row_pairs = tuple(row for row in rows if row.row_idx == row_idx)
        if metric == "dir":
            matched = any(row.same_dir for row in row_pairs)
        elif metric == "file":
            matched = any(row.same_file for row in row_pairs)
        else:
            matched = any(row.same_function for row in row_pairs)
        positives += int(matched)
    return 100.0 * positives / len(row_ids)


def _bool_rate(values: Sequence[bool]) -> float:
    return 100.0 * sum(values) / len(values)


def _threshold_rate(gaps: Sequence[int], threshold: int) -> float | None:
    return 100.0 * sum(gap <= threshold for gap in gaps) / len(gaps) if gaps else None


def _percentile(values: Sequence[int], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _required(row: Mapping[str, str | None], key: str) -> str:
    value = row.get(key)
    if value is None:
        raise KeyError(key)
    return value


def _parse_string_array(raw: str) -> tuple[str, ...]:
    return tuple(_SHA_LIST_ADAPTER.validate_python(json.loads(raw)))
