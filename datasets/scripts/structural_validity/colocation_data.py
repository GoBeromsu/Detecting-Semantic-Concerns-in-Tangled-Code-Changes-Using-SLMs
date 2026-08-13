"""Typed loading and aggregation for the compatibility co-location report."""

from __future__ import annotations

import csv
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Final, Literal

from pydantic import TypeAdapter

from .diff_metrics import pair_diff_metrics, parse_committed_diff

CONCERN_COUNTS: Final[tuple[int, ...]] = (2, 3, 4, 5)
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
    k: int | None
    n_rows: int
    n_pairs: int
    pct_rows_same_file: float
    pct_rows_same_dir: float
    pct_pairs_same_file: float
    pct_pairs_same_dir: float


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
    """Aggregate pair evidence into per-k, per-split pooled, and combined rows.

    Emits one row per (split, k) stratum for k=2..5, followed by one pooled
    `overall` row per split (`k=None`, aggregating that split's own k=2..5
    rows), followed by one combined `all`-split pooled row (`k=None`,
    aggregating both splits together) so the 1,400-row multi-concern
    denominator has a direct row in the output.
    """
    summaries: list[SummaryRow] = []
    for split_name in SPLIT_NAMES:
        split_rows = tuple(row for row in pairs if row.split == split_name)
        for concern_count in CONCERN_COUNTS:
            stratum = tuple(row for row in split_rows if row.concern_count == concern_count)
            if not stratum:
                continue
            summaries.append(_summarize_stratum(split_name, concern_count, stratum))
        if split_rows:
            summaries.append(_summarize_stratum(split_name, None, split_rows))
    if pairs:
        summaries.append(_summarize_stratum("all", None, tuple(pairs)))
    return tuple(summaries)


def _summarize_stratum(
    split_name: str,
    concern_count: int | None,
    rows: Sequence[PairRow],
) -> SummaryRow:
    n_rows = len({(row.split, row.row_idx) for row in rows})
    return SummaryRow(
        split=split_name,
        k=concern_count,
        n_rows=n_rows,
        n_pairs=len(rows),
        pct_rows_same_file=_row_any_rate(rows, "file"),
        pct_rows_same_dir=_row_any_rate(rows, "dir"),
        pct_pairs_same_file=_bool_rate(tuple(row.same_file for row in rows)),
        pct_pairs_same_dir=_bool_rate(tuple(row.same_dir for row in rows)),
    )


def _row_any_rate(rows: Sequence[PairRow], metric: Literal["dir", "file"]) -> float:
    """Share of distinct (split, row_idx) commits with >=1 co-locating pair.

    Keyed by (split, row_idx) rather than row_idx alone so that pooling pairs
    across both splits (the combined `all` row) never conflates a train row
    and a test row that happen to share a row_idx.
    """
    seen: dict[tuple[str, int], bool] = {}
    for row in rows:
        key = (row.split, row.row_idx)
        matched = row.same_dir if metric == "dir" else row.same_file
        seen[key] = seen.get(key, False) or matched
    return 100.0 * sum(seen.values()) / len(seen)


def _bool_rate(values: Sequence[bool]) -> float:
    return 100.0 * sum(values) / len(values)


def _required(row: Mapping[str, str | None], key: str) -> str:
    value = row.get(key)
    if value is None:
        raise KeyError(key)
    return value


def _parse_string_array(raw: str) -> tuple[str, ...]:
    return tuple(_SHA_LIST_ADAPTER.validate_python(json.loads(raw)))
