"""Guarded compatibility CLI for the co-location diagnostic."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from .colocation_data import (
    PairRow,
    SummaryRow,
    load_diff_by_sha,
    load_pairs_by_split_k,
    summarize_by_split_k,
)
from .colocation_report import (
    build_factual_summary,
    build_summary_markdown,
    format_markdown_table,
    write_summary_csv,
)

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
DATA_DIR: Final[Path] = PROJECT_ROOT / "datasets" / "data"
TRAIN_CSV_PATH: Final[Path] = DATA_DIR / "tangled_ccs_dataset_train.csv"
TEST_CSV_PATH: Final[Path] = DATA_DIR / "tangled_ccs_dataset_test.csv"
CCS_DATASET_PATH: Final[Path] = DATA_DIR / "CCS Dataset.csv"
DEFAULT_OUTPUT_DIR: Final[Path] = (
    PROJECT_ROOT / "results" / "analysis" / "repo_dataset_validation"
)
OUTPUT_DIR: Final[Path] = DEFAULT_OUTPUT_DIR
COLOCATION_CSV_PATH: Final[Path] = OUTPUT_DIR / "colocation_by_k.csv"
COLOCATION_SUMMARY_PATH: Final[Path] = OUTPUT_DIR / "colocation_summary.md"


class ExistingOutputError(RuntimeError):
    """Raised when default outputs already exist and would be overwritten."""


@dataclass(frozen=True, slots=True)
class CliOptions:
    output_dir: Path | None
    force_default_output: bool


class _CliNamespace(argparse.Namespace):
    output_dir: Path | None
    force_default_output: bool

    def __init__(self) -> None:
        super().__init__()
        self.output_dir = None
        self.force_default_output = False


def parse_cli_args(argv: Sequence[str] | None = None) -> CliOptions:
    """Parse optional output directory and force flag."""
    parser = argparse.ArgumentParser(
        description="Cross-concern co-location characterization of the reconstructed tangled dataset."
    )
    _ = parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for colocation_by_k.csv and colocation_summary.md",
    )
    _ = parser.add_argument(
        "--force-default-output",
        action="store_true",
        help="Allow overwriting existing default output paths",
    )
    namespace = _CliNamespace()
    parsed = parser.parse_args(list(argv) if argv is not None else None, namespace)
    return CliOptions(parsed.output_dir, parsed.force_default_output)


def resolve_output_paths(
    output_dir: Path | None, *, force_default_output: bool = False
) -> tuple[Path, Path, Path]:
    """Return output paths while guarding existing defaults."""
    directory = DEFAULT_OUTPUT_DIR if output_dir is None else output_dir
    csv_path = directory / "colocation_by_k.csv"
    summary_path = directory / "colocation_summary.md"
    if output_dir is None and not force_default_output:
        existing = tuple(path for path in (csv_path, summary_path) if path.exists())
        if existing:
            names = ", ".join(str(path) for path in existing)
            raise ExistingOutputError(
                "Refusing to overwrite existing default colocation outputs: "
                + names
                + ". Pass --output-dir DIR for a fresh location, or "
                + "--force-default-output to overwrite intentionally."
            )
    return directory, csv_path, summary_path


def run_analysis() -> tuple[tuple[PairRow, ...], tuple[SummaryRow, ...]]:
    """Load committed evidence and compute pair and summary rows."""
    diffs = load_diff_by_sha(CCS_DATASET_PATH)
    split_paths = {"train": TRAIN_CSV_PATH, "test": TEST_CSV_PATH}
    pairs = load_pairs_by_split_k(diffs, split_paths)
    return pairs, summarize_by_split_k(pairs)


def main(argv: Sequence[str] | None = None) -> int:
    options = parse_cli_args(argv)
    try:
        output_dir, csv_path, summary_path = resolve_output_paths(
            options.output_dir,
            force_default_output=options.force_default_output,
        )
    except ExistingOutputError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print("Loading CCS Dataset.csv (per-sha git_diff lookup)...")
    print("Computing pairwise cross-concern co-location for all k>=2 tangled rows...")
    pairs, summaries = run_analysis()
    n_rows = len(frozenset((row.split, row.row_idx) for row in pairs))
    print(f"  {n_rows} tangled rows analyzed (train+test, k=2..5), {len(pairs)} concern pairs total")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_summary_csv(summaries, csv_path)
    summary_markdown = build_summary_markdown(summaries)
    _ = summary_path.write_text(summary_markdown + "\n", encoding="utf-8")
    print(f"Per (split, k) table saved to: {csv_path}")
    print(f"Summary saved to: {summary_path}")
    print("\n" + format_markdown_table(summaries))
    print("\n" + build_factual_summary(summaries))
    return 0
