#!/usr/bin/env python3
"""
Tangled Dataset Type Distribution by Concern Count
Reads the repo-grouped tangled train/test CSVs and shows, per concern count
(k=1..5), the per-type label count in each split. The repo-grouped pipeline
(build_repo_pool.py -> generate_repo_tangled.py) drives adaptive
under-represented-type-first sampling to an EXACT 1/7 uniform label
frequency per concern count, so the visual point of this chart is that all 7
type-bars are identical height at every k - this script validates that
against a known ground truth and renders it two ways: a two-panel bar chart
(PNG) and a terminal ASCII dot-art matrix.

Read-only reporting script: never writes to datasets/data/.
"""

import ast
import sys
from collections import Counter
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd

# Constants - use project root directory (no argparse, per project convention)
PROJECT_ROOT = Path(__file__).parent.parent.parent
TRAIN_CSV_PATH = PROJECT_ROOT / "datasets" / "data" / "tangled_ccs_dataset_train.csv"
TEST_CSV_PATH = PROJECT_ROOT / "datasets" / "data" / "tangled_ccs_dataset_test.csv"
OUTPUT_PNG_PATH = (
    PROJECT_ROOT / "results" / "analysis" / "repo_dataset_validation" / "tangled_type_distribution.png"
)

# Resolve cross-tree import of shared plot styling constants
sys.path.insert(0, str(PROJECT_ROOT / "RQ" / "analysis"))
from plot_utils import COLORS, PLOT_STYLE, setup_plot_style  # noqa: E402

# Canonical CCS type order, matches TYPES in generate_repo_tangled.py
TYPES = ["feat", "fix", "refactor", "test", "docs", "build", "ci"]
CONCERN_COUNTS = [1, 2, 3, 4, 5]

# Ground truth: exact per-(k, type) label counts the repo-grouped pipeline is
# designed to produce (adaptive under-represented-type-first sampling forces
# uniform 1/7 label frequency per split, independently at every concern
# count). Used as a correctness guard - FATAL if a re-run of the pipeline
# ever produced a different distribution than this script assumes.
TRAIN_EXPECTED_PER_TYPE_PER_K = {1: 40, 2: 80, 3: 120, 4: 160, 5: 200}
TEST_EXPECTED_PER_TYPE_PER_K = {1: 10, 2: 20, 3: 30, 4: 40, 5: 50}
TRAIN_EXPECTED_TOTAL_PER_TYPE = sum(TRAIN_EXPECTED_PER_TYPE_PER_K.values())  # 600
TEST_EXPECTED_TOTAL_PER_TYPE = sum(TEST_EXPECTED_PER_TYPE_PER_K.values())  # 150

# Fixed categorical color per type - a validated 7-slot CVD-safe order (dataviz
# skill default palette, adjacent-pair checks passed for grouped bars). Colors
# are assigned by type identity, never by rank/count, so they stay stable
# across panels and future re-runs even if the underlying counts change.
TYPE_COLORS = {
    "feat": "#2a78d6",
    "fix": "#eb6834",
    "refactor": "#1baf7a",
    "test": "#eda100",
    "docs": "#e87ba4",
    "build": "#008300",
    "ci": "#4a3aa7",
}

# ASCII dot-art scale: 1 dot per N labels, chosen so the widest bar (k=5)
# renders as a readable ~20 dots without wrapping typical terminal widths.
TRAIN_DOTS_PER_UNIT = 10
TEST_DOTS_PER_UNIT = 5

CHART_DPI = 200


def load_label_counts(csv_path: Path) -> Dict[int, Counter]:
    """Load one split CSV and tally label counts per (concern_count, type).

    Validates row-wise that concern_count == len(parsed types); exits loudly
    on any mismatch rather than silently mis-tallying.
    """
    df = pd.read_csv(csv_path)
    counts: Dict[int, Counter] = {k: Counter() for k in CONCERN_COUNTS}

    for _, row in df.iterrows():
        k = int(row["concern_count"])
        parsed_types = ast.literal_eval(row["types"])

        if len(parsed_types) != k:
            print(
                f"FATAL: type/count mismatch in {csv_path.name} "
                f"(shas={row['shas']!r}): parsed {len(parsed_types)} types "
                f"{parsed_types} but concern_count={k}"
            )
            sys.exit(1)

        if k not in counts:
            print(
                f"FATAL: unexpected concern_count={k} in {csv_path.name} "
                f"(expected one of {CONCERN_COUNTS})"
            )
            sys.exit(1)

        counts[k].update(parsed_types)

    return counts


def validate_counts(
    counts: Dict[int, Counter], expected_per_k: Dict[int, int], split_name: str
) -> None:
    """Assert exact uniform 1/7 label frequency per concern count against the
    known ground truth. A mismatch means the pipeline's output no longer
    matches this script's assumptions (different seed, changed sampling
    logic, or a stale CSV) - fail loudly rather than plot a wrong chart.
    """
    for k in CONCERN_COUNTS:
        expected = expected_per_k[k]
        for t in TYPES:
            actual = counts[k][t]
            if actual != expected:
                print(
                    f"FATAL: {split_name} k={k} type='{t}': got {actual}, "
                    f"expected {expected}"
                )
                sys.exit(1)

    print(f"{split_name}: all concern-count x type label counts match expected ground truth.")


def plot_distribution(
    train_counts: Dict[int, Counter], test_counts: Dict[int, Counter], output_path: Path
) -> Path:
    """Two-panel grouped bar chart (train, test); x-axis = concern count k,
    one bar per type at each k. Counts are exactly uniform across types at
    every k by design, so the visual point is that all 7 bars are identical
    height at every k.
    """
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    n_types = len(TYPES)
    bar_width = 0.8 / n_types
    x_positions = list(range(len(CONCERN_COUNTS)))

    for ax, counts, split_name in zip(axes, [train_counts, test_counts], ["Train", "Test"]):
        for i, t in enumerate(TYPES):
            offsets = [xi + (i - n_types / 2 + 0.5) * bar_width for xi in x_positions]
            values = [counts[k][t] for k in CONCERN_COUNTS]
            ax.bar(
                offsets,
                values,
                width=bar_width,
                color=TYPE_COLORS[t],
                label=t,
                edgecolor=COLORS["text"],
                linewidth=0.5,
            )

        ax.set_xticks(x_positions)
        ax.set_xticklabels([f"k={k}" for k in CONCERN_COUNTS])
        ax.set_xlabel("Concern Count", fontweight="bold", color=COLORS["text"])
        ax.set_ylabel("Label Count", fontweight="bold", color=COLORS["text"])
        ax.set_title(split_name, fontweight="bold")

        ax.grid(True, axis="y", alpha=PLOT_STYLE["grid_alpha"], linestyle="-", linewidth=0.5)
        ax.grid(False, axis="x")
        ax.set_axisbelow(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=n_types, bbox_to_anchor=(0.5, 1.04), framealpha=0.9
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=CHART_DPI, bbox_inches="tight", facecolor="white")
    plt.close()

    return output_path


def print_ascii_dot_matrix(counts: Dict[int, Counter], dots_per_unit: int, split_name: str) -> None:
    """Print an ASCII dot-art matrix: rows = types, columns = k=1..5, each
    cell showing the count and a proportional dot bar (scale: 1 dot per
    dots_per_unit labels) - a terminal-native sanity check for the exact
    uniformity the pipeline is supposed to guarantee.
    """
    print(f"\n{split_name}")
    print("-" * 70)
    for t in TYPES:
        cells = []
        for k in CONCERN_COUNTS:
            count = counts[k][t]
            n_dots = round(count / dots_per_unit)
            dots = "●" * n_dots
            cells.append(f"k{k} {dots} {count}")
        print(f"{t:<10}| " + " | ".join(cells))


def main() -> None:
    print("Loading and validating train data...")
    train_counts = load_label_counts(TRAIN_CSV_PATH)

    print("Loading and validating test data...")
    test_counts = load_label_counts(TEST_CSV_PATH)

    validate_counts(train_counts, TRAIN_EXPECTED_PER_TYPE_PER_K, "TRAIN")
    validate_counts(test_counts, TEST_EXPECTED_PER_TYPE_PER_K, "TEST")

    print("\nGenerating tangled-dataset type-distribution chart...")
    plot_path = plot_distribution(train_counts, test_counts, OUTPUT_PNG_PATH)
    print(f"Chart saved to: {plot_path}")

    print(
        f"\nASCII dot-art matrix (1 dot per {TRAIN_DOTS_PER_UNIT} train labels / "
        f"{TEST_DOTS_PER_UNIT} test labels):"
    )
    print_ascii_dot_matrix(train_counts, TRAIN_DOTS_PER_UNIT, "TRAIN")
    print_ascii_dot_matrix(test_counts, TEST_DOTS_PER_UNIT, "TEST")


if __name__ == "__main__":
    main()
