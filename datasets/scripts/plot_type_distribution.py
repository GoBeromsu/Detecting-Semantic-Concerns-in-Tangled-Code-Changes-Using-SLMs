#!/usr/bin/env python3
"""
Concern-Type Distribution Bar Chart
Generates a single bar chart showing the combined (train + test) distribution
of CCS concern types across the tangled dataset.
"""

import argparse
import ast
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Constants - Use project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "datasets" / "data" / "tangled_ccs_dataset_train.csv"
DEFAULT_TEST_PATH = PROJECT_ROOT / "datasets" / "data" / "tangled_ccs_dataset_test.csv"
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "results" / "analysis" / "type_distribution" / "type_distribution.png"
)

# Resolve cross-tree import of shared plot styling constants
sys.path.insert(0, str(PROJECT_ROOT / "RQ" / "analysis"))
from plot_utils import COLORS, PLOT_STYLE, setup_plot_style  # noqa: E402

# Canonical CCS type order for the paper (near-uniform counts must not read as a ranking)
TYPE_ORDER = ["feat", "fix", "refactor", "docs", "test", "build", "ci"]

# Expected combined counts (train + test), used as a correctness guard.
# Repo-grouped pipeline (build_repo_pool.py -> generate_repo_tangled.py) drives
# per-type label counts to EXACT uniformity: 600/type train + 150/type test = 750/type.
EXPECTED_COUNTS = {
    "ci": 750,
    "test": 750,
    "build": 750,
    "docs": 750,
    "refactor": 750,
    "fix": 750,
    "feat": 750,
}
EXPECTED_TOTAL = 5250


def load_and_validate(csv_path: Path) -> Counter:
    """Load one CSV, parse `types`, and validate against `concern_count` row-wise."""
    df = pd.read_csv(csv_path)

    counter: Counter = Counter()
    for _, row in df.iterrows():
        parsed_types = ast.literal_eval(row["types"])
        if len(parsed_types) != row["concern_count"]:
            print(
                f"FATAL: type/count mismatch in {csv_path.name} "
                f"(shas={row['shas']!r}): parsed {len(parsed_types)} types "
                f"{parsed_types} but concern_count={row['concern_count']}"
            )
            sys.exit(1)
        counter.update(parsed_types)

    return counter


def create_type_distribution_barplot(counts: Counter, output_path: Path) -> Path:
    """Create a single-color bar chart of concern-type counts in canonical order."""
    setup_plot_style()

    values = [counts[t] for t in TYPE_ORDER]

    _, ax = plt.subplots(figsize=PLOT_STYLE["figure_size"])

    ax.bar(TYPE_ORDER, values, color=COLORS["primary"], width=0.6)

    ax.set_xlabel("Concern Type", fontweight="bold", color=COLORS["text"])
    ax.set_ylabel("Commit-Label Count", fontweight="bold", color=COLORS["text"])

    # Y-axis must start at 0 with generous headroom so bars read as comparable
    ax.set_ylim(0, 1000)

    # Horizontal gridlines only, light
    ax.grid(True, axis="y", alpha=PLOT_STYLE["grid_alpha"], linestyle="-", linewidth=0.5)
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=PLOT_STYLE["dpi"], bbox_inches="tight", facecolor="white")
    plt.close()

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot combined concern-type distribution for the tangled CCS dataset."
    )
    parser.add_argument(
        "--train-csv", type=Path, default=DEFAULT_TRAIN_PATH, help="Path to the train CSV."
    )
    parser.add_argument(
        "--test-csv", type=Path, default=DEFAULT_TEST_PATH, help="Path to the test CSV."
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Path to save the output PNG."
    )
    args = parser.parse_args()

    print("Loading and validating train data...")
    train_counts = load_and_validate(args.train_csv)

    print("Loading and validating test data...")
    test_counts = load_and_validate(args.test_csv)

    combined_counts = train_counts + test_counts

    total = sum(combined_counts.values())
    print("\nCombined concern-type counts (train + test):")
    print(f"{'Type':<10}{'Count':>8}")
    for t in TYPE_ORDER:
        print(f"{t:<10}{combined_counts[t]:>8}")
    print(f"{'TOTAL':<10}{total:>8}")

    for t, expected in EXPECTED_COUNTS.items():
        if combined_counts[t] != expected:
            print(
                f"FATAL: count mismatch for type '{t}': got {combined_counts[t]}, "
                f"expected {expected}"
            )
            sys.exit(1)
    if total != EXPECTED_TOTAL:
        print(f"FATAL: total count mismatch: got {total}, expected {EXPECTED_TOTAL}")
        sys.exit(1)

    print("\nAll counts match expected values.")

    print("\nGenerating concern-type distribution bar chart...")
    plot_path = create_type_distribution_barplot(combined_counts, args.output)
    print(f"Bar chart saved to: {plot_path}")


if __name__ == "__main__":
    main()
