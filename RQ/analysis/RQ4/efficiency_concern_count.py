#!/usr/bin/env python3
"""
Efficiency Analysis: Correlation between Concern Count and Inference Time

Analyzes the relationship between concern count and inference time using Pearson correlation.
Generates box plot visualization showing inference time distribution by concern count.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..plot_utils import COLORS, PLOT_STYLE, boxplot_style, setup_plot_style
from . import ANALYSIS_OUTPUT_DIR
from .utils import (
    P_VALUE_THRESHOLD,
    add_boxplot_legend,
    add_correlation_text,
    load_csv,
    pearson_correlation,
    save_json,
)


def create_boxplot(df: pd.DataFrame, output_dir: Path) -> Path:
    """Create box plot showing inference time distribution and correlation."""
    setup_plot_style()

    fig, ax = plt.subplots(figsize=PLOT_STYLE["figure_size"])

    concern_counts = sorted(df["concern_count"].unique())
    box_data = [
        df[df["concern_count"] == cc]["inference_time"].values for cc in concern_counts
    ]

    # Create box plot with unified colors
    ax.boxplot(box_data, labels=concern_counts, **boxplot_style())

    ax.set_xlabel("Concern Count", fontweight="bold", color=COLORS["text"])
    ax.set_ylabel("Inference Time (seconds)", fontweight="bold", color=COLORS["text"])

    # Add correlation statistics with sample size
    correlation, p_value = pearson_correlation(df, "concern_count")
    add_correlation_text(ax, correlation, p_value, len(df))

    # Add legend for box plot components
    add_boxplot_legend(ax, loc="lower right")

    plt.tight_layout()

    output_path = output_dir / "boxplot_concern_count_inference_time.png"
    plt.savefig(
        output_path, dpi=PLOT_STYLE["dpi"], bbox_inches="tight", facecolor="white"
    )
    plt.close()

    return output_path


def compute_inference_time_stats(df: pd.DataFrame) -> dict:
    """Compute descriptive statistics for inference time."""
    return {
        "mean": df["inference_time"].mean(),
        "std": df["inference_time"].std(),
        "min": df["inference_time"].min(),
        "max": df["inference_time"].max(),
    }


def print_correlation_results(
    sample_size: int, outlier_count: int, correlation: float, p_value: float
) -> None:
    """Print correlation analysis results to console."""
    significant = p_value < P_VALUE_THRESHOLD
    print("\nCorrelation Analysis Results:")
    print(f"- Sample size: {sample_size}")
    print(f"- Outliers detected: {outlier_count} (shown in boxplot)")
    print(f"- Pearson correlation: r = {correlation:.4f}")
    print(f"- P-value: {p_value:.6f}")
    print(f"- Significant: {'Yes' if significant else 'No'}")


def main() -> None:
    """Main entry point for concern count vs inference time analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze concern count vs inference time correlation from CSV files"
    )
    parser.add_argument("csv_files", nargs="+", help="CSV files to analyze")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    args = parser.parse_args()

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else ANALYSIS_OUTPUT_DIR / "ef_concern_count"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Starting efficiency analysis: Concern Count vs Inference Time")
    print("=" * 60)
    print(f"Processing {len(args.csv_files)} CSV file(s)...")

    csv_paths = [Path(f) for f in args.csv_files]
    df, outlier_info = load_csv(csv_paths, ["concern_count", "inference_time"])
    print(f"Loaded {len(df)} data points from CSV files")

    correlation, p_value = pearson_correlation(df, "concern_count")
    print_correlation_results(len(df), outlier_info["count"], correlation, p_value)

    print("\nGenerating visualization...")
    boxplot_path = create_boxplot(df, output_dir)

    json_path = save_json(
        output_path=output_dir / "efficiency_analysis_summary.json",
        analysis_type="efficiency_concern_count_correlation",
        input_files=args.csv_files,
        correlation=correlation,
        p_value=p_value,
        outlier_info=outlier_info,
        inference_time_stats=compute_inference_time_stats(df),
        sample_size=len(df),
    )

    print(f"\nResults saved to: {output_dir}")
    print(f"- Box plot: {boxplot_path.name}")
    print(f"- Summary JSON: {json_path.name}")


if __name__ == "__main__":
    main()
