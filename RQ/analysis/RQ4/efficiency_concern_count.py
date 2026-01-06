#!/usr/bin/env python3
"""
Efficiency Analysis: Correlation between Concern Count and Inference Time
Analyzes the relationship between concern count and inference time using Pearson correlation.
Processes raw CSV data for detailed box plot analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from ..plot_utils import COLORS, PLOT_STYLE, boxplot_style, setup_plot_style
from .utils import (
    load_csv,
    pearson_correlation,
    add_correlation_text,
    add_boxplot_legend,
    save_json,
    P_VALUE_THRESHOLD,
)
from . import PROJECT_ROOT, ANALYSIS_OUTPUT_DIR


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
    ax.set_title(
        "Inference Time: Distribution & Correlation with Concern Count",
        fontweight="bold",
        color=COLORS["text"],
        pad=20,
    )

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


def main():
    parser = argparse.ArgumentParser(
        description="Analyze concern count vs inference time correlation from CSV files"
    )
    parser.add_argument("csv_files", nargs="+", help="CSV files to analyze")
    parser.add_argument("--output-dir", type=str, help="Output directory")

    args = parser.parse_args()

    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else ANALYSIS_OUTPUT_DIR / "ef_concern_count"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Starting efficiency analysis: Concern Count vs Inference Time")
    print("=" * 60)

    # Process CSV files
    print(f"Processing {len(args.csv_files)} CSV file(s)...")
    csv_paths = [Path(f) for f in args.csv_files]

    try:
        # Load data using shared utility
        df, outlier_info = load_csv(csv_paths, ["concern_count", "inference_time"])
        print(f"Loaded {len(df)} data points from CSV files")

        # Calculate correlation
        correlation, p_value = pearson_correlation(df, "concern_count")
        significant = p_value < P_VALUE_THRESHOLD

        # Print results
        print(f"\nCorrelation Analysis Results:")
        print(f"- Sample size: {len(df)}")
        print(f"- Outliers detected: {outlier_info['count']} (shown in boxplot)")
        print(f"- Pearson correlation: r = {correlation:.4f}")
        print(f"- P-value: {p_value:.6f}")
        print(f"- Significant: {'Yes' if significant else 'No'}")

        # Generate visualization
        print(f"\nGenerating visualization...")
        boxplot_path = create_boxplot(df, output_dir)

        # Save results using shared utility
        inference_time_stats = {
            "mean": df["inference_time"].mean(),
            "std": df["inference_time"].std(),
            "min": df["inference_time"].min(),
            "max": df["inference_time"].max(),
        }
        json_path = save_json(
            output_path=output_dir / "efficiency_analysis_summary.json",
            analysis_type="efficiency_concern_count_correlation",
            input_files=args.csv_files,
            correlation=correlation,
            p_value=p_value,
            outlier_info=outlier_info,
            inference_time_stats=inference_time_stats,
            sample_size=len(df),
        )

        print(f"\nResults saved to: {output_dir}")
        print(f"- Box plot: {boxplot_path.name}")
        print(f"- Summary JSON: {json_path.name}")

    except Exception as e:
        print(f"Error processing CSV files: {e}")
        raise


if __name__ == "__main__":
    main()
