#!/usr/bin/env python3
"""
Efficiency Analysis: Correlation between Input Tokens and Inference Time
Analyzes the relationship between context length (input tokens) and inference time using Pearson correlation.
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
    """Create box plot showing inference time distribution with Pearson correlation stats."""
    setup_plot_style()

    fig, ax = plt.subplots(figsize=PLOT_STYLE["figure_size"])

    context_lengths = sorted(df["context_len"].unique())
    box_data = [
        df[df["context_len"] == cl]["inference_time"].values for cl in context_lengths
    ]

    # Create box plot with unified colors (widths=500 for token-scale x-axis)
    ax.boxplot(box_data, positions=context_lengths, **boxplot_style(widths=500))

    ax.set_xlabel("Input Token Length", fontweight="bold", color=COLORS["text"])
    ax.set_ylabel("Inference Time (seconds)", fontweight="bold", color=COLORS["text"])
    ax.set_title(
        "Inference Time: Distribution & Correlation with Input Tokens",
        fontweight="bold",
        color=COLORS["text"],
        pad=20,
    )

    # Set x-axis ticks and labels
    ax.set_xticks(context_lengths)
    ax.set_xticklabels([str(int(cl)) for cl in context_lengths])

    # Add correlation statistics with sample size
    correlation, p_value = pearson_correlation(df, "context_len")
    add_correlation_text(ax, correlation, p_value, len(df))

    # Add legend for box plot components
    add_boxplot_legend(ax, loc="lower right")

    plt.tight_layout()

    output_path = output_dir / "boxplot_input_tokens_inference_time.png"
    plt.savefig(
        output_path, dpi=PLOT_STYLE["dpi"], bbox_inches="tight", facecolor="white"
    )
    plt.close()

    return output_path


def generate_summary_stats(df: pd.DataFrame, outlier_info: dict) -> dict:
    """Generate summary statistics for the analysis."""
    correlation, p_value = pearson_correlation(df, "context_len")

    stats_dict = {
        "sample_size": len(df),
        "correlation": correlation,
        "p_value": p_value,
        "significant": p_value < P_VALUE_THRESHOLD,
        "inference_time_mean": df["inference_time"].mean(),
        "inference_time_std": df["inference_time"].std(),
        "inference_time_min": df["inference_time"].min(),
        "inference_time_max": df["inference_time"].max(),
        "outliers_detected": outlier_info["count"],
    }

    return stats_dict


def main():
    parser = argparse.ArgumentParser(
        description="Analyze input tokens vs inference time correlation from CSV files"
    )
    parser.add_argument("csv_files", nargs="+", help="CSV files to analyze")
    parser.add_argument("--output-dir", type=str, help="Output directory")

    args = parser.parse_args()

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = ANALYSIS_OUTPUT_DIR / "ef_input_tokens"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Starting efficiency analysis: Input Tokens vs Inference Time")
    print("=" * 60)

    # Process CSV files
    print(f"Processing {len(args.csv_files)} CSV file(s)...")
    csv_paths = [Path(f) for f in args.csv_files]

    try:
        # Load data using shared utility
        df, outlier_info = load_csv(
            csv_paths, ["context_len", "inference_time"]
        )
        print(f"Loaded {len(df)} data points from CSV files")

        # Generate statistics
        stats = generate_summary_stats(df, outlier_info)

        # Print basic results
        print(f"\nCorrelation Analysis Results:")
        print(f"- Sample size: {stats['sample_size']}")
        print(f"- Outliers detected: {stats['outliers_detected']} (shown in boxplot)")
        print(f"- Pearson correlation: r = {stats['correlation']:.4f}")
        print(f"- P-value: {stats['p_value']:.6f}")
        print(f"- Significant: {'Yes' if stats['significant'] else 'No'}")

        # Generate visualization
        print(f"\nGenerating visualization...")
        boxplot_path = create_boxplot(df, output_dir)

        # Save results using shared utility
        json_path = save_json(
            output_path=output_dir / "efficiency_input_tokens_summary.json",
            analysis_type="efficiency_input_tokens_correlation",
            input_files=args.csv_files,
            correlation=stats["correlation"],
            p_value=stats["p_value"],
            outlier_info=outlier_info,
            inference_time_stats={
                "mean": stats["inference_time_mean"],
                "std": stats["inference_time_std"],
                "min": stats["inference_time_min"],
                "max": stats["inference_time_max"],
            },
            sample_size=stats["sample_size"],
        )

        print(f"\nResults saved to: {output_dir}")
        print(f"- Box plot: {boxplot_path.name}")
        print(f"- Summary JSON: {json_path.name}")

    except Exception as e:
        print(f"Error processing CSV files: {e}")
        raise


if __name__ == "__main__":
    main()
