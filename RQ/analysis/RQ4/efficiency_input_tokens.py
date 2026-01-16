#!/usr/bin/env python3
"""
Efficiency Analysis: Correlation between Input Tokens and Inference Time

Analyzes the relationship between context length (input tokens) and inference time
using Pearson correlation.
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
    """Create box plot showing inference time distribution with Pearson correlation stats."""
    setup_plot_style()

    fig, ax = plt.subplots(figsize=PLOT_STYLE["figure_size"])

    context_lengths = sorted(df["context_len"].unique())
    box_data = [
        df[df["context_len"] == cl]["inference_time"].values for cl in context_lengths
    ]

    ax.boxplot(box_data, positions=context_lengths, **boxplot_style(widths=500))

    ax.set_xlabel("Input Token Length", fontweight="bold", color=COLORS["text"])
    ax.set_ylabel("Inference Time (seconds)", fontweight="bold", color=COLORS["text"])
    ax.set_xticks(context_lengths)
    ax.set_xticklabels([str(int(cl)) for cl in context_lengths])

    correlation, p_value = pearson_correlation(df, "context_len")
    add_correlation_text(ax, correlation, p_value, len(df))
    add_boxplot_legend(ax, loc="lower right")

    plt.tight_layout()

    output_path = output_dir / "boxplot_input_tokens_inference_time.png"
    plt.savefig(
        output_path, dpi=PLOT_STYLE["dpi"], bbox_inches="tight", facecolor="white"
    )
    plt.close()

    return output_path


def main() -> None:
    """Run input tokens vs inference time correlation analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze input tokens vs inference time correlation from CSV files"
    )
    parser.add_argument("csv_files", nargs="+", help="CSV files to analyze")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else ANALYSIS_OUTPUT_DIR / "ef_input_tokens"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Starting efficiency analysis: Input Tokens vs Inference Time")
    print("=" * 60)
    print(f"Processing {len(args.csv_files)} CSV file(s)...")

    csv_paths = [Path(f) for f in args.csv_files]
    df, outlier_info = load_csv(csv_paths, ["context_len", "inference_time"])
    print(f"Loaded {len(df)} data points from CSV files")

    correlation, p_value = pearson_correlation(df, "context_len")
    significant = p_value < P_VALUE_THRESHOLD

    print("\nCorrelation Analysis Results:")
    print(f"- Sample size: {len(df)}")
    print(f"- Outliers detected: {outlier_info['count']} (shown in boxplot)")
    print(f"- Pearson correlation: r = {correlation:.4f}")
    print(f"- P-value: {p_value:.6f}")
    print(f"- Significant: {'Yes' if significant else 'No'}")

    print("\nGenerating visualization...")
    boxplot_path = create_boxplot(df, output_dir)

    json_path = save_json(
        output_path=output_dir / "efficiency_input_tokens_summary.json",
        analysis_type="efficiency_input_tokens_correlation",
        input_files=args.csv_files,
        correlation=correlation,
        p_value=p_value,
        outlier_info=outlier_info,
        inference_time_stats={
            "mean": df["inference_time"].mean(),
            "std": df["inference_time"].std(),
            "min": df["inference_time"].min(),
            "max": df["inference_time"].max(),
        },
        sample_size=len(df),
    )

    print(f"\nResults saved to: {output_dir}")
    print(f"- Box plot: {boxplot_path.name}")
    print(f"- Summary JSON: {json_path.name}")


if __name__ == "__main__":
    main()
