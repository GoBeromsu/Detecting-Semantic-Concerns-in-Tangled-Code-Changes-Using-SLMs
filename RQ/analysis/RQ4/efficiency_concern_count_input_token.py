#!/usr/bin/env python3
"""
Efficiency Analysis: Concern Count and Input Tokens vs Inference Time
Analyzes correlations between concern count, input tokens, and inference time.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from ..plot_utils import COLORS, GROUP_COLORS, PLOT_STYLE, boxplot_style, setup_plot_style
from . import ANALYSIS_OUTPUT_DIR
from .utils import load_csv, save_json

BOX_WIDTH = 0.15


def create_grouped_boxplot(
    df: pd.DataFrame,
    output_dir: Path,
    x_col: str,
    group_col: str,
    x_label: str,
    group_label_template: str,
    output_filename: str,
) -> Path:
    """Create grouped box plot showing inference time distribution.

    Args:
        df: DataFrame with inference_time, x_col, and group_col columns.
        output_dir: Directory to save the plot.
        x_col: Column for x-axis categories.
        group_col: Column for grouping within each x-axis category.
        x_label: Label for x-axis.
        group_label_template: Template for legend labels (e.g., "{} tokens").
        output_filename: Name of output file.

    Returns:
        Path to saved plot.
    """
    setup_plot_style()

    unique_x = sorted(df[x_col].unique())
    unique_groups = sorted(df[group_col].unique())
    colors = GROUP_COLORS[: len(unique_groups)]
    x_positions = np.arange(len(unique_x))

    fig, ax = plt.subplots(figsize=PLOT_STYLE["figure_size"])

    for i, group_val in enumerate(unique_groups):
        box_data = []
        positions = []

        for j, x_val in enumerate(unique_x):
            subset = df[(df[group_col] == group_val) & (df[x_col] == x_val)]
            if len(subset) > 0:
                box_data.append(subset["inference_time"].values)
                positions.append(x_positions[j] + i * BOX_WIDTH)

        if box_data:
            ax.boxplot(
                box_data,
                positions=positions,
                **boxplot_style(box_color=colors[i], widths=BOX_WIDTH * 0.8),
            )

    ax.set_xlabel(x_label, fontweight="bold", color=COLORS["text"])
    ax.set_ylabel("Inference Time (seconds)", fontweight="bold", color=COLORS["text"])
    ax.set_xticks(x_positions + BOX_WIDTH * (len(unique_groups) - 1) / 2)
    ax.set_xticklabels([str(int(v)) for v in unique_x])

    legend_elements = [
        Patch(
            facecolor=colors[i],
            alpha=PLOT_STYLE["alpha"],
            label=group_label_template.format(int(group_val)),
        )
        for i, group_val in enumerate(unique_groups)
    ]
    ax.legend(handles=legend_elements, loc="upper left", framealpha=0.9, fontsize=14)

    ax.grid(True, alpha=PLOT_STYLE["grid_alpha"], linestyle="-", linewidth=0.5, axis="y")
    ax.set_axisbelow(True)

    plt.tight_layout()

    output_path = output_dir / output_filename
    plt.savefig(output_path, dpi=PLOT_STYLE["dpi"], bbox_inches="tight", facecolor="white")
    plt.close()

    return output_path


def create_boxplot_by_concern_grouped_by_token(df: pd.DataFrame, output_dir: Path) -> Path:
    """Create grouped box plot: x-axis = concern count, groups = token lengths."""
    return create_grouped_boxplot(
        df=df,
        output_dir=output_dir,
        x_col="concern_count",
        group_col="context_len",
        x_label="Concern Count",
        group_label_template="{} tokens",
        output_filename="boxplot_concern_by_token.png",
    )


def create_boxplot_by_token_grouped_by_concern(df: pd.DataFrame, output_dir: Path) -> Path:
    """Create grouped box plot: x-axis = token length, groups = concern counts."""
    return create_grouped_boxplot(
        df=df,
        output_dir=output_dir,
        x_col="context_len",
        group_col="concern_count",
        x_label="Input Token Length",
        group_label_template="{} concerns",
        output_filename="boxplot_token_by_concern.png",
    )


def generate_summary_stats(df: pd.DataFrame) -> dict:
    """Generate summary statistics for the analysis."""
    cols = ["context_len", "inference_time", "concern_count"]
    corr_df = df[cols].corr()

    return {
        "sample_size": len(df),
        "correlations": {
            "tokens_time": float(corr_df.loc["context_len", "inference_time"]),
            "concern_time": float(corr_df.loc["concern_count", "inference_time"]),
            "tokens_concern": float(corr_df.loc["context_len", "concern_count"]),
        },
        "inference_time": {
            "mean": float(df["inference_time"].mean()),
            "std": float(df["inference_time"].std()),
            "min": float(df["inference_time"].min()),
            "max": float(df["inference_time"].max()),
        },
    }


def print_results(stats: dict) -> None:
    """Print analysis results to console."""
    print(f"\nAnalysis Results:")
    print(f"- Sample size: {stats['sample_size']}")
    print(f"\nCorrelations (Pearson r):")
    print(f"- tokens <-> time: {stats['correlations']['tokens_time']:.3f}")
    print(f"- concern <-> time: {stats['correlations']['concern_time']:.3f}")
    print(f"- tokens <-> concern: {stats['correlations']['tokens_concern']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Correlation analysis: concern count, input tokens vs inference time"
    )
    parser.add_argument("csv_files", nargs="+", help="CSV files to analyze")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else ANALYSIS_OUTPUT_DIR / "ef_concern_count_input_tokens"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Starting efficiency analysis: Concern Count + Input Tokens vs Inference Time")
    print("=" * 80)
    print(f"Processing {len(args.csv_files)} CSV file(s)...")

    csv_paths = [Path(f) for f in args.csv_files]
    required_cols = ["context_len", "inference_time", "concern_count"]
    df, outlier_info = load_csv(csv_paths, required_cols)
    print(f"Loaded {len(df)} data points from CSV files")

    stats = generate_summary_stats(df)
    print_results(stats)

    print(f"\nGenerating visualizations...")
    concern_boxplot_path = create_boxplot_by_concern_grouped_by_token(df, output_dir)
    token_boxplot_path = create_boxplot_by_token_grouped_by_concern(df, output_dir)

    correlations = stats["correlations"]
    json_path = save_json(
        output_path=output_dir / "efficiency_concern_count_input_token_summary.json",
        analysis_type="efficiency_concern_count_input_token_analysis",
        input_files=args.csv_files,
        correlation=correlations["tokens_time"],
        p_value=0.0,
        outlier_info=outlier_info,
        inference_time_stats=stats["inference_time"],
        sample_size=stats["sample_size"],
        extra_data={
            "multi_correlation_analysis": {
                key: round(val, 4) for key, val in correlations.items()
            }
        },
    )

    print(f"\nResults saved to: {output_dir}")
    print(f"- Box plot (concern by token): {concern_boxplot_path.name}")
    print(f"- Box plot (token by concern): {token_boxplot_path.name}")
    print(f"- Summary JSON: {json_path.name}")


if __name__ == "__main__":
    main()
