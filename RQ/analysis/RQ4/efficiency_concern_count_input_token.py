#!/usr/bin/env python3
"""
Efficiency Analysis: Concern Count and Input Tokens vs Inference Time
Analyzes correlations between concern count, input tokens, and inference time.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List
import argparse

from ..plot_utils import COLORS, GROUP_COLORS, PLOT_STYLE, boxplot_style, setup_plot_style
from .utils import load_csv, save_json
from . import PROJECT_ROOT, ANALYSIS_OUTPUT_DIR


def create_boxplot_by_concern_grouped_by_token(
    df: pd.DataFrame, output_dir: Path
) -> Path:
    """Create grouped box plot showing inference time distribution by concern count, grouped by token length.

    X-axis: Concern count (1-5)
    Groups: Token lengths (5 groups)
    """
    setup_plot_style()

    # Get unique values and sort them
    unique_tokens = sorted(df["context_len"].unique())
    unique_concerns = sorted(df["concern_count"].unique())

    # Create figure
    fig, ax = plt.subplots(figsize=PLOT_STYLE["figure_size"])

    # Use GROUP_COLORS from plot_utils for token length categories
    token_colors = GROUP_COLORS[: len(unique_tokens)]

    # Calculate positions for grouped box plots
    width = 0.15  # Width of each box
    x_positions = np.arange(len(unique_concerns))

    # Create box plots for each token length
    for i, token_len in enumerate(unique_tokens):
        token_data = []
        positions = []

        for j, concern_count in enumerate(unique_concerns):
            # Get data for this specific combination
            subset = df[
                (df["context_len"] == token_len)
                & (df["concern_count"] == concern_count)
            ]
            if len(subset) > 0:
                token_data.append(subset["inference_time"].values)
                positions.append(x_positions[j] + i * width)

        # Create box plot for this token length
        if token_data and positions:
            group_color = token_colors[i % len(token_colors)]
            ax.boxplot(
                token_data,
                positions=positions,
                **boxplot_style(box_color=group_color, widths=width * 0.8),
            )

    # Set labels
    ax.set_xlabel("Concern Count", fontweight="bold", color=COLORS["text"])
    ax.set_ylabel("Inference Time (seconds)", fontweight="bold", color=COLORS["text"])

    # Set x-axis ticks and labels
    ax.set_xticks(x_positions + width * (len(unique_tokens) - 1) / 2)
    ax.set_xticklabels(unique_concerns)

    # Add legend for token lengths
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(
            facecolor=token_colors[i % len(token_colors)],
            alpha=PLOT_STYLE["alpha"],
            label=f"{int(token_len)} tokens",
        )
        for i, token_len in enumerate(unique_tokens)
    ]
    ax.legend(handles=legend_elements, loc="upper left", framealpha=0.9, fontsize=14)

    # Add sample size info
    stats_text = f"n = {len(df)}"

    ax.text(
        0.98,
        0.02,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor=COLORS["background"],
            alpha=0.9,
            edgecolor=COLORS["primary"],
            linewidth=1,
        ),
    )

    # Clean grid styling
    ax.grid(
        True, alpha=PLOT_STYLE["grid_alpha"], linestyle="-", linewidth=0.5, axis="y"
    )
    ax.set_axisbelow(True)

    plt.tight_layout()

    output_path = output_dir / "boxplot_concern_by_token.png"
    plt.savefig(
        output_path, dpi=PLOT_STYLE["dpi"], bbox_inches="tight", facecolor="white"
    )
    plt.close()

    return output_path


def create_boxplot_by_token_grouped_by_concern(
    df: pd.DataFrame, output_dir: Path
) -> Path:
    """Create grouped box plot showing inference time distribution by token length, grouped by concern count.

    X-axis: Token lengths (5 categories)
    Groups: Concern counts (5 groups)
    """
    setup_plot_style()

    # Get unique values and sort them
    unique_tokens = sorted(df["context_len"].unique())
    unique_concerns = sorted(df["concern_count"].unique())

    # Create figure
    fig, ax = plt.subplots(figsize=PLOT_STYLE["figure_size"])

    # Use GROUP_COLORS from plot_utils for concern count categories
    concern_colors = GROUP_COLORS[: len(unique_concerns)]

    # Calculate positions for grouped box plots
    width = 0.15  # Width of each box
    x_positions = np.arange(len(unique_tokens))

    # Create box plots for each concern count
    for i, concern_count in enumerate(unique_concerns):
        concern_data = []
        positions = []

        for j, token_len in enumerate(unique_tokens):
            # Get data for this specific combination
            subset = df[
                (df["concern_count"] == concern_count)
                & (df["context_len"] == token_len)
            ]
            if len(subset) > 0:
                concern_data.append(subset["inference_time"].values)
                positions.append(x_positions[j] + i * width)

        # Create box plot for this concern count
        if concern_data and positions:
            group_color = concern_colors[i % len(concern_colors)]
            ax.boxplot(
                concern_data,
                positions=positions,
                **boxplot_style(box_color=group_color, widths=width * 0.8),
            )

    # Set labels
    ax.set_xlabel("Input Token Length", fontweight="bold", color=COLORS["text"])
    ax.set_ylabel("Inference Time (seconds)", fontweight="bold", color=COLORS["text"])

    # Set x-axis ticks and labels
    ax.set_xticks(x_positions + width * (len(unique_concerns) - 1) / 2)
    ax.set_xticklabels([str(int(t)) for t in unique_tokens])

    # Add legend for concern counts
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(
            facecolor=concern_colors[i % len(concern_colors)],
            alpha=PLOT_STYLE["alpha"],
            label=f"{int(concern_count)} concerns",
        )
        for i, concern_count in enumerate(unique_concerns)
    ]
    ax.legend(handles=legend_elements, loc="upper left", framealpha=0.9, fontsize=14)

    # Add sample size info
    stats_text = f"n = {len(df)}"

    ax.text(
        0.98,
        0.02,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor=COLORS["background"],
            alpha=0.9,
            edgecolor=COLORS["primary"],
            linewidth=1,
        ),
    )

    # Clean grid styling
    ax.grid(
        True, alpha=PLOT_STYLE["grid_alpha"], linestyle="-", linewidth=0.5, axis="y"
    )
    ax.set_axisbelow(True)

    plt.tight_layout()

    output_path = output_dir / "boxplot_token_by_concern.png"
    plt.savefig(
        output_path, dpi=PLOT_STYLE["dpi"], bbox_inches="tight", facecolor="white"
    )
    plt.close()

    return output_path


def generate_summary_stats(df: pd.DataFrame) -> dict:
    """Generate summary statistics for the analysis."""

    # Calculate correlations using df.corr()
    corr_df = df[["context_len", "inference_time", "concern_count"]].corr()

    correlations = {
        "tokens_time": float(corr_df.loc["context_len", "inference_time"]),
        "concern_time": float(corr_df.loc["concern_count", "inference_time"]),
        "tokens_concern": float(corr_df.loc["context_len", "concern_count"]),
    }

    # Inference time statistics
    inference_time_stats = {
        "mean": float(df["inference_time"].mean()),
        "std": float(df["inference_time"].std()),
        "min": float(df["inference_time"].min()),
        "max": float(df["inference_time"].max()),
    }

    return {
        "sample_size": len(df),
        "correlations": correlations,
        "inference_time": inference_time_stats,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Correlation analysis: concern count, input tokens vs inference time"
    )
    parser.add_argument("csv_files", nargs="+", help="CSV files to analyze")
    parser.add_argument("--output-dir", type=str, help="Output directory")

    args = parser.parse_args()

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = ANALYSIS_OUTPUT_DIR / "ef_concern_count_input_tokens"

    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        "Starting efficiency analysis: Concern Count + Input Tokens vs Inference Time"
    )
    print("=" * 80)

    # Process CSV files
    print(f"Processing {len(args.csv_files)} CSV file(s)...")
    csv_paths = [Path(f) for f in args.csv_files]

    try:
        # Load data using utility function
        required_cols = ["context_len", "inference_time", "concern_count"]
        df, outlier_info = load_csv(csv_paths, required_cols)
        print(f"Loaded {len(df)} data points from CSV files")

        # Generate statistics
        stats = generate_summary_stats(df)

        # Print results
        print(f"\nAnalysis Results:")
        print(f"- Sample size: {stats['sample_size']}")

        print(f"\nCorrelations (Pearson r):")
        print(f"- tokens <-> time: {stats['correlations']['tokens_time']:.3f}")
        print(f"- concern <-> time: {stats['correlations']['concern_time']:.3f}")
        print(f"- tokens <-> concern: {stats['correlations']['tokens_concern']:.3f}")

        # Generate visualizations
        print(f"\nGenerating visualizations...")
        concern_boxplot_path = create_boxplot_by_concern_grouped_by_token(df, output_dir)
        token_boxplot_path = create_boxplot_by_token_grouped_by_concern(df, output_dir)

        # Save results using utility function with extra_data for multi-correlation
        output_path = output_dir / "efficiency_concern_count_input_token_summary.json"
        extra_data = {
            "multi_correlation_analysis": {
                "tokens_time": round(stats["correlations"]["tokens_time"], 4),
                "concern_time": round(stats["correlations"]["concern_time"], 4),
                "tokens_concern": round(stats["correlations"]["tokens_concern"], 4),
            }
        }
        json_path = save_json(
            output_path=output_path,
            analysis_type="efficiency_concern_count_input_token_analysis",
            input_files=args.csv_files,
            correlation=stats["correlations"]["tokens_time"],  # Primary correlation
            p_value=0.0,  # Not computed for df.corr() matrix
            outlier_info=outlier_info,
            inference_time_stats=stats["inference_time"],
            sample_size=stats["sample_size"],
            extra_data=extra_data,
        )

        print(f"\nResults saved to: {output_dir}")
        print(f"- Box plot (concern by token): {concern_boxplot_path.name}")
        print(f"- Box plot (token by concern): {token_boxplot_path.name}")
        print(f"- Summary JSON: {json_path.name}")

    except Exception as e:
        print(f"Error processing CSV files: {e}")
        raise


if __name__ == "__main__":
    main()
