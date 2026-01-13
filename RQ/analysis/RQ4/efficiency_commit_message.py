#!/usr/bin/env python3
"""
Efficiency Analysis: Correlation between Commit Message and Inference Time
Analyzes the relationship between commit message presence and inference time using statistical tests.
Processes raw CSV data for detailed box plot analysis and group comparison.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any
import argparse

from ..plot_utils import COLORS, PLOT_STYLE, boxplot_style, setup_plot_style
from .utils import (
    load_csv,
    pearson_correlation,
    add_correlation_text,
    save_json,
    P_VALUE_THRESHOLD,
)
from . import PROJECT_ROOT, ANALYSIS_OUTPUT_DIR


def calculate_group_comparison(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate statistical comparison between with/without message groups."""
    with_msg = df[df["with_message"] == True]["inference_time"]
    without_msg = df[df["with_message"] == False]["inference_time"]

    # Basic statistics
    stats_dict = {
        "with_message": {
            "count": len(with_msg),
            "mean": float(with_msg.mean()),
        },
        "without_message": {
            "count": len(without_msg),
            "mean": float(without_msg.mean()),
        },
        "mean_difference": float(with_msg.mean() - without_msg.mean()),
    }

    return stats_dict


def create_boxplot(df: pd.DataFrame, output_dir: Path) -> Path:
    """Create box plot showing inference time distribution by commit message presence."""
    setup_plot_style()

    # Use consistent figure size
    fig, ax = plt.subplots(figsize=PLOT_STYLE["figure_size"])

    # Prepare data for box plot
    with_msg_data = df[df["with_message"] == True]["inference_time"]
    without_msg_data = df[df["with_message"] == False]["inference_time"]

    box_data = [without_msg_data, with_msg_data]
    labels = ["Without Message", "With Message"]
    box_colors = [COLORS["secondary"], COLORS["primary"]]

    # Create box plot with adjusted width and positions
    positions = [0, 1]
    box_plot = ax.boxplot(box_data, positions=positions, **boxplot_style(widths=0.4))

    # Override box and flier colors for each group
    for patch, flier, color in zip(box_plot["boxes"], box_plot["fliers"], box_colors):
        patch.set_facecolor(color)
        flier.set_markerfacecolor(color)

    # Set custom x-axis labels and ticks
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.5, 1.5)

    ax.set_xlabel("Commit Message Presence", fontweight="bold", color=COLORS["text"])
    ax.set_ylabel("Inference Time (seconds)", fontweight="bold", color=COLORS["text"])

    # Add statistical information using utility function
    # Convert boolean to int for correlation calculation
    df_numeric = df.assign(with_message_int=df["with_message"].astype(int))
    correlation, p_value = pearson_correlation(df_numeric, "with_message_int")
    add_correlation_text(ax, correlation, p_value, len(df))

    # Add legend for box plot components
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    legend_elements = [
        Patch(
            facecolor=COLORS["secondary"],
            alpha=PLOT_STYLE["alpha"],
            label="Without Message",
        ),
        Patch(
            facecolor=COLORS["primary"],
            alpha=PLOT_STYLE["alpha"],
            label="With Message",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["success"],
            linewidth=PLOT_STYLE["line_width"],
            label="Median",
        ),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.9)

    plt.tight_layout()

    output_path = output_dir / "boxplot_commit_message_inference_time.png"
    plt.savefig(
        output_path, dpi=PLOT_STYLE["dpi"], bbox_inches="tight", facecolor="white"
    )
    plt.close()

    return output_path


def generate_summary_stats(df: pd.DataFrame, outlier_info: dict) -> dict:
    """Generate summary statistics for the analysis."""
    group_stats = calculate_group_comparison(df)

    # Convert boolean to int for correlation calculation
    df_numeric = df.assign(with_message_int=df["with_message"].astype(int))
    correlation, p_value = pearson_correlation(df_numeric, "with_message_int")

    stats_dict = {
        "sample_size": len(df),
        "correlation": correlation,
        "p_value": p_value,
        "significant": p_value < P_VALUE_THRESHOLD,
        "outlier_info": outlier_info,
        "group_statistics": group_stats,
        "inference_time": {
            "mean": float(df["inference_time"].mean()),
            "std": float(df["inference_time"].std()),
            "min": float(df["inference_time"].min()),
            "max": float(df["inference_time"].max()),
        },
    }

    return stats_dict


def main():
    parser = argparse.ArgumentParser(
        description="Analyze commit message vs inference time correlation from CSV files"
    )
    parser.add_argument("csv_files", nargs="+", help="CSV files to analyze")
    parser.add_argument("--output-dir", type=str, help="Output directory")

    args = parser.parse_args()

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = ANALYSIS_OUTPUT_DIR / "ef_commit_message"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Starting efficiency analysis: Commit Message vs Inference Time")
    print("=" * 60)

    # Process CSV files
    print(f"Processing {len(args.csv_files)} CSV file(s)...")
    csv_paths = [Path(f) for f in args.csv_files]

    try:
        # Load data using utility function
        df, outlier_info = load_csv(
            csv_paths, ["with_message", "inference_time"]
        )
        print(f"Loaded {len(df)} data points from CSV files")

        # Generate statistics
        stats = generate_summary_stats(df, outlier_info)

        # Print basic results
        print(f"\nGroup Comparison Results:")
        print(f"- Sample size: {stats['sample_size']}")
        print(
            f"- With message: {stats['group_statistics']['with_message']['count']} samples"
        )
        print(
            f"- Without message: {stats['group_statistics']['without_message']['count']} samples"
        )
        print(f"- Pearson correlation: r = {stats['correlation']:.4f}")
        print(f"- P-value: {stats['p_value']:.6f}")
        print(f"- Significant: {'Yes' if stats['significant'] else 'No'}")

        # Generate visualization
        print(f"\nGenerating visualization...")
        boxplot_path = create_boxplot(df, output_dir)

        # Save results using utility function with group_comparison as extra_data
        group_comparison = {
            "group_comparison": {
                "with_message": {
                    "count": stats["group_statistics"]["with_message"]["count"],
                    "mean": round(stats["group_statistics"]["with_message"]["mean"], 4),
                },
                "without_message": {
                    "count": stats["group_statistics"]["without_message"]["count"],
                    "mean": round(stats["group_statistics"]["without_message"]["mean"], 4),
                },
                "mean_difference": round(stats["group_statistics"]["mean_difference"], 4),
            }
        }

        json_path = save_json(
            output_path=output_dir / "efficiency_commit_message_summary.json",
            analysis_type="efficiency_commit_message_correlation",
            input_files=args.csv_files,
            correlation=stats["correlation"],
            p_value=stats["p_value"],
            outlier_info=outlier_info,
            inference_time_stats=stats["inference_time"],
            sample_size=stats["sample_size"],
            extra_data=group_comparison,
        )

        print(f"\nResults saved to: {output_dir}")
        print(f"- Box plot: {boxplot_path.name}")
        print(f"- Summary JSON: {json_path.name}")

        # Print group means for interpretation
        with_msg_mean = stats["group_statistics"]["with_message"]["mean"]
        without_msg_mean = stats["group_statistics"]["without_message"]["mean"]
        print(f"\nGroup means:")
        print(f"- With message: {with_msg_mean:.4f}s")
        print(f"- Without message: {without_msg_mean:.4f}s")
        print(f"- Difference: {with_msg_mean - without_msg_mean:.4f}s")

    except Exception as e:
        print(f"Error processing CSV files: {e}")
        raise


if __name__ == "__main__":
    main()
