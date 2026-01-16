#!/usr/bin/env python3
"""
Efficiency Analysis: Correlation between Commit Message and Inference Time
Analyzes the relationship between commit message presence and inference time using statistical tests.
Processes raw CSV data for detailed box plot analysis and group comparison.
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path
import argparse

from ..plot_utils import (
    COLORS,
    PLOT_STYLE,
    boxplot_style,
    setup_plot_style,
    get_color_palette,
    get_hatch_palette,
)
from .utils import (
    load_csv,
    pearson_correlation,
    add_correlation_text,
    save_json,
    P_VALUE_THRESHOLD,
)
from . import ANALYSIS_OUTPUT_DIR


def calculate_group_stats(df: pd.DataFrame) -> dict:
    """Calculate statistics for with/without message groups."""
    with_msg = df[df["with_message"]]["inference_time"]
    without_msg = df[~df["with_message"]]["inference_time"]

    return {
        "with_message": {"count": len(with_msg), "mean": float(with_msg.mean())},
        "without_message": {"count": len(without_msg), "mean": float(without_msg.mean())},
        "mean_difference": float(with_msg.mean() - without_msg.mean()),
    }


def create_boxplot(df: pd.DataFrame, output_dir: Path) -> Path:
    """Create box plot showing inference time distribution by commit message presence."""
    setup_plot_style()

    fig, ax = plt.subplots(figsize=PLOT_STYLE["figure_size"])

    # Prepare data and styling
    without_msg_data = df[~df["with_message"]]["inference_time"]
    with_msg_data = df[df["with_message"]]["inference_time"]
    box_data = [without_msg_data, with_msg_data]
    labels = ["Without Message", "With Message"]
    colors = get_color_palette(2)
    hatches = get_hatch_palette(2)

    # Create box plot
    positions = [0, 1]
    box_plot = ax.boxplot(box_data, positions=positions, **boxplot_style(widths=0.4))

    # Apply colorblind-friendly styling to each box
    for patch, flier, color, hatch in zip(
        box_plot["boxes"], box_plot["fliers"], colors, hatches
    ):
        patch.set_facecolor(color)
        patch.set_edgecolor(COLORS["text"])
        patch.set_linewidth(1.5)
        if hatch:
            patch.set_hatch(hatch)
        flier.set_markerfacecolor(color)

    # Configure axes
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.5, 1.5)
    ax.set_xlabel("Commit Message Presence", fontweight="bold", color=COLORS["text"])
    ax.set_ylabel("Inference Time (seconds)", fontweight="bold", color=COLORS["text"])

    # Add correlation statistics
    correlation, p_value = pearson_correlation(
        df.assign(with_message_int=df["with_message"].astype(int)), "with_message_int"
    )
    add_correlation_text(ax, correlation, p_value, len(df))

    # Build legend
    legend_elements = [
        Patch(
            facecolor=colors[i],
            alpha=PLOT_STYLE["alpha"],
            edgecolor=COLORS["text"],
            linewidth=1.5,
            hatch=hatches[i] or None,
            label=labels[i],
        )
        for i in range(len(labels))
    ]
    legend_elements.append(
        Line2D([0], [0], color=COLORS["success"], linewidth=PLOT_STYLE["line_width"], label="Median")
    )
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.9)

    plt.tight_layout()

    output_path = output_dir / "boxplot_commit_message_inference_time.png"
    plt.savefig(output_path, dpi=PLOT_STYLE["dpi"], bbox_inches="tight", facecolor="white")
    plt.close()

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Analyze commit message vs inference time correlation from CSV files"
    )
    parser.add_argument("csv_files", nargs="+", help="CSV files to analyze")
    parser.add_argument("--output-dir", type=str, help="Output directory")

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else ANALYSIS_OUTPUT_DIR / "ef_commit_message"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Starting efficiency analysis: Commit Message vs Inference Time")
    print("=" * 60)
    print(f"Processing {len(args.csv_files)} CSV file(s)...")

    csv_paths = [Path(f) for f in args.csv_files]

    try:
        df, outlier_info = load_csv(csv_paths, ["with_message", "inference_time"])
        print(f"Loaded {len(df)} data points from CSV files")

        # Calculate correlation (convert boolean to int)
        df_numeric = df.assign(with_message_int=df["with_message"].astype(int))
        correlation, p_value = pearson_correlation(df_numeric, "with_message_int")
        significant = p_value < P_VALUE_THRESHOLD

        # Calculate group statistics
        group_stats = calculate_group_stats(df)

        # Print results
        print(f"\nGroup Comparison Results:")
        print(f"- Sample size: {len(df)}")
        print(f"- With message: {group_stats['with_message']['count']} samples")
        print(f"- Without message: {group_stats['without_message']['count']} samples")
        print(f"- Pearson correlation: r = {correlation:.4f}")
        print(f"- P-value: {p_value:.6f}")
        print(f"- Significant: {'Yes' if significant else 'No'}")

        # Generate visualization
        print("\nGenerating visualization...")
        boxplot_path = create_boxplot(df, output_dir)

        # Save results
        inference_time_stats = {
            "mean": df["inference_time"].mean(),
            "std": df["inference_time"].std(),
            "min": df["inference_time"].min(),
            "max": df["inference_time"].max(),
        }
        json_path = save_json(
            output_path=output_dir / "efficiency_commit_message_summary.json",
            analysis_type="efficiency_commit_message_correlation",
            input_files=args.csv_files,
            correlation=correlation,
            p_value=p_value,
            outlier_info=outlier_info,
            inference_time_stats=inference_time_stats,
            sample_size=len(df),
            extra_data={
                "group_comparison": {
                    "with_message": {
                        "count": group_stats["with_message"]["count"],
                        "mean": round(group_stats["with_message"]["mean"], 4),
                    },
                    "without_message": {
                        "count": group_stats["without_message"]["count"],
                        "mean": round(group_stats["without_message"]["mean"], 4),
                    },
                    "mean_difference": round(group_stats["mean_difference"], 4),
                }
            },
        )

        print(f"\nResults saved to: {output_dir}")
        print(f"- Box plot: {boxplot_path.name}")
        print(f"- Summary JSON: {json_path.name}")

        # Print group means for interpretation
        with_msg_mean = group_stats["with_message"]["mean"]
        without_msg_mean = group_stats["without_message"]["mean"]
        print(f"\nGroup means:")
        print(f"- With message: {with_msg_mean:.4f}s")
        print(f"- Without message: {without_msg_mean:.4f}s")
        print(f"- Difference: {group_stats['mean_difference']:.4f}s")

    except Exception as e:
        print(f"Error processing CSV files: {e}")
        raise


if __name__ == "__main__":
    main()
