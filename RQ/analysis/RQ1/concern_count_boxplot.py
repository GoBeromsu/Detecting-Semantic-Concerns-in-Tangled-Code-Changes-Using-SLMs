#!/usr/bin/env python3
"""
Hamming Loss Box Plot Analysis
Generates box plots showing hamming loss distribution by concern count and model.
"""

import pandas as pd
import yaml
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict

from ..plot_utils import COLORS, PLOT_STYLE, setup_plot_style, get_style_by_index, create_legend_patches
from . import PROJECT_ROOT, ANALYSIS_OUTPUT_DIR, CONFIG_PATH


def load_config():
    """Load configuration from config.yaml"""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_individual_csv_data(config):
    """Load individual prediction data from CSV files."""
    csv_data = {}

    # Use concern_count_boxplot config to get CSV paths
    csv_config = config["rq1"]["scripts"]["concern_count_boxplot"]

    for model_name, model_config in csv_config["models"].items():
        csv_path = PROJECT_ROOT / model_config["csv_path"]
        if not csv_path.exists():
            print(f"Warning: CSV file not found: {csv_path}")
            continue

        df = pd.read_csv(csv_path)

        # The CSV already has hamming_loss and concern_count columns
        # Just verify they exist
        required_columns = ["hamming_loss", "concern_count"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"Warning: Missing columns in {csv_path}: {missing_columns}")
            continue

        csv_data[model_name] = df

    return csv_data


def create_hamming_loss_boxplot(
    csv_data: Dict[str, pd.DataFrame], output_dir: Path
) -> Path:
    """Create clean box plot for hamming loss distribution by concern count and model."""
    setup_plot_style()

    if not csv_data:
        print("No CSV data found for boxplot generation")
        return None

    # Prepare data for plotting - organize by concern count
    concern_counts = [1, 2, 3, 4, 5]
    model_names = list(csv_data.keys())

    # Create data structure: concern_count -> model_name -> [hamming_loss_values]
    plot_data = {}
    for concern_count in concern_counts:
        plot_data[concern_count] = {}
        for model_name in model_names:
            # Filter data for this specific concern count
            filtered_data = csv_data[model_name][
                csv_data[model_name]["concern_count"] == concern_count
            ]
            if len(filtered_data) > 0:
                plot_data[concern_count][model_name] = filtered_data[
                    "hamming_loss"
                ].tolist()
            else:
                plot_data[concern_count][model_name] = []

    # Create figure
    fig, ax = plt.subplots(figsize=PLOT_STYLE["figure_size"])

    # Calculate positions for grouped box plots
    width = 0.25  # Width of each box
    x_positions = np.arange(len(concern_counts))

    # Create box plots for each model with sequential styling
    for i, model_name in enumerate(model_names):
        model_data = []
        positions = []

        for j, concern_count in enumerate(concern_counts):
            if (
                concern_count in plot_data
                and model_name in plot_data[concern_count]
                and len(plot_data[concern_count][model_name]) > 0
            ):

                model_data.append(plot_data[concern_count][model_name])
                positions.append(x_positions[j] + i * width)

        # Create box plot for this model following standard boxplot definition
        if model_data and positions:
            color, hatch = get_style_by_index(i)

            boxprops = dict(
                facecolor=color,
                alpha=PLOT_STYLE["alpha"],
                edgecolor=COLORS["text"],
                linewidth=1.5,
            )
            if hatch:
                boxprops["hatch"] = hatch

            ax.boxplot(
                model_data,
                positions=positions,
                widths=width * 0.8,
                patch_artist=True,
                showfliers=True,  # Show outliers (points beyond 1.5*IQR)
                whis=1.5,  # Standard whisker length (1.5 * IQR)
                boxprops=boxprops,
                medianprops=dict(
                    color=COLORS["success"], linewidth=PLOT_STYLE["line_width"]
                ),
                whiskerprops=dict(
                    color=COLORS["text"], linewidth=PLOT_STYLE["line_width"]
                ),
                capprops=dict(color=COLORS["text"], linewidth=PLOT_STYLE["line_width"]),
                flierprops=dict(
                    marker="o",
                    markerfacecolor=color,
                    markersize=4,
                    alpha=0.7,
                    markeredgecolor="none",
                ),
            )

    # Set labels
    ax.set_xlabel("Concern Count", fontweight="bold", color=COLORS["text"])
    ax.set_ylabel("Hamming Loss", fontweight="bold", color=COLORS["text"])

    # Set x-axis ticks and labels
    ax.set_xticks(x_positions + width)  # Center the labels
    ax.set_xticklabels(concern_counts)

    # Add legend for models with sequential colors and hatches
    legend_patches, legend_kwargs = create_legend_patches(model_names)
    ax.legend(handles=legend_patches, **legend_kwargs)

    # Clean grid styling
    ax.grid(True, alpha=PLOT_STYLE["grid_alpha"], linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()

    # Save plot with RQ2 naming convention
    plot_path = output_dir / "boxplot_concern_count_hamming_loss.png"
    plt.savefig(
        plot_path, dpi=PLOT_STYLE["dpi"], bbox_inches="tight", facecolor="white"
    )
    plt.close()

    return plot_path


def print_summary_stats(csv_data):
    """Print summary statistics for each model and concern count."""
    print("\nSummary Statistics:")
    print("=" * 80)

    for model_name, df in csv_data.items():
        print(f"\n{model_name}:")
        print("-" * 50)
        for concern_count in [1, 2, 3, 4, 5]:
            filtered_data = df[df["concern_count"] == concern_count]
            if len(filtered_data) > 0:
                hl_values = filtered_data["hamming_loss"]
                print(
                    f"  Concern Count {concern_count}: "
                    f"n={len(hl_values):3d}, "
                    f"mean={hl_values.mean():.3f}, "
                    f"std={hl_values.std():.3f}, "
                    f"median={hl_values.median():.3f}"
                )


def main():
    parser = argparse.ArgumentParser(
        description="Generate hamming loss box plot from CSV experiment results"
    )
    parser.add_argument("--output-dir", type=str, help="Output directory")
    args = parser.parse_args()

    # Load configuration
    config = load_config()

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = ANALYSIS_OUTPUT_DIR / "pf_concern_count_boxplot"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load individual CSV data
    print("Loading individual CSV data for box plot generation...")
    csv_data = load_individual_csv_data(config)

    if not csv_data:
        print("No valid CSV data found. Please check file paths in config.yaml")
        return

    # Print data summary
    total_samples = sum(len(df) for df in csv_data.values())
    print(f"Loaded {total_samples} total samples from {len(csv_data)} models")

    for model_name, df in csv_data.items():
        print(f"  {model_name}: {len(df)} samples")

    # Generate clean box plot
    print("\nGenerating Hamming Loss box plot...")
    plot_path = create_hamming_loss_boxplot(csv_data, output_dir)

    if plot_path:
        print(f"Box plot saved to: {plot_path}")

        # Print summary statistics
        print_summary_stats(csv_data)

        print(f"\nResults saved to: {output_dir}")
    else:
        print("Failed to generate box plot")


if __name__ == "__main__":
    main()
