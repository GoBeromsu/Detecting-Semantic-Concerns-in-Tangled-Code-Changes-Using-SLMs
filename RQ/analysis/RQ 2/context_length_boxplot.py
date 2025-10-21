#!/usr/bin/env python3
"""
Context Length Hamming Loss Box Plot Analysis
Generates box plots showing hamming loss distribution by context length and model.
"""

import pandas as pd
import yaml
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Any

# Constants - Use root results directory (from project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
ANALYSIS_OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis" / "RQ2"

# Design constants for consistent styling (from RQ2 analysis)
COLORS = {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "accent": "#F18F01",
    "success": "#C73E1D",
    "background": "#F5F5F5",
    "text": "#2C3E50",
}

PLOT_STYLE = {
    "figure_size": (10, 6),
    "dpi": 300,
    "line_width": 2,
    "marker_size": 60,
    "alpha": 0.7,
    "grid_alpha": 0.3,
}


def setup_plot_style():
    """Setup consistent plot styling."""
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
        }
    )


def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_individual_csv_data_by_context(config):
    """Load individual prediction data from CSV files by context length."""
    csv_data = {}

    # Get CSV paths from config
    script_config = config["scripts"]["context_length_boxplot"]
    context_lengths = script_config["context_lengths"]

    for context_length in context_lengths:
        csv_data[context_length] = {}

        for model_name, model_config in script_config["models"].items():
            # Build CSV path pattern
            csv_path_pattern = model_config["csv_path_pattern"]
            csv_path = PROJECT_ROOT / csv_path_pattern.format(context=context_length)

            if not csv_path.exists():
                print(f"Warning: CSV file not found: {csv_path}")
                continue

            df = pd.read_csv(csv_path)

            # Verify required columns exist
            required_columns = ["hamming_loss"]
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                print(f"Warning: Missing columns in {csv_path}: {missing_columns}")
                continue

            csv_data[context_length][model_name] = df

    return csv_data


def create_context_length_boxplot(
    csv_data_by_context: dict, output_dir: Path, metric: str = "hamming_loss"
) -> Path:
    """Create box plots for metric distribution by context length and model."""
    setup_plot_style()

    if not csv_data_by_context:
        print(f"No CSV data found for {metric} boxplot generation")
        return None

    # Prepare data for plotting - organize by context length
    context_lengths = sorted(csv_data_by_context.keys())
    model_names = []

    # Get model names from first available context length
    for context_length in context_lengths:
        if csv_data_by_context[context_length]:
            model_names = list(csv_data_by_context[context_length].keys())
            break

    if not model_names:
        print("No model data found")
        return None

    # Create data structure: context_length -> model_name -> [metric_values]
    plot_data = {}
    for context_length in context_lengths:
        plot_data[context_length] = {}
        for model_name in model_names:
            if (
                context_length in csv_data_by_context
                and model_name in csv_data_by_context[context_length]
            ):
                df = csv_data_by_context[context_length][model_name]
                if metric in df.columns:
                    plot_data[context_length][model_name] = df[metric].tolist()
                else:
                    plot_data[context_length][model_name] = []
            else:
                plot_data[context_length][model_name] = []

    # Create figure
    fig, ax = plt.subplots(figsize=PLOT_STYLE["figure_size"])

    # Define colors for each model
    model_colors = [COLORS["primary"], COLORS["secondary"], COLORS["accent"]]

    # Calculate positions for grouped box plots
    width = 0.25  # Width of each box
    x_positions = np.arange(len(context_lengths))

    # Create box plots for each model
    for i, model_name in enumerate(model_names):
        model_data = []
        positions = []

        for j, context_length in enumerate(context_lengths):
            if (
                context_length in plot_data
                and model_name in plot_data[context_length]
                and len(plot_data[context_length][model_name]) > 0
            ):

                model_data.append(plot_data[context_length][model_name])
                positions.append(x_positions[j] + i * width)

        # Create box plot for this model following standard boxplot definition
        if model_data and positions:
            ax.boxplot(
                model_data,
                positions=positions,
                widths=width * 0.8,
                patch_artist=True,
                showfliers=True,  # Show outliers (points beyond 1.5*IQR)
                whis=1.5,  # Standard whisker length (1.5 * IQR)
                boxprops=dict(
                    facecolor=model_colors[i % len(model_colors)],
                    alpha=PLOT_STYLE["alpha"],
                ),
                medianprops=dict(
                    color=COLORS["success"], linewidth=PLOT_STYLE["line_width"]
                ),
                whiskerprops=dict(
                    color=COLORS["text"], linewidth=PLOT_STYLE["line_width"]
                ),
                capprops=dict(color=COLORS["text"], linewidth=PLOT_STYLE["line_width"]),
                flierprops=dict(
                    marker="o",
                    markerfacecolor=model_colors[i % len(model_colors)],
                    markersize=4,
                    alpha=0.7,
                    markeredgecolor="none",
                ),
            )

    # Set labels and title
    metric_label = "Hamming Loss" if metric == "hamming_loss" else "F1 Score"
    ax.set_xlabel("Context Length (tokens)", fontweight="bold", color=COLORS["text"])
    ax.set_ylabel(metric_label, fontweight="bold", color=COLORS["text"])
    ax.set_title(
        f"{metric_label} Distribution by Context Length and Model",
        fontweight="bold",
        color=COLORS["text"],
        pad=20,
    )

    # Set x-axis ticks and labels
    ax.set_xticks(x_positions + width)  # Center the labels
    ax.set_xticklabels(context_lengths)

    # Add legend for models
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(
            facecolor=model_colors[i % len(model_colors)],
            alpha=PLOT_STYLE["alpha"],
            label=model_name,
        )
        for i, model_name in enumerate(model_names)
    ]
    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.9)

    # Clean grid styling
    ax.grid(True, alpha=PLOT_STYLE["grid_alpha"], linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()

    # Save plot
    plot_path = output_dir / f"boxplot_context_length_{metric}.png"
    plt.savefig(
        plot_path, dpi=PLOT_STYLE["dpi"], bbox_inches="tight", facecolor="white"
    )
    plt.close()

    return plot_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate context length hamming loss box plots from CSV experiment results"
    )
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument(
        "--use-config",
        action="store_true",
        default=True,
        help="Use config.yaml for model configuration",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config()
    script_config = config["scripts"]["context_length_boxplot"]

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = ANALYSIS_OUTPUT_DIR / "pf_context_length_boxplot"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load individual CSV data
    print("Loading individual CSV data for context length box plot generation...")
    csv_data_by_context = load_individual_csv_data_by_context(config)

    if not csv_data_by_context:
        print("No valid CSV data found. Please check file paths in config.yaml")
        return

    # Print data summary
    total_samples = sum(
        len(df)
        for context_data in csv_data_by_context.values()
        for df in context_data.values()
    )
    total_contexts = len(csv_data_by_context)
    print(
        f"Loaded {total_samples} total samples across {total_contexts} context lengths from {len(script_config['models'])} models"
    )

    # Generate Hamming Loss box plot only
    print("\nGenerating Context Length Hamming Loss box plot...")

    # Generate Hamming Loss box plot
    hamming_plot_path = create_context_length_boxplot(
        csv_data_by_context, output_dir, "hamming_loss"
    )
    if hamming_plot_path:
        print(f"Hamming Loss box plot saved to: {hamming_plot_path}")
        print(f"\nResults saved to: {output_dir}")
    else:
        print("Failed to generate box plot")


if __name__ == "__main__":
    main()
