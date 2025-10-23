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
from scipy import stats
from typing import Dict, List, Tuple, Any

# Constants - Use root results directory (from project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
ANALYSIS_OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis" / "RQ2"
P_VALUE_THRESHOLD = 0.05
OUTLIER_THRESHOLD_IQR = 1.5  # IQR multiplier for outlier detection

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
            "font.size": 15,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
        }
    )


def detect_outliers_iqr(data: pd.Series) -> List[int]:
    """Detect outliers using the IQR method."""
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - OUTLIER_THRESHOLD_IQR * IQR
    upper_bound = Q3 + OUTLIER_THRESHOLD_IQR * IQR

    outlier_mask = (data < lower_bound) | (data > upper_bound)
    return data[outlier_mask].index.tolist()


def perform_statistical_tests(csv_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Perform comprehensive statistical tests on hamming loss data."""
    results = {"anova": {}, "pairwise": {}, "descriptive": {}}

    # Prepare data for ANOVA
    all_data = []
    for model_name, df in csv_data.items():
        for concern_count in [1, 2, 3, 4, 5]:
            concern_data = df[df["concern_count"] == concern_count]["hamming_loss"]
            for value in concern_data:
                all_data.append(
                    {
                        "model": model_name,
                        "concern_count": concern_count,
                        "hamming_loss": value,
                    }
                )

    analysis_df = pd.DataFrame(all_data)

    # ANOVA for each concern count
    for concern_count in [1, 2, 3, 4, 5]:
        concern_data = analysis_df[analysis_df["concern_count"] == concern_count]
        groups = [
            concern_data[concern_data["model"] == model]["hamming_loss"].values
            for model in concern_data["model"].unique()
        ]

        try:
            f_stat, p_value = stats.f_oneway(*groups)
            results["anova"][concern_count] = {
                "f_statistic": float(f_stat),
                "p_value": float(p_value),
                "significant": p_value < P_VALUE_THRESHOLD,
            }
        except Exception as e:
            results["anova"][concern_count] = {"error": str(e)}

    # Model-wise correlation with concern count
    for model_name, df in csv_data.items():
        try:
            correlation, p_value = stats.pearsonr(
                df["concern_count"], df["hamming_loss"]
            )
            results["pairwise"][model_name] = {
                "correlation": float(correlation),
                "p_value": float(p_value),
                "significant": p_value < P_VALUE_THRESHOLD,
            }
        except Exception as e:
            results["pairwise"][model_name] = {"error": str(e)}

    # Descriptive statistics
    for model_name, df in csv_data.items():
        model_stats = {}
        for concern_count in [1, 2, 3, 4, 5]:
            concern_data = df[df["concern_count"] == concern_count]["hamming_loss"]
            if len(concern_data) > 0:
                model_stats[concern_count] = {
                    "count": len(concern_data),
                    "mean": float(concern_data.mean()),
                    "std": float(concern_data.std()),
                    "median": float(concern_data.median()),
                    "q25": float(concern_data.quantile(0.25)),
                    "q75": float(concern_data.quantile(0.75)),
                    "min": float(concern_data.min()),
                    "max": float(concern_data.max()),
                }
        results["descriptive"][model_name] = model_stats

    return results


def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_individual_csv_data(config):
    """Load individual prediction data from CSV files."""
    csv_data = {}

    # Use concern_count_boxplot config to get CSV paths
    csv_config = config["scripts"]["concern_count_boxplot"]

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

    # Define colors for each model
    model_colors = [COLORS["primary"], COLORS["secondary"], COLORS["accent"]]

    # Calculate positions for grouped box plots
    width = 0.25  # Width of each box
    x_positions = np.arange(len(concern_counts))

    # Create box plots for each model
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
    ax.set_xlabel("Concern Count", fontweight="bold", color=COLORS["text"])
    ax.set_ylabel("Hamming Loss", fontweight="bold", color=COLORS["text"])
    ax.set_title(
        "Hamming Loss Distribution by Concern Count and Model",
        fontweight="bold",
        color=COLORS["text"],
        pad=20,
    )

    # Set x-axis ticks and labels
    ax.set_xticks(x_positions + width)  # Center the labels
    ax.set_xticklabels(concern_counts)

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
    parser.add_argument(
        "--use-config",
        action="store_true",
        default=True,
        help="Use config.yaml for model configuration",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config()
    script_config = config["scripts"]["concern_count_boxplot"]

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
