#!/usr/bin/env python3
"""
Message Impact Analysis: Model Comparison
Analyzes the impact of commit messages on model performance across GPT-4.1, Qwen, and Qwen(FT).
"""

import pandas as pd
import json
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np

from . import PROJECT_ROOT, ANALYSIS_OUTPUT_DIR
from ..plot_utils import COLORS, PLOT_STYLE, setup_plot_style

MODEL_NAME_MAP = {"GPT-4.1": "GPT-4.1", "Qwen": "Qwen", "Qwen (FT)": "QwenFT"}

METRIC_KEY = "hamming_loss"

DEFAULT_MODEL_CONFIGS = {
    "GPT-4.1": {
        "msg0_path": "results/gpt/avg_result/msg0/12288_zs.csv",
        "msg1_path": "results/gpt/avg_result/msg1/12288_zs.csv",
    },
    "Qwen": {
        "msg0_path": "results/Qwen/avg_result/msg0/12288_zs.csv",
        "msg1_path": "results/Qwen/avg_result/msg1/12288_zs.csv",
    },
    "Qwen (FT)": {
        "msg0_path": "results/Qwen3-14B-LoRA/avg_result/msg0/12288_zs.csv",
        "msg1_path": "results/Qwen3-14B-LoRA/avg_result/msg1/12288_zs.csv",
    },
}


def load_config_from_file(config_path: Path) -> dict:
    """Load model configuration from JSON file."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config
    except Exception as e:
        raise ValueError(f"Failed to load config from {config_path}: {e}")


def build_model_configs(config: dict = None) -> dict:
    """Build model configurations with absolute paths."""
    if config is None:
        # Use default configuration
        model_configs = DEFAULT_MODEL_CONFIGS.copy()
        project_root = PROJECT_ROOT
    else:
        # Use provided configuration
        model_configs = config.get("models", DEFAULT_MODEL_CONFIGS).copy()
        project_root = Path(config.get("project_root", PROJECT_ROOT))

    # Convert relative paths to absolute paths
    for model_name, paths in model_configs.items():
        for key, path in paths.items():
            if isinstance(path, str):
                model_configs[model_name][key] = project_root / path
            else:
                model_configs[model_name][key] = Path(path)

    return model_configs


def load_metrics(csv_path: Path) -> dict:
    """Load metrics from CSV file and calculate macro average for Hamming Loss."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Calculate macro average for Hamming Loss only
    metrics_macro = {
        "hamming_loss": float(df["hamming_loss"].mean()),
    }

    return metrics_macro


def generate_comparison(model_configs: dict) -> pd.DataFrame:
    """Generate comparison table for Hamming Loss with models as columns."""
    data = {"Condition": ["Without Msg", "With Msg", "Delta"]}

    for model_name, paths in model_configs.items():
        msg0_metrics = load_metrics(paths["msg0_path"])
        msg1_metrics = load_metrics(paths["msg1_path"])

        without_msg = round(msg0_metrics[METRIC_KEY], 2)
        with_msg = round(msg1_metrics[METRIC_KEY], 2)
        delta = (
            without_msg - with_msg
        )  # Positive means improvement (lower HS is better)

        clean_name = MODEL_NAME_MAP.get(model_name, model_name)
        data[clean_name] = [without_msg, with_msg, delta]

    return pd.DataFrame(data)


def generate_delta_only_comparison(model_configs: dict) -> pd.DataFrame:
    """Generate comparison table with only delta values."""
    data = {"Metric": "Delta"}

    for model_name, paths in model_configs.items():
        msg0_metrics = load_metrics(paths["msg0_path"])
        msg1_metrics = load_metrics(paths["msg1_path"])

        clean_name = MODEL_NAME_MAP.get(model_name, model_name)
        without_msg = round(msg0_metrics[METRIC_KEY], 2)
        with_msg = round(msg1_metrics[METRIC_KEY], 2)
        delta = (
            without_msg - with_msg
        )  # Positive means improvement (lower HS is better)

        data[clean_name] = delta

    return pd.DataFrame([data])


def load_individual_csv_data(model_configs: dict) -> dict:
    """Load individual prediction data from CSV files."""
    csv_data = {}

    for model_name, paths in model_configs.items():
        clean_name = MODEL_NAME_MAP.get(model_name, model_name)
        csv_data[clean_name] = {}

        # Load msg0 (without message) data
        msg0_path = PROJECT_ROOT / paths["msg0_path"]
        if not msg0_path.exists():
            print(f"Warning: CSV file not found: {msg0_path}")
            continue

        df_msg0 = pd.read_csv(msg0_path)

        # Load msg1 (with message) data
        msg1_path = PROJECT_ROOT / paths["msg1_path"]
        if not msg1_path.exists():
            print(f"Warning: CSV file not found: {msg1_path}")
            continue

        df_msg1 = pd.read_csv(msg1_path)

        # Verify required columns exist
        required_columns = ["hamming_loss"]
        missing_columns_msg0 = [
            col for col in required_columns if col not in df_msg0.columns
        ]
        missing_columns_msg1 = [
            col for col in required_columns if col not in df_msg1.columns
        ]

        if missing_columns_msg0:
            print(f"Warning: Missing columns in {msg0_path}: {missing_columns_msg0}")
            continue
        if missing_columns_msg1:
            print(f"Warning: Missing columns in {msg1_path}: {missing_columns_msg1}")
            continue

        # Store both datasets with condition labels
        csv_data[clean_name] = {"Without Msg": df_msg0, "With Msg": df_msg1}

    return csv_data


def create_msg_impact_boxplot(csv_data: dict, output_dir: Path) -> Path:
    """Create clean box plot for hamming loss distribution by message condition and model."""
    setup_plot_style()

    if not csv_data:
        print("No CSV data found for boxplot generation")
        return None

    # Prepare data for plotting - organize by condition
    conditions = ["Without Msg", "With Msg"]
    model_names = list(csv_data.keys())

    # Create data structure: condition -> model_name -> [hamming_loss_values]
    plot_data = {}
    for condition in conditions:
        plot_data[condition] = {}
        for model_name in model_names:
            # Filter data for this specific condition
            if condition in csv_data[model_name]:
                filtered_data = csv_data[model_name][condition]
                if len(filtered_data) > 0:
                    plot_data[condition][model_name] = filtered_data[
                        "hamming_loss"
                    ].tolist()
                else:
                    plot_data[condition][model_name] = []
            else:
                plot_data[condition][model_name] = []

    # Create figure
    fig, ax = plt.subplots(figsize=PLOT_STYLE["figure_size"])

    # Define colors for each condition
    condition_colors = [COLORS["secondary"], COLORS["primary"]]

    # Calculate positions for grouped box plots
    width = 0.25  # Width of each box
    x_positions = np.arange(len(model_names))

    # Create box plots for each condition
    for i, condition in enumerate(conditions):
        condition_data = []
        positions = []

        for j, model_name in enumerate(model_names):
            if (
                condition in plot_data
                and model_name in plot_data[condition]
                and len(plot_data[condition][model_name]) > 0
            ):

                condition_data.append(plot_data[condition][model_name])
                positions.append(x_positions[j] + i * width)

        # Create box plot for this condition following standard boxplot definition
        if condition_data and positions:
            ax.boxplot(
                condition_data,
                positions=positions,
                widths=width * 0.8,
                patch_artist=True,
                showfliers=True,  # Show outliers (points beyond 1.5*IQR)
                whis=1.5,  # Standard whisker length (1.5 * IQR)
                boxprops=dict(facecolor=condition_colors[i], alpha=PLOT_STYLE["alpha"]),
                medianprops=dict(
                    color=COLORS["success"], linewidth=PLOT_STYLE["line_width"]
                ),
                whiskerprops=dict(
                    color=COLORS["text"], linewidth=PLOT_STYLE["line_width"]
                ),
                capprops=dict(color=COLORS["text"], linewidth=PLOT_STYLE["line_width"]),
                flierprops=dict(
                    marker="o",
                    markerfacecolor=condition_colors[i],
                    markersize=4,
                    alpha=0.7,
                    markeredgecolor="none",
                ),
            )

    # Set labels and title
    ax.set_xlabel("Model", fontweight="bold", color=COLORS["text"])
    ax.set_ylabel("Hamming Loss", fontweight="bold", color=COLORS["text"])
    ax.set_title(
        "Impact of Commit Messages on Model Performance",
        fontweight="bold",
        color=COLORS["text"],
        pad=20,
    )

    # Set x-axis ticks and labels
    ax.set_xticks(x_positions + width)  # Center the labels
    ax.set_xticklabels(model_names)

    # Add legend for conditions
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=condition_colors[i], alpha=PLOT_STYLE["alpha"], label=condition)
        for i, condition in enumerate(conditions)
    ]
    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.9)

    # Clean grid styling
    ax.grid(True, alpha=PLOT_STYLE["grid_alpha"], linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()

    # Save plot with RQ2 naming convention
    plot_path = output_dir / "boxplot_msg_impact_hamming_loss.png"
    plt.savefig(
        plot_path, dpi=PLOT_STYLE["dpi"], bbox_inches="tight", facecolor="white"
    )
    plt.close()

    return plot_path


def print_summary_stats(csv_data):
    """Print summary statistics for each model and condition."""
    print("\nSummary Statistics:")
    print("=" * 80)

    for model_name, model_data in csv_data.items():
        print(f"\n{model_name}:")
        print("-" * 50)
        for condition in ["Without Msg", "With Msg"]:
            if condition in model_data:
                df = model_data[condition]
                hl_values = df["hamming_loss"]
                print(
                    f"  {condition}: "
                    f"n={len(hl_values):3d}, "
                    f"mean={hl_values.mean():.3f}, "
                    f"std={hl_values.std():.3f}, "
                    f"median={hl_values.median():.3f}"
                )


def main():
    parser = argparse.ArgumentParser(
        description="Generate message impact analysis from avg_result JSON files"
    )
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument(
        "--config", type=str, help="JSON configuration file with model paths"
    )

    args = parser.parse_args()

    if args.config:
        config = load_config_from_file(Path(args.config))
        model_configs = build_model_configs(config)
        print(f"✅ Loaded configuration from {args.config}")
    else:
        model_configs = build_model_configs()
        print("📋 Using default model configuration")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = ANALYSIS_OUTPUT_DIR / "pf_msg_impact"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate main comparison with all conditions
    comparison_df = generate_comparison(model_configs)
    comparison_csv_path = output_dir / "msg_impact_comparison.csv"
    comparison_df.to_csv(comparison_csv_path, index=False, float_format="%.2f")

    # Generate delta-only comparison
    delta_df = generate_delta_only_comparison(model_configs)
    delta_csv_path = output_dir / "msg_impact_delta.csv"
    delta_df.to_csv(delta_csv_path, index=False, float_format="%.2f")

    # Load individual CSV data for box plot
    print("Loading individual CSV data for message impact box plot...")
    csv_data = load_individual_csv_data(model_configs)

    if not csv_data:
        print("No valid CSV data found. Please check file paths in configuration")
        return

    # Print data summary
    total_samples = sum(
        len(df) for model_data in csv_data.values() for df in model_data.values()
    )
    print(f"Loaded {total_samples} total samples from {len(csv_data)} models")

    for model_name, model_data in csv_data.items():
        print(f"  {model_name}: {sum(len(df) for df in model_data.values())} samples")

    # Generate clean box plot
    print("\nGenerating Message Impact box plot...")
    plot_path = create_msg_impact_boxplot(csv_data, output_dir)

    if plot_path:
        print(f"Box plot saved to: {plot_path}")

        # Print summary statistics
        print_summary_stats(csv_data)

        print(f"\nResults saved to: {output_dir}")
    else:
        print("Failed to generate box plot")

    # Print main comparison table
    print("\n=== Message Impact Analysis ===")
    columns = list(comparison_df.columns)
    print(" ".join(f"{h:<12}" for h in columns))
    print("-" * (len(columns) * 12))
    for _, row in comparison_df.iterrows():
        print(
            " ".join(
                f"{row[h]:<12.2f}" if h != "Condition" else f"{row[h]:<12}"
                for h in columns
            )
        )

    # Print delta table
    print("\n=== Delta Values ===")
    delta_columns = list(delta_df.columns)
    print(" ".join(f"{h:<12}" for h in delta_columns))
    print("-" * (len(delta_columns) * 12))
    for _, row in delta_df.iterrows():
        print(
            " ".join(
                f"{row[h]:+12.2f}" if h != "Metric" else f"{row[h]:<12}"
                for h in delta_columns
            )
        )

    print(f"\nResults saved to:")
    print(f"  Comparison: {comparison_csv_path}")
    print(f"  Delta: {delta_csv_path}")


if __name__ == "__main__":
    main()
