#!/usr/bin/env python3
"""
Performance Summary: Model Comparison Analysis
Analyzes performance metrics from CSV experiment results and generates summary table with box plot.
"""

import pandas as pd
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import Dict

from ..plot_utils import COLORS, PLOT_STYLE, setup_plot_style, display_model_name, get_style_by_index
from . import PROJECT_ROOT, ANALYSIS_OUTPUT_DIR, CONFIG_PATH

MODEL_NAME_MAP = {
    "GPT-4.1": "GPT-4.1",
    "Qwen": "Qwen",
    "Qwen (FT)": "QwenFT",
    "Qwen (Fine-tuned)": "QwenFT",
}

METRIC_KEY = "hamming_loss"


def load_config():
    """Load configuration from config.yaml"""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_individual_csv_data(csv_paths: list) -> Dict[str, pd.DataFrame]:
    """Load individual prediction data from CSV files."""
    csv_data = {}

    for csv_path in csv_paths:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            print(f"Warning: CSV file not found: {csv_path}")
            continue

        df = pd.read_csv(csv_path)

        # Verify required columns exist
        if "hamming_loss" not in df.columns:
            print(f"Warning: Missing 'hamming_loss' column in {csv_path}")
            continue

        # Determine model name based on CSV path
        if "gpt" in str(csv_path).lower():
            model_name = "GPT-4.1"
        # Match on the adapter marker, not the model size: the base and fine-tuned runs
        # share a directory prefix, so a version-specific string silently sends the LoRA
        # results into the base model's column instead of failing.
        elif "LoRA" in str(csv_path):
            model_name = "Qwen (Fine-tuned)"
        elif "Qwen" in str(csv_path):
            model_name = "Qwen"
        else:
            model_name = csv_path.stem

        clean_name = MODEL_NAME_MAP.get(model_name, model_name)
        csv_data[clean_name] = df

    return csv_data


def create_performance_boxplot(
    csv_data: Dict[str, pd.DataFrame], output_dir: Path
) -> Path:
    """Create box plot for hamming loss distribution by model."""
    setup_plot_style()

    if not csv_data:
        print("No CSV data found for boxplot generation")
        return None

    # Prepare data for plotting
    model_names = list(csv_data.keys())

    # Create figure
    fig, ax = plt.subplots(figsize=PLOT_STYLE["figure_size"])

    # Prepare box plot data with sequential styling
    box_data = []
    positions = []
    colors_list = []
    hatches_list = []

    for i, model_name in enumerate(model_names):
        if model_name in csv_data and len(csv_data[model_name]) > 0:
            box_data.append(csv_data[model_name]["hamming_loss"].tolist())
            positions.append(i)
            color, hatch = get_style_by_index(i)
            colors_list.append(color)
            hatches_list.append(hatch)

    # Create box plot
    if box_data:
        bp = ax.boxplot(
            box_data,
            positions=positions,
            widths=0.5,
            patch_artist=True,
            showfliers=True,  # Show outliers (points beyond 1.5*IQR)
            whis=1.5,  # Standard whisker length (1.5 * IQR)
            boxprops=dict(alpha=PLOT_STYLE["alpha"], edgecolor=COLORS["text"], linewidth=1.5),
            medianprops=dict(
                color=COLORS["success"], linewidth=PLOT_STYLE["line_width"]
            ),
            whiskerprops=dict(color=COLORS["text"], linewidth=PLOT_STYLE["line_width"]),
            capprops=dict(color=COLORS["text"], linewidth=PLOT_STYLE["line_width"]),
        )

        # Color each box with its corresponding model color and hatch pattern
        for patch, color, hatch in zip(bp["boxes"], colors_list, hatches_list):
            patch.set_facecolor(color)
            if hatch:
                patch.set_hatch(hatch)

        # Set flier properties for each model's color
        for flier, color in zip(bp["fliers"], colors_list):
            flier.set(
                marker="o",
                markerfacecolor=color,
                markersize=4,
                alpha=0.7,
                markeredgecolor="none",
            )

    # Set labels
    ax.set_xlabel("Model", fontweight="bold", color=COLORS["text"])
    ax.set_ylabel("Hamming Loss", fontweight="bold", color=COLORS["text"])

    # Set x-axis ticks and labels
    ax.set_xticks(positions)
    ax.set_xticklabels([display_model_name(n) for n in model_names])

    # Clean grid styling
    ax.grid(True, alpha=PLOT_STYLE["grid_alpha"], linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()

    # Save plot
    plot_path = output_dir / "boxplot_model_performance.png"
    plt.savefig(
        plot_path, dpi=PLOT_STYLE["dpi"], bbox_inches="tight", facecolor="white"
    )
    plt.close()

    return plot_path


def print_summary_stats(csv_data: Dict[str, pd.DataFrame]):
    """Print summary statistics for each model."""
    print("\nSummary Statistics:")
    print("=" * 80)

    for model_name, df in csv_data.items():
        hl_values = df["hamming_loss"]
        print(
            f"{model_name}: "
            f"n={len(hl_values):3d}, "
            f"mean={hl_values.mean():.3f}, "
            f"std={hl_values.std():.3f}, "
            f"median={hl_values.median():.3f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate performance summary comparison from CSV experiment results"
    )
    parser.add_argument("csv_files", nargs="*", help="CSV files to analyze (optional, reads from config if not provided)")
    parser.add_argument("--output-dir", type=str, help="Output directory")

    args = parser.parse_args()

    # Load configuration
    config = load_config()
    script_config = config["rq1"]["scripts"]["performance_summary"]

    # Determine CSV files to use
    if args.csv_files:
        csv_files = args.csv_files
    else:
        # Read CSV paths from config
        csv_files = [
            str(PROJECT_ROOT / model_config["csv_path"])
            for model_config in script_config["models"].values()
        ]

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = ANALYSIS_OUTPUT_DIR / "pf_summary"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load individual CSV data for box plot
    print("Loading individual CSV data for performance box plot...")
    csv_data = load_individual_csv_data(csv_files)

    if not csv_data:
        print("No valid CSV data found. Please check file paths")
        return

    # Print data summary
    total_samples = sum(len(df) for df in csv_data.values())
    print(f"Loaded {total_samples} total samples from {len(csv_data)} models")
    for model_name, df in csv_data.items():
        print(f"  {model_name}: {len(df)} samples")

    # Generate box plot
    print("\nGenerating Performance box plot...")
    plot_path = create_performance_boxplot(csv_data, output_dir)

    if plot_path:
        print(f"Box plot saved to: {plot_path}")

        # Print summary statistics
        print_summary_stats(csv_data)

    # Generate summary CSV
    rows = []
    for model_name, df in csv_data.items():
        metrics = {
            "hamming_loss": float(df["hamming_loss"].mean()),
        }
        rows.append(
            {"Model": model_name, "Value": round(metrics.get(METRIC_KEY, 0), 2)}
        )

    # Transform to wide format with models as columns
    data_dict = {"Metric": "Hamming Loss"}
    for row in rows:
        data_dict[row["Model"]] = row["Value"]

    df = pd.DataFrame([data_dict])
    csv_path = output_dir / "performance_summary.csv"
    df.to_csv(csv_path, index=False, float_format="%.2f")

    # Long-format avg/min/max summary for the manuscript table (one row per model)
    latex_rows = []
    for model_name, model_df in csv_data.items():
        hl_values = model_df["hamming_loss"]
        latex_rows.append(
            {
                "group_key": display_model_name(model_name),
                "avg": round(float(hl_values.mean()), 2),
                "min": round(float(hl_values.min()), 2),
                "max": round(float(hl_values.max()), 2),
            }
        )
    latex_csv_path = output_dir / "performance_summary_latex.csv"
    pd.DataFrame(latex_rows).to_csv(latex_csv_path, index=False, float_format="%.2f")

    # Print summary table
    print("\n=== Performance Summary ===")
    columns = list(df.columns)
    print(" ".join(f"{h:<12}" for h in columns))
    print("-" * (len(columns) * 12))
    for _, row in df.iterrows():
        print(
            " ".join(
                f"{row[h]:<12.2f}" if h != "Metric" else f"{row[h]:<12}"
                for h in columns
            )
        )

    print(f"\nResults saved to:")
    print(f"  Summary CSV: {csv_path}")
    print(f"  Table CSV: {latex_csv_path}")
    print(f"  Box plot: {plot_path}")


if __name__ == "__main__":
    main()
