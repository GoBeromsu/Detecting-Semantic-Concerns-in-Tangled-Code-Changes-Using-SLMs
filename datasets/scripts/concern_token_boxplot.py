#!/usr/bin/env python3
"""
Token Length by Concern Count Box Plot Analysis
Generates box plots showing token length distribution by concern count.
Uses tiktoken to calculate token lengths from diff content.
"""

import pandas as pd
import json
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np
import tiktoken
from typing import Dict, List, Tuple, Any

# Constants - Use project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATASET_PATH = PROJECT_ROOT / "datasets" / "data" / "tangled_ccs_dataset_test.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis" / "concern_token_boxplot"

# Tiktoken encoding model (GPT-4 encoding)
ENCODING_MODEL = "cl100k_base"

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


def calculate_token_length(text: str, encoding) -> int:
    """
    Calculate token length using tiktoken encoding.

    Args:
        text: The text to tokenize
        encoding: The tiktoken encoding object

    Returns:
        Number of tokens in the text
    """
    try:
        # Parse the diff if it's in JSON format
        if text.startswith('[') and text.endswith(']'):
            diffs = json.loads(text)
            # Join all diffs with newlines
            text = '\n'.join(diffs) if isinstance(diffs, list) else text
    except (json.JSONDecodeError, TypeError):
        pass  # Use text as-is if not valid JSON

    return len(encoding.encode(text))


def load_and_process_data(csv_path: Path) -> pd.DataFrame:
    """
    Load CSV data and calculate token lengths for diff + commit message.

    Args:
        csv_path: Path to the CSV file

    Returns:
        DataFrame with added token_length column
    """
    df = pd.read_csv(csv_path)

    # Initialize tiktoken encoding
    encoding = tiktoken.get_encoding(ENCODING_MODEL)

    # Calculate token lengths for diff + commit message
    def calculate_combined_length(row):
        diff_text = row['diff']
        commit_msg = row['commit_message'] if pd.notna(row['commit_message']) else ''

        # Combine diff and commit message
        combined_text = f"{commit_msg}\n\n{diff_text}" if commit_msg else diff_text

        return calculate_token_length(combined_text, encoding)

    df['token_length'] = df.apply(calculate_combined_length, axis=1)

    return df


def create_token_length_boxplot(df: pd.DataFrame, output_dir: Path) -> Path:
    """
    Create clean box plot for token length distribution by concern count.

    Args:
        df: DataFrame with concern_count and token_length columns
        output_dir: Directory to save the plot

    Returns:
        Path to the saved plot
    """
    setup_plot_style()

    # Get unique concern counts and sort them
    concern_counts = sorted(df['concern_count'].unique())

    # Prepare data for box plot
    plot_data = []
    for concern_count in concern_counts:
        concern_data = df[df['concern_count'] == concern_count]['token_length'].tolist()
        plot_data.append(concern_data)

    # Create figure
    fig, ax = plt.subplots(figsize=PLOT_STYLE["figure_size"])

    # Create box plot following standard boxplot definition
    bp = ax.boxplot(
        plot_data,
        positions=concern_counts,
        widths=0.6,
        patch_artist=True,
        showfliers=True,  # Show outliers (points beyond 1.5*IQR)
        whis=1.5,  # Standard whisker length (1.5 * IQR)
        boxprops=dict(
            facecolor=COLORS["primary"],
            alpha=PLOT_STYLE["alpha"],
        ),
        medianprops=dict(
            color=COLORS["success"],
            linewidth=PLOT_STYLE["line_width"]
        ),
        whiskerprops=dict(
            color=COLORS["text"],
            linewidth=PLOT_STYLE["line_width"]
        ),
        capprops=dict(
            color=COLORS["text"],
            linewidth=PLOT_STYLE["line_width"]
        ),
        flierprops=dict(
            marker="o",
            markerfacecolor=COLORS["primary"],
            markersize=4,
            alpha=0.7,
            markeredgecolor="none",
        ),
    )

    # Set labels and title
    ax.set_xlabel("Concern Count", fontweight="bold", color=COLORS["text"])
    ax.set_ylabel("Input Token Length", fontweight="bold", color=COLORS["text"])
    ax.set_title(
        "Token Length Distribution by Concern Count",
        fontweight="bold",
        color=COLORS["text"],
        pad=20,
    )

    # Set x-axis ticks
    ax.set_xticks(concern_counts)
    ax.set_xticklabels(concern_counts)

    # Clean grid styling
    ax.grid(True, alpha=PLOT_STYLE["grid_alpha"], linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()

    # Save plot
    plot_path = output_dir / "boxplot_concern_count_token_length.png"
    plt.savefig(
        plot_path,
        dpi=PLOT_STYLE["dpi"],
        bbox_inches="tight",
        facecolor="white"
    )
    plt.close()

    return plot_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate token length box plot from tangled dataset"
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        help="Path to CSV file (default: tangled_ccs_dataset_test.csv)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for plots"
    )

    args = parser.parse_args()

    # Set paths
    csv_path = Path(args.csv_path) if args.csv_path else DATASET_PATH
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and process data
    df = load_and_process_data(csv_path)

    # Generate box plot
    plot_path = create_token_length_boxplot(df, output_dir)

    if plot_path:
        print(f"Box plot saved to: {plot_path}")
    else:
        print("Failed to generate box plot")


if __name__ == "__main__":
    main()