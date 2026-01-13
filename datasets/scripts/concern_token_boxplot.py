#!/usr/bin/env python3
"""
Token Length by Concern Count Box Plot Analysis
Generates box plots showing token length distribution by concern count.
Uses tiktoken to calculate token lengths from diff content.
"""

import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
import tiktoken

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


def calculate_row_tokens(
    commit_msg: str, diff_text: str, encoding: tiktoken.Encoding
) -> int:
    """
    Calculate total token length for commit message and diffs.

    Args:
        commit_msg: Commit message text (may be empty)
        diff_text: Diff content (may be JSON array or plain text)
        encoding: Tiktoken encoding instance

    Returns:
        Total token count
    """
    total_tokens = 0

    # Calculate tokens for commit message
    if commit_msg:
        total_tokens += len(encoding.encode(commit_msg))

    # Calculate tokens for diff
    try:
        # Parse JSON list of diffs
        if diff_text.startswith("[") and diff_text.endswith("]"):
            diffs = json.loads(diff_text)
            if isinstance(diffs, list):
                # Sum tokens for each diff individually
                for diff in diffs:
                    total_tokens += len(encoding.encode(diff))
            else:
                total_tokens += len(encoding.encode(diff_text))
        else:
            total_tokens += len(encoding.encode(diff_text))
    except (json.JSONDecodeError, TypeError):
        total_tokens += len(encoding.encode(diff_text))

    return total_tokens


def load_and_process_data(csv_path: Path) -> pd.DataFrame:
    """
    Load CSV data and calculate token lengths for diff + commit message.

    Raises:
        ValueError: If the dataset is empty or missing required columns
    """
    df = pd.read_csv(csv_path)

    # Validate data at load time
    if df.empty:
        raise ValueError(f"Dataset is empty: {csv_path}")

    required_columns = ["diff", "commit_message", "concern_count"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Initialize tiktoken encoding once
    encoding = tiktoken.get_encoding(ENCODING_MODEL)

    # Calculate token lengths
    df["token_length"] = df.apply(
        lambda row: calculate_row_tokens(
            row["commit_message"] if pd.notna(row["commit_message"]) else "",
            row["diff"],
            encoding,
        ),
        axis=1,
    )

    return df


def create_token_length_boxplot(df: pd.DataFrame, output_dir: Path) -> Path:
    """Create box plot for token length distribution by concern count."""
    setup_plot_style()

    # Get unique concern counts and sort them
    concern_counts = sorted(df["concern_count"].unique())

    # Prepare data for box plot
    plot_data = []
    for concern_count in concern_counts:
        concern_data = df[df["concern_count"] == concern_count]["token_length"].tolist()
        plot_data.append(concern_data)

    # Create figure
    _, ax = plt.subplots(figsize=PLOT_STYLE["figure_size"])

    # Create box plot following standard boxplot definition
    ax.boxplot(
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
        medianprops=dict(color=COLORS["success"], linewidth=PLOT_STYLE["line_width"]),
        whiskerprops=dict(color=COLORS["text"], linewidth=PLOT_STYLE["line_width"]),
        capprops=dict(color=COLORS["text"], linewidth=PLOT_STYLE["line_width"]),
        flierprops=dict(
            marker="o",
            markerfacecolor=COLORS["primary"],
            markersize=4,
            alpha=0.7,
            markeredgecolor="none",
        ),
    )

    # Set labels
    ax.set_xlabel("Concern Count", fontweight="bold", color=COLORS["text"])
    ax.set_ylabel("Input Token Length", fontweight="bold", color=COLORS["text"])

    # Set x-axis ticks and labels
    ax.set_xticks(concern_counts)
    ax.set_xticklabels(concern_counts)

    # Clean grid styling
    ax.grid(True, alpha=PLOT_STYLE["grid_alpha"], linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()

    # Save plot
    plot_path = output_dir / "boxplot_concern_count_token_length.png"
    plt.savefig(
        plot_path, dpi=PLOT_STYLE["dpi"], bbox_inches="tight", facecolor="white"
    )
    plt.close()

    return plot_path


def main():
    """Generate token length box plot from tangled dataset."""
    # Setup: Prepare output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Input: Load and process data
    print("Loading and processing data...")
    df = load_and_process_data(DATASET_PATH)

    # Process: Generate visualization
    print("\nGenerating Token Length box plot...")
    plot_path = create_token_length_boxplot(df, OUTPUT_DIR)

    # Output: Report results
    print(f"Box plot saved to: {plot_path}")
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
