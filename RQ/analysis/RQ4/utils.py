"""
Utility functions for RQ4 efficiency analysis.

Common functions shared across efficiency analysis scripts:
- load_efficiency_csv: Load CSV with column validation and outlier detection
- save_efficiency_json: Save analysis results as JSON with standard structure
- add_correlation_text: Add correlation stats text to plot
- add_boxplot_legend: Add standard legend to boxplot
"""

import json
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple
from scipy import stats

from ..stats_utils import detect_outliers_iqr
from ..plot_utils import COLORS, PLOT_STYLE

# Constants
P_VALUE_THRESHOLD = 0.05


def load_csv(
    csv_paths: List[Path],
    required_cols: List[str],
) -> Tuple[pd.DataFrame, dict]:
    """Load CSV files with column validation and outlier detection.

    Args:
        csv_paths: List of CSV file paths
        required_cols: Required column names (must include 'inference_time')

    Returns:
        Tuple of (combined_dataframe, outlier_info)
    """
    all_data = []

    for csv_path in csv_paths:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in {csv_path}: {missing_cols}")

        df["source_file"] = csv_path.name
        all_data.append(df[required_cols + ["source_file"]])

    combined_df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

    # Detect outliers on inference_time
    _, outlier_info = detect_outliers_iqr(combined_df["inference_time"])
    print(f"Detected {outlier_info['count']} outliers (shown in boxplot, not removed)")

    return combined_df, outlier_info


def pearson_correlation(
    df: pd.DataFrame, x_col: str, y_col: str = "inference_time"
) -> Tuple[float, float]:
    """Calculate Pearson correlation between two columns.

    Args:
        df: DataFrame
        x_col: Column name for x variable
        y_col: Column name for y variable (default: inference_time)

    Returns:
        Tuple of (correlation, p_value)
    """
    if len(df) < 2:
        return 0.0, 1.0
    return stats.pearsonr(df[x_col], df[y_col])


def significance_stars(p_value: float) -> str:
    """Get significance stars for p-value."""
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < P_VALUE_THRESHOLD:
        return "*"
    return ""


def add_correlation_text(ax, correlation: float, p_value: float, n: int) -> None:
    """Add correlation statistics text box to plot.

    Args:
        ax: Matplotlib axes
        correlation: Pearson correlation coefficient
        p_value: P-value from correlation test
        n: Sample size
    """
    stars = significance_stars(p_value)
    stats_text = f"Pearson r = {correlation:.3f} {stars}\np = {p_value:.4f}\nn = {n}"

    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor=COLORS["background"],
            alpha=0.9,
            edgecolor=COLORS["primary"],
            linewidth=1,
        ),
    )


def add_boxplot_legend(ax, loc: str = "lower right") -> None:
    """Add standard boxplot legend.

    Args:
        ax: Matplotlib axes
        loc: Legend location (default: lower right)
    """
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    legend_elements = [
        Patch(
            facecolor=COLORS["primary"],
            alpha=PLOT_STYLE["alpha"],
            label="Distribution (IQR)",
        ),
        Line2D(
            [0], [0],
            color=COLORS["success"],
            linewidth=PLOT_STYLE["line_width"],
            label="Median",
        ),
    ]
    ax.legend(handles=legend_elements, loc=loc, framealpha=0.9)


def save_json(
    output_path: Path,
    analysis_type: str,
    input_files: List[str],
    correlation: float,
    p_value: float,
    outlier_info: dict,
    inference_time_stats: dict,
    sample_size: int,
    extra_data: dict = None,
) -> Path:
    """Save efficiency analysis results as JSON.

    Args:
        output_path: Path to save JSON file
        analysis_type: Type of analysis (e.g., 'efficiency_concern_count')
        input_files: List of input CSV files
        correlation: Pearson correlation coefficient
        p_value: P-value from correlation test
        outlier_info: Outlier detection results from detect_outliers_iqr
        inference_time_stats: Dict with mean, std, min, max
        sample_size: Number of samples
        extra_data: Additional data to include (optional)

    Returns:
        Path to saved JSON file
    """
    summary = {
        "analysis_info": {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "analysis_type": analysis_type,
            "input_files": input_files,
        },
        "correlation_analysis": {
            "pearson_r": round(float(correlation), 4),
            "p_value": float(p_value),
            "significant": bool(p_value < P_VALUE_THRESHOLD),
        },
        "outlier_analysis": {
            "method": "IQR",
            "count": outlier_info["count"],
            "values": (
                [round(v, 2) for v in outlier_info["values"]]
                if outlier_info["values"]
                else []
            ),
        },
        "inference_time": {
            "mean": round(float(inference_time_stats["mean"]), 4),
            "std": round(float(inference_time_stats["std"]), 4),
            "min": round(float(inference_time_stats["min"]), 4),
            "max": round(float(inference_time_stats["max"]), 4),
            "unit": "seconds",
        },
        "sample_size": int(sample_size),
    }

    if extra_data:
        summary.update(extra_data)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return output_path
