"""
Plot utilities for RQ analysis.

Provides consistent styling across all visualization scripts:
- COLORS: Color palette for plots
- GROUP_COLORS: Colors for multi-category visualizations
- PLOT_STYLE: Common plot configuration
- boxplot_style: Function returning boxplot styling configuration
- setup_plot_style: Matplotlib rcParams setup
"""

import matplotlib.pyplot as plt
from typing import Optional

# Human‑readable display names for model labels in figures
# Keeps CSV/model keys intact while unifying how names appear in legends/ticks
DISPLAY_NAME_MAP = {
    "Qwen": "Qwen3-14B",
    "Qwen (FT)": "Qwen3-14B (LoRA)",
    "QwenFT": "Qwen3-14B (LoRA)",
    "LLM": "GPT-4.1",
    "SLM": "Qwen3-14B",
    "Fine-tuned SLM": "Qwen3-14B (LoRA)",
}


def display_model_name(name: str) -> str:
    """Map internal model key to canonical display label for figures."""
    return DISPLAY_NAME_MAP.get(name, name)

# Color palette for consistent styling across all RQ visualizations
COLORS = {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "accent": "#F18F01",
    "success": "#C73E1D",
    "background": "#F5F5F5",
    "text": "#2C3E50",
}

# Group colors for multi-category visualizations (boxplots, grouped charts)
GROUP_COLORS = [
    COLORS["primary"],
    COLORS["secondary"],
    COLORS["accent"],
    COLORS["success"],
    COLORS["text"],
]

# Plot styling configuration
PLOT_STYLE = {
    "figure_size": (10, 6),
    "dpi": 300,
    "line_width": 2,
    "marker_size": 60,
    "alpha": 0.7,
    "grid_alpha": 0.3,
}


def boxplot_style(
    box_color: Optional[str] = None,
    widths: float = 0.5,
) -> dict:
    """Get boxplot style configuration for matplotlib.

    Args:
        box_color: Facecolor for boxes and fliers. If None, uses COLORS["primary"].
        widths: Width of boxes. Default 0.5.

    Returns:
        Dict ready to be unpacked into ax.boxplot(**style).

    Example:
        ax.boxplot(data, **boxplot_style())
        ax.boxplot(data, **boxplot_style(box_color=GROUP_COLORS[i], widths=0.4))
    """
    color = box_color or COLORS["primary"]

    return {
        "patch_artist": True,
        "widths": widths,
        "showfliers": True,
        "whis": 1.5,
        "boxprops": {"facecolor": color, "alpha": PLOT_STYLE["alpha"]},
        "medianprops": {"color": COLORS["success"], "linewidth": 2},
        "whiskerprops": {"color": COLORS["text"], "linewidth": 1},
        "capprops": {"color": COLORS["text"], "linewidth": 1},
        "flierprops": {
            "marker": "o",
            "markerfacecolor": color,
            "markersize": 4,
            "alpha": 0.7,
            "markeredgecolor": "none",
        },
    }


def setup_plot_style():
    """Setup consistent matplotlib plot styling for all RQ visualizations."""
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
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
