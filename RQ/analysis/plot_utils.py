"""
Plot utilities for RQ analysis.

Provides consistent styling across all visualization scripts:
- COLORS: Color palette for plots
- PLOT_STYLE: Common plot configuration
- setup_plot_style: Matplotlib rcParams setup
"""

import matplotlib.pyplot as plt

# Color palette for consistent styling across all RQ visualizations
COLORS = {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "accent": "#F18F01",
    "success": "#C73E1D",
    "background": "#F5F5F5",
    "text": "#2C3E50",
}

# Plot styling configuration
PLOT_STYLE = {
    "figure_size": (10, 6),
    "dpi": 300,
    "line_width": 2,
    "marker_size": 60,
    "alpha": 0.7,
    "grid_alpha": 0.3,
}


def setup_plot_style():
    """Setup consistent matplotlib plot styling for all RQ visualizations."""
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
