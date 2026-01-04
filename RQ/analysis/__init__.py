"""RQ Analysis package."""

from .stats_utils import vargha_delaney_a, mann_whitney_u, detect_outliers_iqr
from .plot_utils import COLORS, PLOT_STYLE, setup_plot_style

__all__ = [
    # Statistics
    "vargha_delaney_a",
    "mann_whitney_u",
    "detect_outliers_iqr",
    # Plotting
    "COLORS",
    "PLOT_STYLE",
    "setup_plot_style",
]
