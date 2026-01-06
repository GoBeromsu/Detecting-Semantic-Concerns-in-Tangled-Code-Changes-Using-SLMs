"""RQ Analysis package."""

from pathlib import Path

from .stats_utils import vargha_delaney_a, detect_outliers_iqr
from .plot_utils import COLORS, PLOT_STYLE, setup_plot_style

# Central path definitions
PROJECT_ROOT = Path(__file__).parent.parent.parent
ANALYSIS_OUTPUT_BASE = PROJECT_ROOT / "results" / "analysis"
CONFIG_PATH = Path(__file__).parent / "config.yaml"

__all__ = [
    # Paths
    "PROJECT_ROOT",
    "ANALYSIS_OUTPUT_BASE",
    "CONFIG_PATH",
    # Statistics
    "vargha_delaney_a",
    "detect_outliers_iqr",
    # Plotting
    "COLORS",
    "PLOT_STYLE",
    "setup_plot_style",
]
