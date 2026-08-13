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
    "Qwen": "Qwen3.6",
    "Qwen (FT)": "Qwen3.6-FT",
    "QwenFT": "Qwen3.6-FT",
    "LLM": "GPT-4.1",
    "SLM": "Qwen3.6",
    "Fine-tuned SLM": "Qwen3.6-FT",
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

# Hatch patterns for colorblind accessibility (distinguishable in grayscale)
HATCH_PATTERNS = [
    "",      # Solid (no hatch) - GPT-4.1
    "///",   # Dense diagonal - Qwen3
    "xxx",   # Dense cross - Qwen3-FT
    "|||",   # Vertical lines
    "...",   # Dots
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
    hatch: Optional[str] = None,
) -> dict:
    """Get boxplot style configuration for matplotlib.

    Args:
        box_color: Facecolor for boxes and fliers. If None, uses COLORS["primary"].
        widths: Width of boxes. Default 0.5.
        hatch: Hatch pattern for boxes (e.g., '//', '\\\\', 'xx'). Default None.

    Returns:
        Dict ready to be unpacked into ax.boxplot(**style).

    Example:
        ax.boxplot(data, **boxplot_style())
        ax.boxplot(data, **boxplot_style(box_color=GROUP_COLORS[i], widths=0.4))
        ax.boxplot(data, **boxplot_style(box_color=GROUP_COLORS[i], hatch=HATCH_PATTERNS[i]))
    """
    color = box_color or COLORS["primary"]

    boxprops = {
        "facecolor": color,
        "alpha": PLOT_STYLE["alpha"],
        "edgecolor": COLORS["text"],
        "linewidth": 1.5,
    }
    if hatch:
        boxprops["hatch"] = hatch

    return {
        "patch_artist": True,
        "widths": widths,
        "showfliers": True,
        "whis": 1.5,
        "boxprops": boxprops,
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


# =============================================================================
# Sequential Style Helpers (colorblind-friendly, order-based assignment)
# =============================================================================


def get_style_by_index(index: int) -> tuple[str, str]:
    """Get (color, hatch) tuple by sequential index.

    Ensures consistent colorblind-friendly styling across all visualizations.
    First item = primary + solid, second = secondary + ///, etc.

    Args:
        index: Zero-based index for the data series.

    Returns:
        Tuple of (color_hex, hatch_pattern).

    Example:
        color, hatch = get_style_by_index(0)  # ('#2E86AB', '')
        color, hatch = get_style_by_index(1)  # ('#A23B72', '///')
    """
    return GROUP_COLORS[index % len(GROUP_COLORS)], HATCH_PATTERNS[index % len(HATCH_PATTERNS)]


def get_color_palette(n: int = 3) -> list[str]:
    """Get color palette for n data series.

    Args:
        n: Number of colors needed.

    Returns:
        List of color hex codes.
    """
    return GROUP_COLORS[:n]


def get_hatch_palette(n: int = 3) -> list[str]:
    """Get hatch pattern palette for n data series.

    Args:
        n: Number of patterns needed.

    Returns:
        List of hatch pattern strings.
    """
    return HATCH_PATTERNS[:n]


def create_legend_patches(
    labels: list[str],
    loc: str = "upper right",
    use_display_names: bool = True,
) -> tuple[list, dict]:
    """Create legend patches with sequential colors and hatches.

    Args:
        labels: List of labels for legend entries.
        loc: Legend location (default: "upper right").
        use_display_names: Whether to apply display_model_name() to labels.

    Returns:
        Tuple of (legend_elements, legend_kwargs) for ax.legend().

    Example:
        patches, kwargs = create_legend_patches(["GPT-4.1", "Qwen", "QwenFT"])
        ax.legend(handles=patches, **kwargs)
    """
    from matplotlib.patches import Patch

    n = len(labels)
    colors = get_color_palette(n)
    hatches = get_hatch_palette(n)

    patches = [
        Patch(
            facecolor=colors[i],
            alpha=PLOT_STYLE["alpha"],
            edgecolor=COLORS["text"],
            linewidth=1.5,
            hatch=hatches[i] if hatches[i] else None,
            label=display_model_name(labels[i]) if use_display_names else labels[i],
        )
        for i in range(n)
    ]

    return patches, {"loc": loc, "framealpha": 0.9}
