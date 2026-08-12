"""The C1.1 structural-proximity figure.

`forest` is the manuscript figure and the default output: one panel carrying
all three proximity measures -- same file, same directory, directory depth --
against both nulls, so the exploratory study costs the paper a single figure.
The claim reads in one glance: every observed dot on the 1.0 line, every
repository-free diamond far to its left.

`--exploratory` additionally renders three working views that were used to
check the headline and are not meant for the manuscript: `ladder` (the graded
shared-depth distribution), `bars` (the same numbers as percentages), and
`by_k` (concern share stratified by k).

Reads the JSON written by `proximity_study`; runs no analysis of its own.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from plot_utils import COLORS, HATCH_PATTERNS, PLOT_STYLE, setup_plot_style

Artifact = dict[str, Any]
DPI = PLOT_STYLE["dpi"]

SHORT_LABELS = {
    "commit_file_any": "Commit\nsame file",
    "commit_directory_any": "Commit\nsame dir.",
    "mean_file_share": "Concern share\nsame file",
    "mean_directory_share": "Concern share\nsame dir.",
    "pair_file_rate": "Pair\nsame file",
    "pair_directory_rate": "Pair\nsame dir.",
}
PERCENT_KEYS = tuple(SHORT_LABELS)
# The seven estimands are three measures asked three ways, not seven findings.
# Grouping them says so on the figure instead of in the caption.
GROUPS = (
    ("Same file", (("At least one pair", "commit_file_any"),
                   ("Typical pair", "pair_file_rate"),
                   ("Concern share", "mean_file_share"))),
    ("Same directory", (("At least one pair", "commit_directory_any"),
                        ("Typical pair", "pair_directory_rate"),
                        ("Concern share", "mean_directory_share"))),
    ("Directory depth", (("Mean deepest level", "mean_max_depth"),)),
)


def _forest_rows() -> list[tuple[str, str | None]]:
    """Row order top to bottom: a bold heading, its estimands, then a blank spacer.

    A row with no key is a heading or a spacer and carries no data.
    """
    rows: list[tuple[str, str | None]] = []
    for index, (title, members) in enumerate(GROUPS):
        if index:
            rows.append(("", None))
        rows.append((title, None))
        rows.extend((f"    {family}", key) for family, key in members)
    return rows


def _summary(artifact: Artifact, name: str) -> dict[str, Any]:
    """Nulls nest their summary under a key; observed and per-k blocks are bare."""
    block = artifact[name]
    return block.get("summary", block)


def _interval(artifact: Artifact, name: str) -> dict[str, Sequence[float]]:
    return artifact[name]["interval"]


def plot_forest(artifact: Artifact, out: Path) -> None:
    """Every estimand relative to its own same-repository null mean.

    Dividing by the null mean is what lets a percentage and a path depth share
    one axis. The 1.0 line is "exactly what the sampling constraints produce";
    the log scale keeps the repository-free null visible without crushing it.
    """
    observed = _summary(artifact, "observed")
    null, band = _summary(artifact, "null_same_repo"), _interval(artifact, "null_same_repo")
    free = _summary(artifact, "null_repo_free")
    rows = _forest_rows()
    figure, axes = plt.subplots(figsize=(10, 7))
    for row, (_text, key) in enumerate(rows):
        if key is None:
            continue
        scale = null[key]
        low, high = band[key][0] / scale, band[key][1] / scale
        _ = axes.plot([low, high], [row, row], color=COLORS["primary"], linewidth=7, alpha=0.35,
                      solid_capstyle="butt", zorder=2)
        # The guide line runs to the annotation column, so it is drawn per estimand
        # row rather than by the y grid, which would also stripe the headings.
        _ = axes.axhline(row, color=COLORS["text"], alpha=PLOT_STYLE["grid_alpha"], linewidth=0.8, zorder=0)
        _ = axes.scatter([observed[key] / scale], [row], s=110, marker="o",
                         color=COLORS["secondary"], zorder=5)
        _ = axes.scatter([free[key] / scale], [row], s=110, marker="D", facecolors="none",
                         edgecolors=COLORS["text"], linewidths=1.6, zorder=4)
        # The ratio axis hides magnitude, so the absolute observed value rides along.
        unit = "" if key == "mean_max_depth" else "%"
        _ = axes.annotate(f"{observed[key]:.2f}{unit}", (2.6, row), va="center", fontsize=13,
                          color=COLORS["text"], annotation_clip=False)
    _ = axes.axvline(1.0, color=COLORS["text"], linestyle="--", linewidth=1.2, zorder=1)
    _ = axes.set_xscale("log")
    _ = axes.set_xlim(0.06, 2.2)
    _ = axes.set_xticks([0.1, 0.25, 0.5, 1.0, 2.0])
    _ = axes.set_xticklabels(["0.1x", "0.25x", "0.5x", "1x", "2x"])
    _ = axes.set_yticks(range(len(rows)))
    _ = axes.set_yticklabels([text for text, _key in rows])
    for label, (_text, key) in zip(axes.get_yticklabels(), rows, strict=True):
        if key is None:
            label.set_fontweight("bold")
    _ = axes.set_ylim(len(rows) - 0.4, -0.6)  # replaces invert_yaxis; trims the dead margin
    _ = axes.set_xlabel("Value relative to the same-repository null mean")
    _ = axes.set_title("Structural proximity is fully explained by the sampling constraints")
    # Every mark is drawn once per row, so the legend is built from proxies rather
    # than from labels attached to whichever row happened to be plotted first.
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=11, color=COLORS["secondary"]),
        plt.Line2D([], [], marker="D", linestyle="", markersize=11, markerfacecolor="none",
                   markeredgecolor=COLORS["text"], markeredgewidth=1.6),
        plt.Line2D([], [], color=COLORS["primary"], linewidth=7, alpha=0.35),
    ]
    _ = axes.legend(handles, ["Observed corpus", "Null: repository constraint dropped",
                              "Null: repository and type fixed (95%)"],
                    loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False, fontsize=12)
    _ = axes.grid(axis="x", alpha=PLOT_STYLE["grid_alpha"])
    _ = axes.grid(axis="y", visible=False)
    axes.tick_params(axis="y", length=0)
    figure.tight_layout()
    figure.savefig(out, dpi=DPI)
    plt.close(figure)


def plot_ladder(artifact: Artifact, out: Path) -> None:
    """How deep the shared directory prefix goes, over all 7,000 concern pairs.

    Drawn as a survival curve rather than a histogram. The depth-0 bin holds
    four fifths of the pairs, which on a linear histogram flattens every rung
    that actually distinguishes the three conditions. Reading "at least depth d"
    on a log axis keeps the rare deep rungs legible, and the two curves that lie
    on top of each other are the finding.
    """
    series = [
        ("Observed corpus", "observed", COLORS["secondary"], "-", "o"),
        ("Null: repository and type fixed", "null_same_repo", COLORS["primary"], "--", "s"),
        ("Null: repository constraint dropped", "null_repo_free", COLORS["accent"], ":", "D"),
    ]
    pairs = int(_summary(artifact, "observed")["pairs"])
    curves: dict[str, list[float]] = {}
    for _label, name, *_style in series:
        histogram = _summary(artifact, name)["depth_histogram"]
        curves[name] = [sum(histogram[depth:]) for depth in range(len(histogram))]
    # Past the point where the observed corpus holds fewer than ten pairs, a log
    # axis magnifies single-pair noise into an apparent divergence. The tail is
    # cut and its total stated instead of being drawn as if it were resolved.
    observed_curve = curves["observed"]
    span = next((depth for depth, percent in enumerate(observed_curve) if percent / 100 * pairs < 10),
                len(observed_curve) - 1)
    figure, axes = plt.subplots(figsize=PLOT_STYLE["figure_size"])
    for label, name, color, dashes, marker in series:
        survival = curves[name][:span]
        _ = axes.plot(range(len(survival)), survival, label=label, color=color, linestyle=dashes,
                      marker=marker, markersize=7, linewidth=PLOT_STYLE["line_width"])
    _ = axes.annotate(f"depth $\\geq${span}: {observed_curve[span]:.2f}% of observed pairs "
                      f"({round(observed_curve[span] * pairs / 100)} of {pairs:,})",
                      (0.5, 0.03), xycoords="axes fraction", ha="center", fontsize=12, color=COLORS["text"])
    _ = axes.set_yscale("log")
    _ = axes.set_xticks(range(span))
    _ = axes.set_xlabel("Shared directory prefix of at least this depth")
    _ = axes.set_ylabel("Concern pairs (%, log scale)")
    _ = axes.set_title("Sharing any directory at all is rare, and matches the constrained null")
    _ = axes.legend(frameon=True, loc="upper right")
    _ = axes.grid(alpha=PLOT_STYLE["grid_alpha"])
    figure.tight_layout()
    figure.savefig(out, dpi=DPI)
    plt.close(figure)


def plot_bars(artifact: Artifact, out: Path) -> None:
    """The percentage estimands as grouped bars, with the null's 95% interval."""
    observed = _summary(artifact, "observed")
    null, band = _summary(artifact, "null_same_repo"), _interval(artifact, "null_same_repo")
    free = _summary(artifact, "null_repo_free")
    positions = np.arange(len(PERCENT_KEYS))
    width = 0.27
    figure, axes = plt.subplots(figsize=(11, 6))
    _ = axes.bar(positions - width, [free[key] for key in PERCENT_KEYS], width,
                 label="Null: repository constraint dropped", color=COLORS["accent"],
                 hatch=HATCH_PATTERNS[2], edgecolor="white", linewidth=0.8)
    _ = axes.bar(positions, [observed[key] for key in PERCENT_KEYS], width, label="Observed corpus",
                 color=COLORS["secondary"], hatch=HATCH_PATTERNS[0], edgecolor="white", linewidth=0.8)
    heights = [null[key] for key in PERCENT_KEYS]
    errors = np.array([[null[key] - band[key][0] for key in PERCENT_KEYS],
                       [band[key][1] - null[key] for key in PERCENT_KEYS]])
    _ = axes.bar(positions + width, heights, width, yerr=errors, capsize=4,
                 label="Null: repository and type fixed (95%)", color=COLORS["primary"],
                 hatch=HATCH_PATTERNS[1], edgecolor="white", linewidth=0.8,
                 error_kw={"ecolor": COLORS["text"], "linewidth": 1.4})
    _ = axes.set_xticks(positions)
    _ = axes.set_xticklabels([SHORT_LABELS[key] for key in PERCENT_KEYS], fontsize=13)
    _ = axes.set_ylabel("Percent")
    _ = axes.set_title("Observed proximity matches the constrained null on every estimand")
    _ = axes.legend(frameon=True)
    _ = axes.grid(axis="y", alpha=PLOT_STYLE["grid_alpha"])
    figure.tight_layout()
    figure.savefig(out, dpi=DPI)
    plt.close(figure)


def plot_by_k(artifact: Artifact, out: Path) -> None:
    """Concern share by k: normalising the denominator does not remove k-inflation."""
    by_k = artifact["by_concern_count"]
    counts = sorted(by_k, key=int)
    positions = np.arange(len(counts))
    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 5))
    for axes, key, title in ((left, "mean_file_share", "Same file"), (right, "mean_directory_share", "Same directory")):
        _ = axes.bar(positions, [by_k[k][key] for k in counts], 0.7, color=COLORS["secondary"],
                     hatch=HATCH_PATTERNS[0], edgecolor="white", linewidth=0.8)
        _ = axes.set_xticks(positions)
        _ = axes.set_xticklabels([f"k={k}\n(n={int(by_k[k]['commits'])})" for k in counts], fontsize=12)
        _ = axes.set_title(title)
        _ = axes.set_ylabel("Concerns sharing ground with another (%)")
        _ = axes.grid(axis="y", alpha=PLOT_STYLE["grid_alpha"])
    figure.tight_layout()
    figure.savefig(out, dpi=DPI)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the C1.1 structural-proximity figure")
    _ = parser.add_argument("--artifact", type=Path, default=Path("results/structural_validity/proximity_seed43.json"))
    _ = parser.add_argument("--out", type=Path, default=Path("results/structural_validity/figures"))
    _ = parser.add_argument("--exploratory", action="store_true",
                            help="also render the working views that are not manuscript figures")
    arguments = parser.parse_args(argv)
    out: Path = arguments.out
    artifact = json.loads(Path(arguments.artifact).read_text(encoding="utf-8"))

    figures = [("forest", plot_forest)]
    if arguments.exploratory:
        figures += [("ladder", plot_ladder), ("bars", plot_bars), ("by_k", plot_by_k)]

    setup_plot_style()
    out.mkdir(parents=True, exist_ok=True)
    for name, render in figures:
        target = out / f"structural_proximity_{name}.png"
        render(artifact, target)
        print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
