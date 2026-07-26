#!/usr/bin/env python3
"""
Exploratory analysis for rebuttal of reviewer comment C1.1 ("synthetic
concatenated tangles may not reflect real interleaved tangling").

Compares structural properties of two sets of tangled-commit diffs:
  Set 1 (synthetic): our generated tangled commits, built by concatenating
      k distinct-concern atomic diffs from the same repo
      (data/tangled_ccs_dataset_{train,test}.csv).
  Set 2 (reference): 110 commits removed from the original CCS dataset
      during curation because they appeared to contain multiple concerns;
      used as an indicative reference set (single-annotator flag, not
      independently validated), following the reviewer's suggested
      comparison design (files / functions / lines)
      (results/analysis/tangling_comparison/real_tangled_shas.csv,
      joined to data/CCS Dataset.csv on sha).

Circularity control: 65 of the 110 Set 2 SHAs also appear as one of the
k atomics inside some Set 1 rows - 263/1750 (15%) of synthetic commits
share >=1 SHA with Set 2, making a naive comparison partially
self-referential. Every metric/statistic below is therefore computed
twice: once on the full Set 1 (as originally), and once on Set 1' = Set 1
minus those 263 overlapping rows (the "disjoint" variant). Both are
reported side by side - see metrics.csv's `set1_variant` column and
summary.md's two comparison tables.

Metrics computed identically for both sets, per commit diff:
  FILES:     number of files changed (count of `diff --git` sections).
             Set 1 additionally: cross-concern same-file overlap rate
             (whether atomics of different concern types in the same
             tangled commit touch a common file).
  FUNCTIONS: distinct function contexts parsed from hunk headers
             (`@@ -a,b +c,d @@ <context>`), function-context coverage
             (share of hunks with a non-empty context), and the share of
             commits with within-function co-editing (>=2 hunks landing
             in the same file+context).
  LINES:     hunks per commit, hunks per file, median line-gap between
             consecutive hunks in the same file, total changed lines (+/-).

Statistics: Mann-Whitney U (two-sided) + Cliff's delta (implemented
directly) comparing Set 1 (k=2 stratum, and all k>=2 pooled) vs Set 2,
for every metric, with Holm correction across all tests. This is a
structural comparability check, not a validation of Set 1's realism -
Set 2 is an indicative reference, not ground truth.

Read-only reporting script: never writes to datasets/data/.
Run with: uv run python datasets/scripts/compare_synthetic_vs_real_tangled.py
"""

import ast
import json
import re
import sys
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "datasets" / "data"

TRAIN_CSV_PATH = DATA_DIR / "tangled_ccs_dataset_train.csv"
TEST_CSV_PATH = DATA_DIR / "tangled_ccs_dataset_test.csv"
CCS_DATASET_PATH = DATA_DIR / "CCS Dataset.csv"
POOL_CSV_PATH = DATA_DIR / "repo_grouped_pool.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis" / "tangling_comparison"
METRICS_CSV_PATH = OUTPUT_DIR / "metrics.csv"
SUMMARY_MD_PATH = OUTPUT_DIR / "summary.md"
FIGURE_PATH = OUTPUT_DIR / "tangling_comparison.png"
# 110 commits removed from the original CCS dataset during curation because
# they appeared to contain multiple concerns; used as an indicative reference
# set for Set 2 (single-annotator flag, not independently validated) - not a
# pipeline input, lives alongside this analysis's own outputs.
REAL_TANGLED_SHAS_PATH = OUTPUT_DIR / "real_tangled_shas.csv"

sys.path.insert(0, str(PROJECT_ROOT / "RQ" / "analysis"))
from plot_utils import COLORS, GROUP_COLORS, HATCH_PATTERNS, PLOT_STYLE, boxplot_style, setup_plot_style  # noqa: E402

CONCERN_COUNTS: List[int] = [1, 2, 3, 4, 5]
HEADLINE_K = 2  # primary Set-1 stratum compared against Set 2
CHART_DPI = 200

# Metrics reported per commit and statistically tested (Set 1 vs Set 2)
METRIC_LABELS: Dict[str, str] = {
    "n_files": "Files changed",
    "n_hunks": "Hunks per commit",
    "hunks_per_file": "Hunks per file",
    "distinct_function_contexts": "Distinct function contexts",
    "function_context_coverage": "Function-context coverage",
    "within_function_coediting": "Within-function co-editing (0/1)",
    "median_line_gap": "Median line-gap (same file)",
    "total_changed_lines": "Total changed lines (+/-)",
}

# Effect-size magnitude thresholds (Romano et al. 2006)
CLIFFS_DELTA_THRESHOLDS = [(0.147, "negligible"), (0.33, "small"), (0.474, "medium")]

DIFF_FILE_SPLIT_RE = re.compile(r"(?=^diff --git )", re.MULTILINE)
DIFF_FILE_HEADER_RE = re.compile(r"^diff --git a/(.*?) b/(.*)$", re.MULTILINE)
HUNK_HEADER_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@[ \t]?(.*)$", re.MULTILINE)
ADDED_LINE_RE = re.compile(r"^\+(?!\+\+)", re.MULTILINE)
REMOVED_LINE_RE = re.compile(r"^-(?!--)", re.MULTILINE)


# --- Stage 1: unified-diff parsing -----------------------------------------


def split_diff_into_files(diff_text: str) -> List[str]:
    """Split one combined unified diff into per-file blocks."""
    blocks = DIFF_FILE_SPLIT_RE.split(diff_text)
    return [b for b in blocks if b.strip().startswith("diff --git ")]


def file_path_from_block(block: str) -> Optional[str]:
    """Extract the changed file's path from a `diff --git a/X b/Y` header."""
    m = DIFF_FILE_HEADER_RE.match(block)
    if not m:
        return None
    a_path, b_path = m.group(1), m.group(2)
    return a_path if b_path == "/dev/null" else b_path


def extract_file_sets(diff_text: str) -> set:
    """Return the set of file paths touched by a combined unified diff."""
    files = set()
    for block in split_diff_into_files(diff_text):
        path = file_path_from_block(block)
        if path:
            files.add(path)
    return files


def compute_commit_metrics(diff_text: str) -> Dict[str, float]:
    """Compute the FILES/FUNCTIONS/LINES metric set for one commit diff."""
    blocks = split_diff_into_files(diff_text)
    n_files = len(blocks)

    n_hunks = 0
    hunks_with_context = 0
    context_groups: Dict[Tuple[str, str], int] = {}
    line_gap_samples: List[int] = []
    added_lines = 0
    removed_lines = 0

    for block in blocks:
        path = file_path_from_block(block) or ""
        added_lines += len(ADDED_LINE_RE.findall(block))
        removed_lines += len(REMOVED_LINE_RE.findall(block))

        hunk_starts = []
        for m in HUNK_HEADER_RE.finditer(block):
            n_hunks += 1
            new_start = int(m.group(1))
            context = m.group(2).strip()
            hunk_starts.append(new_start)
            if context:
                hunks_with_context += 1
                key = (path, context)
                context_groups[key] = context_groups.get(key, 0) + 1

        hunk_starts.sort()
        for a, b in zip(hunk_starts, hunk_starts[1:]):
            line_gap_samples.append(b - a)

    distinct_function_contexts = len(context_groups)
    within_function_coediting = 1 if any(v >= 2 for v in context_groups.values()) else 0
    function_context_coverage = (hunks_with_context / n_hunks) if n_hunks else np.nan
    hunks_per_file = (n_hunks / n_files) if n_files else np.nan
    median_line_gap = median(line_gap_samples) if line_gap_samples else np.nan
    total_changed_lines = added_lines + removed_lines

    return {
        "n_files": n_files,
        "n_hunks": n_hunks,
        "hunks_per_file": hunks_per_file,
        "distinct_function_contexts": distinct_function_contexts,
        "function_context_coverage": function_context_coverage,
        "within_function_coediting": within_function_coediting,
        "median_line_gap": median_line_gap,
        "total_changed_lines": total_changed_lines,
    }


# --- Stage 2: Set 1 (synthetic) construction -------------------------------


def load_synthetic_set(pool_diff_by_sha: Dict[str, str], reference_sha_set: set) -> pd.DataFrame:
    """Load train+test tangled CSVs, compute per-commit metrics and, for
    k>=2 rows, the cross-concern same-file overlap flag. Also tags every
    row with `overlaps_reference_set`: whether any of its k atomic SHAs is
    one of the 110 Set-2 reference SHAs (circularity control - see
    `run_comparisons`'s disjoint variant)."""
    rows = []
    for split_name, csv_path in [("train", TRAIN_CSV_PATH), ("test", TEST_CSV_PATH)]:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            k = int(row["concern_count"])
            shas = ast.literal_eval(row["shas"])
            types = ast.literal_eval(row["types"])
            combined_diff = "\n".join(json.loads(row["diff"]))
            metrics = compute_commit_metrics(combined_diff)
            overlaps_reference_set = bool(set(shas) & reference_sha_set)

            overlap = np.nan
            if k >= 2:
                # File sets per atomic, looked up from the pool by sha (the
                # authoritative per-atomic source, independent of how the
                # combined `diff` column was concatenated).
                file_sets = [extract_file_sets(pool_diff_by_sha[s]) for s in shas]
                overlap = 0
                for i in range(len(shas)):
                    for j in range(i + 1, len(shas)):
                        if types[i] != types[j] and file_sets[i] & file_sets[j]:
                            overlap = 1
                            break
                    if overlap:
                        break

            rows.append(
                {
                    "set": "synthetic",
                    "split": split_name,
                    "concern_count": k,
                    "cross_concern_file_overlap": overlap,
                    "overlaps_reference_set": overlaps_reference_set,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


# --- Stage 3: Set 2 (reference set) construction ---------------------------


def load_real_tangled_set() -> Tuple[pd.DataFrame, Dict[str, int], set]:
    """Load real_tangled_shas.csv (110 commits removed from the original CCS
    dataset during curation because they appeared to contain multiple
    concerns - an indicative reference set, single-annotator flag, not
    independently validated), optionally filter by a reason/category field
    if present (none exists on disk), join to CCS Dataset.csv on sha,
    compute per-commit metrics. Also returns the kept SHA set for the
    Set-1 circularity control (see `load_synthetic_set`)."""
    real_shas = pd.read_csv(REAL_TANGLED_SHAS_PATH)
    join_report = {"real_shas_total": len(real_shas)}

    reason_cols = [c for c in real_shas.columns if "reason" in c.lower()]
    if reason_cols:
        # A reason/category field exists: keep only rows indicating tangling/mixed purposes.
        col = reason_cols[0]
        keyword_mask = real_shas[col].astype(str).str.contains(
            "tangl|mixed|multiple", case=False, na=False
        )
        kept = real_shas[keyword_mask]
        join_report["reason_column"] = col
        join_report["kept_by_reason"] = len(kept)
        join_report["dropped_by_reason"] = len(real_shas) - len(kept)
    else:
        # No reason/category column exists in real_tangled_shas.csv (columns =
        # ['sha'] only) - all rows are kept, and this is reported explicitly
        # rather than silently assumed.
        kept = real_shas
        join_report["reason_column"] = None
        join_report["kept_by_reason"] = len(kept)
        join_report["dropped_by_reason"] = 0

    ccs = pd.read_csv(CCS_DATASET_PATH)
    merged = kept.merge(ccs[["sha", "git_diff"]], on="sha", how="left", indicator=True)
    join_report["join_matched"] = int((merged["_merge"] == "both").sum())
    join_report["join_unmatched"] = int((merged["_merge"] != "both").sum())

    matched = merged[merged["_merge"] == "both"]
    rows = []
    for _, row in matched.iterrows():
        metrics = compute_commit_metrics(str(row["git_diff"]))
        rows.append({"set": "real", "split": None, "concern_count": np.nan, **metrics})
    return pd.DataFrame(rows), join_report, set(kept["sha"])


# --- Stage 4: statistics -----------------------------------------------------


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta: P(x>y) - P(x<y), computed via sorted-rank counting."""
    x = np.asarray(x, dtype=float)
    y_sorted = np.sort(np.asarray(y, dtype=float))
    ny = len(y_sorted)
    more = 0
    less = 0
    for xi in x:
        more += ny - np.searchsorted(y_sorted, xi, side="right")
        less += np.searchsorted(y_sorted, xi, side="left")
    return (more - less) / (len(x) * ny)


def delta_magnitude(delta: float) -> str:
    """Map |Cliff's delta| to a magnitude label."""
    ad = abs(delta)
    for threshold, label in CLIFFS_DELTA_THRESHOLDS:
        if ad < threshold:
            return label
    return "large"


def iqr(values: np.ndarray) -> Tuple[float, float]:
    return float(np.percentile(values, 25)), float(np.percentile(values, 75))


def holm_correction(pvalues: List[float]) -> List[float]:
    """Holm-Bonferroni step-down correction. Returns adjusted p-values in
    the original input order."""
    n = len(pvalues)
    order = sorted(range(n), key=lambda i: pvalues[i])
    adjusted = [0.0] * n
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (n - rank) * pvalues[idx])
        running_max = max(running_max, val)
        adjusted[idx] = running_max
    return adjusted


def run_comparisons(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Mann-Whitney U + Cliff's delta for every metric, comparing Set 1
    (k=2 stratum, and all k>=2 pooled) vs Set 2. Holm-corrects across the
    full set of tests."""
    real = metrics_df[metrics_df["set"] == "real"]
    synth_k2 = metrics_df[(metrics_df["set"] == "synthetic") & (metrics_df["concern_count"] == HEADLINE_K)]
    synth_kge2 = metrics_df[(metrics_df["set"] == "synthetic") & (metrics_df["concern_count"] >= 2)]

    comparisons = [("k=2 vs real", synth_k2), ("k>=2 pooled vs real", synth_kge2)]
    results = []
    for metric in METRIC_LABELS:
        for comp_name, synth_subset in comparisons:
            x = synth_subset[metric].dropna().to_numpy()
            y = real[metric].dropna().to_numpy()
            if len(x) < 2 or len(y) < 2:
                continue
            u_stat, p_value = mannwhitneyu(x, y, alternative="two-sided")
            delta = cliffs_delta(x, y)
            x_q1, x_q3 = iqr(x)
            y_q1, y_q3 = iqr(y)
            results.append(
                {
                    "metric": metric,
                    "comparison": comp_name,
                    "n_synthetic": len(x),
                    "median_synthetic": float(np.median(x)),
                    "iqr_synthetic": f"[{x_q1:.2f}, {x_q3:.2f}]",
                    "n_real": len(y),
                    "median_real": float(np.median(y)),
                    "iqr_real": f"[{y_q1:.2f}, {y_q3:.2f}]",
                    "U": float(u_stat),
                    "p_raw": float(p_value),
                    "cliffs_delta": float(delta),
                    "magnitude": delta_magnitude(delta),
                }
            )

    results_df = pd.DataFrame(results)
    results_df["p_holm"] = holm_correction(results_df["p_raw"].tolist())
    return results_df


# --- Stage 5: figure ----------------------------------------------------------


def render_figure(metrics_df: pd.DataFrame, output_path: Path) -> Path:
    """Per-metric box panels: Set 1 (k=2, full variant) vs Set 2 (indicative
    reference), dpi=200. A structural comparability check, not a claim that
    Set 2 validates Set 1's realism."""
    setup_plot_style()
    real = metrics_df[metrics_df["set"] == "real"]
    synth_k2 = metrics_df[(metrics_df["set"] == "synthetic") & (metrics_df["concern_count"] == HEADLINE_K)]

    metric_names = list(METRIC_LABELS.keys())
    n = len(metric_names)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.2 * nrows))
    axes = np.atleast_1d(axes).flatten()

    labels = [f"Set 1 (k={HEADLINE_K}, synthetic)", "Set 2 (indicative reference)"]
    colors = [GROUP_COLORS[0], GROUP_COLORS[1]]
    hatches = [HATCH_PATTERNS[0], HATCH_PATTERNS[1]]

    for ax, metric in zip(axes, metric_names):
        data = [synth_k2[metric].dropna().to_numpy(), real[metric].dropna().to_numpy()]
        for i, d in enumerate(data):
            bp = ax.boxplot([d], positions=[i], **boxplot_style(box_color=colors[i], hatch=hatches[i]))
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Set 1", "Set 2"])
        ax.set_title(METRIC_LABELS[metric], fontsize=12, fontweight="bold")
        ax.grid(True, axis="y", alpha=PLOT_STYLE["grid_alpha"], linestyle="-", linewidth=0.5)
        ax.set_axisbelow(True)

    for ax in axes[n:]:
        ax.axis("off")

    from matplotlib.patches import Patch

    legend_patches = [
        Patch(facecolor=colors[i], alpha=PLOT_STYLE["alpha"], edgecolor=COLORS["text"],
              linewidth=1.5, hatch=hatches[i] if hatches[i] else None, label=labels[i])
        for i in range(2)
    ]
    fig.legend(handles=legend_patches, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02), framealpha=0.9)
    fig.suptitle("Synthetic (Set 1, k=2) vs Indicative Reference Set (Set 2): Structural Comparability Check", y=1.06, fontweight="bold")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=CHART_DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    return output_path


# --- Stage 6: summary table ---------------------------------------------------


def _comparisons_table_lines(comparisons_df: pd.DataFrame) -> List[str]:
    header = ["metric", "comparison", "n_synth", "median_synth", "IQR_synth", "n_real", "median_real", "IQR_real", "U", "p_raw", "p_holm", "cliffs_delta", "magnitude"]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for _, r in comparisons_df.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    METRIC_LABELS[r["metric"]],
                    r["comparison"],
                    str(r["n_synthetic"]),
                    f"{r['median_synthetic']:.2f}",
                    r["iqr_synthetic"],
                    str(r["n_real"]),
                    f"{r['median_real']:.2f}",
                    r["iqr_real"],
                    f"{r['U']:.1f}",
                    f"{r['p_raw']:.4g}",
                    f"{r['p_holm']:.4g}",
                    f"{r['cliffs_delta']:.3f}",
                    r["magnitude"],
                ]
            )
            + " |"
        )
    return lines


def _conclusion_shift_lines(variant_results: Dict[str, pd.DataFrame]) -> List[str]:
    """Compare magnitude label + significance (p_holm < 0.05) per
    metric/comparison cell between the full and disjoint variants."""
    full_df = variant_results["full"].set_index(["metric", "comparison"])
    disjoint_df = variant_results["disjoint"].set_index(["metric", "comparison"])
    shifts = []
    for key in full_df.index:
        if key not in disjoint_df.index:
            continue
        m_full, m_disj = full_df.loc[key, "magnitude"], disjoint_df.loc[key, "magnitude"]
        sig_full = full_df.loc[key, "p_holm"] < 0.05
        sig_disj = disjoint_df.loc[key, "p_holm"] < 0.05
        if m_full != m_disj or sig_full != sig_disj:
            shifts.append((key, m_full, m_disj, sig_full, sig_disj))

    lines = ["## Do conclusions change between the full and disjoint variants?", ""]
    if shifts:
        lines.append(
            f"- {len(shifts)} of {len(full_df)} metric/comparison cells shift "
            "(magnitude label and/or Holm-corrected significance) between variants:"
        )
        for (metric, comp), m_full, m_disj, sig_full, sig_disj in shifts:
            lines.append(
                f"  - {METRIC_LABELS[metric]} ({comp}): {m_full} "
                f"(p_holm {'<' if sig_full else '>='} 0.05) -> {m_disj} "
                f"(p_holm {'<' if sig_disj else '>='} 0.05)"
            )
    else:
        lines.append(
            "- No metric/comparison cell changes magnitude label or Holm-corrected "
            "significance (p_holm < 0.05) between the full and disjoint variants - "
            "excluding the 263 circularity-overlapping rows does not change any conclusion."
        )
    return lines


def build_summary_markdown(
    variant_results: Dict[str, pd.DataFrame],
    join_report: Dict[str, int],
    n_overlap: int,
) -> str:
    lines = ["# Synthetic vs Reference Set Structural Comparison", ""]
    lines.append(
        "Set 2 is 110 commits removed from the original CCS dataset during "
        "curation because they appeared to contain multiple concerns. This is "
        "an **indicative reference set** (single-annotator flag, not "
        "independently validated), following the reviewer's suggested "
        "comparison design (files / functions / lines). What follows is a "
        "**structural comparability check**, not a claim that Set 2 validates "
        "Set 1's realism or constitutes ground truth."
    )
    lines.append("")

    lines.append("## Set 2 (reference set) construction")
    lines.append(f"- real_tangled_shas.csv total rows: {join_report['real_shas_total']}")
    if join_report["reason_column"]:
        lines.append(f"- reason column found: `{join_report['reason_column']}`")
        lines.append(f"- kept by reason filter: {join_report['kept_by_reason']}, dropped: {join_report['dropped_by_reason']}")
    else:
        lines.append(
            "- no reason/category column exists in real_tangled_shas.csv "
            f"(columns = ['sha'] only) - all {join_report['kept_by_reason']} rows kept, 0 dropped"
        )
    lines.append(f"- join to CCS Dataset.csv on sha: matched {join_report['join_matched']}, unmatched {join_report['join_unmatched']}")
    lines.append("")

    lines.append("## Circularity control")
    lines.append(
        f"- {n_overlap} synthetic (Set 1) rows share >=1 SHA with Set 2 - these "
        "are excluded from the disjoint variant (Set 1') reported below"
    )
    lines.append("")

    lines.append("## Mann-Whitney U + Cliff's delta - full Set 1 (as originally, includes rows overlapping with Set 2)")
    lines.append("")
    lines.extend(_comparisons_table_lines(variant_results["full"]))
    lines.append("")

    lines.append("## Mann-Whitney U + Cliff's delta - disjoint Set 1' (circularity-excluded)")
    lines.append("")
    lines.extend(_comparisons_table_lines(variant_results["disjoint"]))
    lines.append("")

    lines.extend(_conclusion_shift_lines(variant_results))
    lines.append("")

    return "\n".join(lines)


# --- Main ---------------------------------------------------------------------


def main() -> None:
    print("Loading atomic pool (per-sha git_diff lookup)...")
    pool = pd.read_csv(POOL_CSV_PATH)
    pool_diff_by_sha = dict(zip(pool["sha"], pool["git_diff"]))

    print("Building Set 2 (reference set: commits removed during curation for appearing multi-concern)...")
    real_df, join_report, reference_sha_set = load_real_tangled_set()
    print(f"  Set 2: {len(real_df)} commits after join")
    print(f"  join report: {join_report}")

    print("Building Set 1 (synthetic tangled commits)...")
    synthetic_df = load_synthetic_set(pool_diff_by_sha, reference_sha_set)
    print(f"  Set 1: {len(synthetic_df)} commits (train+test, k=1..5)")

    overlap_mask = synthetic_df["overlaps_reference_set"].astype(bool)
    n_overlap = int(overlap_mask.sum())
    print(
        f"  Circularity check: {n_overlap} synthetic rows sharing >=1 SHA with the "
        "reference set were excluded from the disjoint variant"
    )

    variant_results: Dict[str, pd.DataFrame] = {}
    variant_metrics_frames = []
    for variant_name, synth_subset in [
        ("full", synthetic_df),
        ("disjoint", synthetic_df[~overlap_mask]),
    ]:
        variant_metrics_df = pd.concat([synth_subset, real_df], ignore_index=True)
        variant_metrics_df["set1_variant"] = variant_name
        variant_metrics_frames.append(variant_metrics_df)

        print(f"Running Mann-Whitney U + Cliff's delta (Holm-corrected) [{variant_name} variant]...")
        variant_results[variant_name] = run_comparisons(variant_metrics_df)

    metrics_df = pd.concat(variant_metrics_frames, ignore_index=True)

    print("Rendering comparison figure (full variant)...")
    fig_path = render_figure(metrics_df[metrics_df["set1_variant"] == "full"], FIGURE_PATH)
    print(f"  Figure saved to: {fig_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(METRICS_CSV_PATH, index=False)
    print(f"Metrics CSV saved to: {METRICS_CSV_PATH}")

    summary_md = build_summary_markdown(variant_results, join_report, n_overlap)
    SUMMARY_MD_PATH.write_text(summary_md)
    print(f"Summary saved to: {SUMMARY_MD_PATH}")

    print("\n" + summary_md)

    # Set-1-only metric: cross-concern same-file overlap rate (full variant, no
    # Set-2 equivalent, since real commits have no concern labels to check
    # "cross-concern" against)
    overlap = synthetic_df[synthetic_df["concern_count"] >= 2]["cross_concern_file_overlap"]
    print(f"\nSet 1 cross-concern same-file overlap rate (k>=2, full): {overlap.mean():.4f} ({int(overlap.sum())}/{len(overlap)})")


if __name__ == "__main__":
    main()
