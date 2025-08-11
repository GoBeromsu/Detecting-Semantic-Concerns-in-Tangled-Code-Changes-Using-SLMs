from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib
from datetime import datetime, timezone

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Constants
RESULTS_DIR: Path = Path(__file__).parent / "results"
ANALYSIS_DIR: Path = RESULTS_DIR / "analysis"
RELATION_VALUES: Tuple[str, str, str] = ("LLM>SLM", "Equal", "SLM>LLM")
METRICS: Tuple[str, str, str, str] = ("precision", "recall", "f1", "exact_match")
 


def load_and_prepare(csv_path: Path, role_suffix: str) -> pd.DataFrame:
    """Load CSV and prepare columns for comparison.

    Returns a DataFrame with: row_id, <metric>_<role_suffix> for each metric.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    row_id = df.index.astype(int)

    prepared = pd.DataFrame({"row_id": row_id})

    for metric in METRICS:
        if metric not in df.columns:
            raise ValueError(f"Missing metric column '{metric}' in {csv_path}")
        prepared[f"{metric}_{role_suffix}"] = pd.to_numeric(df[metric], errors="coerce").astype(float)

    return prepared



def save_metric_winners_csv(df: pd.DataFrame, metric: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{metric}_winner.csv"
    df.to_csv(out_path, index=False)
    return out_path


def save_overall_classification(merged: pd.DataFrame, output_dir: Path) -> Path:
    relation_cols = [f"relation_{m}" for m in METRICS]
    base = merged[["row_id", *relation_cols]].copy()

    llm_better = (base[relation_cols] == RELATION_VALUES[0]).sum(axis=1)
    equal = (base[relation_cols] == RELATION_VALUES[1]).sum(axis=1)
    slm_better = (base[relation_cols] == RELATION_VALUES[2]).sum(axis=1)

    base["llm_better"] = llm_better
    base["equal"] = equal
    base["slm_better"] = slm_better
    base["label"] = (
        "LLM_" + llm_better.astype(str)
        + "_Equal_" + equal.astype(str)
        + "_SLM_" + slm_better.astype(str)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "overall_classification.csv"
    base.to_csv(out_path, index=False)
    return out_path


def plot_scatter(metric_df: pd.DataFrame, metric: str, output_dir: Path) -> Path:
    x_col = f"{metric}_SLM"
    y_col = f"{metric}_LLM"

    color_map = {"LLM>SLM": "#2ca02c", "Equal": "#7f7f7f", "SLM>LLM": "#ff7f0e"}
    colors = metric_df[f"relation_{metric}"].map(color_map)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(metric_df[x_col], metric_df[y_col], c=colors, alpha=0.7, edgecolor="none")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(f"SLM: {metric}")
    ax.set_ylabel(f"LLM: {metric}")
    ax.set_title(f"Scatter — {metric}")

    # Guideline at 0.5 and y=x line
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.plot([0, 1], [0, 1], color="red", linestyle=":", linewidth=1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"scatter_{metric}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_winner_counts(metric_df: pd.DataFrame, metric: str, output_dir: Path) -> Path:
    relation_series = metric_df[f"relation_{metric}"].astype(str).copy()
    categories: List[str] = list(RELATION_VALUES)
    colors = ["#1f77b4", "#7f7f7f", "#ff7f0e"]

    counts = relation_series.value_counts().reindex(categories, fill_value=0).reset_index()
    counts.columns = ["relation", "count"]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(counts["relation"], counts["count"], color=colors)
    ax.set_xlabel("Comparison")
    ax.set_ylabel("Count")
    ax.set_title(f"Comparison distribution — {metric}")
    for i, v in enumerate(counts["count"]):
        ax.text(i, v + max(1, counts["count"].max() * 0.02), str(int(v)), ha="center")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"countplot_{metric}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_accuracy_categories(metric_df: pd.DataFrame, metric: str, output_dir: Path) -> Path:
    """Plot bar chart showing accuracy categories: both correct, only one correct, both wrong."""
    slm_col = f"{metric}_SLM"
    llm_col = f"{metric}_LLM"
    
    if slm_col not in metric_df.columns or llm_col not in metric_df.columns:
        raise KeyError(f"Missing expected columns: {slm_col}, {llm_col}")
    
    slm_vals = metric_df[slm_col].astype(float)
    llm_vals = metric_df[llm_col].astype(float)
    
    both_correct = (slm_vals == 1.0) & (llm_vals == 1.0)
    both_wrong = (slm_vals == 0.0) & (llm_vals == 0.0)
    only_llm_correct = (slm_vals == 0.0) & (llm_vals == 1.0)
    only_slm_correct = (slm_vals == 1.0) & (llm_vals == 0.0)
    
    categories = ["Both Correct", "Only LLM Correct", "Only SLM Correct", "Both Wrong"]
    counts = [
        both_correct.sum(),
        only_llm_correct.sum(),
        only_slm_correct.sum(),
        both_wrong.sum()
    ]
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]
    
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(categories, counts, color=colors)
    ax.set_xlabel("Prediction Result")
    ax.set_ylabel("Count")
    ax.set_title(f"{metric} Distribution")
    
    plt.xticks(rotation=30, ha='right')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"accuracy_bar_{metric}.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def run(
    slm_csv: Path,
    llm_csv: Path,
) -> None:
    # Per-run output directory under analysis
    created_at = datetime.now(tz=timezone.utc)
    run_dir = ANALYSIS_DIR / created_at.strftime("%Y%m%d_%H%M")

    # Load and prepare (no meta/shas)
    slm_df = load_and_prepare(slm_csv, role_suffix="SLM")
    llm_df = load_and_prepare(llm_csv, role_suffix="LLM")

    merged = pd.merge(slm_df, llm_df, on="row_id", how="inner")

    for metric in METRICS:
        slm_col = f"{metric}_SLM"
        llm_col = f"{metric}_LLM"
        relation_col = f"relation_{metric}"
        
        # Start with default assumption: SLM performs better
        relation = pd.Series(np.full(len(merged), RELATION_VALUES[2], dtype=object))  # "SLM>LLM"
        # Override where LLM actually performs better
        relation.loc[merged[llm_col] > merged[slm_col]] = RELATION_VALUES[0]  # "LLM>SLM"
        # Override where both perform equally
        relation.loc[merged[llm_col] == merged[slm_col]] = RELATION_VALUES[1]  # "Equal"
        merged[relation_col] = relation

    for metric in METRICS:
        cols = [
            "row_id",
            f"{metric}_SLM",
            f"{metric}_LLM",
            f"relation_{metric}",
        ]
        metric_df = merged[cols].copy()
        save_metric_winners_csv(metric_df, metric, output_dir=run_dir)
        plot_scatter(metric_df, metric, output_dir=run_dir)
        plot_winner_counts(merged, metric, output_dir=run_dir)
        if metric == "exact_match":
            plot_accuracy_categories(metric_df, metric, output_dir=run_dir)

    save_overall_classification(merged, output_dir=run_dir)


def main() -> None:

    # If not provided, try defaults used in the example
    slm_csv = RESULTS_DIR / "Phi-4_202508101101" / "huggingface" / "Phi-4_12288_zs_msg1.csv"
    llm_csv = RESULTS_DIR / "gpt" / "gpt-4.1-2025-04-14_12288_os_msg1.csv"

    run(slm_csv=slm_csv, llm_csv=llm_csv)


if __name__ == "__main__":
    main()


