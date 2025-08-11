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
 


def load_and_prepare_data(csv_path: Path, model_type: str) -> pd.DataFrame:
    """Load CSV and prepare columns for comparison.

    Returns a DataFrame with: row_id, <metric>_<role_suffix> for each metric.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    row_id = df.index.astype(int)

    data = pd.DataFrame({"row_id": row_id})

    for metric in METRICS:
        if metric not in df.columns:
            raise ValueError(f"Missing metric column '{metric}' in {csv_path}")
        data[f"{metric}_{model_type}"] = pd.to_numeric(df[metric], errors="coerce").astype(float)

    return data



def save_metric_comparison_csv(comparison_data: pd.DataFrame, metric_name: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{metric_name}_results.csv"
    comparison_data.to_csv(csv_path, index=False)
    return csv_path


def save_overall_summary(comparison_results: pd.DataFrame, output_dir: Path) -> Path:
    comparison_columns = [f"relation_{m}" for m in METRICS]
    summary = comparison_results[["row_id", *comparison_columns]].copy()

    llm_wins_count = (summary[comparison_columns] == RELATION_VALUES[0]).sum(axis=1)
    equal_count = (summary[comparison_columns] == RELATION_VALUES[1]).sum(axis=1)
    slm_wins_count = (summary[comparison_columns] == RELATION_VALUES[2]).sum(axis=1)

    summary["llm_wins_count"] = llm_wins_count
    summary["equal_count"] = equal_count
    summary["slm_wins_count"] = slm_wins_count
    summary["overall_pattern"] = (
        "LLM_" + llm_wins_count.astype(str)
        + "_Equal_" + equal_count.astype(str)
        + "_SLM_" + slm_wins_count.astype(str)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "comparison_summary.csv"
    summary.to_csv(summary_path, index=False)
    return summary_path


def create_scatter_plot(comparison_data: pd.DataFrame, metric_name: str, output_dir: Path) -> Path:
    slm_values_col = f"{metric_name}_SLM"
    llm_values_col = f"{metric_name}_LLM"

    relation_colors = {"LLM>SLM": "#2ca02c", "Equal": "#7f7f7f", "SLM>LLM": "#ff7f0e"}
    point_colors = comparison_data[f"relation_{metric_name}"].map(relation_colors)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(comparison_data[slm_values_col], comparison_data[llm_values_col], c=point_colors, alpha=0.7, edgecolor="none")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(f"SLM: {metric_name}")
    ax.set_ylabel(f"LLM: {metric_name}")
    ax.set_title(f"Scatter — {metric_name}")

    # Guideline at 0.5 and y=x line
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.plot([0, 1], [0, 1], color="red", linestyle=":", linewidth=1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"{metric_name}_scatter.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def create_comparison_bar_chart(comparison_data: pd.DataFrame, metric_name: str, output_dir: Path) -> Path:
    relation_values = comparison_data[f"relation_{metric_name}"].astype(str).copy()
    comparison_categories: List[str] = list(RELATION_VALUES)
    bar_colors = ["#1f77b4", "#7f7f7f", "#ff7f0e"]

    category_counts = relation_values.value_counts().reindex(comparison_categories, fill_value=0).reset_index()
    category_counts.columns = ["comparison_type", "count"]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(category_counts["comparison_type"], category_counts["count"], color=bar_colors)
    ax.set_xlabel("Comparison Result")
    ax.set_ylabel("Count")
    ax.set_title(f"Comparison distribution — {metric_name}")
    for i, count_value in enumerate(category_counts["count"]):
        ax.text(i, count_value + max(1, category_counts["count"].max() * 0.02), str(int(count_value)), ha="center")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    chart_path = output_dir / f"{metric_name}_distribution.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    return chart_path


def create_accuracy_breakdown_chart(comparison_data: pd.DataFrame, metric_name: str, output_dir: Path) -> Path:
    """Plot bar chart showing accuracy categories: both correct, only one correct, both wrong."""
    slm_scores_col = f"{metric_name}_SLM"
    llm_scores_col = f"{metric_name}_LLM"
    
    if slm_scores_col not in comparison_data.columns or llm_scores_col not in comparison_data.columns:
        raise KeyError(f"Missing expected columns: {slm_scores_col}, {llm_scores_col}")
    
    slm_scores = comparison_data[slm_scores_col].astype(float)
    llm_scores = comparison_data[llm_scores_col].astype(float)
    
    both_models_correct = (slm_scores == 1.0) & (llm_scores == 1.0)
    both_models_wrong = (slm_scores == 0.0) & (llm_scores == 0.0)
    only_llm_correct = (slm_scores == 0.0) & (llm_scores == 1.0)
    only_slm_correct = (slm_scores == 1.0) & (llm_scores == 0.0)
    
    accuracy_categories = ["Both Correct", "Only LLM Correct", "Only SLM Correct", "Both Wrong"]
    category_counts = [
        both_models_correct.sum(),
        only_llm_correct.sum(),
        only_slm_correct.sum(),
        both_models_wrong.sum()
    ]
    category_colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]
    
    fig, ax = plt.subplots(figsize=(6, 4))
    accuracy_bars = ax.bar(accuracy_categories, category_counts, color=category_colors)
    ax.set_xlabel("Accuracy Outcome")
    ax.set_ylabel("Count")
    ax.set_title(f"{metric_name} Accuracy Breakdown")
    
    plt.xticks(rotation=30, ha='right')
    
    for bar in accuracy_bars:
        bar_height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., bar_height,
                f'{int(bar_height)}',
                ha='center', va='bottom')
    
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    breakdown_chart_path = output_dir / f"{metric_name}_accuracy_breakdown.png"
    fig.savefig(breakdown_chart_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return breakdown_chart_path


def run_model_comparison(
    slm_results_path: Path,
    llm_results_path: Path,
) -> None:
    # Create timestamped output directory
    timestamp = datetime.now(tz=timezone.utc)
    output_dir = ANALYSIS_DIR / f"comparison_{timestamp.strftime('%Y%m%d_%H%M')}"

    # Load model results
    slm_data = load_and_prepare_data(slm_results_path, model_type="SLM")
    llm_data = load_and_prepare_data(llm_results_path, model_type="LLM")

    # Combine results for comparison
    combined_results = pd.merge(slm_data, llm_data, on="row_id", how="inner")

    # Calculate performance comparisons for each metric
    for metric_name in METRICS:
        slm_metric_col = f"{metric_name}_SLM"
        llm_metric_col = f"{metric_name}_LLM"
        comparison_col = f"relation_{metric_name}"
        
        # Start with default assumption: SLM performs better
        comparison_result = pd.Series(np.full(len(combined_results), RELATION_VALUES[2], dtype=object))  # "SLM>LLM"
        # Override where LLM actually performs better
        comparison_result.loc[combined_results[llm_metric_col] > combined_results[slm_metric_col]] = RELATION_VALUES[0]  # "LLM>SLM"
        # Override where both perform equally
        comparison_result.loc[combined_results[llm_metric_col] == combined_results[slm_metric_col]] = RELATION_VALUES[1]  # "Equal"
        combined_results[comparison_col] = comparison_result

    # Additionally: export rows where only one model is exactly correct (exact_match)
    slm_full_df = pd.read_csv(slm_results_path)
    llm_full_df = pd.read_csv(llm_results_path)

    exact_slm_col = "exact_match_SLM"
    exact_llm_col = "exact_match_LLM"
    if exact_slm_col not in combined_results.columns or exact_llm_col not in combined_results.columns:
        raise KeyError("Missing exact_match columns in combined results")

    only_slm_correct_mask = (combined_results[exact_slm_col] == 1.0) & (combined_results[exact_llm_col] == 0.0)
    only_llm_correct_mask = (combined_results[exact_slm_col] == 0.0) & (combined_results[exact_llm_col] == 1.0)

    only_slm_row_ids = combined_results.loc[only_slm_correct_mask, "row_id"].astype(int).tolist()
    only_llm_row_ids = combined_results.loc[only_llm_correct_mask, "row_id"].astype(int).tolist()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save SLM-only-correct rows (from SLM source CSV)
    slm_only_path = output_dir / "exact_match_only_slm_correct.csv"
    if len(only_slm_row_ids) > 0:
        slm_only_rows = slm_full_df.iloc[only_slm_row_ids].copy()
        slm_only_rows.insert(0, "row_id", only_slm_row_ids)
        slm_only_rows.to_csv(slm_only_path, index=False)
    else:
        pd.DataFrame(columns=["row_id", *slm_full_df.columns]).to_csv(slm_only_path, index=False)

    # Save LLM-only-correct rows (from LLM source CSV)
    llm_only_path = output_dir / "exact_match_only_llm_correct.csv"
    if len(only_llm_row_ids) > 0:
        llm_only_rows = llm_full_df.iloc[only_llm_row_ids].copy()
        llm_only_rows.insert(0, "row_id", only_llm_row_ids)
        llm_only_rows.to_csv(llm_only_path, index=False)
    else:
        pd.DataFrame(columns=["row_id", *llm_full_df.columns]).to_csv(llm_only_path, index=False)

    # Generate outputs for each metric
    for metric_name in METRICS:
        metric_columns = [
            "row_id",
            f"{metric_name}_SLM",
            f"{metric_name}_LLM",
            f"relation_{metric_name}",
        ]
        metric_comparison_data = combined_results[metric_columns].copy()
        save_metric_comparison_csv(metric_comparison_data, metric_name, output_dir=output_dir)
        create_scatter_plot(metric_comparison_data, metric_name, output_dir=output_dir)
        create_comparison_bar_chart(combined_results, metric_name, output_dir=output_dir)
        if metric_name == "exact_match":
            create_accuracy_breakdown_chart(metric_comparison_data, metric_name, output_dir=output_dir)

    save_overall_summary(combined_results, output_dir=output_dir)


def main() -> None:
    # Default model result files for comparison
    default_slm_results = RESULTS_DIR / "Phi-4_202508101101" / "huggingface" / "Phi-4_12288_zs_msg1.csv"
    default_llm_results = RESULTS_DIR / "gpt" / "gpt-4.1-2025-04-14_12288_os_msg1.csv"

    run_model_comparison(slm_results_path=default_slm_results, llm_results_path=default_llm_results)


if __name__ == "__main__":
    main()


