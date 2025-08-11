#!/usr/bin/env python3
"""
Correlation Analysis: Context Length vs Concern Count Impact on Performance Metrics
Analyzes which factor has stronger correlation with precision, recall, f1 scores, and inference_time.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# Constants
ANALYSIS_OUTPUT_DIR_BASE = Path("results/analysis")
P_VALUE_SIGNIFICANCE_THRESHOLD: float = 0.05

# User-editable input: add your CSV paths here
INPUT_CSV_PATHS: List[Path] = [
    # Path("results/gpt/gpt-4.1-2025-04-14_1024_zs_msg1.csv"),
    # Path("results/gpt/gpt-4.1-2025-04-14_2048_zs_msg1.csv"),
    # Path("results/gpt/gpt-4.1-2025-04-14_4096_zs_msg1.csv"),
    # Path("results/gpt/gpt-4.1-2025-04-14_8192_zs_msg1.csv"),
    # Path("results/gpt/gpt-4.1-2025-04-14_12288_zs_msg1.csv"),
    Path("results/phi-4_202508101101/huggingface/Phi-4_1024_zs_msg1.csv"),
    Path("results/phi-4_202508101101/huggingface/Phi-4_2048_zs_msg1.csv"),
    Path("results/phi-4_202508101101/huggingface/Phi-4_4096_zs_msg1.csv"),
    Path("results/phi-4_202508101101/huggingface/Phi-4_8192_zs_msg1.csv"),
    Path("results/phi-4_202508101101/huggingface/Phi-4_12288_zs_msg1.csv"),
]

PERFORMANCE_METRICS = ["precision", "recall", "f1", "inference_time"]
EXPLANATORY_FACTORS = ["context_len", "concern_count"]


def load_zero_shot_msg1_results(csv_paths: List[Path]) -> pd.DataFrame:
    """Load and combine provided zero-shot msg1 CSV files."""
    if not csv_paths:
        raise ValueError("No CSV paths provided. Use --files to pass one or more CSV files.")

    results_frames: List[pd.DataFrame] = []

    for csv_path in csv_paths:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        file_results_df = pd.read_csv(csv_path)
        results_frames.append(file_results_df)
        print(f"Loaded {csv_path.name}: {len(file_results_df)} records")

    combined_results_df = pd.concat(results_frames, ignore_index=True)
    print(f"Total combined records: {len(combined_results_df)}")

    return combined_results_df


def compute_pearson_correlations(combined_results_df: pd.DataFrame) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Calculate Pearson correlations between factors and metrics."""
    pearson_correlation_by_factor_metric: Dict[str, Dict[str, Tuple[float, float]]] = {}

    for factor_name in EXPLANATORY_FACTORS:
        pearson_correlation_by_factor_metric[factor_name] = {}
        for metric_name in PERFORMANCE_METRICS:
            correlation_coefficient, p_value = stats.pearsonr(
                combined_results_df[factor_name], combined_results_df[metric_name]
            )
            pearson_correlation_by_factor_metric[factor_name][metric_name] = (
                correlation_coefficient,
                p_value,
            )

    return pearson_correlation_by_factor_metric


# Heatmap removed per user request; focus on factor-to-metric scatter relationships


def compute_explanatory_factor_pair_correlation(
    combined_results_df: pd.DataFrame,
) -> Tuple[float, float]:
    """Compute Pearson correlation between context_len and concern_count."""
    correlation_coefficient, p_value = stats.pearsonr(
        combined_results_df["context_len"], combined_results_df["concern_count"]
    )
    return correlation_coefficient, p_value


def plot_correlation_visualizations(
    combined_results_df: pd.DataFrame,
    pearson_correlation_by_factor_metric: Dict[str, Dict[str, Tuple[float, float]]],
    analysis_output_dir: Path,
) -> None:
    """Create and save individual scatter plots for each factor-metric pair."""
    analysis_output_dir.mkdir(parents=True, exist_ok=True)

    for factor_name in EXPLANATORY_FACTORS:
        for metric_name in PERFORMANCE_METRICS:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))

            ax.scatter(
                combined_results_df[factor_name],
                combined_results_df[metric_name],
                alpha=0.6,
                s=30,
            )

            linear_coefficients = np.polyfit(
                combined_results_df[factor_name], combined_results_df[metric_name], 1
            )
            linear_model = np.poly1d(linear_coefficients)
            ax.plot(
                combined_results_df[factor_name],
                linear_model(combined_results_df[factor_name]),
                "r--",
                alpha=0.8,
            )

            correlation_coefficient, p_value = pearson_correlation_by_factor_metric[factor_name][metric_name]
            significance_stars = (
                "***"
                if p_value < 0.001
                else "**"
                if p_value < 0.01
                else "*"
                if p_value < P_VALUE_SIGNIFICANCE_THRESHOLD
                else ""
            )
            ax.set_title(
                f"{factor_name} vs {metric_name}\nr={correlation_coefficient:.3f}, p={p_value:.4f} {significance_stars}"
            )
            ax.set_xlabel(factor_name)
            ax.set_ylabel(metric_name)

            if factor_name == "context_len":
                ax.set_xscale("log")

            fig.text(
                0.5,
                0.01,
                f"Significance threshold: p < {P_VALUE_SIGNIFICANCE_THRESHOLD}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

            plt.tight_layout(rect=(0, 0.03, 1, 1))
            output_path = analysis_output_dir / f"{factor_name}_vs_{metric_name}.png"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)


def generate_report(
    pearson_correlation_by_factor_metric: Dict[str, Dict[str, Tuple[float, float]]],
    combined_results_df: pd.DataFrame,
    num_input_files: int,
    factor_pair_correlation: Tuple[float, float],
) -> str:
    """Generate text report of correlation analysis."""
    report_lines: List[str] = ["Correlation Analysis Report", "=" * 50, ""]

    report_lines.append(
        f"Dataset: {len(combined_results_df)} records from {num_input_files} input CSV files"
    )
    report_lines.append(
        f"Context lengths: {sorted(combined_results_df['context_len'].unique())}"
    )
    report_lines.append(
        f"Concern counts: {sorted(combined_results_df['concern_count'].unique())}"
    )
    report_lines.append("")

    report_lines.append("Correlation Results:")
    report_lines.append("-" * 30)

    for factor_name in EXPLANATORY_FACTORS:
        report_lines.append(f"\n{factor_name.upper()}:")
        for metric_name in PERFORMANCE_METRICS:
            correlation_coefficient, p_value = pearson_correlation_by_factor_metric[factor_name][metric_name]
            significance = (
                "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            )
            report_lines.append(
                f"  {metric_name}: r={correlation_coefficient:6.3f}, p={p_value:6.4f} {significance}"
            )

    # Summary comparison
    report_lines.append("\nSummary Comparison:")
    report_lines.append("-" * 20)

    for metric_name in PERFORMANCE_METRICS:
        context_abs_corr = abs(pearson_correlation_by_factor_metric["context_len"][metric_name][0])
        concern_abs_corr = abs(pearson_correlation_by_factor_metric["concern_count"][metric_name][0])

        stronger_factor = "context_len" if context_abs_corr > concern_abs_corr else "concern_count"
        report_lines.append(
            f"{metric_name}: {stronger_factor} has stronger correlation ({context_abs_corr:.3f} vs {concern_abs_corr:.3f})"
        )

    # Factor-to-factor correlation
    report_lines.append("\nFactor-to-Factor Correlation:")
    report_lines.append("-" * 28)
    ff_r, ff_p = factor_pair_correlation
    ff_significance = (
        "***" if ff_p < 0.001 else "**" if ff_p < 0.01 else "*" if ff_p < 0.05 else ""
    )
    report_lines.append(
        f"context_len vs concern_count: r={ff_r:6.3f}, p={ff_p:6.4f} {ff_significance}"
    )

    return "\n".join(report_lines)


def save_results(
    pearson_correlation_by_factor_metric: Dict[str, Dict[str, Tuple[float, float]]],
    combined_results_df: pd.DataFrame,
    analysis_output_dir: Path,
) -> None:
    """Save correlation results to CSV."""
    # Create results DataFrame
    correlation_result_rows: List[Dict[str, object]] = []
    for factor_name in EXPLANATORY_FACTORS:
        for metric_name in PERFORMANCE_METRICS:
            correlation_coefficient, p_value = pearson_correlation_by_factor_metric[factor_name][metric_name]
            correlation_result_rows.append(
                {
                    "factor": factor_name,
                    "metric": metric_name,
                    "correlation": correlation_coefficient,
                    "p_value": p_value,
                    "abs_correlation": abs(correlation_coefficient),
                    "significant": p_value < 0.05,
                }
            )

    correlation_results_df = pd.DataFrame(correlation_result_rows)
    correlation_results_df.to_csv(analysis_output_dir / "correlation_results.csv", index=False)

    # Save summary statistics
    summary_statistics_df = combined_results_df[EXPLANATORY_FACTORS + PERFORMANCE_METRICS].describe()
    summary_statistics_df.to_csv(analysis_output_dir / "summary_statistics.csv")


def plot_explanatory_factor_pair_visualization(
    combined_results_df: pd.DataFrame,
    factor_pair_correlation: Tuple[float, float],
    analysis_output_dir: Path,
) -> None:
    """Create and save scatter plot for context_len vs concern_count."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    ax.scatter(
        combined_results_df["context_len"],
        combined_results_df["concern_count"],
        alpha=0.6,
        s=30,
    )

    linear_coefficients = np.polyfit(
        combined_results_df["context_len"], combined_results_df["concern_count"], 1
    )
    linear_model = np.poly1d(linear_coefficients)
    ax.plot(
        combined_results_df["context_len"],
        linear_model(combined_results_df["context_len"]),
        "r--",
        alpha=0.8,
    )

    correlation_coefficient, p_value = factor_pair_correlation
    significance_stars = (
        "***"
        if p_value < 0.001
        else "**"
        if p_value < 0.01
        else "*"
        if p_value < P_VALUE_SIGNIFICANCE_THRESHOLD
        else ""
    )
    ax.set_title(
        f"context_len vs concern_count\nr={correlation_coefficient:.3f}, p={p_value:.4f} {significance_stars}"
    )
    ax.set_xlabel("context_len")
    ax.set_ylabel("concern_count")
    ax.set_xscale("log")

    fig.text(
        0.5,
        0.01,
        f"Significance threshold: p < {P_VALUE_SIGNIFICANCE_THRESHOLD}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

    plt.tight_layout(rect=(0, 0.03, 1, 1))
    output_path = analysis_output_dir / "context_len_vs_concern_count.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_explanatory_factor_pair_results(
    factor_pair_correlation: Tuple[float, float], analysis_output_dir: Path
) -> None:
    """Save the context_len vs concern_count correlation to CSV."""
    ff_r, ff_p = factor_pair_correlation
    df = pd.DataFrame(
        [
            {
                "factor_x": "context_len",
                "factor_y": "concern_count",
                "correlation": ff_r,
                "p_value": ff_p,
                "abs_correlation": abs(ff_r),
                "significant": ff_p < 0.05,
            }
        ]
    )
    df.to_csv(analysis_output_dir / "factor_pair_correlation.csv", index=False)


def main() -> None:
    """Main analysis function."""
    print("Starting correlation analysis...")

    input_csv_paths = INPUT_CSV_PATHS
    if not input_csv_paths:
        raise ValueError("INPUT_CSV_PATHS is empty. Please populate it with CSV Path values.")

    # Load data
    combined_results_df = load_zero_shot_msg1_results(input_csv_paths)

    # Validate data
    required_columns = EXPLANATORY_FACTORS + PERFORMANCE_METRICS
    missing_columns = [col for col in required_columns if col not in combined_results_df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    # Calculate correlations
    pearson_correlation_by_factor_metric = compute_pearson_correlations(combined_results_df)
    factor_pair_correlation = compute_explanatory_factor_pair_correlation(combined_results_df)

    # Create output directory
    analysis_output_dir = ANALYSIS_OUTPUT_DIR_BASE / f"correlation_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}"
    analysis_output_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualizations (scatter plots only)
    plot_correlation_visualizations(combined_results_df, pearson_correlation_by_factor_metric, analysis_output_dir)
    plot_explanatory_factor_pair_visualization(combined_results_df, factor_pair_correlation, analysis_output_dir)

    # Generate and print report
    correlation_text_report = generate_report(
        pearson_correlation_by_factor_metric,
        combined_results_df,
        num_input_files=len(input_csv_paths),
        factor_pair_correlation=factor_pair_correlation,
    )
    print("\n" + correlation_text_report)

    # Save results
    save_results(pearson_correlation_by_factor_metric, combined_results_df, analysis_output_dir)
    save_explanatory_factor_pair_results(factor_pair_correlation, analysis_output_dir)

    print(f"\nResults saved to: {analysis_output_dir}")


if __name__ == "__main__":
    main()
