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
from sklearn.linear_model import LinearRegression

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

    # Additional factor-to-factor plot: concern_count vs contextlength
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.scatter(
        combined_results_df["concern_count"],
        combined_results_df["context_len"],
        alpha=0.6,
        s=30,
    )
    linear_coefficients = np.polyfit(
        combined_results_df["concern_count"], combined_results_df["context_len"], 1
    )
    linear_model = np.poly1d(linear_coefficients)
    ax.plot(
        combined_results_df["concern_count"],
        linear_model(combined_results_df["concern_count"]),
        "r--",
        alpha=0.8,
    )
    corr_coef, p_value = stats.pearsonr(
        combined_results_df["concern_count"], combined_results_df["context_len"]
    )
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
        f"concern_count vs contextlength\nr={corr_coef:.3f}, p={p_value:.4f} {significance_stars}"
    )
    ax.set_xlabel("concern_count")
    ax.set_ylabel("contextlength")
    ax.set_yscale("log")
    fig.text(
        0.5,
        0.01,
        f"Significance threshold: p < {P_VALUE_SIGNIFICANCE_THRESHOLD}",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    pair_output_path = analysis_output_dir / "concern_count_vs_contextlength.png"
    plt.savefig(pair_output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_report(
    pearson_correlation_by_factor_metric: Dict[str, Dict[str, Tuple[float, float]]],
    combined_results_df: pd.DataFrame,
    num_input_files: int,
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

    # Save regression and partial-correlation summary
    x1_log2_context = np.log2(combined_results_df["context_len"].astype(float).to_numpy())
    x2_concern = combined_results_df["concern_count"].astype(float).to_numpy()

    summary_rows: List[Dict[str, float]] = []
    for metric_name in PERFORMANCE_METRICS:
        y_metric = combined_results_df[metric_name].astype(float).to_numpy()

        # Pearson r and p (from earlier computation)
        pearson_r_ctx, pearson_p_ctx = pearson_correlation_by_factor_metric["context_len"][metric_name]
        pearson_r_cnc, pearson_p_cnc = pearson_correlation_by_factor_metric["concern_count"][metric_name]

        # Partial correlations via residuals
        # context_len partial (control concern_count)
        y_on_x2 = LinearRegression().fit(x2_concern.reshape(-1, 1), y_metric)
        y_resid_ctx = y_metric - y_on_x2.predict(x2_concern.reshape(-1, 1))
        x1_on_x2 = LinearRegression().fit(x2_concern.reshape(-1, 1), x1_log2_context)
        x1_resid = x1_log2_context - x1_on_x2.predict(x2_concern.reshape(-1, 1))
        partial_r_ctx, partial_p_ctx = stats.pearsonr(x1_resid, y_resid_ctx)

        # concern_count partial (control log2(context_len))
        y_on_x1 = LinearRegression().fit(x1_log2_context.reshape(-1, 1), y_metric)
        y_resid_cnc = y_metric - y_on_x1.predict(x1_log2_context.reshape(-1, 1))
        x2_on_x1 = LinearRegression().fit(x1_log2_context.reshape(-1, 1), x2_concern)
        x2_resid = x2_concern - x2_on_x1.predict(x1_log2_context.reshape(-1, 1))
        partial_r_cnc, partial_p_cnc = stats.pearsonr(x2_resid, y_resid_cnc)

        # Full and reduced models for coefficients and delta R2
        X_full = np.column_stack([x1_log2_context, x2_concern])
        full_model = LinearRegression().fit(X_full, y_metric)
        beta_context_len = float(full_model.coef_[0])
        beta_concern_count = float(full_model.coef_[1])
        r2_full = float(full_model.score(X_full, y_metric))

        r2_only_ctx = float(LinearRegression().fit(x1_log2_context.reshape(-1, 1), y_metric).score(x1_log2_context.reshape(-1, 1), y_metric))
        r2_only_cnc = float(LinearRegression().fit(x2_concern.reshape(-1, 1), y_metric).score(x2_concern.reshape(-1, 1), y_metric))
        delta_r2_context_len = r2_full - r2_only_cnc
        delta_r2_concern_count = r2_full - r2_only_ctx

        # VIFs for predictors
        r2_x1_on_x2 = float(LinearRegression().fit(x2_concern.reshape(-1, 1), x1_log2_context).score(x2_concern.reshape(-1, 1), x1_log2_context))
        r2_x2_on_x1 = float(LinearRegression().fit(x1_log2_context.reshape(-1, 1), x2_concern).score(x1_log2_context.reshape(-1, 1), x2_concern))
        vif_context_len = float("inf") if (1.0 - r2_x1_on_x2) == 0.0 else 1.0 / (1.0 - r2_x1_on_x2)
        vif_concern_count = float("inf") if (1.0 - r2_x2_on_x1) == 0.0 else 1.0 / (1.0 - r2_x2_on_x1)

        summary_rows.append(
            {
                "metric": metric_name,
                "pearson_r_context_len_metric": pearson_r_ctx,
                "pearson_p_context_len_metric": pearson_p_ctx,
                "pearson_r_concern_count_metric": pearson_r_cnc,
                "pearson_p_concern_count_metric": pearson_p_cnc,
                "partial_r_context_len_metric": partial_r_ctx,
                "partial_p_context_len_metric": partial_p_ctx,
                "partial_r_concern_count_metric": partial_r_cnc,
                "partial_p_concern_count_metric": partial_p_cnc,
                "beta_context_len": beta_context_len,
                "beta_concern_count": beta_concern_count,
                "delta_r2_context_len": delta_r2_context_len,
                "delta_r2_concern_count": delta_r2_concern_count,
                "vif_context_len": vif_context_len,
                "vif_concern_count": vif_concern_count,
            }
        )

    regression_partial_results_df = pd.DataFrame(summary_rows)
    regression_partial_results_df.to_csv(
        analysis_output_dir / "regression_partial_results.csv", index=False
    )


def plot_partial_regressions(
    combined_results_df: pd.DataFrame,
    analysis_output_dir: Path,
) -> None:
    """Fit y ~ log2(context_len) + concern_count for each metric and plot partial regression.

    For each y in PERFORMANCE_METRICS, generate two plots:
    - Partial effect of context_len (controlling concern_count)
    - Partial effect of concern_count (controlling log2(context_len))
    """
    x1_log2_context = np.log2(combined_results_df["context_len"].astype(float).to_numpy())
    x2_concern = combined_results_df["concern_count"].astype(float).to_numpy()

    for metric_name in PERFORMANCE_METRICS:
        y_metric = combined_results_df[metric_name].astype(float).to_numpy()

        # Fit full model to obtain coefficients
        X_full = np.column_stack([x1_log2_context, x2_concern])
        full_model = LinearRegression()
        full_model.fit(X_full, y_metric)
        beta1_context, beta2_concern = float(full_model.coef_[0]), float(full_model.coef_[1])
        # Print raw and standardized coefficients for influence comparison
        y_std = float(np.std(y_metric)) if float(np.std(y_metric)) != 0.0 else 1.0
        std_beta_context = beta1_context * (float(np.std(x1_log2_context)) / y_std)
        std_beta_concern = beta2_concern * (float(np.std(x2_concern)) / y_std)

        # Partial for context_len (control concern_count)
        y_on_x2 = LinearRegression().fit(x2_concern.reshape(-1, 1), y_metric)
        y_resid_ctx = y_metric - y_on_x2.predict(x2_concern.reshape(-1, 1))
        x1_on_x2 = LinearRegression().fit(x2_concern.reshape(-1, 1), x1_log2_context)
        x1_resid = x1_log2_context - x1_on_x2.predict(x2_concern.reshape(-1, 1))
        partial_r_ctx, partial_p_ctx = stats.pearsonr(x1_resid, y_resid_ctx)

        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.scatter(x1_resid, y_resid_ctx, alpha=0.6, s=30)
        slope_ctx, intercept_ctx = np.polyfit(x1_resid, y_resid_ctx, 1)
        x_line_ctx = np.linspace(x1_resid.min(), x1_resid.max(), 100)
        ax.plot(x_line_ctx, slope_ctx * x_line_ctx + intercept_ctx, "r--", alpha=0.8)
        ax.axhline(0.0, color="gray", alpha=0.3, linewidth=1)
        ax.axvline(0.0, color="gray", alpha=0.3, linewidth=1)
        ax.set_title(
            f"Partial: contextlength vs {metric_name} (control concern_count)\nr={partial_r_ctx:.3f}, p={partial_p_ctx:.4f}"
        )
        ax.set_xlabel("contextlength residuals (control concern_count)")
        ax.set_ylabel(f"{metric_name} residuals")
        plt.tight_layout()
        out_ctx = analysis_output_dir / f"partial_contextlength_vs_{metric_name}.png"
        plt.savefig(out_ctx, dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Partial for concern_count (control log2(context_len))
        y_on_x1 = LinearRegression().fit(x1_log2_context.reshape(-1, 1), y_metric)
        y_resid_cnc = y_metric - y_on_x1.predict(x1_log2_context.reshape(-1, 1))
        x2_on_x1 = LinearRegression().fit(x1_log2_context.reshape(-1, 1), x2_concern)
        x2_resid = x2_concern - x2_on_x1.predict(x1_log2_context.reshape(-1, 1))
        partial_r_cnc, partial_p_cnc = stats.pearsonr(x2_resid, y_resid_cnc)

        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.scatter(x2_resid, y_resid_cnc, alpha=0.6, s=30)
        slope_cnc, intercept_cnc = np.polyfit(x2_resid, y_resid_cnc, 1)
        x_line_cnc = np.linspace(x2_resid.min(), x2_resid.max(), 100)
        ax.plot(x_line_cnc, slope_cnc * x_line_cnc + intercept_cnc, "r--", alpha=0.8)
        ax.axhline(0.0, color="gray", alpha=0.3, linewidth=1)
        ax.axvline(0.0, color="gray", alpha=0.3, linewidth=1)
        ax.set_title(
            f"Partial: concern_count vs {metric_name} (control log2(contextlength))\nr={partial_r_cnc:.3f}, p={partial_p_cnc:.4f}"
        )
        ax.set_xlabel("concern_count residuals (control log2(contextlength))")
        ax.set_ylabel(f"{metric_name} residuals")
        plt.tight_layout()
        out_cnc = analysis_output_dir / f"partial_concern_count_vs_{metric_name}.png"
        plt.savefig(out_cnc, dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Minimal, readable weight printout per metric
        stronger = (
            "log2_context_len" if abs(std_beta_context) > abs(std_beta_concern) else "concern_count"
        )
        print(
            f"[{metric_name}] beta_log2_context_len={beta1_context:.6f}, "
            f"beta_concern_count={beta2_concern:.6f} | std_beta_log2_context_len={std_beta_context:.6f}, "
            f"std_beta_concern_count={std_beta_concern:.6f} -> stronger={stronger}"
        )

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

    # Create output directory
    analysis_output_dir = ANALYSIS_OUTPUT_DIR_BASE / f"correlation_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}"
    analysis_output_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualizations (scatter plots only)
    plot_correlation_visualizations(combined_results_df, pearson_correlation_by_factor_metric, analysis_output_dir)
    plot_partial_regressions(combined_results_df, analysis_output_dir)

    # Generate and print report
    correlation_text_report = generate_report(
        pearson_correlation_by_factor_metric,
        combined_results_df,
        num_input_files=len(input_csv_paths),
    )
    print("\n" + correlation_text_report)

    # Save results
    save_results(pearson_correlation_by_factor_metric, combined_results_df, analysis_output_dir)

    print(f"\nResults saved to: {analysis_output_dir}")


if __name__ == "__main__":
    main()
