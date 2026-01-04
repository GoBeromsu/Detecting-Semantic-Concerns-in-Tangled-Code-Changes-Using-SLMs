#!/usr/bin/env python3
"""
Message Impact Pairwise P-Value Table Generator

Statistical Test Choice:
- Wilcoxon Signed-Rank Test (paired, non-parametric) is used because:
  1. Each commit is evaluated with/without message, creating naturally paired samples
  2. Hamming loss distributions are often non-normal and bounded [0, 1]
  3. Commit-level pairing controls for per-commit difficulty variance

Effect Size Interpretation:
- Vargha-Delaney Â₁₂ = (R₁/m - (m+1)/2) / n (rank-based formula)
- Â₁₂ > 0.5: msg0 (without message) tends to have higher HS (worse) → with message is better
- Â₁₂ < 0.5: msg0 (without message) tends to have lower HS (better) → without message is better
- Â₁₂ = 0.5: no difference
- |Â₁₂ - 0.5|: < 0.06 negligible, 0.06-0.14 small, 0.14-0.21 medium, >= 0.21 large
"""

import pandas as pd
import yaml
import json
from pathlib import Path
import argparse
import numpy as np

from ..stats_utils import vargha_delaney_a, wilcoxon_signed_rank

# Constants
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
RQ_NAME = Path(__file__).parent.name
ANALYSIS_OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis" / RQ_NAME
P_VALUE_THRESHOLD = 0.05


# =============================================================================
# Input
# =============================================================================

def load_data():
    """Load configuration and CSV data for all models."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    script_config = config["rq2"]["scripts"]["msg_impact_pairwise_pvalue"]
    csv_data = {}

    for model_name, model_config in script_config["models"].items():
        msg0_path = PROJECT_ROOT / model_config["msg0_path"]
        msg1_path = PROJECT_ROOT / model_config["msg1_path"]

        if msg0_path.exists() and msg1_path.exists():
            df_msg0 = pd.read_csv(msg0_path)
            df_msg1 = pd.read_csv(msg1_path)

            if "hamming_loss" in df_msg0.columns and "hamming_loss" in df_msg1.columns:
                csv_data[model_name] = {
                    "msg0": df_msg0,
                    "msg1": df_msg1
                }

    return csv_data


# =============================================================================
# Process
# =============================================================================

def perform_pairwise_tests(csv_data: dict) -> dict:
    """Perform Wilcoxon Signed-Rank tests for msg0 vs msg1 for each model."""
    model_names = list(csv_data.keys())

    results = {"by_model": {}, "summary": {
        "test_method": "Wilcoxon Signed-Rank Test (two-sided, paired)",
        "effect_size_metric": "Vargha-Delaney A",
        "significance_threshold": P_VALUE_THRESHOLD,
        "models": model_names,
        "comparison": "Without Message vs With Message"
    }}

    for model_name in model_names:
        data_msg0 = csv_data[model_name]["msg0"]["hamming_loss"].values
        data_msg1 = csv_data[model_name]["msg1"]["hamming_loss"].values

        if len(data_msg0) == 0:
            continue

        # Statistical tests
        p_value = wilcoxon_signed_rank(data_msg0, data_msg1)
        effect_size = vargha_delaney_a(data_msg0, data_msg1)

        # Sign-support
        msg1_wins = int(np.sum(data_msg1 < data_msg0))  # msg1 better (lower HS)
        msg0_wins = int(np.sum(data_msg0 < data_msg1))  # msg0 better (lower HS)
        n = len(data_msg0)

        results["by_model"][model_name] = {
            "p_value": float(p_value),
            "p_formatted": "0.001" if p_value < 0.001 else f"{p_value:.3f}",
            "effect_size": float(effect_size),
            "significant": bool(p_value < P_VALUE_THRESHOLD),
            "msg0_wins": msg0_wins, "msg1_wins": msg1_wins, "n": n,
            "msg0_win_pct": float(msg0_wins / n * 100),
            "msg1_win_pct": float(msg1_wins / n * 100),
            "msg0_mean": float(data_msg0.mean()),
            "msg1_mean": float(data_msg1.mean())
        }

    return results


# =============================================================================
# Output
# =============================================================================

def save_results(results: dict, output_dir: Path):
    """Save results as JSON and LaTeX-friendly CSV only."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    with open(output_dir / "msg_impact_pairwise_pvalues.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # -------------------------------------------------------------------------
    # LaTeX-friendly CSV for pgfplotstabletypeset (one row per model)
    # Columns: model, r, p
    # -------------------------------------------------------------------------
    model_names = results["summary"]["models"]

    def fmt_p(p: float) -> str:
        """Format p-value: 0.001 floor or 3 decimal places."""
        return "0.001" if p < 0.001 else f"{p:.3f}"

    latex_rows = []
    for model_name in model_names:
        data = results["by_model"].get(model_name, {})
        if data:
            latex_rows.append({
                "model": model_name,
                "A": f"{data['effect_size']:.2f}",
                "p": fmt_p(data["p_value"])
            })

    pd.DataFrame(latex_rows).to_csv(output_dir / "msg_impact_pairwise_latex.csv", index=False)


def print_summary(results: dict):
    """Print summary to console."""
    print(f"\nTest: {results['summary']['test_method']}")
    print(f"Comparison: {results['summary']['comparison']}")
    print("Note: A > 0.5 means 'With Message' has lower HS (better).")
    print("-" * 80)

    for model_name, data in results["by_model"].items():
        sig = "*" if data["significant"] else ""
        winner = "With Msg" if data["effect_size"] > 0.5 else "Without Msg"
        print(f"\n{model_name}:")
        print(f"  p={data['p_formatted']}{sig}, A={data['effect_size']:.2f}")
        print(f"  Without Msg: mean={data['msg0_mean']:.3f}, wins={data['msg0_win_pct']:.1f}%")
        print(f"  With Msg: mean={data['msg1_mean']:.3f}, wins={data['msg1_win_pct']:.1f}%")
        print(f"  Better: {winner}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate pairwise p-value tables for message impact")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else ANALYSIS_OUTPUT_DIR / "pf_msg_impact_pairwise"

    # Input
    print("Loading data...")
    csv_data = load_data()
    print(f"Loaded {len(csv_data)} models: {list(csv_data.keys())}")

    if len(csv_data) < 1:
        print("Error: Need at least 1 model")
        return

    # Process
    print("Performing Wilcoxon Signed-Rank tests...")
    results = perform_pairwise_tests(csv_data)

    # Output
    save_results(results, output_dir)
    print_summary(results)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
