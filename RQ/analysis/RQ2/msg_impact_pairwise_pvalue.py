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

from ..stats_utils import compute_pairwise_stats

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

        results["by_model"][model_name] = compute_pairwise_stats(data_msg0, data_msg1)

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

    latex_rows = []
    for model_name in model_names:
        data = results["by_model"].get(model_name, {})
        if data:
            latex_rows.append({
                "model": model_name,
                "A": data.get('effect_size', ''),
                "p": data.get('p_value', '')
            })

    pd.DataFrame(latex_rows).to_csv(output_dir / "msg_impact_pairwise_latex.csv", index=False)


def print_summary(results: dict):
    """Print summary to console."""
    print(f"\nTest: {results['summary']['test_method']}")
    print(f"Comparison: {results['summary']['comparison']}")
    print("-" * 60)

    for model_name, data in results["by_model"].items():
        sig = "*" if data["significant"] else ""
        print(f"  {model_name}: p={data['p_value']}{sig}, A={data['effect_size']}")


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
