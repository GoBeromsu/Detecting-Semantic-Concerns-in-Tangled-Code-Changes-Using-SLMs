#!/usr/bin/env python3
"""
Concern Count Pairwise P-Value Table Generator

Statistical Test Choice:
- Wilcoxon Signed-Rank Test (paired, non-parametric) is used because:
  1. Each commit is evaluated by all models, creating naturally paired samples
  2. Hamming loss distributions are often non-normal and bounded [0, 1]
  3. Commit-level pairing controls for per-commit difficulty variance

Effect Size Interpretation:
- Vargha-Delaney Â₁₂ = (R₁/m - (m+1)/2) / n (rank-based formula)
- Â₁₂ > 0.5: model_a tends to have higher HS (worse performance)
- Â₁₂ < 0.5: model_a tends to have lower HS (better performance)
- Â₁₂ = 0.5: no difference
- |Â₁₂ - 0.5|: < 0.06 negligible, 0.06-0.14 small, 0.14-0.21 medium, >= 0.21 large
"""

import pandas as pd
import yaml
import json
from pathlib import Path
import argparse
from itertools import combinations
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

    script_config = config["rq1"]["scripts"]["concern_count_pairwise_pvalue"]
    csv_data = {}

    for model_name, model_config in script_config["models"].items():
        csv_path = PROJECT_ROOT / model_config["csv_path"]
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if all(col in df.columns for col in ["hamming_loss", "concern_count"]):
                csv_data[model_name] = df

    return csv_data


# =============================================================================
# Process
# =============================================================================

def perform_pairwise_tests(csv_data: dict) -> dict:
    """Perform Wilcoxon Signed-Rank tests for all model pairs at each concern count."""
    model_names = list(csv_data.keys())
    first_model = list(csv_data.values())[0]
    concern_counts = [int(x) for x in sorted(first_model["concern_count"].unique())]

    results = {"by_concern_count": {}, "summary": {
        "test_method": "Wilcoxon Signed-Rank Test (two-sided, paired)",
        "effect_size_metric": "Vargha-Delaney A",
        "significance_threshold": P_VALUE_THRESHOLD,
        "models": model_names,
        "concern_counts": concern_counts
    }}

    for cc in concern_counts:
        results["by_concern_count"][cc] = {}

        for model_a, model_b in combinations(model_names, 2):
            data_a = csv_data[model_a][csv_data[model_a]["concern_count"] == cc]["hamming_loss"].values
            data_b = csv_data[model_b][csv_data[model_b]["concern_count"] == cc]["hamming_loss"].values

            if len(data_a) == 0:
                continue

            results["by_concern_count"][cc][f"{model_a} vs {model_b}"] = compute_pairwise_stats(data_a, data_b)

    return results


# =============================================================================
# Output
# =============================================================================

def save_results(results: dict, output_dir: Path):
    """Save results as JSON and LaTeX-friendly CSV only."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    with open(output_dir / "concern_count_pairwise_pvalues.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    concern_counts = results["summary"]["concern_counts"]

    pair_short = {
        "LLM vs SLM": "LLM_SLM",
        "LLM vs Fine-tuned SLM": "LLM_FT",
        "SLM vs Fine-tuned SLM": "SLM_FT"
    }

    latex_rows = []
    for cc in concern_counts:
        cc_data = results["by_concern_count"][cc]
        row = {"concerns": cc}
        for pair, short in pair_short.items():
            data = cc_data.get(pair, {})
            row[f"A_{short}"] = data.get('effect_size', '')
            row[f"p_{short}"] = data.get('p_value', '')
        latex_rows.append(row)

    pd.DataFrame(latex_rows).to_csv(output_dir / "concern_count_pairwise_latex.csv", index=False)


def print_summary(results: dict):
    """Print summary to console."""
    print(f"\nTest: {results['summary']['test_method']}")
    print(f"Models: {', '.join(results['summary']['models'])}")
    print("-" * 60)

    for cc in results["summary"]["concern_counts"]:
        print(f"\nConcern Count {cc}:")
        for pair, data in results["by_concern_count"][cc].items():
            sig = "*" if data["significant"] else ""
            print(f"  {pair}: p={data['p_value']}{sig}, A={data['effect_size']}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate pairwise p-value tables")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else ANALYSIS_OUTPUT_DIR / "pf_concern_count_pairwise"

    # Input
    print("Loading data...")
    csv_data = load_data()
    print(f"Loaded {len(csv_data)} models: {list(csv_data.keys())}")

    if len(csv_data) < 2:
        print("Error: Need at least 2 models")
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
