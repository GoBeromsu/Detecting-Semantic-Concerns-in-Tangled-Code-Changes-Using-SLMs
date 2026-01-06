#!/usr/bin/env python3
"""
Context Length Pairwise P-Value Table Generator

Statistical Test Choice (following Arcuri & Briand 2014):
- Wilcoxon Signed-Rank Test (paired, non-parametric) is used because:
  1. Same commits are evaluated by different models (paired design)
  2. Tests whether differences are symmetric around zero
  3. Pairs naturally with Vargha-Delaney A12 effect size

Effect Size Interpretation (Vargha & Delaney 2000):
- Vargha-Delaney Â₁₂ = P(X > Y) + 0.5 * P(X = Y)
- Â₁₂ > 0.5: model_a tends to have higher HL (worse performance)
- Â₁₂ < 0.5: model_a tends to have lower HL (better performance)
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

from . import PROJECT_ROOT, ANALYSIS_OUTPUT_DIR, CONFIG_PATH
from ..stats_utils import compute_pairwise_stats

# Constants
P_VALUE_THRESHOLD = 0.05


# =============================================================================
# Input
# =============================================================================

def load_data():
    """Load configuration and CSV data for all models at each context length."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    script_config = config["rq3"]["scripts"]["context_length_pairwise_pvalue"]
    context_lengths = script_config["context_lengths"]
    csv_data = {}

    for context_length in context_lengths:
        csv_data[context_length] = {}

        for model_name, model_config in script_config["models"].items():
            csv_path_pattern = model_config["csv_path_pattern"]
            csv_path = PROJECT_ROOT / csv_path_pattern.format(context=context_length)

            if csv_path.exists():
                df = pd.read_csv(csv_path)
                if "hamming_loss" in df.columns:
                    csv_data[context_length][model_name] = df

    return csv_data, context_lengths


# =============================================================================
# Process
# =============================================================================

def perform_pairwise_tests(csv_data: dict, context_lengths: list) -> dict:
    """Perform Wilcoxon signed-rank tests for all model pairs at each context length."""
    # Get model names from first available context
    model_names = []
    for cl in context_lengths:
        if csv_data[cl]:
            model_names = list(csv_data[cl].keys())
            break

    results = {"by_context_length": {}, "summary": {
        "test_method": "Wilcoxon Signed-Rank Test (two-sided, paired)",
        "effect_size_metric": "Vargha-Delaney A12",
        "significance_threshold": P_VALUE_THRESHOLD,
        "models": model_names,
        "context_lengths": context_lengths
    }}

    for cl in context_lengths:
        results["by_context_length"][cl] = {}

        for model_a, model_b in combinations(model_names, 2):
            if model_a not in csv_data[cl] or model_b not in csv_data[cl]:
                continue

            data_a = csv_data[cl][model_a]["hamming_loss"].values
            data_b = csv_data[cl][model_b]["hamming_loss"].values

            if len(data_a) == 0:
                continue

            results["by_context_length"][cl][f"{model_a} vs {model_b}"] = compute_pairwise_stats(data_a, data_b)

    return results


# =============================================================================
# Output
# =============================================================================

def save_results(results: dict, output_dir: Path):
    """Save results as JSON and LaTeX-friendly CSV only."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    with open(output_dir / "context_length_pairwise_pvalues.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # -------------------------------------------------------------------------
    # LaTeX-friendly CSV for pgfplotstabletypeset (one row per context length)
    # Columns: context, r_X, p_X for each comparison pair
    # -------------------------------------------------------------------------
    context_lengths = results["summary"]["context_lengths"]

    if not context_lengths or not results["by_context_length"].get(context_lengths[0]):
        print("No data to save")
        return

    pair_short = {
        "LLM vs SLM": "LLM_SLM",
        "LLM vs Fine-tuned SLM": "LLM_FT",
        "SLM vs Fine-tuned SLM": "SLM_FT"
    }

    latex_rows = []
    for cl in context_lengths:
        cl_data = results["by_context_length"][cl]
        row = {"context": cl}
        for pair, short in pair_short.items():
            data = cl_data.get(pair, {})
            row[f"A_{short}"] = data.get('effect_size', '')
            row[f"p_{short}"] = data.get('p_value', '')
        latex_rows.append(row)

    pd.DataFrame(latex_rows).to_csv(output_dir / "context_length_pairwise_latex.csv", index=False)


def print_summary(results: dict):
    """Print summary to console."""
    print(f"\nTest: {results['summary']['test_method']}")
    print(f"Models: {', '.join(results['summary']['models'])}")
    print("-" * 60)

    for cl in results["summary"]["context_lengths"]:
        print(f"\nContext Length {cl}:")
        for pair, data in results["by_context_length"][cl].items():
            sig = "*" if data["significant"] else ""
            print(f"  {pair}: p={data['p_value']}{sig}, A={data['effect_size']}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate pairwise p-value tables for context length")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else ANALYSIS_OUTPUT_DIR / "pf_context_length_pairwise"

    # Input
    print("Loading data...")
    csv_data, context_lengths = load_data()
    total_models = len(set(m for cl_data in csv_data.values() for m in cl_data.keys()))
    print(f"Loaded data for {len(context_lengths)} context lengths, {total_models} models")

    if total_models < 2:
        print("Error: Need at least 2 models")
        return

    # Process
    print("Performing Wilcoxon signed-rank tests...")
    results = perform_pairwise_tests(csv_data, context_lengths)

    # Output
    save_results(results, output_dir)
    print_summary(results)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
