#!/usr/bin/env python3
"""
Context Length Pairwise P-Value Table Generator

Statistical Test Choice:
- Wilcoxon Signed-Rank Test (paired, non-parametric) is used because:
  1. Each commit is evaluated by all models at each context length, creating naturally paired samples
  2. Hamming loss distributions are often non-normal and bounded [0, 1]
  3. Commit-level pairing controls for per-commit difficulty variance

Effect Size Interpretation:
- Rank-biserial correlation (r) is computed from diff = HS_a - HS_b
- r < 0: model_a tends to have lower HS (better performance)
- r > 0: model_b tends to have lower HS (better performance)
- |r| > 0.5: large effect, 0.3-0.5: medium, < 0.3: small
"""

import pandas as pd
import yaml
import json
from pathlib import Path
import argparse
from scipy.stats import wilcoxon, rankdata
from itertools import combinations
import numpy as np

# Constants
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
RQ_NAME = Path(__file__).parent.name
ANALYSIS_OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis" / RQ_NAME
P_VALUE_THRESHOLD = 0.05


# =============================================================================
# Input
# =============================================================================

def load_data():
    """Load configuration and CSV data for all models at each context length."""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    script_config = config["scripts"]["context_length_pairwise_pvalue"]
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
    """Perform Wilcoxon Signed-Rank tests for all model pairs at each context length."""
    # Get model names from first available context
    model_names = []
    for cl in context_lengths:
        if csv_data[cl]:
            model_names = list(csv_data[cl].keys())
            break

    results = {"by_context_length": {}, "summary": {
        "test_method": "Wilcoxon Signed-Rank Test (two-sided, paired)",
        "effect_size_metric": "rank-biserial correlation",
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

            # Fail fast if pairing is broken
            if len(data_a) != len(data_b):
                raise ValueError(f"Length mismatch for {model_a} vs {model_b} at context={cl}: "
                                 f"{len(data_a)} vs {len(data_b)}")

            # Wilcoxon signed-rank test
            try:
                result = wilcoxon(data_a, data_b, alternative='two-sided')
                p_value = result.pvalue
            except ValueError:
                p_value = 1.0

            # Rank-biserial correlation: r = (W+ - W-) / (W+ + W-)
            # Based on diff = HS_a - HS_b, so r < 0 means model_a has lower HS (better)
            diff = data_a - data_b
            diff_nz = diff[diff != 0]
            if len(diff_nz) > 0:
                ranks = rankdata(np.abs(diff_nz), method='average')
                w_plus, w_minus = np.sum(ranks[diff_nz > 0]), np.sum(ranks[diff_nz < 0])
                effect_size = (w_plus - w_minus) / (w_plus + w_minus) if (w_plus + w_minus) > 0 else 0.0
            else:
                effect_size = 0.0

            # Sign-support
            a_wins, b_wins = int(np.sum(data_a < data_b)), int(np.sum(data_b < data_a))
            n = len(data_a)

            results["by_context_length"][cl][f"{model_a} vs {model_b}"] = {
                "p_value": float(p_value),
                "p_formatted": "0.001" if p_value < 0.001 else f"{p_value:.3f}",
                "effect_size": float(effect_size),
                "significant": bool(p_value < P_VALUE_THRESHOLD),
                "a_wins": a_wins, "b_wins": b_wins, "n": n,
                "a_win_pct": float(a_wins / n * 100),
                "b_win_pct": float(b_wins / n * 100)
            }

    return results


# =============================================================================
# Output
# =============================================================================

def save_results(results: dict, output_dir: Path):
    """Save results as JSON and CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    with open(output_dir / "context_length_pairwise_pvalues.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # CSV with p-values only: numeric, no special characters
    # p < 0.001 shown as 0.001 (caption should note this)
    context_lengths = results["summary"]["context_lengths"]

    if not context_lengths or not results["by_context_length"].get(context_lengths[0]):
        print("No data to save")
        return

    pairs = list(results["by_context_length"][context_lengths[0]].keys())

    # Column name mapping for CSV
    pair_to_col = {
        "LLM vs SLM": "p_LLM_SLM",
        "LLM vs Fine-tuned SLM": "p_LLM_FT_SLM",
        "SLM vs Fine-tuned SLM": "p_SLM_FT_SLM"
    }

    rows = []
    for cl in context_lengths:
        row = {"Context": cl}
        for pair in pairs:
            data = results["by_context_length"][cl].get(pair, {})
            col = pair_to_col.get(pair, f"p_{pair.replace(' vs ', '_vs_')}")
            row[col] = data["p_formatted"] if data else ""
        rows.append(row)

    pd.DataFrame(rows).to_csv(output_dir / "context_length_pairwise_pvalues.csv", index=False)


def print_summary(results: dict):
    """Print summary to console."""
    print(f"\nTest: {results['summary']['test_method']}")
    print(f"Models: {', '.join(results['summary']['models'])}")
    print("Note: lower HS is better; r < 0 favours the first model in the pair.")
    print("-" * 80)

    for cl in results["summary"]["context_lengths"]:
        print(f"\nContext Length {cl}:")
        for pair, data in results["by_context_length"][cl].items():
            sig = "*" if data["significant"] else ""
            model_a, model_b = pair.split(" vs ")
            winner = f"{model_a} {data['a_win_pct']:.1f}%" if data["a_wins"] > data["b_wins"] else f"{model_b} {data['b_win_pct']:.1f}%"
            print(f"  {pair}: p={data['p_formatted']}{sig}, r={data['effect_size']:.2f}, {winner}")


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
    print("Performing Wilcoxon Signed-Rank tests...")
    results = perform_pairwise_tests(csv_data, context_lengths)

    # Output
    save_results(results, output_dir)
    print_summary(results)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
