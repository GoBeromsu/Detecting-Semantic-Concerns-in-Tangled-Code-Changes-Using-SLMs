#!/usr/bin/env python3
"""
Message Impact Pairwise P-Value Table Generator

Statistical Test Choice:
- Wilcoxon Signed-Rank Test (paired, non-parametric) is used because:
  1. Each commit is evaluated with/without message, creating naturally paired samples
  2. Hamming loss distributions are often non-normal and bounded [0, 1]
  3. Commit-level pairing controls for per-commit difficulty variance

Effect Size Interpretation:
- Rank-biserial correlation (r) is computed from diff = HS_without - HS_with
- r > 0: with message tends to have lower HS (better performance)
- r < 0: without message tends to have lower HS (better performance)
- |r| > 0.5: large effect, 0.3-0.5: medium, < 0.3: small
"""

import pandas as pd
import yaml
import json
from pathlib import Path
import argparse
from scipy.stats import wilcoxon, rankdata
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
        "effect_size_metric": "rank-biserial correlation",
        "significance_threshold": P_VALUE_THRESHOLD,
        "models": model_names,
        "comparison": "Without Message vs With Message"
    }}

    for model_name in model_names:
        data_msg0 = csv_data[model_name]["msg0"]["hamming_loss"].values
        data_msg1 = csv_data[model_name]["msg1"]["hamming_loss"].values

        # Fail fast if pairing is broken
        if len(data_msg0) != len(data_msg1):
            raise ValueError(f"Length mismatch for {model_name}: "
                             f"msg0={len(data_msg0)} vs msg1={len(data_msg1)}")

        if len(data_msg0) == 0:
            continue

        # Wilcoxon signed-rank test
        try:
            result = wilcoxon(data_msg0, data_msg1, alternative='two-sided')
            p_value = result.pvalue
        except ValueError:
            p_value = 1.0

        # Rank-biserial correlation: r = (W+ - W-) / (W+ + W-)
        # Based on diff = HS_msg0 - HS_msg1, so r > 0 means msg1 has lower HS (better)
        diff = data_msg0 - data_msg1
        diff_nz = diff[diff != 0]
        if len(diff_nz) > 0:
            ranks = rankdata(np.abs(diff_nz), method='average')
            w_plus, w_minus = np.sum(ranks[diff_nz > 0]), np.sum(ranks[diff_nz < 0])
            effect_size = (w_plus - w_minus) / (w_plus + w_minus) if (w_plus + w_minus) > 0 else 0.0
        else:
            effect_size = 0.0

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
    """Save results as JSON and CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    with open(output_dir / "msg_impact_pairwise_pvalues.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # CSV with p-values only: numeric, no special characters
    # p < 0.001 shown as 0.001 (caption should note this)
    model_names = results["summary"]["models"]

    # Column name mapping for CSV
    model_to_col = {
        "GPT-4.1": "p_GPT",
        "Qwen": "p_Qwen",
        "Qwen (FT)": "p_QwenFT"
    }

    row = {"Comparison": "msg0 vs msg1"}
    for model_name in model_names:
        data = results["by_model"].get(model_name, {})
        col = model_to_col.get(model_name, f"p_{model_name}")
        row[col] = data["p_formatted"] if data else ""

    pd.DataFrame([row]).to_csv(output_dir / "msg_impact_pairwise_pvalues.csv", index=False)

    # Appendix CSV: detailed statistics (p-value, effect size, win counts, means)
    # Convention: p (scientific 2 decimals), r (2 decimals), mean (3 decimals), pct (1 decimal)
    appendix_rows = []
    for model_name in model_names:
        data = results["by_model"].get(model_name, {})
        if data:
            appendix_rows.append({
                "Model": model_name,
                "p": f"{data['p_value']:.2e}",
                "r": f"{data['effect_size']:.2f}",
                "n": data["n"],
                "msg0_mean": f"{data['msg0_mean']:.3f}",
                "msg1_mean": f"{data['msg1_mean']:.3f}",
                "msg0_wins": data["msg0_wins"],
                "msg1_wins": data["msg1_wins"],
                "msg0_pct": f"{data['msg0_win_pct']:.1f}",
                "msg1_pct": f"{data['msg1_win_pct']:.1f}"
            })

    pd.DataFrame(appendix_rows).to_csv(output_dir / "msg_impact_pairwise_appendix.csv", index=False)


def print_summary(results: dict):
    """Print summary to console."""
    print(f"\nTest: {results['summary']['test_method']}")
    print(f"Comparison: {results['summary']['comparison']}")
    print("Note: r > 0 means 'With Message' has lower HS (better).")
    print("-" * 80)

    for model_name, data in results["by_model"].items():
        sig = "*" if data["significant"] else ""
        winner = "With Msg" if data["effect_size"] > 0 else "Without Msg"
        print(f"\n{model_name}:")
        print(f"  p={data['p_formatted']}{sig}, r={data['effect_size']:.2f}")
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
