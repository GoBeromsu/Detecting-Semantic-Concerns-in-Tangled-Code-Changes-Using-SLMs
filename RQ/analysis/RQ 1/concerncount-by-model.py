#!/usr/bin/env python3
"""
Performance by Concern Count: Model Comparison Analysis
Analyzes performance metrics by concern count from JSON experiment results and generates CSV table.
"""

import pandas as pd
import json
import yaml
from pathlib import Path
import argparse

# Constants
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
ANALYSIS_OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis" / "RQ1"

MODEL_NAME_MAP = {"GPT-4.1": "GPT-4.1", "Qwen": "Qwen", "Qwen (FT)": "QwenFT"}

METRIC_KEY = "hamming_loss"


def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Generate concern count performance comparison from JSON experiment results"
    )
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument(
        "--use-config",
        action="store_true",
        default=True,
        help="Use config.yaml for model configuration",
    )

    args = parser.parse_args()

    config = load_config()
    script_config = config["scripts"]["concerncount_by_model"]

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        model_names = list(script_config["models"].keys())
        models_summary = "_".join(model_names)[:50]
        output_dir = (
            ANALYSIS_OUTPUT_DIR / f"pf_concerncount_{models_summary}_{timestamp}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    models_data = {}
    for model_name, model_config in script_config["models"].items():
        csv_path = PROJECT_ROOT / model_config["path"]
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        # Calculate macro average for Hamming Loss by concern count
        metrics_by_concern = []
        for concern_count in sorted(df["concern_count"].unique()):
            subset = df[df["concern_count"] == concern_count]
            metrics_by_concern.append(
                {
                    "concern_count": int(concern_count),
                    "hamming_loss": float(subset["hamming_loss"].mean()),
                }
            )

        clean_name = MODEL_NAME_MAP.get(model_name, model_name)
        models_data[clean_name] = metrics_by_concern

    rows = []
    for concern_count in [1, 2, 3, 4, 5]:
        row = {"Count": concern_count}
        for clean_name, concern_list in models_data.items():
            concern_data = next(
                (
                    item
                    for item in concern_list
                    if item["concern_count"] == concern_count
                ),
                None,
            )
            if concern_data is None:
                raise ValueError(
                    f"Missing concern_count={concern_count} for model {clean_name}"
                )

            if METRIC_KEY not in concern_data:
                raise ValueError(
                    f"Missing {METRIC_KEY} for model {clean_name}, concern_count={concern_count}"
                )
            row[clean_name] = round(concern_data[METRIC_KEY], 3)
        rows.append(row)

    df_complete = pd.DataFrame(rows)
    columns = ["Count"] + list(models_data.keys())
    df_hs = df_complete[columns].copy()
    csv_path = output_dir / "performance_by_concern_count.csv"
    df_hs.to_csv(csv_path, index=False, float_format="%.2f")

    print(" ".join(f"{h:<12}" for h in columns))
    print("-" * (len(columns) * 12))
    for _, row in df_hs.iterrows():
        print(
            " ".join(
                f"{row[h]:<12.2f}" if h != "Count" else f"{int(row[h]):<12}"
                for h in columns
            )
        )
    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()
