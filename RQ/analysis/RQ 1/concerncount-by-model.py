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

# Constants - Use root results directory (from project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
ANALYSIS_OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis" / "RQ1"


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

    # Load configuration
    config = load_config()
    script_config = config["scripts"]["concerncount_by_model"]

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        # Generate descriptive output directory name
        model_names = list(script_config["models"].keys())
        models_summary = "_".join(model_names)[:50]
        output_dir = (
            ANALYSIS_OUTPUT_DIR / f"pf_concerncount_{models_summary}_{timestamp}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Process models from config and collect data
    models_data = {}
    for model_name, model_config in script_config["models"].items():
        json_path = PROJECT_ROOT / model_config["path"]
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "metrics_by_concern" not in data:
            raise ValueError(f"Missing 'metrics_by_concern' in {json_path.name}")
        models_data[model_name] = data["metrics_by_concern"]

    metrics = [
        ("F1", "f1"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("Accuracy", "accuracy"),
        ("HS", "hamming_loss"),
    ]

    rows = []
    for concern_count in [1, 2, 3, 4, 5]:
        row = {"Count": concern_count}
        for model_name in script_config["models"].keys():
            concern_data = next(
                (
                    item
                    for item in models_data[model_name]
                    if item["concern_count"] == concern_count
                ),
                None,
            )
            if concern_data is None:
                raise ValueError(
                    f"Missing concern_count={concern_count} for model {model_name}"
                )
            for label, key in metrics:
                if key not in concern_data:
                    raise ValueError(
                        f"Missing metric '{key}' for model {model_name}, concern_count={concern_count}"
                    )
                row[f"{model_name} {label}"] = round(concern_data[key], 3)
        rows.append(row)

    # Create DataFrame and save CSV (2 decimal places)
    df = pd.DataFrame(rows)
    csv_path = output_dir / "performance_by_concern_count.csv"
    df.to_csv(csv_path, index=False, float_format="%.2f")

    # Dynamic header/print
    header = ["Count"] + [
        f"{model} {label}"
        for model in script_config["models"].keys()
        for label, _ in metrics
    ]
    print(" ".join(f"{h:<12}" for h in header))
    print("-" * (len(header) * 12))
    for _, row in df.iterrows():
        print(
            " ".join(
                f"{row[h]:<12.2f}" if h != "Count" else f"{int(row[h]):<12}"
                for h in header
            )
        )
    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()
