#!/usr/bin/env python3
"""
Context Length Performance Analysis: Model Comparison
Analyzes model performance across different context lengths (input tokens).
"""

import pandas as pd
import json
from pathlib import Path
import argparse
import yaml

# Constants
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
ANALYSIS_OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis" / "RQ1"

MODEL_NAME_MAP = {
    "GPT-4.1": "GPT4_1",
    "Qwen": "Qwen",
    "Qwen (FT)": "QwenFT"
}

METRIC_KEY = "hamming_loss"


def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_metrics(json_path: Path) -> dict:
    """Load metrics from JSON file."""
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "metrics_macro" not in data:
        raise ValueError(f"Missing 'metrics_macro' in {json_path.name}")

    return data["metrics_macro"]


def generate_context_length_comparison(config: dict) -> pd.DataFrame:
    """Generate context length performance comparison table (Hamming Loss only)."""

    rows = []
    for context_length in config["context_lengths"]:
        row = {"ContextLength": context_length}
        for model_name, model_config in config["models"].items():
            # Build file path using pattern
            file_path = PROJECT_ROOT / model_config["path_pattern"].format(
                context=context_length
            )
            # Qwen 파일 경로 분기(기존 로직 유지)
            if model_name == "Qwen":
                base_path = (
                    PROJECT_ROOT / f"results/Qwen/avg_result/msg1/json/{context_length}"
                )
                if (base_path.parent / f"{context_length}_zs_filtered.json").exists():
                    file_path = base_path.parent / f"{context_length}_zs_filtered.json"
                else:
                    file_path = base_path.parent / f"{context_length}_zs.json"
            metrics_data = load_metrics(file_path)
            clean_model_name = MODEL_NAME_MAP.get(model_name, model_name)
            if METRIC_KEY not in metrics_data:
                raise ValueError(
                    f"Missing metric '{METRIC_KEY}' for model {model_name}, context_length={context_length}"
                )
            row[clean_model_name] = round(metrics_data[METRIC_KEY], 3)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Generate context length performance comparison table"
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
    script_config = config["scripts"]["context_length_performance"]

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        model_names = list(script_config["models"].keys())
        models_summary = "_".join(model_names)[:50]
        output_dir = ANALYSIS_OUTPUT_DIR / f"context_length_{models_summary}_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)

    df = generate_context_length_comparison(script_config)
    csv_path = output_dir / "context_length_performance.csv"
    df.to_csv(csv_path, index=False, float_format="%.2f")

    header = ["ContextLength"] + [MODEL_NAME_MAP.get(name, name) for name in script_config["models"].keys()]
    print(" ".join(f"{h:<12}" for h in header))
    print("-" * (len(header) * 12))
    for _, row in df.iterrows():
        print(
            " ".join(
                f"{row[h]:<12.2f}" if h != "ContextLength" else f"{int(row[h]):<12}"
                for h in header
            )
        )

    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()
