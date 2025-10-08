#!/usr/bin/env python3
"""
Message Impact Analysis: Model Comparison
Analyzes the impact of commit messages on model performance across GPT-4.1, Qwen, and Qwen(FT).
"""

import pandas as pd
import json
from pathlib import Path
import argparse

# Constants
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
ANALYSIS_OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis" / "RQ1"

MODEL_NAME_MAP = {
    "GPT-4.1": "GPT4_1",
    "Qwen": "Qwen",
    "Qwen (FT)": "QwenFT"
}

METRIC_KEY = "hamming_loss"

DEFAULT_MODEL_CONFIGS = {
    "GPT-4.1": {
        "msg0_path": "results/gpt/avg_result/msg0/json/12288_zs.json",
        "msg1_path": "results/gpt/avg_result/msg1/json/12288_zs.json",
    },
    "Qwen": {
        "msg0_path": "results/Qwen/avg_result/msg0/json/12288_zs_filtered.json",
        "msg1_path": "results/Qwen/avg_result/msg1/json/12288_zs.json",
    },
    "Qwen (FT)": {
        "msg0_path": "results/Qwen3-14B-LoRA/avg_result/msg0/json/12288_zs.json",
        "msg1_path": "results/Qwen3-14B-LoRA/avg_result/msg1/json/12288_zs.json",
    },
}


def load_config_from_file(config_path: Path) -> dict:
    """Load model configuration from JSON file."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config
    except Exception as e:
        raise ValueError(f"Failed to load config from {config_path}: {e}")


def build_model_configs(config: dict = None) -> dict:
    """Build model configurations with absolute paths."""
    if config is None:
        # Use default configuration
        model_configs = DEFAULT_MODEL_CONFIGS.copy()
        project_root = PROJECT_ROOT
    else:
        # Use provided configuration
        model_configs = config.get("models", DEFAULT_MODEL_CONFIGS).copy()
        project_root = Path(config.get("project_root", PROJECT_ROOT))

    # Convert relative paths to absolute paths
    for model_name, paths in model_configs.items():
        for key, path in paths.items():
            if isinstance(path, str):
                model_configs[model_name][key] = project_root / path
            else:
                model_configs[model_name][key] = Path(path)

    return model_configs


def load_metrics(json_path: Path) -> dict:
    """Load metrics from JSON file."""
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "metrics_macro" not in data:
        raise ValueError(f"Missing 'metrics_macro' in {json_path.name}")

    return data["metrics_macro"]


def generate_comparison(model_configs: dict) -> pd.DataFrame:
    """Generate comparison table for Hamming Loss with models as columns."""
    data = {
        "Condition": ["Without Msg", "With Msg", "Delta"]
    }

    for model_name, paths in model_configs.items():
        msg0_metrics = load_metrics(paths["msg0_path"])
        msg1_metrics = load_metrics(paths["msg1_path"])

        without_msg = round(msg0_metrics[METRIC_KEY], 2)
        with_msg = round(msg1_metrics[METRIC_KEY], 2)
        delta = with_msg - without_msg  # Delta from rounded values

        clean_name = MODEL_NAME_MAP.get(model_name, model_name)
        data[clean_name] = [without_msg, with_msg, delta]

    return pd.DataFrame(data)


def generate_delta_only_comparison(model_configs: dict) -> pd.DataFrame:
    """Generate comparison table with only delta values."""
    data = {"Metric": "Delta"}

    for model_name, paths in model_configs.items():
        msg0_metrics = load_metrics(paths["msg0_path"])
        msg1_metrics = load_metrics(paths["msg1_path"])

        clean_name = MODEL_NAME_MAP.get(model_name, model_name)
        without_msg = round(msg0_metrics[METRIC_KEY], 2)
        with_msg = round(msg1_metrics[METRIC_KEY], 2)
        delta = with_msg - without_msg  # Delta from rounded values

        data[clean_name] = delta

    return pd.DataFrame([data])




def main():
    parser = argparse.ArgumentParser(
        description="Generate message impact analysis from avg_result JSON files"
    )
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument(
        "--config", type=str, help="JSON configuration file with model paths"
    )

    args = parser.parse_args()

    if args.config:
        config = load_config_from_file(Path(args.config))
        model_configs = build_model_configs(config)
        print(f"✅ Loaded configuration from {args.config}")
    else:
        model_configs = build_model_configs()
        print("📋 Using default model configuration")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_dir = ANALYSIS_OUTPUT_DIR / f"msg_impact_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate main comparison with all conditions
    comparison_df = generate_comparison(model_configs)
    comparison_csv_path = output_dir / "msg_impact_comparison.csv"
    comparison_df.to_csv(comparison_csv_path, index=False, float_format="%.2f")

    # Generate delta-only comparison
    delta_df = generate_delta_only_comparison(model_configs)
    delta_csv_path = output_dir / "msg_impact_delta.csv"
    delta_df.to_csv(delta_csv_path, index=False, float_format="%.2f")

    # Print main comparison table
    print("=== Message Impact Analysis ===")
    columns = list(comparison_df.columns)
    print(" ".join(f"{h:<12}" for h in columns))
    print("-" * (len(columns) * 12))
    for _, row in comparison_df.iterrows():
        print(
            " ".join(
                f"{row[h]:<12.2f}" if h != "Condition" else f"{row[h]:<12}"
                for h in columns
            )
        )

    # Print delta table
    print("\n=== Delta Values ===")
    delta_columns = list(delta_df.columns)
    print(" ".join(f"{h:<12}" for h in delta_columns))
    print("-" * (len(delta_columns) * 12))
    for _, row in delta_df.iterrows():
        print(
            " ".join(
                f"{row[h]:+12.2f}" if h != "Metric" else f"{row[h]:<12}"
                for h in delta_columns
            )
        )

    print(f"\nResults saved to:")
    print(f"  Comparison: {comparison_csv_path}")
    print(f"  Delta: {delta_csv_path}")


if __name__ == "__main__":
    main()
