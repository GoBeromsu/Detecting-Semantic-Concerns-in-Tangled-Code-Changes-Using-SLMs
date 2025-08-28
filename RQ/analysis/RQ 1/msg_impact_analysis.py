#!/usr/bin/env python3
"""
Message Impact Analysis: Model Comparison
Analyzes the impact of commit messages on model performance across GPT-4.1, Phi-4, and Phi-4(FT).
"""

import pandas as pd
import json
from pathlib import Path
import argparse

# Constants - Use root results directory (from project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
ANALYSIS_OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis" / "RQ1"

# Default model configurations (fallback if no config provided)
DEFAULT_MODEL_CONFIGS = {
    'GPT-4.1': {
        'msg0_path': "results/gpt/avg_result/msg0/json/16384_zs.json",
        'msg1_path': "results/gpt/avg_result/msg1/json/16384_zs.json"
    },
    'Phi-4': {
        'msg0_path': "results/phi/avg_result/msg0/json/16384_zs_filtered.json",
        'msg1_path': "results/phi/avg_result/msg1/json/16384_zs_filtered.json"
    },
    'Phi-4 (FT)': {
        'msg0_path': "results/phi_lora/avg_result/with_message_msg0/json/Phi4_16384.json",
        'msg1_path': "results/phi_lora/avg_result/with_message_msg1/json/Phi4_16384.json"
    }
}


def load_config_from_file(config_path: Path) -> dict:
    """Load model configuration from JSON file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
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
        model_configs = config.get('models', DEFAULT_MODEL_CONFIGS).copy()
        project_root = Path(config.get('project_root', PROJECT_ROOT))

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
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'metrics_macro' not in data:
        raise ValueError(f"Missing 'metrics_macro' in {json_path.name}")
    
    return data['metrics_macro']


def generate_f1_comparison(model_configs: dict) -> pd.DataFrame:
    """Generate F1 comparison table."""
    rows = []
    
    for model_name, paths in model_configs.items():
        # Load metrics
        msg0_metrics = load_metrics(paths['msg0_path'])
        msg1_metrics = load_metrics(paths['msg1_path'])
        
        # Extract F1 scores
        without_msg_f1 = msg0_metrics['f1']
        with_msg_f1 = msg1_metrics['f1']
        delta_f1 = with_msg_f1 - without_msg_f1
        
        row = {
            'Model': model_name,
            'Without Msg': f"{without_msg_f1:.3f}",
            'With Msg': f"{with_msg_f1:.3f}",
            'Delta': f"{delta_f1:+.3f}"
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


def generate_full_comparison(model_configs: dict) -> pd.DataFrame:
    """Generate full metrics comparison in long format."""
    rows = []
    metrics_to_compare = ['f1', 'precision', 'recall', 'accuracy']
    
    for model_name, paths in model_configs.items():
        # Load metrics
        msg0_metrics = load_metrics(paths['msg0_path'])
        msg1_metrics = load_metrics(paths['msg1_path'])
        
        for metric in metrics_to_compare:
            without_msg = msg0_metrics[metric]
            with_msg = msg1_metrics[metric]
            delta = with_msg - without_msg
            
            row = {
                'Model': model_name,
                'Metric': metric.upper(),
                'Without Msg': f"{without_msg:.3f}",
                'With Msg': f"{with_msg:.3f}",
                'Delta': f"{delta:+.3f}"
            }
            rows.append(row)
    
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description='Generate message impact analysis from avg_result JSON files')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--config', type=str, help='JSON configuration file with model paths')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config_from_file(Path(args.config))
        model_configs = build_model_configs(config)
        print(f"✅ Loaded configuration from {args.config}")
    else:
        model_configs = build_model_configs()
        print("📋 Using default model configuration")
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        output_dir = ANALYSIS_OUTPUT_DIR / f"msg_impact_{timestamp}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate F1 comparison
    f1_df = generate_f1_comparison(model_configs)
    f1_csv_path = output_dir / "msg_impact_f1.csv"
    f1_df.to_csv(f1_csv_path, index=False)
    
    # Generate full comparison
    full_df = generate_full_comparison(model_configs)
    full_csv_path = output_dir / "msg_impact_full.csv"
    full_df.to_csv(full_csv_path, index=False)
    
    # Print F1 comparison table
    print("=== F1 Score Impact Analysis ===")
    print(f"{'Model':<15} {'Without Msg':<12} {'With Msg':<10} {'Delta':<10}")
    print("-" * 50)
    
    for _, row in f1_df.iterrows():
        print(f"{row['Model']:<15} {row['Without Msg']:<12} {row['With Msg']:<10} {row['Delta']:<10}")
    
    print(f"\n=== Full Metrics Comparison (Sample) ===")
    print(full_df.head(12).to_string(index=False))
    
    print(f"\nResults saved to:")
    print(f"  F1 comparison: {f1_csv_path}")
    print(f"  Full comparison: {full_csv_path}")


if __name__ == "__main__":
    main()
