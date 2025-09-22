#!/usr/bin/env python3
"""
Context Length Performance Analysis: Model Comparison
Analyzes model performance across different context lengths (input tokens).
"""

import pandas as pd
import json
from pathlib import Path
import argparse

# Constants - Use root results directory (from project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
ANALYSIS_OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis" / "RQ1"

# Default configuration
DEFAULT_CONFIG = {
    'models': {
        'GPT-4.1': {
            'path_pattern': "results/gpt/avg_result/msg1/json/{context}_zs.json"
        },
        'Qwen': {
            'path_pattern': "results/Qwen/avg_result/msg1/json/{context}_zs_filtered.json"
        },
        'Qwen (FT)': {
            'path_pattern': "results/Qwen3-14B-LoRA/avg_result/msg1/json/{context}_zs.json"
        }
    },
    'context_lengths': [1024, 2048, 4096, 8192, 12288]
}


def load_metrics(json_path: Path) -> dict:
    """Load metrics from JSON file."""
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'metrics_macro' not in data:
        raise ValueError(f"Missing 'metrics_macro' in {json_path.name}")
    
    return data['metrics_macro']


def generate_context_length_comparison(config: dict) -> pd.DataFrame:
    """Generate context length performance comparison table."""
    rows = []
    metrics_order = ['f1', 'precision', 'recall', 'accuracy', 'hamming_loss']
    metric_names = ['F1', 'Precision', 'Recall', 'Accuracy', 'HS']
    
    # Model name mapping for cleaner column names
    model_name_map = {
        'GPT-4.1': 'GPT41',
        'Qwen': 'Qwen',
        'Qwen (FT)': 'QwenFT'
    }
    
    for context_length in config['context_lengths']:
        row = {'ContextLength': context_length}
        
        for model_name, model_config in config['models'].items():
            # Build file path using pattern
            file_path = PROJECT_ROOT / model_config['path_pattern'].format(context=context_length)

            # Get clean model name
            clean_model_name = model_name_map.get(model_name, model_name)

            try:
                # For Qwen model, determine file pattern based on context length
                if model_name == 'Qwen':
                    # Check which file exists for this context length
                    base_path = PROJECT_ROOT / f"results/Qwen/avg_result/msg1/json/{context_length}"
                    if (base_path.parent / f"{context_length}_zs_filtered.json").exists():
                        file_path = base_path.parent / f"{context_length}_zs_filtered.json"
                    else:
                        file_path = base_path.parent / f"{context_length}_zs.json"

                metrics = load_metrics(file_path)

                # Add metrics for this model
                for metric_key, metric_name in zip(metrics_order, metric_names):
                    column_name = f'{clean_model_name}_{metric_name}'
                    if metric_key == 'hamming_loss':
                        value = metrics.get('hamming_loss', 0)
                    else:
                        value = metrics.get(metric_key, 0)

                    row[column_name] = f"{value:.4f}"

            except (FileNotFoundError, ValueError) as e:
                print(f"⚠️  Warning: Could not load {file_path}: {e}")
                # Fill with NaN values for missing data (will be displayed as empty in CSV)
                for metric_name in metric_names:
                    column_name = f'{clean_model_name}_{metric_name}'
                    row[column_name] = None  # This will become NaN in pandas
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description='Generate context length performance comparison')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    
    args = parser.parse_args()
    
    # Use default configuration
    config = DEFAULT_CONFIG
    print("📋 Using default configuration for context length analysis")
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        output_dir = ANALYSIS_OUTPUT_DIR / f"context_length_performance_{timestamp}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate comparison
    df = generate_context_length_comparison(config)
    csv_path = output_dir / "context_length_performance.csv"
    # Save to CSV with NaN values displayed as empty strings
    df.to_csv(csv_path, index=False, na_rep='')
    
    # Print table
    print("=== Context Length Performance Analysis ===")
    print(df.to_string(index=False))
    
    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()
