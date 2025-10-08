#!/usr/bin/env python3
"""
Performance Summary: Model Comparison Analysis
Analyzes performance metrics from JSON experiment results and generates CSV summary table.
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
    "Qwen (FT)": "QwenFT",
    "Qwen (Fine-tuned)": "QwenFT"
}

METRIC_KEY = "hamming_loss"


def main():
    parser = argparse.ArgumentParser(description='Generate performance summary comparison from JSON experiment results')
    parser.add_argument('json_files', nargs='+', help='JSON files to analyze')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    
    args = parser.parse_args()
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        
        file_stems = [Path(f).stem for f in args.json_files]
        files_summary = "_".join(file_stems)[:50]
        output_dir = ANALYSIS_OUTPUT_DIR / f"pf_{files_summary}_{timestamp}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rows = []
    json_paths = [Path(f) for f in args.json_files]
    
    for json_path in json_paths:
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")
            
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'metrics_macro' not in data:
            raise ValueError(f"Missing 'metrics_macro' in {json_path.name}")
        
        metrics = data['metrics_macro']
        
        # Determine model name based on JSON content and path
        if 'gpt' in str(json_path):
            model_name = 'GPT-4.1'
        elif 'Qwen3-14B-LoRA' in str(json_path):
            model_name = 'Qwen (Fine-tuned)'
        elif 'Qwen' in str(json_path):
            model_name = 'Qwen'
        else:
            model_name = json_path.stem  # fallback to filename
        
        clean_name = MODEL_NAME_MAP.get(model_name, model_name)
        rows.append({
            'Model': clean_name,
            'Value': round(metrics.get(METRIC_KEY, 0), 2)
        })

    # Transform to wide format with models as columns
    df_long = pd.DataFrame(rows)
    data_dict = {'Metric': 'Hamming Loss'}
    for row in rows:
        data_dict[row['Model']] = row['Value']

    df = pd.DataFrame([data_dict])
    csv_path = output_dir / "performance_summary.csv"
    df.to_csv(csv_path, index=False, float_format="%.2f")

    # Print in the same format as concerncount-by-model.py
    columns = list(df.columns)
    print(" ".join(f"{h:<12}" for h in columns))
    print("-" * (len(columns) * 12))
    for _, row in df.iterrows():
        print(
            " ".join(
                f"{row[h]:<12.2f}" if h != "Metric" else f"{row[h]:<12}"
                for h in columns
            )
        )
    
    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()