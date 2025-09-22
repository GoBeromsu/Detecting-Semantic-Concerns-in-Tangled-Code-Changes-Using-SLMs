#!/usr/bin/env python3
"""
Performance Summary: Model Comparison Analysis
Analyzes performance metrics from JSON experiment results and generates CSV summary table.
"""

import pandas as pd
import json
from pathlib import Path
import argparse

# Constants - Use root results directory (from project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
ANALYSIS_OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis" / "RQ1"


def main():
    parser = argparse.ArgumentParser(description='Generate performance summary comparison from JSON experiment results')
    parser.add_argument('json_files', nargs='+', help='JSON files to analyze')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        
        # Generate descriptive output directory name following efficiency_input_tokens.py convention
        file_stems = [Path(f).stem for f in args.json_files]
        files_summary = "_".join(file_stems)[:50]
        output_dir = ANALYSIS_OUTPUT_DIR / f"pf_{files_summary}_{timestamp}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process JSON files
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
        
        # Extract metrics, handle missing hamming_loss
        row = {
            'Model': model_name,
            'F1': round(metrics.get('f1', 0), 4),
            'Precision': round(metrics.get('precision', 0), 4),
            'Recall': round(metrics.get('recall', 0), 4),
            'Accuracy': round(metrics.get('accuracy', 0), 4),
            'Hamming Loss': round(metrics.get('hamming_loss', 0), 4) if 'hamming_loss' in metrics else None
        }
        rows.append(row)
    
    # Create DataFrame and save CSV
    df = pd.DataFrame(rows)
    csv_path = output_dir / "performance_summary_comparison.csv"
    
    # Format numeric columns to 2 decimal places
    numeric_columns = ['F1', 'Precision', 'Recall', 'Accuracy', 'Hamming Loss']
    for col in numeric_columns:
        df[col] = df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    
    df.to_csv(csv_path, index=False)
    
    # Print table
    print(f"{'Model':<20} {'F1':<8} {'Precision':<12} {'Recall':<8} {'Accuracy':<10} {'Hamming Loss':<12}")
    print("-" * 80)
    
    for _, row in df.iterrows():
        f1_str = row['F1'] if row['F1'] else "N/A"
        prec_str = row['Precision'] if row['Precision'] else "N/A"
        rec_str = row['Recall'] if row['Recall'] else "N/A"
        acc_str = row['Accuracy'] if row['Accuracy'] else "N/A"
        ham_str = row['Hamming Loss'] if row['Hamming Loss'] else "N/A"
        
        print(f"{row['Model']:<20} {f1_str:<8} {prec_str:<12} {rec_str:<8} {acc_str:<10} {ham_str:<12}")
    
    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()