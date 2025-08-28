#!/usr/bin/env python3
"""
Performance by Concern Count: Model Comparison Analysis
Analyzes performance metrics by concern count from JSON experiment results and generates CSV table.
"""

import pandas as pd
import json
from pathlib import Path
import argparse

# Constants - Use root results directory (from project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
ANALYSIS_OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis"


def main():
    parser = argparse.ArgumentParser(description='Generate concern count performance comparison from JSON experiment results')
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
        output_dir = ANALYSIS_OUTPUT_DIR / f"pf_concerncount_{files_summary}_{timestamp}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process JSON files and collect data
    models_data = {}
    json_paths = [Path(f) for f in args.json_files]
    
    for json_path in json_paths:
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")
            
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'metrics_by_concern' not in data:
            raise ValueError(f"Missing 'metrics_by_concern' in {json_path.name}")
        
        # Determine model name
        if 'gpt' in str(json_path) or data.get('model') == '16384':
            model_name = 'GPT-4.1'
        elif 'Lora' in str(json_path) or data.get('model') == 'Phi4':
            model_name = 'Phi-4 (FT)'
        elif data.get('model') == 'Phi-4' or 'zero_shot' in str(json_path):
            model_name = 'Phi-4'
        else:
            model_name = json_path.stem
        
        models_data[model_name] = data['metrics_by_concern']
    
    # Build CSV rows directly from metrics_by_concern data
    rows = []
    for concern_count in [1, 2, 3, 4, 5]:
        row = {'Count': concern_count}
        
        for model_name in ['GPT-4.1', 'Phi-4', 'Phi-4 (FT)']:
            if model_name in models_data:
                # Find the concern count data
                concern_data = next((item for item in models_data[model_name] 
                                   if item['concern_count'] == concern_count), {})
                
                row[f'{model_name} F1'] = f"{concern_data.get('f1', 0):.3f}" if concern_data else ""
                row[f'{model_name} Precision'] = f"{concern_data.get('precision', 0):.3f}" if concern_data else ""
                row[f'{model_name} Recall'] = f"{concern_data.get('recall', 0):.3f}" if concern_data else ""
                row[f'{model_name} Accuracy'] = f"{concern_data.get('accuracy', 0):.3f}" if concern_data else ""
                row[f'{model_name} HS'] = f"{concern_data.get('hamming_loss', 0):.3f}" if concern_data and 'hamming_loss' in concern_data else ""
        
        rows.append(row)
    
    # Create DataFrame and save CSV
    df = pd.DataFrame(rows)
    csv_path = output_dir / "performance_by_concern_count.csv"
    df.to_csv(csv_path, index=False)
    
    # Print table
    print(f"{'Count':<5} {'GPT-4.1 F1':<10} {'GPT-4.1 Prec':<12} {'GPT-4.1 Rec':<11} {'GPT-4.1 Acc':<11} {'GPT-4.1 HS':<10} {'Phi-4 F1':<9} {'Phi-4 Prec':<11} {'Phi-4 Rec':<10} {'Phi-4 Acc':<10} {'Phi-4 HS':<9} {'Phi-4(FT) F1':<12} {'Phi-4(FT) Prec':<14} {'Phi-4(FT) Rec':<13} {'Phi-4(FT) Acc':<13} {'Phi-4(FT) HS':<11}")
    print("-" * 160)
    
    for _, row in df.iterrows():
        print(f"{row['Count']:<5} {row.get('GPT-4.1 F1', ''):<10} {row.get('GPT-4.1 Precision', ''):<12} {row.get('GPT-4.1 Recall', ''):<11} {row.get('GPT-4.1 Accuracy', ''):<11} {row.get('GPT-4.1 HS', ''):<10} {row.get('Phi-4 F1', ''):<9} {row.get('Phi-4 Precision', ''):<11} {row.get('Phi-4 Recall', ''):<10} {row.get('Phi-4 Accuracy', ''):<10} {row.get('Phi-4 HS', ''):<9} {row.get('Phi-4 (FT) F1', ''):<12} {row.get('Phi-4 (FT) Precision', ''):<14} {row.get('Phi-4 (FT) Recall', ''):<13} {row.get('Phi-4 (FT) Accuracy', ''):<13} {row.get('Phi-4 (FT) HS', ''):<11}")
    
    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()
