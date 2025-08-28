#!/usr/bin/env python3

import json
from pathlib import Path
from collections import defaultdict

def average_numbers(values):
    return sum(values) / len(values)

def average_json_files(json_files):
    # Load all files
    data = [json.load(open(f)) for f in json_files]
    
    # Use first file as template
    result = data[0].copy()
    
    # Average macro metrics
    for field in ['accuracy', 'precision', 'recall', 'f1', 'hamming_loss', 'inference_time_avg']:
        values = [d['metrics_macro'][field] for d in data]
        result['metrics_macro'][field] = average_numbers(values)
    
    # Average concern metrics
    concern_groups = defaultdict(list)
    for d in data:
        for metric in d['metrics_by_concern']:
            concern_groups[metric['concern_count']].append(metric)
    
    result['metrics_by_concern'] = []
    for concern_count in sorted(concern_groups.keys()):
        metrics = concern_groups[concern_count]
        avg_metric = {'concern_count': concern_count}
        
        for field in ['accuracy', 'precision', 'recall', 'f1', 'hamming_loss', 'inference_time_avg']:
            values = [m[field] for m in metrics]
            avg_metric[field] = average_numbers(values)
        
        avg_metric['num_samples'] = sum(m['num_samples'] for m in metrics)
        result['metrics_by_concern'].append(avg_metric)
    
    # Update metadata
    result['num_samples'] = sum(d['num_samples'] for d in data)
    result['experiment_id'] = f"avg_{result['experiment_id']}"
    result['created_at'] = "averaged_results"
    
    return result

def main():
    base_path = Path("/Users/beomsu/Documents/Concern-is-All-You-Need/results/gpt")
    
    # Find all JSON files grouped by relative path
    file_groups = defaultdict(list)
    for timestamp_dir in base_path.iterdir():
        if timestamp_dir.is_dir() and timestamp_dir.name.startswith('202508'):
            for json_file in timestamp_dir.rglob("*.json"):
                relative_path = json_file.relative_to(timestamp_dir)
                file_groups[str(relative_path)].append(json_file)
    
    # Process each group
    for relative_path, files in file_groups.items():
        if len(files) < 2:
            continue
            
        print(f"Averaging {len(files)} files: {relative_path}")
        
        # Average and save
        averaged = average_json_files(files)
        output_file = base_path / "avg_result" / relative_path
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(averaged, f, indent=2)

if __name__ == "__main__":
    main()
