#!/usr/bin/env python3
"""
Aggregate Phi-LoRA Experiments
Aggregates multiple Phi-LoRA experiment results by averaging inference times
and combining all context lengths into unified datasets.
"""

import pandas as pd
from pathlib import Path
import argparse
from typing import List, Dict

# Constants
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
PHI_LORA_DIR = RESULTS_DIR / "phi_lora"
OUTPUT_DIR = PHI_LORA_DIR / "avg_result"

# Experiment folders to aggregate
EXPERIMENT_FOLDERS = ["phi-4-Lora", "phi-4-Lora_2", "phi-4-Lora_3"]

# Context lengths available
CONTEXT_LENGTHS = [1024, 2048, 4096, 8192, 16384]

# Message types
MESSAGE_TYPES = ["with_message_msg0", "with_message_msg1"]


def aggregate_inference_times(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Aggregate multiple dataframes by averaging inference times.
    Other deterministic values are taken from the first dataframe.
    """
    if not dfs:
        raise ValueError("No dataframes to aggregate")
    
    # Use first dataframe as base (all other values are deterministic)
    result_df = dfs[0].copy()
    
    # Average inference times across all experiments
    inference_times = []
    for df in dfs:
        inference_times.append(df['inference_time'])
    
    # Calculate mean inference time
    result_df['inference_time'] = pd.concat(inference_times, axis=1).mean(axis=1)
    
    return result_df


def process_message_type(message_type: str) -> pd.DataFrame:
    """Process a single message type across all experiments and context lengths."""
    print(f"Processing {message_type}...")
    
    all_data = []
    
    for context_length in CONTEXT_LENGTHS:
        print(f"  Processing context length: {context_length}")
        
        # Collect dataframes from all experiments for this context length
        dfs_for_context = []
        
        for exp_folder in EXPERIMENT_FOLDERS:
            csv_path = PHI_LORA_DIR / exp_folder / message_type / f"Phi4_{context_length}.csv"
            
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                dfs_for_context.append(df)
                print(f"    Loaded {len(df)} rows from {exp_folder}")
            else:
                print(f"    Warning: {csv_path} not found")
        
        if dfs_for_context:
            # Aggregate inference times for this context length
            aggregated_df = aggregate_inference_times(dfs_for_context)
            all_data.append(aggregated_df)
            print(f"    Aggregated {len(aggregated_df)} rows for context {context_length}")
    
    if all_data:
        # Combine all context lengths into single dataframe
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"  Total combined rows: {len(combined_df)}")
        return combined_df
    else:
        raise ValueError(f"No data found for {message_type}")


def main():
    parser = argparse.ArgumentParser(description='Aggregate Phi-LoRA experiment results')
    parser.add_argument('--output-dir', type=str, help='Output directory (optional)')
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = OUTPUT_DIR
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🔄 Aggregating Phi-LoRA Experiments")
    print("=" * 50)
    print(f"Input directory: {PHI_LORA_DIR}")
    print(f"Output directory: {output_dir}")
    print(f"Experiment folders: {EXPERIMENT_FOLDERS}")
    print(f"Context lengths: {CONTEXT_LENGTHS}")
    
    # Process each message type
    for message_type in MESSAGE_TYPES:
        try:
            print(f"\n🚀 Processing {message_type}")
            
            # Aggregate data for this message type
            combined_df = process_message_type(message_type)
            
            # Create output directory for this message type
            msg_output_dir = output_dir / message_type
            msg_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save aggregated CSV (all context lengths combined)
            csv_output_path = msg_output_dir / "phi4_aggregated.csv"
            combined_df.to_csv(csv_output_path, index=False)
            
            print(f"✅ Saved aggregated data: {csv_output_path}")
            print(f"   Total rows: {len(combined_df)}")
            print(f"   Context lengths: {sorted(combined_df['context_len'].unique())}")
            print(f"   Average inference time: {combined_df['inference_time'].mean():.4f}s")
            
        except Exception as e:
            print(f"❌ Error processing {message_type}: {e}")
    
    print("\n🎉 Aggregation completed!")


if __name__ == "__main__":
    main()
