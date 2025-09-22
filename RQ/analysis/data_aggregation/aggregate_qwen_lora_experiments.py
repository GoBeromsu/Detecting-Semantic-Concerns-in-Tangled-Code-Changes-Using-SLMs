#!/usr/bin/env python3
"""
Aggregate Qwen3-14B-LoRA Experiments
Aggregates multiple Qwen3-14B-LoRA experiment results by averaging inference times
and combining all context lengths into unified datasets.
"""

import pandas as pd
from pathlib import Path
import argparse
from typing import List, Dict

# Constants
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
TARGET_DIR = RESULTS_DIR / "Qwen3-14B-LoRA"
OUTPUT_DIR = TARGET_DIR / "avg_result"

# Experiment folders to aggregate
EXPERIMENT_FOLDERS = ["20250918142638", "20250921185011", "20250921192953"]

# Message types
MESSAGE_TYPES = ["msg0", "msg1"]


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
    """Process a single message type across all experiments by finding CSV files."""
    print(f"Processing {message_type}...")

    all_data = []

    # Find all CSV files for this message type across experiments
    for exp_folder in EXPERIMENT_FOLDERS:
        message_dir = TARGET_DIR / exp_folder / message_type

        if not message_dir.exists():
            print(f"  Warning: Directory {message_dir} not found")
            continue

        # Find all CSV files in this message type directory
        csv_files = list(message_dir.glob("*.csv"))

        if not csv_files:
            print(f"  Warning: No CSV files found in {message_dir}")
            continue

        for csv_file in csv_files:
            print(f"    Processing {csv_file.name}")

            try:
                df = pd.read_csv(csv_file)
                all_data.append(df)
                print(f"      Loaded {len(df)} rows")
            except Exception as e:
                print(f"      Error loading {csv_file.name}: {e}")

    if all_data:
        # Combine all dataframes into single dataframe
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"  Total combined rows: {len(combined_df)}")
        return combined_df
    else:
        raise ValueError(f"No data found for {message_type}")


def main():
    parser = argparse.ArgumentParser(description='Aggregate Qwen3-14B-LoRA experiment results')
    parser.add_argument('--output-dir', type=str, help='Output directory (optional)')

    args = parser.parse_args()

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = OUTPUT_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    print("🔄 Aggregating Qwen3-14B-LoRA Experiments")
    print("=" * 50)
    print(f"Input directory: {TARGET_DIR}")
    print(f"Output directory: {output_dir}")
    print(f"Experiment folders: {EXPERIMENT_FOLDERS}")


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
            csv_output_path = msg_output_dir / "qwen_aggregated.csv"
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