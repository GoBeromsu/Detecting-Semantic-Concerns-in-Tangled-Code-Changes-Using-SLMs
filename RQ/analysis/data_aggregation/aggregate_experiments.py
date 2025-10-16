#!/usr/bin/env python3
"""
Generic Experiment Aggregation Script
Concatenates experiment results by token length prefix and creates full aggregated files.
Groups files by token length (e.g., 1024_zs.csv, 1024_zs_filtered.csv -> 1024 group).
"""

import pandas as pd
from pathlib import Path
import argparse
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
MESSAGE_TYPES = ["msg0", "msg1"]


def extract_token_length(filename: str) -> int:
    """Extract token length from filename (e.g., '1024_zs.csv' -> 1024)."""
    parts = filename.split("_")
    if parts and parts[0].isdigit():
        return int(parts[0])
    return None


def setup(model_dir: Path, message_type: str) -> Tuple[List[Path], List[int]]:
    """
    Setup: discover experiment folders and token lengths.
    Returns (experiment_folders, token_lengths)
    """
    if not model_dir.exists():
        raise ValueError(f"Model directory not found: {model_dir}")

    # Discover experiment folders (exclude avg_result)
    experiment_folders = sorted(
        [
            folder
            for folder in model_dir.iterdir()
            if folder.is_dir() and folder.name != "avg_result"
        ]
    )

    if not experiment_folders:
        return [], []

    # Discover unique token lengths across all experiments
    token_lengths = set()
    for exp_folder in experiment_folders:
        msg_dir = exp_folder / message_type
        if msg_dir.exists():
            for csv_file in msg_dir.glob("*.csv"):
                token_len = extract_token_length(csv_file.name)
                if token_len:
                    token_lengths.add(token_len)

    return experiment_folders, sorted(token_lengths)


def process_model(model_name: str, aggregated_basename: str = None) -> None:
    """
    Process all experiments for a single model.
    """
    print(f"\n{'='*60}")
    print(f"Processing model: {model_name}")
    print(f"{'='*60}")

    model_dir = RESULTS_DIR / model_name
    output_dir = model_dir / "avg_result"

    # Determine aggregated filename
    if aggregated_basename:
        full_csv_basename = aggregated_basename
    else:
        base_name = model_name.split("-")[0].split(" ")[0].lower()
        full_csv_basename = f"{base_name}_aggregated"

    for message_type in MESSAGE_TYPES:
        print(f"\n📂 Processing {message_type}...")

        # Setup: discover folders and token lengths
        experiment_folders, token_lengths = setup(model_dir, message_type)

        if not experiment_folders:
            print(f"  ❌ No experiment folders found")
            continue

        if not token_lengths:
            print(f"  ❌ No CSV files found")
            continue

        print(
            f"  Found {len(experiment_folders)} experiments, {len(token_lengths)} token lengths"
        )

        msg_output_dir = output_dir / message_type
        msg_output_dir.mkdir(parents=True, exist_ok=True)

        all_token_dfs = []

        # Process each token length
        for token_length in token_lengths:
            print(f"\n  🔄 Token length: {token_length}...")

            # Collect all CSV files matching this token length prefix from all experiments
            csv_paths = []
            for exp_folder in experiment_folders:
                msg_dir = exp_folder / message_type
                if msg_dir.exists():
                    # Find all files starting with this token length
                    matching_files = [f for f in msg_dir.glob(f"{token_length}_*.csv")]
                    csv_paths.extend(matching_files)

            if not csv_paths:
                print(f"     ⚠️  No files found, skipping")
                continue

            # Concatenate all matching files
            dfs = [pd.read_csv(p) for p in csv_paths]
            aggregated_df = pd.concat(dfs, ignore_index=True)

            # Save token-length specific CSV (use standard naming)
            output_filename = f"{token_length}_zs.csv"
            output_path = msg_output_dir / output_filename
            aggregated_df.to_csv(output_path, index=False)

            print(
                f"     ✅ {output_filename}: {len(aggregated_df)} rows ({len(csv_paths)} files)"
            )

            all_token_dfs.append(aggregated_df)

        # Create full aggregated CSV (all token lengths)
        if all_token_dfs:
            print(f"\n  🔄 Creating full aggregated...")
            full_aggregated_df = pd.concat(all_token_dfs, ignore_index=True)

            full_output_path = msg_output_dir / f"{full_csv_basename}.csv"
            full_aggregated_df.to_csv(full_output_path, index=False)

            print(f"     ✅ {full_csv_basename}.csv")
            print(f"     Rows: {len(full_aggregated_df)}")
            print(
                f"     Context lengths: {sorted(full_aggregated_df['context_len'].unique())}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate experiment results across multiple runs"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["gpt", "Qwen", "Qwen3-14B-LoRA", "all"],
        help="Model to process (default: all)",
    )
    parser.add_argument(
        "--aggregated-basename",
        type=str,
        default=None,
        help="Base name for full aggregated CSV (default: auto-generated from model name)",
    )

    args = parser.parse_args()

    print("🚀 Generic Experiment Aggregation")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Target model(s): {args.model}")

    if args.model == "all":
        models = ["gpt", "Qwen", "Qwen3-14B-LoRA"]
    else:
        models = [args.model]

    for model in models:
        try:
            process_model(model, args.aggregated_basename)
        except Exception as e:
            print(f"❌ Error processing {model}: {e}")

    print("\n🎉 Aggregation completed!")


if __name__ == "__main__":
    main()
