#!/usr/bin/env python3
"""
Unified token distribution analysis for all dataset files.
Analyzes token lengths using tiktoken for multiple CSV datasets.
"""

import pandas as pd
import json
import tiktoken
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Any


# Initialize tiktoken encoding
ENCODING_MODEL = "cl100k_base"  # GPT-4 encoding


def calculate_token_length(text: str, encoding) -> int:
    """Calculate token length for given text."""
    if not text or pd.isna(text):
        return 0

    # Handle JSON formatted diffs
    if isinstance(text, str) and text.startswith('[') and text.endswith(']'):
        try:
            diffs = json.loads(text)
            text = '\n'.join(diffs) if isinstance(diffs, list) else text
        except (json.JSONDecodeError, TypeError):
            pass

    return len(encoding.encode(str(text)))


def analyze_tangled_dataset(df: pd.DataFrame, encoding) -> None:
    """Analyze tangled CCS dataset with concern_count."""
    print("\n" + "="*80)
    print("TANGLED DATASET ANALYSIS")
    print("="*80)

    # Calculate token lengths
    print("\nCalculating token lengths...")
    token_lengths_combined = []

    for idx, row in df.iterrows():
        diff_text = row['diff']
        commit_msg = row['commit_message'] if pd.notna(row['commit_message']) else ''

        # Handle JSON formatted diffs
        if diff_text.startswith('[') and diff_text.endswith(']'):
            try:
                diffs = json.loads(diff_text)
                diff_text = '\n'.join(diffs) if isinstance(diffs, list) else diff_text
            except:
                pass

        combined = f"{commit_msg}\n\n{diff_text}" if commit_msg else diff_text
        combined_tokens = calculate_token_length(combined, encoding)
        token_lengths_combined.append(combined_tokens)

    df['token_length'] = token_lengths_combined

    # Basic statistics
    print(f"\n=== Overall Statistics ===")
    print(f"Total samples: {len(df)}")
    print(f"Token length range: {min(token_lengths_combined)} - {max(token_lengths_combined)}")
    print(f"Mean: {np.mean(token_lengths_combined):.1f} tokens")
    print(f"Median: {np.median(token_lengths_combined):.1f} tokens")

    # Distribution by concern count
    print(f"\n=== Token Length by Concern Count ===")
    for cc in sorted(df['concern_count'].unique()):
        subset = df[df['concern_count'] == cc]['token_length']
        print(f"\nConcern count {cc} (n={len(subset)}):")
        print(f"  Mean: {subset.mean():.1f}, Median: {subset.median():.1f}")
        print(f"  Min: {subset.min()}, Max: {subset.max()}")

    # Token range distribution
    print_token_distribution(token_lengths_combined)

    # Percentiles
    print_percentiles(token_lengths_combined)


def analyze_atomic_dataset(df: pd.DataFrame, encoding, dataset_type: str) -> None:
    """Analyze atomic CCS dataset (sampled or full)."""
    print("\n" + "="*80)
    print(f"{dataset_type.upper()} DATASET ANALYSIS")
    print("="*80)

    # Detect columns
    diff_col = None
    msg_col = None
    type_col = None

    for col in df.columns:
        col_lower = col.lower()
        if 'git_diff' in col_lower or col == 'diff':
            diff_col = col
        if 'masked_commit_message' in col_lower or 'message' in col_lower:
            msg_col = col
        if 'annotated_type' in col_lower or col == 'type':
            type_col = col

    print(f"\nDetected columns:")
    print(f"  Diff: {diff_col}")
    print(f"  Message: {msg_col}")
    print(f"  Type: {type_col}")

    if not diff_col or not msg_col:
        print("ERROR: Could not detect required columns")
        return

    # Calculate token lengths
    print("\nCalculating token lengths...")
    token_lengths_combined = []
    token_lengths_diff = []
    token_lengths_msg = []

    for idx, row in df.iterrows():
        diff_text = str(row[diff_col]) if pd.notna(row[diff_col]) else ''
        msg_text = str(row[msg_col]) if pd.notna(row[msg_col]) else ''

        diff_tokens = calculate_token_length(diff_text, encoding)
        msg_tokens = calculate_token_length(msg_text, encoding)

        combined = f"{msg_text}\n\n{diff_text}" if msg_text else diff_text
        combined_tokens = calculate_token_length(combined, encoding)

        token_lengths_diff.append(diff_tokens)
        token_lengths_msg.append(msg_tokens)
        token_lengths_combined.append(combined_tokens)

    df['token_length'] = token_lengths_combined

    # Basic statistics
    print(f"\n=== Overall Statistics ===")
    print(f"Total samples: {len(df)}")
    print(f"\nDiff only:")
    print(f"  Range: {min(token_lengths_diff)} - {max(token_lengths_diff)}")
    print(f"  Mean: {np.mean(token_lengths_diff):.1f}")

    print(f"\nMessage only:")
    print(f"  Range: {min(token_lengths_msg)} - {max(token_lengths_msg)}")
    print(f"  Mean: {np.mean(token_lengths_msg):.1f}")

    print(f"\nCombined (Message + Diff):")
    print(f"  Range: {min(token_lengths_combined)} - {max(token_lengths_combined)}")
    print(f"  Mean: {np.mean(token_lengths_combined):.1f}")
    print(f"  Median: {np.median(token_lengths_combined):.1f}")

    # Distribution by type if available
    if type_col and type_col in df.columns:
        print(f"\n=== Token Length by Type ===")
        type_counts = df[type_col].value_counts()
        for atype, count in type_counts.items():
            subset = df[df[type_col] == atype]['token_length']
            print(f"\n{atype} (n={count}):")
            print(f"  Mean: {subset.mean():.1f}, Median: {subset.median():.1f}")
            print(f"  Max: {subset.max()}")

    # Token range distribution
    print_token_distribution(token_lengths_combined)

    # Percentiles
    print_percentiles(token_lengths_combined)

    # Extreme samples
    print(f"\n=== Samples with >12288 tokens ===")
    high_token_samples = df[df['token_length'] > 12288].sort_values('token_length', ascending=False)
    print(f"Found {len(high_token_samples)} samples ({len(high_token_samples)/len(df)*100:.1f}%)")

    if len(high_token_samples) > 0:
        for idx, row in high_token_samples.head(5).iterrows():
            msg_preview = str(row[msg_col])[:50] if pd.notna(row[msg_col]) else "No message"
            print(f"  {row['token_length']:,} tokens: {msg_preview}...")


def print_token_distribution(token_lengths: List[int]) -> None:
    """Print token distribution in ranges."""
    print(f"\n=== Token Distribution ===")
    ranges = [
        (0, 1024),
        (1024, 2048),
        (2048, 4096),
        (4096, 8192),
        (8192, 12288),
        (12288, 16384),
        (16384, float('inf'))
    ]

    for min_r, max_r in ranges:
        if max_r == float('inf'):
            count = sum(1 for t in token_lengths if t > min_r)
            label = f">{min_r:,}"
        else:
            count = sum(1 for t in token_lengths if min_r < t <= max_r)
            label = f"{min_r+1:,}-{max_r:,}"

        pct = (count / len(token_lengths) * 100) if len(token_lengths) > 0 else 0
        bar = '█' * int(pct / 2)  # Simple bar chart
        print(f"  {label:>14}: {count:4d} ({pct:5.1f}%) {bar}")


def print_percentiles(token_lengths: List[int]) -> None:
    """Print percentile statistics."""
    print(f"\n=== Percentiles ===")
    percentiles = [25, 50, 75, 90, 95, 99, 99.9]
    for p in percentiles:
        val = np.percentile(token_lengths, p)
        print(f"  {p:5.1f}%: {val:,.0f} tokens")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze token distribution for CCS datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["tangled", "sampled", "full", "all"],
        default="all",
        help="Which dataset to analyze (default: all)"
    )
    parser.add_argument(
        "--path",
        type=str,
        help="Custom path to CSV file"
    )

    args = parser.parse_args()

    # Initialize encoding
    encoding = tiktoken.get_encoding(ENCODING_MODEL)

    # Define dataset paths
    # NOTE: "sampled" points to the repo-grouped pool (build_repo_pool.py output),
    # which superseded the old sampled_ccs_dataset.csv (removed). Same required
    # columns (git_diff, masked_commit_message, annotated_type, sha), so the
    # atomic-dataset analysis path below works unchanged.
    data_dir = Path(__file__).parent.parent / "data"
    datasets = {
        "tangled": data_dir / "tangled_ccs_dataset_test.csv",
        "sampled": data_dir / "repo_grouped_pool.csv",
        "full": data_dir / "CCS Dataset.csv"
    }

    # Handle custom path
    if args.path:
        df = pd.read_csv(args.path)
        print(f"\n=== Analyzing: {args.path} ===")
        print(f"Columns: {df.columns.tolist()}")

        # Detect dataset type by columns
        if 'concern_count' in df.columns:
            analyze_tangled_dataset(df, encoding)
        else:
            analyze_atomic_dataset(df, encoding, "CUSTOM")
    else:
        # Analyze selected datasets
        datasets_to_analyze = []
        if args.dataset == "all":
            datasets_to_analyze = ["tangled", "sampled", "full"]
        else:
            datasets_to_analyze = [args.dataset]

        for dataset_name in datasets_to_analyze:
            dataset_path = datasets[dataset_name]
            if not dataset_path.exists():
                print(f"\n⚠️  Dataset not found: {dataset_path}")
                continue

            print(f"\n{'='*80}")
            print(f"Analyzing: {dataset_path.name}")
            print(f"{'='*80}")

            df = pd.read_csv(dataset_path)

            if dataset_name == "tangled":
                analyze_tangled_dataset(df, encoding)
            elif dataset_name == "sampled":
                analyze_atomic_dataset(df, encoding, "SAMPLED")
            elif dataset_name == "full":
                analyze_atomic_dataset(df, encoding, "FULL CCS")

    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)


if __name__ == "__main__":
    main()