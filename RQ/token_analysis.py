#!/usr/bin/env python3
"""
Analyze token length distribution by concern count from tangled dataset.
"""

import pandas as pd
import tiktoken
from pathlib import Path
from typing import Final

# Constants
ENCODING_MODEL: Final[str] = "cl100k_base"  # GPT-4 encoding
TANGLED_DATASET_PATH: Final[Path] = Path(
    "../datasets/data/tangled_ccs_dataset_test.csv"
)


def calculate_token_length(text: str) -> int:
    """Calculate token length using tiktoken encoding."""
    encoding = tiktoken.get_encoding(ENCODING_MODEL)
    return len(encoding.encode(text))


def analyze_token_distribution_by_concern_count() -> None:
    """
    Analyze and print token length distribution by concern count ranges.
    """
    print("Loading tangled dataset...")
    df = pd.read_csv(TANGLED_DATASET_PATH)
    print(f"Loaded {len(df)} samples")

    print("Calculating token lengths...")
    # Calculate token lengths for each diff
    df["token_length"] = df["diff"].apply(calculate_token_length)

    # Define token ranges
    ranges = [
        ("≤1024", 0, 1024),
        ("1025-2048", 1025, 2048),
        ("2049-4096", 2049, 4096),
        ("4097-8192", 4097, 8192),
        ("8193-16384", 8193, 16384),
    ]

    print("\n=== Token Length Distribution by Concern Count (%) ===")
    print(
        f"{'Concern Count':<15} {'≤1024':<15} {'1025-2048':<17} {'2049-4096':<17} {'4097-8192':<17} {'8193-16384':<17}"
    )
    print("-" * 105)

    for concern_count in sorted(df["concern_count"].unique()):
        concern_df = df[df["concern_count"] == concern_count]
        total_samples = len(concern_df)

        if total_samples == 0:
            continue

        row_data = [f"{concern_count}"]

        for range_name, min_val, max_val in ranges:
            if range_name == "≤1024":
                count = len(concern_df[concern_df["token_length"] <= max_val])
            else:
                count = len(
                    concern_df[
                        (concern_df["token_length"] >= min_val)
                        & (concern_df["token_length"] <= max_val)
                    ]
                )

            percentage = (count / total_samples) * 100
            row_data.append(f"{percentage:.1f}% ({count})")

        print(
            f"{row_data[0]:<15} {row_data[1]:<15} {row_data[2]:<17} {row_data[3]:<17} {row_data[4]:<17} {row_data[5]:<17}"
        )

    # Print basic statistics
    print("\n=== Basic Statistics ===")
    concern_distribution = df["concern_count"].value_counts().sort_index()
    print("Concern count distribution:")
    for concern_count, count in concern_distribution.items():
        print(f"  Concern {concern_count}: {count} samples")

    print(f"\nTotal samples: {len(df)}")
    print(
        f"Token length range: {df['token_length'].min()} - {df['token_length'].max()}"
    )
    print(f"Average token length: {df['token_length'].mean():.1f}")


if __name__ == "__main__":
    analyze_token_distribution_by_concern_count()
