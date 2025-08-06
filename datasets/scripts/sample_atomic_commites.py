#!/usr/bin/env python3
"""
Sample atomic commits from CCS dataset for concern extraction.
Implements balanced sampling pipeline with filtering, normalization, and deduplication.
"""

import pandas as pd

import tiktoken
from typing import Dict, List, Set, Tuple

# Random seed for reproducibility
RANDOM_SEED: int = 42

# Processing configuration
CONVENTIONAL_COMMIT_TYPES: List[str] = ["feat", "fix", "refactor", "test", "docs", "build", "cicd"]
SAMPLES_PER_TYPE: int = 50
TARGET_TOKEN_LIMIT: int = 12288  # 16384 - 4096
ENCODING_MODEL: str = "cl100k_base"  # GPT-4 encoding

# Column name constants
COLUMN_SHA: str = "sha"
COLUMN_ANNOTATED_TYPE: str = "annotated_type"
COLUMN_GIT_DIFF: str = "git_diff"
COLUMN_MASKED_COMMIT_MESSAGE: str = "masked_commit_message"
OUTPUT_COLUMNS: List[str] = [
    COLUMN_ANNOTATED_TYPE,
    COLUMN_MASKED_COMMIT_MESSAGE,
    COLUMN_GIT_DIFF,
    COLUMN_SHA,
]

# Data transformation constants
CI_TO_CICD_REPLACEMENT: str = "cicd"

# File paths
CCS_SOURCE_PATH: str = "data/CCS Dataset.csv"
SAMPLED_CSV_PATH: str = "data/sampled_ccs_dataset.csv"
EXCLUDED_COMMITS_PATH: str = "data/excluded_commits.csv"
DIFF_OUTPUT_DIR: str = "data/types"


def normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize CI commit type labels to CICD for consistent categorization."""
    df[COLUMN_ANNOTATED_TYPE] = (
        df[COLUMN_ANNOTATED_TYPE]
        .str.lower()
        .str.strip()
        .replace("ci", CI_TO_CICD_REPLACEMENT)
    )
    print("Applied CI -> CICD normalization")
    return df


def remove_long_token_commits(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out commits exceeding TARGET_TOKEN_LIMIT to prevent model context overflow."""
    encoding = tiktoken.get_encoding(ENCODING_MODEL)

    combined_text = (
        df[COLUMN_GIT_DIFF].astype(str)
        + " "
        + df[COLUMN_MASKED_COMMIT_MESSAGE].astype(str)
    )

    token_counts = combined_text.apply(lambda x: len(encoding.encode(x)))
    filtered_df = df[token_counts <= TARGET_TOKEN_LIMIT].copy()

    removed_count = len(df) - len(filtered_df)
    if removed_count > 0:
        print(f"Token filtering: removed {removed_count} commits exceeding {TARGET_TOKEN_LIMIT} tokens")

    print(f"Token filtering: kept {len(filtered_df)} commits")
    return filtered_df


def remove_existing_commits(df: pd.DataFrame, excluded_shas: Set[str]) -> pd.DataFrame:
    """Remove commits with SHAs that already exist in the sampled dataset."""
    original_count = len(df)
    
    sha_mask = ~df[COLUMN_SHA].astype(str).isin(excluded_shas)
    filtered_df = df[sha_mask].copy()

    removed_count = original_count - len(filtered_df)
    print(f"SHA deduplication: removed {removed_count} duplicate commits")
    return filtered_df


def load_shas_and_type_counts(file_path: str) -> Tuple[Set[str], Dict[str, int]]:
    """Load commit SHAs and type counts from CSV file for deduplication and intelligent sampling."""
    try:
        df = pd.read_csv(file_path)
        sha_set = set(df[COLUMN_SHA].astype(str))
        type_counts = df[COLUMN_ANNOTATED_TYPE].value_counts().to_dict()
        print(f"Loaded {len(sha_set)} SHAs for deduplication")
        print(f"Existing type counts: {type_counts}")
        return sha_set, type_counts
    except FileNotFoundError:
        print(f"No existing samples found at {file_path}")
        return set(), {}
    except Exception as e:
        print(f"Error loading existing data: {e}")
        return set(), {}


def load_ccs_dataset(file_path: str) -> pd.DataFrame:
    """Load CCS dataset CSV and validate required columns exist."""
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            raise ValueError("Dataset is empty")

        missing_columns = set(OUTPUT_COLUMNS) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        print(f"Loaded {len(df)} records from CCS dataset")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise


def save_to_csv(
    data: List[Dict[str, str]], output_path: str, columns: List[str]
) -> None:
    """Save sampled commit data to CSV file, appending if file exists."""
    import os

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if data:
        df = pd.DataFrame(data, columns=columns)
        file_exists = os.path.exists(output_path)

        df.to_csv(
            output_path,
            mode="a" if file_exists else "w",
            header=not file_exists,
            index=False,
        )

    print(f"Saved {len(data)} records to {output_path}")


def group_commits_by_type(
    df: pd.DataFrame, valid_types: List[str]
) -> Dict[str, pd.DataFrame]:
    """Filter commits by valid types and group into separate DataFrames by type."""
    type_mask = df[COLUMN_ANNOTATED_TYPE].isin(valid_types)
    valid_df = df[type_mask].copy()

    excluded_count = len(df) - len(valid_df)
    print(f"Type filtering: excluded {excluded_count} records (invalid types)")

    commits_by_type = {}
    for commit_type, group_df in valid_df.groupby(COLUMN_ANNOTATED_TYPE):
        commits_by_type[commit_type] = group_df
        print(f"  {commit_type}: {len(group_df)} commits")

    return commits_by_type


def sample_commits_for_type(
    df: pd.DataFrame, count: int, output_columns: List[str]
) -> List[Dict[str, str]]:
    """Randomly sample specified count of commits from DataFrame."""
    sampled_df = df.sample(n=count, random_state=RANDOM_SEED)
    return sampled_df[output_columns].to_dict("records")


def extract_diffs(sampled_data: List[Dict[str, str]], output_dir: str) -> None:
    """Create individual diff files organized by commit type in subdirectories."""
    import os

    type_counts = {}

    for record in sampled_data:
        commit_type = record[COLUMN_ANNOTATED_TYPE]

        # Create type directory if needed
        type_dir = os.path.join(output_dir, commit_type)
        os.makedirs(type_dir, exist_ok=True)

        # Count entries for this type
        if commit_type not in type_counts:
            type_counts[commit_type] = 0
        type_counts[commit_type] += 1

        # Generate filename
        filename = f"{commit_type}_{type_counts[commit_type]}_{record[COLUMN_SHA]}.diff"
        filepath = os.path.join(type_dir, filename)

        # Create file content with metadata
        content_lines = [
            f"# Type: {commit_type}",
            f"# Commit Message: {record[COLUMN_MASKED_COMMIT_MESSAGE]}",
            f"# SHA: {record[COLUMN_SHA]}",
            "",
            "# === Git Diff Content ===",
            "",
            record[COLUMN_GIT_DIFF],
        ]

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(content_lines))

    print(f"Extracted {len(sampled_data)} diff files to {output_dir}")

def remove_excluded_commits(df: pd.DataFrame, excluded_shas: Set[str]) -> pd.DataFrame:
    """Remove commits with SHAs listed in the excluded commits file."""
    before_count = len(df)
    print(f"Initial commit count: {before_count}")
    
    mask = ~df[COLUMN_SHA].astype(str).isin(excluded_shas)
    excluded_count = before_count - mask.sum()
    print(f"Excluded {excluded_count} commits by SHA")
    
    filtered_df = df[mask].copy()
    print(f"Remaining commit count: {len(filtered_df)}")
    return filtered_df


def main() -> None:
    """
    Execute atomic sampling pipeline for CCS dataset:
    1. Load dataset, existing SHAs and type counts for deduplication and sampling
    2. Remove excluded commits by SHA
    3. Remove existing commits to prevent duplicates
    4. Normalize CI commit types to CICD
    5. Filter commits exceeding token limits
    6. Sample needed amounts per type to reach target
    7. Save results and extract individual diff files (new samples only)
    """
    print("Starting atomic sampling strategy for CCS dataset")
    print("=" * 50)

    # Step 1: Load dataset, backup SHAs and existing type counts
    print("Step 1: Loading dataset, backup SHAs and existing type counts")
    existing_shas, existing_type_counts = load_shas_and_type_counts(SAMPLED_CSV_PATH)
    excluded_shas, _ = load_shas_and_type_counts(EXCLUDED_COMMITS_PATH)
    ccs_df = load_ccs_dataset(CCS_SOURCE_PATH)

    # Step 2: Remove excluded commits
    print("\nStep 2: Removing excluded commits")
    ccs_df = remove_excluded_commits(ccs_df, excluded_shas)

    # Step 3: Remove existing commits
    print("\nStep 3: Removing existing commits")
    ccs_df = remove_existing_commits(ccs_df, existing_shas)

    # Step 4: Apply CI->CICD normalization
    print("\nStep 4: Applying CI->CICD normalization")
    ccs_df = normalize_dataset(ccs_df)


    # Step 5: Apply token-based filtering
    print("\nStep 5: Applying token-based filtering")
    ccs_df = remove_long_token_commits(ccs_df)

    # Step 6: Group by type and sample
    print("\nStep 6: Grouping by type and random sampling")
    commits_by_type = group_commits_by_type(ccs_df, CONVENTIONAL_COMMIT_TYPES)

    all_sampled_data = []
    
    for commit_type, commits_df in commits_by_type.items():
        existing_type_count = existing_type_counts.get(commit_type, 0)
        needed_count = max(0, SAMPLES_PER_TYPE - existing_type_count)
        available_type_count = len(commits_df)
        actual_sample_count = min(needed_count, available_type_count)
        
        if needed_count == 0:
            print(f"  {commit_type}: target reached, skipping")
            continue
        if actual_sample_count <= 0:
            print(f"  {commit_type}: no commits available")
            continue
    
        sampled_data = sample_commits_for_type(
            commits_df, actual_sample_count, OUTPUT_COLUMNS
        )
        all_sampled_data.extend(sampled_data)
        print(f"  {commit_type}: sampled {actual_sample_count} commits")

    print(f"Random sampling: generated {len(all_sampled_data)} samples total")

    # Step 7: Save results and extract diffs
    print("\nStep 7: Saving results and extracting diffs")
    if all_sampled_data:
        save_to_csv(all_sampled_data, SAMPLED_CSV_PATH, OUTPUT_COLUMNS)
        extract_diffs(all_sampled_data, DIFF_OUTPUT_DIR)
    else:
        print("No new samples to save - all types have reached target counts")

    # Final summary
    print("\n" + "=" * 50)
    print("Atomic sampling completed successfully!")
    print(f"New samples added: {len(all_sampled_data)}")


if __name__ == "__main__":
    main()
