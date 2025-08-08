"""Dataset management utilities for concern classification evaluation."""

import os
import glob
import streamlit as st
import pandas as pd
from typing import List

# Dataset constants
DIFF_COLUMN: str = "diff"
TYPES_COLUMN: str = "types"
SHAS_COLUMN: str = "shas"

DATASET_SEARCH_PATTERNS = [
    "datasets/**/*.csv",
    "../datasets/**/*.csv",
]

REQUIRED_COLUMNS = [DIFF_COLUMN, TYPES_COLUMN, SHAS_COLUMN]


def find_dataset_files() -> List[str]:
    """Find all available CSV dataset files using search patterns."""
    available_files = []
    for search_pattern in DATASET_SEARCH_PATTERNS:
        matched_files = glob.glob(search_pattern, recursive=True)
        available_files.extend(matched_files)
    return sorted(available_files)


def validate_dataset_columns(df: pd.DataFrame) -> List[str]:
    """
    Validate that dataset contains required columns.

    Returns:
        List of missing column names, empty if all required columns present
    """
    return [col for col in REQUIRED_COLUMNS if col not in df.columns]


def load_dataset_from_file(file_path: str) -> pd.DataFrame:
    """
    Load dataset from CSV file with validation.

    Args:
        file_path: Path to CSV file

    Returns:
        DataFrame with validated columns, empty DataFrame on error
    """
    try:
        df = pd.read_csv(file_path)
        missing_columns = validate_dataset_columns(df)

        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            return pd.DataFrame()

        return df

    except Exception as e:
        st.error(f"Error loading CSV dataset: {str(e)}")
        return pd.DataFrame()


@st.cache_data
def get_available_datasets() -> List[str]:
    """Get list of available dataset files with Streamlit caching."""
    return find_dataset_files()


@st.cache_data
def load_dataset(file_path: str) -> pd.DataFrame:
    """Load concern classification dataset from CSV file with Streamlit caching."""
    return load_dataset_from_file(file_path)
