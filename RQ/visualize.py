#!/usr/bin/env python3

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Final
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json

sys.path.append(str(Path(__file__).parent.parent))

import utils.eval as eval_utils

RESULTS_BASE_DIR = Path("results")
ANALYSIS_OUTPUT_DIR = Path("results/analysis")
CONTEXT_LENGTH: Final[Tuple[int, ...]] = (1024, 2048, 4096, 8192, 12288)
EXPERIMENT_TYPES: Final[Tuple[str, ...]] = ("with_message", "diff_only")
DF_COLUMNS: Final[Tuple[str, ...]] = (
    "predicted_types",
    "actual_types",
    "inference_time",
    "shas",
    "context_len",
    "with_message",
    "model",
    "precision",
    "recall",
    "f1",
    "accuracy",
)
METRICS: Final[Tuple[str, ...]] = ("accuracy", "f1", "precision", "recall")
AGGREGATION_METRICS: Final[Tuple[str, ...]] = (*METRICS, "inference_time")

plt.style.use("default")
sns.set_palette("husl")


def parse_filename(filepath: Path) -> Tuple[str, int]:
    """
    Parse model name and context length from filename.
    Example: microsoft_phi-4_4096.csv -> ('microsoft_phi-4', 4096)
    """
    filename = filepath.stem  # Remove .csv extension
    parts = filename.split("_")

    # Get context_len from the last part
    context_len = int(parts[-1])

    # Get model name by joining all parts except the last one
    model = "_".join(parts[:-1])

    return model, context_len


def preprocess_experimental_data() -> pd.DataFrame:
    """
    Load experimental data and add metadata columns.
    Returns DataFrame with: predicted_types, actual_types, inference_time, shas, context_len, with_message, model
    """
    all_dataframes = []

    for experiment_type in EXPERIMENT_TYPES:
        exp_folder = RESULTS_BASE_DIR / experiment_type

        if not exp_folder.exists():
            continue

        with_message = 1 if experiment_type == "with_message" else 0

        for csv_file in exp_folder.glob("*.csv"):
            try:
                model, context_len = parse_filename(csv_file)
            except (ValueError, IndexError):
                continue

            df = pd.read_csv(csv_file)
            
            # Skip empty dataframes
            if df.empty:
                continue

            # Add only the required metadata columns
            df["context_len"] = context_len
            df["with_message"] = with_message
            df["model"] = model
            
            # Parse string lists to actual lists
            df["predicted_types"] = df["predicted_types"].apply(ast.literal_eval)
            df["actual_types"] = df["actual_types"].apply(ast.literal_eval)
            df["count"] = df["actual_types"].apply(len)

            # Use pandas-optimized metric calculation
            df = eval_utils.calculate_metrics_pandas(df, "predicted_types", "actual_types")
            
            # Rename exact_match to accuracy for consistency
            df["accuracy"] = df["exact_match"].astype(float)
            df = df.drop("exact_match", axis=1)
            
            all_dataframes.append(df)

    if not all_dataframes:
        raise ValueError("No valid data files found to process")
    
    result = pd.concat(all_dataframes, ignore_index=True)
    return result


def create_results_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary results table with aggregated metrics.

    Args:
        df: Input DataFrame with raw experimental results

    Returns:
        DataFrame with columns:
        - Model: Model name
        - Context Length: Input context length
        - With Message: Yes/No for message inclusion
        - Accuracy, F1, Precision, Recall: Aggregated metrics
        - inference_time: Average inference time in ms
    """
    summary_df = (
        df.groupby(["model", "context_len", "with_message"])
        .agg({metric: "mean" for metric in AGGREGATION_METRICS})
        .reset_index()
    )

    # Transform columns directly (no duplication)
    summary_df = summary_df.rename(
        columns={
            "model": "Model",
            "context_len": "Context Length",
            "with_message": "With Message",
        }
    )

    # Format specific columns
    summary_df["With Message"] = summary_df["With Message"].map({1: "Yes", 0: "No"})
    summary_df["Inference Time"] = summary_df["inference_time"]

    # Round metrics to 2 decimal places
    for metric in METRICS:
        summary_df[metric.capitalize()] = summary_df[metric].round(2)

    # Select and order final columns
    display_columns = [
        "Model",
        "Context Length",
        "With Message",
        "Accuracy",
        "F1",
        "Precision",
        "Recall",
        "Inference Time",
    ]

    return summary_df[display_columns].sort_values(
        ["Model", "Context Length", "With Message"]
    )


def save_results_table(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save the results table to a CSV file.

    Args:
        df: Results DataFrame to save
        output_path: Path to save the CSV file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    """Generate comprehensive visualization report with clear data preprocessing."""
    df = preprocess_experimental_data()
    results_table = create_results_table(df)
    save_results_table(results_table, ANALYSIS_OUTPUT_DIR / "results_table.csv")
