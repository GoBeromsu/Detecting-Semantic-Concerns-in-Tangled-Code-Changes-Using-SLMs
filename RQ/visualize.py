#!/usr/bin/env python3

import ast
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Final
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

import utils.eval as eval

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

            # Skip empty DataFrames
            if df.empty:
                print(f"Skipping empty file: {csv_file}")
                continue

            # Add only the required metadata columns
            df["context_len"] = context_len
            df["with_message"] = with_message
            df["model"] = model

            # Parse string lists to actual lists
            df["predicted_types"] = df["predicted_types"].apply(ast.literal_eval)
            df["actual_types"] = df["actual_types"].apply(ast.literal_eval)

            # Count ground truth labels (concern count)
            df["concern_count"] = df["actual_types"].apply(len)

            # Print concern count distribution for this CSV file
            concern_dist = df["concern_count"].value_counts().sort_index()
            print(f"{csv_file.name}: {dict(concern_dist)} (total: {len(df)})")

            metrics_df = df.apply(
                lambda row: pd.Series(
                    eval.calculate_metrics(row["predicted_types"], row["actual_types"])
                ),
                axis=1,
            )
            # Add metrics as new columns
            df["precision"] = metrics_df["precision"]
            df["recall"] = metrics_df["recall"]
            df["f1"] = metrics_df["f1"]
            df["accuracy"] = metrics_df["exact_match"]

            all_dataframes.append(df)

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


def create_results_table_with_concern_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary results table with aggregated metrics grouped by concern count.

    Args:
        df: Input DataFrame with raw experimental results

    Returns:
        DataFrame with columns:
        - Model: Model name
        - Context Length: Input context length
        - With Message: Yes/No for message inclusion
        - Concern Count: Number of concerns in the commit
        - Accuracy, F1, Precision, Recall: Aggregated metrics
        - Inference Time: Average inference time in ms
    """
    summary_df = (
        df.groupby(["model", "context_len", "with_message", "concern_count"])
        .agg({metric: "mean" for metric in AGGREGATION_METRICS})
        .reset_index()
    )

    # Transform columns directly (no duplication)
    summary_df = summary_df.rename(
        columns={
            "model": "Model",
            "context_len": "Context Length",
            "with_message": "With Message",
            "concern_count": "Concern Count",
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
        "Concern Count",
        "Accuracy",
        "F1",
        "Precision",
        "Recall",
        "Inference Time",
    ]

    return summary_df[display_columns].sort_values(
        ["Model", "Context Length", "With Message", "Concern Count"]
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

    # Print concern count distribution for verification
    print("=== Concern Count Distribution ===")
    concern_distribution = df["concern_count"].value_counts().sort_index()
    print(concern_distribution)
    print(f"Total samples: {len(df)}")
    print(
        f"Concern count range: {df['concern_count'].min()} - {df['concern_count'].max()}"
    )
    print()

    # Generate original results table (without concern count grouping)
    results_table = create_results_table(df)
    save_results_table(results_table, ANALYSIS_OUTPUT_DIR / "results_table.csv")
    print(f"Generated results_table.csv with {len(results_table)} rows")

    # Generate results table with concern count grouping
    results_table_with_concern = create_results_table_with_concern_count(df)
    save_results_table(
        results_table_with_concern,
        ANALYSIS_OUTPUT_DIR / "results_table_with_concern_count.csv",
    )
    print(
        f"Generated results_table_with_concern_count.csv with {len(results_table_with_concern)} rows"
    )
