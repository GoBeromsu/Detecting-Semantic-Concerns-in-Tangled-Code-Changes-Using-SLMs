"""Evaluation utilities for parsing outputs and calculating metrics."""

import time
from typing import Dict, Any, Set, List, Tuple, Callable
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter


def measure_inference_time(func: Callable) -> Tuple[Any, float]:
    """Measure the execution time of a function."""
    start_time = time.time()
    result = func()
    execution_time = time.time() - start_time
    return result, execution_time


def load_dataset(dataset_split: str) -> pd.DataFrame:
    """Load dataset from local CSV file."""
    if dataset_split == "test":
        csv_path = Path("../datasets/data/tangled_ccs_dataset_test.csv")
    elif dataset_split == "train":
        csv_path = Path("../datasets/data/tangled_ccs_dataset_train.csv")
    elif dataset_split == "test_small":
        csv_path = Path("../datasets/data/tangled_ccs_dataset_test_small.csv")
    elif dataset_split == "atomic":
        csv_path = Path("../datasets/data/sampled_ccs_dataset.csv")
    else:
        csv_path = Path("../datasets/data/tangled_ccs_dataset.csv")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {csv_path}")
    return df


def calculate_batch_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate macro-averaged metrics for batch evaluation.

    Args:
        df: DataFrame with 'predictions' and 'ground_truth' columns containing lists

    Returns:
        Dict with macro-averaged precision, recall, f1
    """
    case_metrics = []
    for _, row in df.iterrows():
        metrics = calculate_metrics(row["predictions"], row["ground_truth"])
        case_metrics.append(metrics)

    # Macro average: average per-case metrics
    avg_precision = sum(m["precision"] for m in case_metrics) / len(case_metrics)
    avg_recall = sum(m["recall"] for m in case_metrics) / len(case_metrics)
    avg_f1 = sum(m["f1"] for m in case_metrics) / len(case_metrics)

    return {
        "f1": avg_f1,
        "precision": avg_precision,
        "recall": avg_recall,
    }


def get_tp_fp_fn(
    predicted_types: List[str], actual_types: List[str]
) -> Tuple[int, int, int]:
    """
    Core logic: Calculate TP, FP, FN using strict multiset matching.

    This exposes the fundamental metric calculation that counts exact occurrences.

    Args:
        predicted_types: List of predicted concern types
        actual_types: List of actual concern types

    Returns:
        Tuple of (true_positives, false_positives, false_negatives)
    """
    predicted_counter = Counter(predicted_types)
    actual_counter = Counter(actual_types)

    all_types = set(predicted_counter.keys()) | set(actual_counter.keys())

    tp = sum(min(predicted_counter[t], actual_counter[t]) for t in all_types)
    fp = sum(max(0, predicted_counter[t] - actual_counter[t]) for t in all_types)
    fn = sum(max(0, actual_counter[t] - predicted_counter[t]) for t in all_types)

    return tp, fp, fn


def calculate_metrics(
    predicted_types: List[str], actual_types: List[str]
) -> Dict[str, float]:
    """
    Calculate precision, recall, F1 from predicted and actual types.

    This is the canonical metric calculation function for concern type evaluation.

    Args:
        predicted_types: List of predicted concern types
        actual_types: List of actual concern types

    Returns:
        Dict with precision, recall, f1, and exact_match
    """
    tp, fp, fn = get_tp_fp_fn(predicted_types, actual_types)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # Exact match check
    exact_match = Counter(predicted_types) == Counter(actual_types)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": exact_match,
    }


def calculate_metrics_pandas(df: pd.DataFrame, pred_col: str, actual_col: str) -> pd.DataFrame:
    """
    Calculate metrics for each row in a DataFrame using pandas vectorization.
    
    Args:
        df: DataFrame containing predictions and actuals
        pred_col: Column name containing predicted types
        actual_col: Column name containing actual types
        
    Returns:
        DataFrame with metric columns added
    """
    # Calculate metrics using vectorized operations
    metrics_list = [
        calculate_metrics(pred, actual) 
        for pred, actual in zip(df[pred_col], df[actual_col])
    ]
    
    # Convert to DataFrame and ensure numeric types
    metrics_df = pd.DataFrame(metrics_list)
    for col in ['precision', 'recall', 'f1']:
        metrics_df[col] = metrics_df[col].astype(float)
    metrics_df['exact_match'] = metrics_df['exact_match'].astype(float)
    
    # Add metrics to original dataframe
    return pd.concat([df, metrics_df], axis=1)


def save_results(df: pd.DataFrame, metrics: Dict[str, float], output_dir: str) -> None:
    """Save DataFrame as predictions.csv and metrics as metrics.json."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path / "predictions.csv", index=False)

    with open(output_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


def plot_graph(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    output_path: str,
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
) -> None:
    """Create and save a line plot for analysis."""
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x=x_col, y=y_col, marker="o")

    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
