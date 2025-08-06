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
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer

from utils.llms.constant import COMMIT_TYPES


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
    Calculate TP, FP, FN using sklearn multilabel_confusion_matrix.

    Args:
        predicted_types: List of predicted concern types
        actual_types: List of actual concern types

    Returns:
        Tuple of (true_positives, false_positives, false_negatives)
    """
    # Convert to binary format
    mlb = MultiLabelBinarizer(classes=COMMIT_TYPES)
    mlb.fit([COMMIT_TYPES])  # Fit with all possible classes
    y_true = mlb.transform([actual_types])
    y_pred = mlb.transform([predicted_types])

    # Get confusion matrices for each label
    mcm = multilabel_confusion_matrix(y_true, y_pred, labels=range(len(COMMIT_TYPES)))

    # Sum across all labels: mcm shape is (n_labels, 2, 2)
    # mcm[i] = [[tn, fp], [fn, tp]] for label i
    tp = int(mcm[:, 1, 1].sum())  # True positives
    fp = int(mcm[:, 0, 1].sum())  # False positives
    fn = int(mcm[:, 1, 0].sum())  # False negatives

    return tp, fp, fn


def calculate_metrics(
    predicted_types: List[str], actual_types: List[str]
) -> Dict[str, float]:
    """
    Calculate precision, recall, F1 using sklearn multilabel metrics.

    Args:
        predicted_types: List of predicted concern types
        actual_types: List of actual concern types

    Returns:
        Dict with precision, recall, f1, and exact_match
    """
    tp, fp, fn = get_tp_fp_fn(predicted_types, actual_types)

    # Exact match check
    exact_match = Counter(predicted_types) == Counter(actual_types)

    # Calculate precision, recall, f1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": exact_match,
    }


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
