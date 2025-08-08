"""Evaluation utilities for parsing outputs and calculating metrics."""

import time
from typing import Dict, Any, List, Tuple, Callable
from collections import Counter
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer

from utils.llms.constant import COMMIT_TYPES


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


