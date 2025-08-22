"""Evaluation utilities for parsing outputs and calculating metrics."""

from typing import Dict, List, Tuple
from collections import Counter

from utils.llms.constant import COMMIT_TYPES


def get_tp_fp_fn(
    predicted_types: List[str], actual_types: List[str]
) -> Tuple[int, int, int]:
    """
    Calculate TP, FP, FN using set operations filtered by valid classes.

    This avoids any dependency on label ordering and ensures unknown labels
    are ignored rather than distorting metrics.

    Args:
        predicted_types: List of predicted concern types
        actual_types: List of actual concern types

    Returns:
        Tuple of (true_positives, false_positives, false_negatives)
    """
    valid_labels = set(COMMIT_TYPES)
    predicted_set = {t for t in predicted_types if t in valid_labels}
    actual_set = {t for t in actual_types if t in valid_labels}

    tp = len(predicted_set & actual_set)
    fp = len(predicted_set - actual_set)
    fn = len(actual_set - predicted_set)

    return tp, fp, fn


def calculate_metrics(
    predicted_types: List[str], actual_types: List[str]
) -> Dict[str, float]:
    """
    Calculate precision, recall, F1, and Hamming loss using set operations.

    Args:
        predicted_types: List of predicted concern types
        actual_types: List of actual concern types

    Returns:
        Dict with precision, recall, f1, exact_match, and hamming_loss
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

    valid_labels = set(COMMIT_TYPES)
    hamming_loss = (fp + fn) / len(valid_labels) if len(valid_labels) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": exact_match,
        "hamming_loss": hamming_loss,
    }


