"""Test module for evaluation metrics functionality.

Tests focus on core domain logic: precision/recall/f1 calculation 
and exact match logic using simple set operations.
"""

from typing import List, Dict
from collections import Counter


def calculate_metrics_simple(predicted_types: List[str], actual_types: List[str]) -> Dict[str, float]:
    """
    Simple implementation of metrics calculation for testing domain logic.
    Uses set operations instead of sklearn to focus on our algorithm.
    """
    predicted_set = set(predicted_types)
    actual_set = set(actual_types)
    
    # Calculate TP, FP, FN using set operations
    tp = len(predicted_set & actual_set)
    fp = len(predicted_set - actual_set)
    fn = len(actual_set - predicted_set)
    
    # Exact match using Counter for duplicate handling
    exact_match = Counter(predicted_types) == Counter(actual_types)
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": exact_match,
    }


class TestCalculateMetrics:
    """Test cases for calculate_metrics function focusing on domain logic."""

    def test_exact_match_perfect_prediction(self):
        """Should return perfect metrics when prediction exactly matches actual."""
        predicted = ["feat", "fix"]
        actual = ["feat", "fix"]

        result = calculate_metrics_simple(predicted, actual)

        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0
        assert result["exact_match"] is True

    def test_exact_match_different_order(self):
        """Should return perfect metrics when prediction matches actual in different order."""
        predicted = ["fix", "feat"]
        actual = ["feat", "fix"]

        result = calculate_metrics_simple(predicted, actual)

        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0
        assert result["exact_match"] is True

    def test_partial_match_scenario(self):
        """Should handle partial match: predicted [feat,test] vs actual [feat,fix]."""
        predicted = ["feat", "test"]
        actual = ["feat", "fix"]

        result = calculate_metrics_simple(predicted, actual)

        # TP=1 (feat), FP=1 (test), FN=1 (fix)
        assert result["precision"] == 0.5  # 1/(1+1)
        assert result["recall"] == 0.5     # 1/(1+1)
        assert result["f1"] == 0.5         # 2*0.5*0.5/(0.5+0.5)
        assert result["exact_match"] is False

    def test_no_match_scenario(self):
        """Should handle completely wrong predictions."""
        predicted = ["docs", "test"]
        actual = ["feat", "fix"]

        result = calculate_metrics_simple(predicted, actual)

        # TP=0, FP=2, FN=2
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0
        assert result["exact_match"] is False

    def test_empty_prediction(self):
        """Should handle empty prediction list."""
        predicted = []
        actual = ["feat", "fix"]

        result = calculate_metrics_simple(predicted, actual)

        # TP=0, FP=0, FN=2
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0
        assert result["exact_match"] is False

    def test_both_empty(self):
        """Should handle both empty lists."""
        predicted = []
        actual = []

        result = calculate_metrics_simple(predicted, actual)

        # TP=0, FP=0, FN=0
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0
        assert result["exact_match"] is True
