"""Test module for evaluation metrics functionality."""

from unittest.mock import patch, MagicMock
import sys
import numpy as np

# Mock only the heaviest dependencies that we don't need for the core eval logic
mock_modules = {
    "outlines": MagicMock(),
    "torch": MagicMock(),
    "transformers": MagicMock(),
    "huggingface_hub": MagicMock(),
}

for name, mock_module in mock_modules.items():
    mock_module.__spec__ = MagicMock()
    sys.modules[name] = mock_module

# Mock visualization modules that eval.py imports but we don't use in tests
matplotlib_mock = MagicMock()
matplotlib_mock.__spec__ = MagicMock()
matplotlib_mock.pyplot = MagicMock()
sys.modules["matplotlib"] = matplotlib_mock
sys.modules["matplotlib.pyplot"] = matplotlib_mock.pyplot

seaborn_mock = MagicMock()
seaborn_mock.__spec__ = MagicMock()
sys.modules["seaborn"] = seaborn_mock


# Mock sklearn's multilabel_confusion_matrix to avoid dependency issues
def mock_multilabel_confusion_matrix(y_true, y_pred, labels=None):
    """Mock implementation of multilabel_confusion_matrix for testing."""
    # Simple TP/FP/FN calculation based on binary arrays
    y_true_arr = y_true[0] if len(y_true) > 0 else []
    y_pred_arr = y_pred[0] if len(y_pred) > 0 else []

    n_labels = len(labels) if labels else 7  # Default to 7 commit types
    mcm = np.zeros((n_labels, 2, 2), dtype=int)

    for i in range(n_labels):
        true_i = y_true_arr[i] if i < len(y_true_arr) else 0
        pred_i = y_pred_arr[i] if i < len(y_pred_arr) else 0

        # Confusion matrix for label i: [[tn, fp], [fn, tp]]
        if true_i == 1 and pred_i == 1:  # TP
            mcm[i, 1, 1] = 1
        elif true_i == 0 and pred_i == 1:  # FP
            mcm[i, 0, 1] = 1
        elif true_i == 1 and pred_i == 0:  # FN
            mcm[i, 1, 0] = 1
        else:  # TN
            mcm[i, 0, 0] = 1

    return mcm


# Now we can import the functions we need to test
with patch.dict(
    "sys.modules",
    {
        "utils.model": MagicMock(),
        "utils.llms.hugging_face": MagicMock(),
        "utils.llms.lmstudio": MagicMock(),
        "utils.llms.openai": MagicMock(),
    },
):
    with patch(
        "sklearn.metrics.multilabel_confusion_matrix", mock_multilabel_confusion_matrix
    ):
        from utils.eval import calculate_metrics, get_tp_fp_fn


@patch("sklearn.metrics.multilabel_confusion_matrix", mock_multilabel_confusion_matrix)
class TestCalculateMetrics:
    """Test cases for calculate_metrics function."""

    def test_exact_match_perfect_prediction(self):
        """Should return perfect metrics when prediction exactly matches actual."""
        predicted = ["feat", "fix"]
        actual = ["feat", "fix"]

        result = calculate_metrics(predicted, actual)

        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0
        assert result["exact_match"] is True

    def test_exact_match_different_order(self):
        """Should return perfect metrics when prediction matches actual in different order."""
        predicted = ["fix", "feat"]
        actual = ["feat", "fix"]

        result = calculate_metrics(predicted, actual)

        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0
        assert result["exact_match"] is True

    def test_partial_match_scenario(self):
        """Should handle partial match scenario: actual [feat,fix] vs predicted [feat,test]."""
        predicted = ["feat", "test"]
        actual = ["feat", "fix"]

        result = calculate_metrics(predicted, actual)

        # TP=1 (feat), FP=1 (test), FN=1 (fix)
        assert result["precision"] == 0.5  # 1/(1+1)
        assert result["recall"] == 0.5  # 1/(1+1)
        assert result["f1"] == 0.5  # 2*0.5*0.5/(0.5+0.5)
        assert result["exact_match"] is False

    def test_no_match_scenario(self):
        """Should handle completely wrong predictions."""
        predicted = ["docs", "test"]
        actual = ["feat", "fix"]

        result = calculate_metrics(predicted, actual)

        # TP=0, FP=2, FN=2
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0
        assert result["exact_match"] is False

    def test_empty_prediction(self):
        """Should handle empty prediction list."""
        predicted = []
        actual = ["feat", "fix"]

        result = calculate_metrics(predicted, actual)

        # TP=0, FP=0, FN=2
        assert result["precision"] == 0.0  # 0/(0+0) -> 0.0 by default
        assert result["recall"] == 0.0  # 0/(0+2)
        assert result["f1"] == 0.0
        assert result["exact_match"] is False

    def test_empty_actual(self):
        """Should handle empty actual list."""
        predicted = ["feat", "fix"]
        actual = []

        result = calculate_metrics(predicted, actual)

        # TP=0, FP=2, FN=0
        assert result["precision"] == 0.0  # 0/(0+2)
        assert result["recall"] == 0.0  # 0/(0+0) -> 0.0 by default
        assert result["f1"] == 0.0
        assert result["exact_match"] is False

    def test_both_empty(self):
        """Should handle both empty lists."""
        predicted = []
        actual = []

        result = calculate_metrics(predicted, actual)

        # TP=0, FP=0, FN=0
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0
        assert result["exact_match"] is True

    def test_duplicate_types_in_prediction(self):
        """Should handle duplicate types in prediction list."""
        predicted = ["feat", "feat", "fix"]
        actual = ["feat", "fix"]

        result = calculate_metrics(predicted, actual)

        # Counter should handle duplicates correctly
        assert (
            result["exact_match"] is False
        )  # Counter(["feat", "feat", "fix"]) != Counter(["feat", "fix"])
        assert isinstance(result["precision"], float)
        assert isinstance(result["recall"], float)
        assert isinstance(result["f1"], float)

    def test_single_type_correct(self):
        """Should handle single type prediction correctly."""
        predicted = ["feat"]
        actual = ["feat"]

        result = calculate_metrics(predicted, actual)

        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0
        assert result["exact_match"] is True

    def test_overprediction_scenario(self):
        """Should handle when prediction has more types than actual."""
        predicted = ["feat", "fix", "docs", "test"]
        actual = ["feat", "fix"]

        result = calculate_metrics(predicted, actual)

        # TP=2 (feat, fix), FP=2 (docs, test), FN=0
        assert result["precision"] == 0.5  # 2/(2+2)
        assert result["recall"] == 1.0  # 2/(2+0)
        assert result["f1"] == 2 / 3  # 2*0.5*1.0/(0.5+1.0)
        assert result["exact_match"] is False

    def test_underprediction_scenario(self):
        """Should handle when prediction has fewer types than actual."""
        predicted = ["feat"]
        actual = ["feat", "fix", "docs"]

        result = calculate_metrics(predicted, actual)

        # TP=1 (feat), FP=0, FN=2 (fix, docs)
        assert result["precision"] == 1.0  # 1/(1+0)
        assert result["recall"] == 1 / 3  # 1/(1+2)
        assert result["f1"] == 0.5  # 2*1.0*(1/3)/(1.0+(1/3))
        assert result["exact_match"] is False

    def test_invalid_commit_types(self):
        """Should handle invalid commit types gracefully."""
        predicted = ["invalid_type", "feat"]
        actual = ["feat", "fix"]

        # Should not raise exception due to MultiLabelBinarizer with classes parameter
        result = calculate_metrics(predicted, actual)

        assert isinstance(result, dict)
        assert "precision" in result
        assert "recall" in result
        assert "f1" in result
        assert "exact_match" in result


@patch("sklearn.metrics.multilabel_confusion_matrix", mock_multilabel_confusion_matrix)
class TestGetTpFpFn:
    """Test cases for get_tp_fp_fn helper function."""

    def test_perfect_match(self):
        """Should return correct TP/FP/FN for perfect match."""
        predicted = ["feat", "fix"]
        actual = ["feat", "fix"]

        tp, fp, fn = get_tp_fp_fn(predicted, actual)

        assert tp == 2  # Both feat and fix correctly predicted
        assert fp == 0  # No false positives
        assert fn == 0  # No false negatives

    def test_partial_match(self):
        """Should return correct TP/FP/FN for partial match."""
        predicted = ["feat", "test"]
        actual = ["feat", "fix"]

        tp, fp, fn = get_tp_fp_fn(predicted, actual)

        assert tp == 1  # feat correctly predicted
        assert fp == 1  # test is false positive
        assert fn == 1  # fix is false negative

    def test_no_match(self):
        """Should return correct TP/FP/FN for no match."""
        predicted = ["docs", "test"]
        actual = ["feat", "fix"]

        tp, fp, fn = get_tp_fp_fn(predicted, actual)

        assert tp == 0  # No correct predictions
        assert fp == 2  # docs and test are false positives
        assert fn == 2  # feat and fix are false negatives

    def test_empty_lists(self):
        """Should handle empty lists correctly."""
        predicted = []
        actual = []

        tp, fp, fn = get_tp_fp_fn(predicted, actual)

        assert tp == 0
        assert fp == 0
        assert fn == 0

    def test_type_consistency(self):
        """Should return integer types for TP/FP/FN."""
        predicted = ["feat"]
        actual = ["fix"]

        tp, fp, fn = get_tp_fp_fn(predicted, actual)

        assert isinstance(tp, int)
        assert isinstance(fp, int)
        assert isinstance(fn, int)
