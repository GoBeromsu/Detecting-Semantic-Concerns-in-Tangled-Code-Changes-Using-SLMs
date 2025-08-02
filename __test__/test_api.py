"""Test module for API call functionality."""

import pytest
from unittest.mock import patch, MagicMock, Mock
import sys

# Mock external dependencies before importing
sys.modules["lmstudio"] = Mock()
sys.modules["outlines"] = Mock()
sys.modules["torch"] = Mock()
sys.modules["transformers"] = Mock()
sys.modules["huggingface_hub"] = Mock()

from utils.llms.lmstudio import api_call
from utils.llms.constant import COMMIT_TYPES


class TestApiCall:
    """Test cases for api_call function."""

    @patch("utils.llms.lmstudio.load_model")
    def test_api_call_success(self, mock_load_model):
        """Should successfully call API and return concern types."""
        # Mock model and response
        mock_model = MagicMock()
        mock_response = MagicMock()
        expected_types = [COMMIT_TYPES[0], COMMIT_TYPES[1]]  # ["docs", "test"]
        mock_response.parsed.get.return_value = expected_types
        mock_model.respond.return_value = mock_response
        mock_load_model.return_value = mock_model

        result = api_call("test_model", "test commit", "test system prompt")

        # Verify result
        assert result == expected_types
        assert isinstance(result, list)
        assert all(commit_type in COMMIT_TYPES for commit_type in result)

    def test_api_call_raises_runtime_error_on_failure(self):
        """Should raise RuntimeError when API call fails."""
        with patch("utils.llms.lmstudio.load_model") as mock_load:
            mock_load.side_effect = Exception("Load failed")

            with pytest.raises(
                RuntimeError, match="An error occurred while calling LM Studio"
            ):
                api_call("test_model", "test commit", "test prompt")
