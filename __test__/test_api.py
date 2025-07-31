"""Test module for API call functionality."""

import pytest
from unittest.mock import patch, MagicMock, Mock
import sys

# Mock external dependencies before importing
sys.modules["outlines"] = Mock()
sys.modules["torch"] = Mock()
sys.modules["transformers"] = Mock()
sys.modules["huggingface_hub"] = Mock()

from utils.llms.hugging_face import api_call


class TestApiCall:
    """Test cases for api_call function."""

    @patch("utils.llms.lmstudio.Generator")
    @patch("utils.llms.lmstudio.load_model")
    def test_api_call_success(self, mock_load_model, mock_generator_class):
        """Should successfully call API and return concern types."""
        # Mock model
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model.tokenizer = mock_tokenizer
        mock_tokenizer.apply_chat_template.return_value = "formatted_prompt"
        mock_load_model.return_value = mock_model

        # Mock generator instance and response
        mock_generator_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.types = ["security", "performance"]
        mock_generator_instance.return_value = mock_response
        mock_generator_class.return_value = mock_generator_instance

        result = api_call("test_model", "test commit", "test system prompt")

        # Verify result
        assert result == ["security", "performance"]
        assert isinstance(result, list)

    def test_api_call_raises_runtime_error_on_failure(self):
        """Should raise RuntimeError when API call fails."""
        with patch("utils.llms.lmstudio.load_model") as mock_load:
            mock_load.side_effect = Exception("Load failed")

            with pytest.raises(
                RuntimeError, match="An error occurred while calling Hugging Face API"
            ):
                api_call("test_model", "test commit", "test prompt")
