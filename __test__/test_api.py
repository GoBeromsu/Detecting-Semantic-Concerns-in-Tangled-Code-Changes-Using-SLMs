"""Test module for API call functionality."""

import pytest
from unittest.mock import MagicMock


@pytest.fixture
def lmstudio_mock(monkeypatch):
    """Mock the lmstudio module chain for API testing."""
    mock_lms = MagicMock()
    monkeypatch.setattr("utils.llms.lmstudio.lms", mock_lms)
    return mock_lms


class TestApiCall:
    """Test cases for api_call function."""

    def test_api_call_success(self, lmstudio_mock):
        """Should successfully call API and return concern types."""
        # Import after fixtures are applied
        from utils.llms.lmstudio import api_call
        from utils.llms.constant import COMMIT_TYPES
        
        # Arrange
        expected_types = [COMMIT_TYPES[0], COMMIT_TYPES[1]]  # ["docs", "test"]
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.parsed = {"types": expected_types}
        mock_model.respond.return_value = mock_response
        lmstudio_mock.llm.return_value = mock_model

        # Act
        result = api_call("test_model", "test commit", "test system prompt")

        # Assert
        assert result == expected_types
        assert isinstance(result, list)
        assert all(commit_type in COMMIT_TYPES for commit_type in result)
        lmstudio_mock.llm.assert_called_once()
        mock_model.respond.assert_called_once()

    def test_api_call_raises_runtime_error_on_failure(self, lmstudio_mock):
        """Should raise RuntimeError when API call fails."""
        # Import after fixtures are applied
        from utils.llms.lmstudio import api_call
        
        # Arrange
        lmstudio_mock.llm.side_effect = Exception("Load failed")

        # Act & Assert
        with pytest.raises(
            RuntimeError, match="An error occurred while calling LM Studio"
        ):
            api_call("test_model", "test commit", "test prompt")
