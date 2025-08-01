"""Session state management utilities for consistent state handling."""

import streamlit as st
import pandas as pd
from typing import Optional, List, Literal

# Session state keys
API_PROVIDER_KEY = "selected_api"
MODEL_NAME_KEY = "selected_model"

# Type definitions
ApiProvider = Literal["openai", "lmstudio"]


def get_api_provider() -> ApiProvider:
    """Get currently selected API provider with type safety."""
    return st.session_state.get(API_PROVIDER_KEY, "openai")


def set_api_provider(provider: ApiProvider, model_name: Optional[str] = None) -> None:
    """Set API provider and associated model with type safety."""
    st.session_state[API_PROVIDER_KEY] = provider
    st.session_state[MODEL_NAME_KEY] = model_name


def get_model_name() -> str:
    """Get currently selected model name."""
    return st.session_state.get(MODEL_NAME_KEY, "")


def get_available_models() -> List[str]:
    """Get list of available LM Studio models."""
    return st.session_state.get(AVAILABLE_MODELS_KEY, [])


def set_evaluation_results(results_df: pd.DataFrame) -> None:
    """Store final evaluation results in session."""
    st.session_state[EVALUATION_RESULTS_KEY] = results_df
