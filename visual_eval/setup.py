"""API setup and configuration utilities for sidebar management."""

import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from utils.llms.lmstudio import get_models
from utils.llms.constant import DEFAULT_OPENAI_MODEL
from .session import (
    set_api_provider,
)

load_dotenv()


def validate_openai_key(api_key: str) -> tuple[bool, str]:
    """Validate OpenAI API key by making a simple API call."""
    try:
        client = OpenAI(api_key=api_key)
        client.models.list()
        return True, ""
    except Exception as e:
        return False, str(e)


def setup_openai_api() -> bool:
    """
    Setup OpenAI API configuration.

    Returns:
        True if setup successful, False otherwise
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("❌ No OpenAI API Key found. Please set OPENAI_API_KEY in .env file")
        return False

    with st.spinner("Validating API key..."):
        is_valid, error_msg = validate_openai_key(api_key)

    if not is_valid:
        st.error(f"❌ Invalid OpenAI API Key: {error_msg}")
        return False

    set_api_provider("openai", DEFAULT_OPENAI_MODEL)
    st.success(f"✅ OpenAI API configured with model: **{DEFAULT_OPENAI_MODEL}**")
    return True


def setup_lmstudio_api() -> bool:
    """
    Setup LM Studio API configuration.

    Returns:
        True if setup successful, False otherwise
    """
    with st.spinner("Loading available models..."):
        models, error_msg = get_models()

        if error_msg:
            st.warning(f"⚠️ LM Studio auto-detection failed: {error_msg}")
            st.info("💡 You can still manually enter your model name below")
        else:
            st.success(f"✅ Found {len(models)} available models")

    selected_model = st.selectbox(
        "Select Model:",
        models,
        help="Choose a model loaded in LM Studio",
    )

    set_api_provider("lmstudio", selected_model)
    st.success(f"✅ LM Studio configured with model: **{selected_model}**")
    return True


def render_api_setup_sidebar() -> bool:
    """
    Render API setup in sidebar with provider selection.

    Returns:
        True if setup successful, False otherwise
    """
    st.header("🔧 Setup")

    api_provider = st.selectbox(
        "Select API Provider:",
        ["OpenAI", "LM Studio"],
        help="Choose between OpenAI API or local LM Studio",
    )

    if api_provider == "OpenAI":
        return setup_openai_api()
    else:  # LM Studio
        return setup_lmstudio_api()
