"""Unified LM Studio API utilities for all experiments."""

from typing import List, Tuple, Dict, Any
import lmstudio as lms

from .constant import (
    RESPONSE_SCHEMA,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    LMSTUDIO_MODEL_CONFIG,
)

# Global state for loaded models
loaded_models: Dict[str, Any] = {}


def get_models() -> Tuple[List[str], str]:
    """
    Get available LLM models from LM Studio.

    Returns:
        Tuple of (model_names_list, error_message)
    """
    try:
        downloaded_models = lms.list_downloaded_models()
        models = [model.model_key for model in downloaded_models]
        if not models:
            return (
                [],
                "No models downloaded in LM Studio. Please download a model first.",
            )
        return models, ""
    except (ConnectionError, ImportError, AssertionError) as e:
        error_type = type(e).__name__
        error_msg = str(e) if str(e) else f"LM Studio {error_type} occurred"
        return [], f"LM Studio error: {error_msg}"
    except Exception as e:
        return [], f"LM Studio error: {str(e)}"


def load_model(model_name: str) -> Any:
    """
    Load model with optimized configuration.

    Args:
        model_name: Name of the model to load

    Returns:
        Loaded model instance
    """
    try:
        model = lms.llm(model_name, config=LMSTUDIO_MODEL_CONFIG)
        loaded_models[model_name] = model
    except Exception as e:
        raise RuntimeError(f"Failed to load model {model_name}: {e}")

    return loaded_models[model_name]


def api_call(
    model_name: str,
    commit: str,
    system_prompt: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> List[str]:
    """
    Call LM Studio API for commit classification with inference time measurement.

    Args:
        model_name: Name of the model to use
        commit: Commit content to analyze
        system_prompt: System prompt for the model
        temperature: Sampling temperature (0.0 for greedy decoding)
        max_tokens: Maximum tokens to generate

    Returns:
        List of concern types
    """
    try:
        model = load_model(model_name)

        messages = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": commit},
            ]
        }

        response = model.respond(
            messages,
            config={
                "temperature": temperature,
                "maxTokens": max_tokens,
                "structured": {
                    "type": "json",
                    "jsonSchema": RESPONSE_SCHEMA,
                },
            },
        )

        return response.parsed.get("types", [])

    except Exception as e:
        raise RuntimeError(f"An error occurred while calling LM Studio: {e}")


def clear_cache() -> None:
    """Clear the model cache."""
    global loaded_models
    loaded_models.clear()
