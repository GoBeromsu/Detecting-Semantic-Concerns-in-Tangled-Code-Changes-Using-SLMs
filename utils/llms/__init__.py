# LLM handlers package for common use across all RQ experiments

from .openai import (
    api_call as openai_api_call,
)
from .hugging_face import (
    get_models,
    load_model,
)
from .hugging_face import api_call as hugging_face_api_call
from .lmstudio import api_call as lmstudio_api_call
from .constant import (
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    COMMIT_TYPES,
    RESPONSE_SCHEMA,
    OPENAI_STRUCTURED_OUTPUT_FORMAT,
    LMSTUDIO_STRUCTURED_OUTPUT_FORMAT,
)

from typing import List


def api_call(
    provider: str,
    model_name: str,
    commit: str,
    system_prompt: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    api_key: str = "",
) -> List[str]:
    """
    Unified API call that routes to appropriate provider.

    Args:
        model_name: Name of the model (determines provider automatically)
        commit: Commit content to analyze
        system_prompt: System prompt for the model
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate

    Returns:
        List of concern types
    """
    if provider == "openai":
        return openai_api_call(
            api_key=api_key,
            commit=commit,
            system_prompt=system_prompt,
            model=model_name,
            temperature=temperature,
        )

    elif provider == "hugging_face":  # hugging_face
        return hugging_face_api_call(
            model_name=model_name,
            commit=commit,
            system_prompt=system_prompt,
            temperature=temperature,
        )

    elif provider == "lmstudio":
        return lmstudio_api_call(
            model_name=model_name,
            commit=commit,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        raise ValueError(f"Invalid provider: {provider}")


__all__ = [
    "api_call",  # Unified API call (primary interface)
    "get_models",
    "load_model",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_MAX_TOKENS",
    "COMMIT_TYPES",
    "RESPONSE_SCHEMA",
    "OPENAI_STRUCTURED_OUTPUT_FORMAT",
    "LMSTUDIO_STRUCTURED_OUTPUT_FORMAT",
]
