# LLM handlers package for common use across all RQ experiments

from collections.abc import Callable
from types import FunctionType
from typing import TYPE_CHECKING, Protocol

from .constant import (
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    COMMIT_TYPES,
    RESPONSE_SCHEMA,
    OPENAI_STRUCTURED_OUTPUT_FORMAT,
    LMSTUDIO_STRUCTURED_OUTPUT_FORMAT,
)


if TYPE_CHECKING:
    from .hugging_face import get_models, load_model


class _CallableProvider(Protocol):
    def __call__(
        self, *args: FunctionType, **kwargs: FunctionType
    ) -> FunctionType: ...


def _cache_export(name: str, value: FunctionType) -> FunctionType:
    globals()[name] = value
    return value


def _load_get_models() -> FunctionType:
    from .hugging_face import get_models

    return get_models


def _load_model() -> FunctionType:
    from .hugging_face import load_model

    return load_model


def _load_openai_api_call() -> FunctionType:
    from .openai import api_call

    return api_call


def _load_hugging_face_api_call() -> FunctionType:
    from .hugging_face import api_call

    return api_call


def _load_lmstudio_api_call() -> FunctionType:
    from .lmstudio import api_call

    return api_call


def _lazy_provider_export(
    name: str, provider_loader: Callable[[], _CallableProvider]
) -> FunctionType:
    def resolve(*args: FunctionType, **kwargs: FunctionType) -> FunctionType:
        provider: _CallableProvider = provider_loader()
        globals()[name] = provider
        return provider(*args, **kwargs)

    return _cache_export(name, resolve)


def __getattr__(name: str) -> FunctionType:
    if name == "get_models":
        return _lazy_provider_export(name, _load_get_models)
    if name == "load_model":
        return _lazy_provider_export(name, _load_model)
    if name == "openai_api_call":
        return _lazy_provider_export(name, _load_openai_api_call)
    if name == "hugging_face_api_call":
        return _lazy_provider_export(name, _load_hugging_face_api_call)
    if name == "lmstudio_api_call":
        return _lazy_provider_export(name, _load_lmstudio_api_call)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def api_call(
    provider: str,
    model_name: str,
    commit: str,
    system_prompt: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    api_key: str = "",
) -> list[str]:
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
        from .openai import api_call as openai_api_call

        return openai_api_call(
            api_key=api_key,
            commit=commit,
            system_prompt=system_prompt,
            model=model_name,
            temperature=temperature,
        )

    elif provider == "hugging_face":  # hugging_face
        from .hugging_face import api_call as provider_api_call

        hugging_face_api_call: Callable[..., list[str]] = provider_api_call

        return hugging_face_api_call(
            model_name=model_name,
            commit=commit,
            system_prompt=system_prompt,
            temperature=temperature,
        )

    elif provider == "lmstudio":
        from .lmstudio import api_call as lmstudio_api_call

        return lmstudio_api_call(
            model_name=model_name,
            commit=commit,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        message = f"Invalid provider: {provider}"
        raise ValueError(message)


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
