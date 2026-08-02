"""
Utility modules for commit untangler experiments.
"""

from types import FunctionType
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .eval import calculate_metrics
    from .llms import api_call, get_models, load_model
    from .model import get_prediction, load_model_and_tokenizer


__all__ = [
    "load_model_and_tokenizer",
    "get_prediction",
    # Unified LLM interfaces
    "get_models",
    "load_model",
    "api_call",
    # Evaluation utilities
    "calculate_metrics",
]


def _cache_export(name: str, value: FunctionType) -> FunctionType:
    globals()[name] = value
    return value


def __getattr__(name: str) -> FunctionType:
    if name == "load_model_and_tokenizer":
        from .model import load_model_and_tokenizer

        return _cache_export(name, load_model_and_tokenizer)
    if name == "get_prediction":
        from .model import get_prediction

        return _cache_export(name, get_prediction)
    if name == "get_models":
        from .llms import get_models

        return _cache_export(name, get_models)
    if name == "load_model":
        from .llms import load_model

        return _cache_export(name, load_model)
    if name == "api_call":
        from .llms import api_call

        return _cache_export(name, api_call)
    if name == "calculate_metrics":
        from .eval import calculate_metrics

        return _cache_export(name, calculate_metrics)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
