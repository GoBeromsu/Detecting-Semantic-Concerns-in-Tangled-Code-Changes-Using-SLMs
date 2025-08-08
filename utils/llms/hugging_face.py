"""Unified Hugging Face API utilities for all experiments."""

from typing import List, Tuple
from pydantic import BaseModel, Field
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import scan_cache_dir
from .constant import DEFAULT_TEMPERATURE


class ConcernResponse(BaseModel):
    types: List[str] = Field(description="List of concern types")


def get_models() -> Tuple[List[str], str]:
    """
    Get available LLM models from Hugging Face cache.

    Returns:
        Tuple of (model_names_list, error_message)
    """
    try:
        cache_info = scan_cache_dir()
        if not cache_info.repos:
            return (
                [],
                "No Hugging Face models found in cache. Please download a model first.",
            )

        models = [repo.repo_id for repo in cache_info.repos]
        return models, ""

    except Exception as e:
        return [], f"Error scanning Hugging Face cache: {str(e)}"


def load_model(name: str):
    """
    Load model with optimized configuration.

    Args:
        name: Name of the model to load

    Returns:
        Outlines model instance
    """
    try:
        hf_model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        hf_tokenizer = AutoTokenizer.from_pretrained(name)

        # Lazy import to avoid hard dependency at module import time
        try:
            import outlines  # type: ignore
        except ImportError as e:
            raise ImportError(
                "The 'outlines' package is required for the Hugging Face provider. "
                "Install dependencies with 'uv sync' or add it via 'uv add outlines'."
            ) from e

        outlines_model = outlines.from_transformers(hf_model, hf_tokenizer)
        return outlines_model
    except Exception as e:
        raise RuntimeError(f"Failed to load model {name}: {e}")


def api_call(
    model_name: str,
    commit: str,
    system_prompt: str,
    temperature: float = DEFAULT_TEMPERATURE,
) -> List[str]:
    """
    Call Hugging Face API for commit classification with ChatML format.
    """
    try:
        # Lazy imports to avoid module import failure when outlines is not installed
        try:
            import outlines  # type: ignore
            from outlines.inputs import Chat  # type: ignore
        except ImportError as e:
            raise ImportError(
                "The 'outlines' package is required for the Hugging Face provider. "
                "Install dependencies with 'uv sync' or add it via 'uv add outlines'."
            ) from e

        model = load_model(model_name)

        chat = Chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": commit},
            ]
        )

        generator = outlines.Generator(model, ConcernResponse)
        response = generator(chat, temperature=temperature)
        return response.types

    except Exception as e:
        raise RuntimeError(f"An error occurred while calling Hugging Face API: {e}")
