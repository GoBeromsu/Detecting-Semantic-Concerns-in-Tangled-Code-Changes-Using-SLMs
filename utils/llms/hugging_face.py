"""Unified Hugging Face API utilities for all experiments."""

from typing import List, Tuple
from pydantic import BaseModel, Field
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from outlines import Generator
from huggingface_hub import scan_cache_dir
from .constant import (
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
)


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


def load_model(name: str) -> Generator:
    """
    Load model with optimized configuration.

    Args:
        model_name: Name of the model to load

    Returns:
        Loaded model instance
    """
    try:
        model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(name)
        outlines_model = Generator(model, tokenizer)
        return outlines_model
    except Exception as e:
        raise RuntimeError(f"Failed to load model {name}: {e}")


def api_call(
    model_name: str,
    commit: str,
    system_prompt: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> List[str]:
    """
    Call LM Studio API for commit classification with inference time measurement.
    """
    try:
        model = load_model(model_name)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": commit},
        ]
        # apply_chat_template handles the specific formatting for the chosen model (e.g., ChatML)
        prompt = model.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Use `outlines` to generate a structured JSON response
        # This enforces the output to match the `ConcernResponse` Pydantic model.
        generator = Generator(model, ConcernResponse)
        response = generator(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=1234,  # for reproducibility
        )
        return response.types

    except Exception as e:
        raise RuntimeError(f"An error occurred while calling Hugging Face API: {e}")
