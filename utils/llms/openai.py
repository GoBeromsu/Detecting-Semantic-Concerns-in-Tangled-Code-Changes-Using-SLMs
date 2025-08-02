"""Unified OpenAI API utilities for all experiments."""

import openai
import json
from typing import List

from .constant import (
    OPENAI_STRUCTURED_OUTPUT_FORMAT,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    RANDOM_SEED,
)


def api_call(
    api_key: str,
    commit: str,
    system_prompt: str,
    model: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> List[str]:
    """
    Call OpenAI API with cached client.

    Args:
        api_key: OpenAI API key
        commit: Commit content to analyze
        system_prompt: System prompt for the model
        model: OpenAI model name
        temperature: Sampling temperature

    Returns:
        List of concern types
    """
    client = openai.OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": commit},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            seed=RANDOM_SEED,
            response_format=OPENAI_STRUCTURED_OUTPUT_FORMAT,
        )
        response_json = response.choices[0].message.content or "{'types': []}"
        response_data = json.loads(response_json)
        return response_data.get("types", [])
    except openai.APIError as e:
        raise RuntimeError(f"OpenAI API error: {e}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON response: {e}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")
