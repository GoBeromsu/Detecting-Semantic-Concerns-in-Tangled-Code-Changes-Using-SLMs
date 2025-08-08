"""Unified Hugging Face API utilities for all experiments."""

from typing import List, Tuple
from huggingface_hub import scan_cache_dir, hf_hub_download
from .constant import DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS, RANDOM_SEED, RESPONSE_SCHEMA
from llama_cpp import Llama, LlamaGrammar  
import json

_loaded_models = {}

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


def load_model(repo_id: str, filename: str, seed: int = RANDOM_SEED) -> Llama:
    """Load a llama.cpp model from Hugging Face and return a ready Llama instance.

    Args:
        repo_id: Hugging Face repository ID (e.g., "microsoft/phi-4-gguf").
        filename: GGUF file name inside the repository (e.g., "phi-4-Q4_K.gguf").
        seed: Random seed for sampling; set a fixed value for reproducible outputs.

    Returns:
        A `Llama` instance configured with `n_ctx=DEFAULT_MAX_TOKENS` and the given seed.
    """
    local_path = hf_hub_download(repo_id=repo_id, filename=filename)

    # Create llama instance from local file path
    llm = Llama(
        model_path=local_path,
        n_ctx=DEFAULT_MAX_TOKENS,
        verbose=False,
        seed=seed,
    )

    cache_key = f"{repo_id}:{filename}"
    _loaded_models[cache_key] = llm

    return llm


def api_call(
    repo_id: str,
    filename: str,
    commit: str,
    system_prompt: str,
    temperature: float = DEFAULT_TEMPERATURE,
    seed: int = RANDOM_SEED,
    use_schema: bool = False,
) -> List[str]:
    """Run chat inference via llama.cpp and return predicted commit types.

    Args:
        repo_id: Hugging Face repository ID (e.g., "microsoft/phi-4-gguf").
        filename: GGUF file name inside the repository.
        commit: Input text to analyze (e.g., truncated diff + message).
        system_prompt: Instructional system prompt to steer the model.
        temperature: Sampling temperature (higher = more random generation).
        seed: RNG seed used for sampling in this call for reproducibility.

    Returns:
        List of predicted commit types extracted from the model's JSON output.

    Notes:
        This function expects the model to output JSON with a top-level key
        "types" (array of strings). If decoding fails, it attempts to parse the
        first valid JSON object found in the output and returns an empty list if
        none is found.
    """
    cache_key = f"{repo_id}:{filename}"
    llm = _loaded_models[cache_key]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": commit},
    ]

    try:
        grammar = None
        if use_schema:
            grammar = LlamaGrammar.from_json_schema(json.dumps(RESPONSE_SCHEMA))

        result = llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=DEFAULT_MAX_TOKENS,
            seed=seed,
            response_format={"type": "json_object"} if not use_schema else None,
            grammar=grammar,
        )
        output_text = (
            result.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
    except Exception as exc:
        raise RuntimeError(f"llama-cpp-python inference failed: {exc}") from exc

    # Parse JSON {"types": [...]}
    try:
        data = json.loads(output_text)
        types = data.get("types", [])
        return [str(t) for t in types] if isinstance(types, list) else []
    except json.JSONDecodeError:
        start = output_text.find("{")
        end = output_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(output_text[start : end + 1])
                types = data.get("types", [])
                return [str(t) for t in types] if isinstance(types, list) else []
            except Exception:
                return []
        return []
