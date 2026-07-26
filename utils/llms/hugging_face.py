"""
Hugging Face API utilities for HPC inference (RQ/SLM).

Used by: RQ/SLM/infer.py for fine-tuned model inference on Stanage cluster.
NOT used by: visual_eval dashboard (uses OpenAI + LM Studio instead).
"""

from typing import List, Tuple, Optional
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


def load_model(
    repo_id: str,
    filename: str,
    seed: int = RANDOM_SEED,
    chat_format: Optional[str] = None,
    revision: str = "main",
    *,
    # Performance tuning (optional)
    n_gpu_layers: Optional[int] = -1,  # Offload all layers to GPU
    main_gpu: Optional[int] = 0,  # Use first GPU
    n_batch: Optional[int] = 2048,  # Maximize prompt processing efficiency
    flash_attn: Optional[bool] = True,  # Enable Flash Attention
    offload_kqv: Optional[bool] = True,  # Process K/Q/V tensors on GPU
    use_mlock: Optional[bool] = False,  # Do NOT pin weights in host RAM; all layers live in VRAM (n_gpu_layers=-1), so a locked host copy is redundant and OOMs low-RAM hosts (e.g. 32GB Blackwell box). The old True was tuned for the 128GB H100/parscratch cluster.
    use_mmap: Optional[bool] = True,  # mmap the GGUF so the OS pages weights on demand and evicts after GPU upload -> negligible steady-state host RAM. No effect on inference latency (weights are on GPU).
    swa_full: Optional[bool] = True,  # Use full SWA cache size
) -> "Llama":
    """Load a llama.cpp model from Hugging Face and return a ready Llama instance.

    Args:
        repo_id: Hugging Face repository ID (e.g., "microsoft/phi-4-gguf").
        filename: GGUF file name inside the repository (e.g., "phi-4-Q4_K.gguf").
        seed: Random seed for sampling; set a fixed value for reproducible outputs.
        chat_format: Optional chat format to use.
        revision: HuggingFace revision (branch, tag, or commit) for the model (default: main).

    Returns:
        A `Llama` instance configured with `n_ctx=DEFAULT_MAX_TOKENS` and the given seed.
    """
    local_path = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)

    # Create llama instance from local file path
    llm = Llama(
        model_path=local_path,
        n_ctx=DEFAULT_MAX_TOKENS,
        verbose=False,
        seed=seed,
        chat_format=chat_format,
        n_gpu_layers=n_gpu_layers,
        main_gpu=main_gpu,
        offload_kqv=offload_kqv,
        flash_attn=flash_attn,
        swa_full=swa_full,
        n_batch=n_batch,
        use_mlock=use_mlock,
        use_mmap=use_mmap,
    )

    cache_key = f"{repo_id}:{filename}:{chat_format or ''}"
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
    chat_format: Optional[str] = None,
) -> List[str]:
    """Run chat inference via llama.cpp and return predicted commit types.

    Args:
        repo_id: Hugging Face repository ID (e.g., "microsoft/phi-4-gguf").
        filename: GGUF file name inside the repository.
        commit: Input text to analyze (e.g., truncated diff + message).
        system_prompt: Instructional system prompt to steer the model.
        temperature: Sampling temperature (higher = more random generation).
        seed: RNG seed used for sampling in this call for reproducibility.
        use_schema: Whether to use JSON schema for structured output.
        chat_format: Optional chat format to use.

    Returns:
        List of predicted commit types extracted from the model's JSON output.

    Notes:
        This function expects the model to output JSON with a top-level key
        "types" (array of strings). If decoding fails, it attempts to parse the
        first valid JSON object found in the output and returns an empty list if
        none is found.
    """
    cache_key = f"{repo_id}:{filename}:{chat_format or ''}"
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
