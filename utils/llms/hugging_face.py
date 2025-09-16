"""Unified Hugging Face API utilities for all experiments."""

from typing import List, Tuple, Optional, Union, Dict, Any
from huggingface_hub import scan_cache_dir, hf_hub_download
from .constant import DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS, RANDOM_SEED, RESPONSE_SCHEMA
from llama_cpp import Llama, LlamaGrammar
import json

# ========== LEGACY GGUF-BASED IMPLEMENTATION (TO BE REMOVED) ==========
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
    *,
    # Performance tuning (optional)
    n_gpu_layers: Optional[int] = -1,  # Offload all layers to GPU
    main_gpu: Optional[int] = 0,  # Use first GPU
    n_batch: Optional[int] = 2048,  # Maximize prompt processing efficiency
    flash_attn: Optional[bool] = True,  # Enable Flash Attention
    offload_kqv: Optional[bool] = True,  # Process K/Q/V tensors on GPU
    use_mlock: Optional[bool] = True,  # Lock model in RAM to eliminate disk I/O delays
    use_mmap: Optional[bool] = False,  # Disable memory mapping for RAM-fixed model
    swa_full: Optional[bool] = True,  # Use full SWA cache size
) -> "Llama":
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


# ========== NEW TRANSFORMERS NATIVE IMPLEMENTATION ==========

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed as transformers_set_seed
)
from peft import PeftModel

# Cache for loaded models and tokenizers
_transformer_models = {}


def load_transformer_model(
    model_id: str,
    adapter_id: Optional[str] = None,
    device_map: str = "auto",
    torch_dtype: Optional[torch.dtype] = None,
    trust_remote_code: bool = True,
) -> Tuple[Any, Any]:
    """
    Load a Transformers model and tokenizer.
    Supports both base models and LoRA adapters.

    Args:
        model_id: HuggingFace model ID (e.g., "microsoft/phi-4")
        adapter_id: Optional LoRA adapter ID (e.g., "Berom0227/Semantic-Concern-SLM-Phi-adapter")
        device_map: Device mapping strategy
        torch_dtype: Model precision (None for auto-detection)
        trust_remote_code: Whether to trust remote code

    Returns:
        Tuple of (model, tokenizer)
    """
    cache_key = f"{model_id}:{adapter_id or 'base'}"

    # Return cached model if available
    if cache_key in _transformer_models:
        return _transformer_models[cache_key]

    # Auto-detect dtype if not specified
    if torch_dtype is None:
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    print(f"Loading model: {model_id} with dtype={torch_dtype}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )

    # Ensure padding token is set
    tokenizer.padding_side = "left"

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
    )

    # Load LoRA adapter if specified
    if adapter_id:
        print(f"Loading LoRA adapter: {adapter_id}")
        model = PeftModel.from_pretrained(
            model,
            adapter_id,
            torch_dtype=torch_dtype,
        )

    # Cache the model and tokenizer
    _transformer_models[cache_key] = (model, tokenizer)

    return model, tokenizer


def transformer_api_call(
    model_id: str,
    adapter_id: Optional[str] = None,
    commit: str = None,
    system_prompt: str = None,
    temperature: float = DEFAULT_TEMPERATURE,
    seed: int = RANDOM_SEED,
    use_schema: bool = True,
) -> List[str]:
    """
    Run inference using Transformers models and return predicted commit types.
    Direct replacement for the GGUF-based api_call function.

    Args:
        model_id: HuggingFace model ID (e.g., "microsoft/phi-4")
        adapter_id: Optional LoRA adapter ID for fine-tuned models
        commit: Input text to analyze (truncated diff + message)
        system_prompt: System prompt to guide the model
        temperature: Sampling temperature
        seed: Random seed for reproducibility
        use_schema: Whether to enforce JSON schema (always True in our case)

    Returns:
        List of predicted commit types
    """
    # Set seed for reproducibility
    transformers_set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Load model and tokenizer
    model, tokenizer = load_transformer_model(
        model_id=model_id,
        adapter_id=adapter_id,
    )

    # Build messages in chat format
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": commit},
    ]

    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Generate with JSON format
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=DEFAULT_MAX_TOKENS,
            temperature=temperature,
            do_sample=True if temperature > 0 else False,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated part
    input_length = inputs["input_ids"].shape[1]
    generated_ids = outputs[:, input_length:]
    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # Parse JSON output (same logic as GGUF version)
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
