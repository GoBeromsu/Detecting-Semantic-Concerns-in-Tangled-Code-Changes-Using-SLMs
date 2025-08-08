from typing import Dict, List, Any

COMMIT_MESSAGE = "commit_message"
GIT_DIFF = "git_diff"
MASKED_COMMIT_MESSAGE_KEY = "masked_commit_message"
TYPES_KEY = "types"

# API Configuration Constants
DEFAULT_LMSTUDIO_URL: str = "localhost:1234"
DEFAULT_OPENAI_MODEL: str = "gpt-4.1-2025-04-14"
DEFAULT_TEMPERATURE: float = 0.3 # 
DEFAULT_MAX_TOKENS: int = 16384
CONNECTION_TIMEOUT_SECONDS: int = 30
RANDOM_SEED: int = 42

DEFAULT_DF_COLUMNS: List[str] = [
    "predicted_types",
    "actual_types",
    "inference_time",
    "shas",
    "precision",
    "recall",
    "f1",
    "exact_match",
    # Additional metadata columns appended for analysis/reporting
    "context_len",
    "with_message",
    "concern_count",
]
# LM Studio Model Load Configuration (based on https://lmstudio.ai/docs/typescript/api-reference/llm-load-model-config)
LMSTUDIO_MODEL_CONFIG: Dict[str, Any] = {
    "contextLength": DEFAULT_MAX_TOKENS,
    "gpu": {
        "ratio": 1.0,  # Use maximum GPU
    },
    "flashAttention": True,  # Enable Flash Attention for better performance
    "keepModelInMemory": True,  # Keep model in memory for faster access
    "useFp16ForKVCache": True,  # Use FP16 for KV cache to save memory
    "evalBatchSize": 1024,  # Optimal batch size for evaluation
    "seed": RANDOM_SEED,
}

# UI Configuration Constants
CODE_DIFF_INPUT_HEIGHT: int = 300
SYSTEM_PROMPT_INPUT_HEIGHT: int = 200
RECENT_RESULTS_DISPLAY_LIMIT: int = 15

# Commit types for classification
COMMIT_TYPES: List[str] = [
    "docs",
    "test",
    "cicd",
    "build",
    "refactor",
    "feat",
    "fix",
]

# Base response schema for commit classification (visual_eval standard)
RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "types": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": COMMIT_TYPES,
            },
        },
    },
    "required": ["types"],
    "additionalProperties": False,
}


def create_structured_output_format(
    schema_name: str = "commit_classification_response",
    schema: Dict[str, Any] = RESPONSE_SCHEMA,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Create structured output format for API calls.

    Args:
        schema_name: Name for the JSON schema
        schema: The JSON schema definition
        strict: Whether to use strict mode

    Returns:
        Structured output format dict
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "schema": schema,
            "strict": strict,
        },
    }


# Pre-configured structured output formats for different providers
OPENAI_STRUCTURED_OUTPUT_FORMAT: Dict[str, Any] = create_structured_output_format()
LMSTUDIO_STRUCTURED_OUTPUT_FORMAT: Dict[str, Any] = create_structured_output_format()


# Legacy alias for backwards compatibility
STRUCTURED_OUTPUT_FORMAT: Dict[str, Any] = OPENAI_STRUCTURED_OUTPUT_FORMAT
