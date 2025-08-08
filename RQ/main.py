import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List
import tiktoken
import pandas as pd
from dotenv import load_dotenv
from datasets import load_dataset
# Load environment variables from .env file
load_dotenv()

sys.path.append(str(Path(__file__).parent.parent))

from utils.llms import constant
import utils.llms as llms
import utils.eval as eval_utils
import utils.prompt as prompt

DATASET_REPO_ID = "Berom0227/Detecting-Semantic-Concerns-in-Tangled-Code-Changes-Using-SLMs"# Data key constants

PROVIDER = "lmstudio"

# API configuration
OPENAI_KEY = os.getenv("OPENAI_API_KEY") or ""

# Model configuration
MODEL_NAMES = [
    # "microsoft/phi-4",  # LM Studio model
    # "gpt-4o-mini",  # OpenAI model
    # "gpt-4.1-2025-04-14",
    # "phi4-commit"
    # "microsoft/phi-4-reasoning-plus",
    "qwen/qwen3-14b",
    "qwen2.5-coder-14b-instruct-mlx@8bit",
]

# Context window sizes for testing
CONTEXT_WINDOW = [1024, 2048, 4096, 8192, 12288]

# Encoding configuration
ENCODING_NAME = "cl100k_base"  # GPT-4 encoding


def _truncate_diffs_equally(diffs: List[str], available_tokens: int, encoding: tiktoken.Encoding) -> str:
    """Truncate a list of diffs to fit within available_tokens, allocating equally per diff."""
    if available_tokens <= 0 or len(diffs) == 0:
        return ""

    tokens_per_diff: int = max(available_tokens // len(diffs), 0)
    tokenized: List[List[int]] = [encoding.encode(d) for d in diffs]
    truncated_tokens: List[List[int]] = [t[:tokens_per_diff] for t in tokenized]
    truncated_texts: List[str] = [encoding.decode(t) for t in truncated_tokens]
    return "\n".join(truncated_texts)


def measure_performance(
    model_name: str,
    truncated_dataset: pd.DataFrame,
    system_prompt: str,
    csv_path: Path,
) -> None:

    for idx, row in truncated_dataset.iterrows():
        commit: str = row["truncated_commit"]
        actual_types: List[str] = json.loads(row["types"])
        try:
            start_time = time.time()
            predicted_types = llms.api_call(
                provider=PROVIDER,
                model_name=model_name,
                commit=commit,
                system_prompt=system_prompt,
            )
            end_time = time.time()
            inference_time = end_time - start_time
        except Exception as e:
            print(f"Unexpected error processing row {idx}: {e}")
            predicted_types = []
            inference_time = 0.0

        metrics = eval_utils.calculate_metrics(predicted_types=predicted_types, actual_types=actual_types)

        result_df = pd.DataFrame(
            {
                "predicted_types": [predicted_types],
                "actual_types": [actual_types],
                "inference_time": [inference_time],
                "shas": [row["shas"]],
                "precision": [metrics["precision"]],
                "recall": [metrics["recall"]],
                "f1": [metrics["f1"]],
                "exact_match": [metrics["exact_match"]],
            },
            columns=constant.DEFAULT_DF_COLUMNS,
        )

        result_df.to_csv(csv_path, mode="a", header=False, index=False)

        print(
            f"Row {idx}: inference_time(sec): {inference_time:.2f} - saved to {csv_path}"
        )


def truncate_dataset(
    tangled_df: pd.DataFrame,
    context_window: int,
    include_message: bool,
) -> pd.DataFrame:
    encoding = tiktoken.get_encoding(ENCODING_NAME)
    truncated_texts: List[str] = []

    for _, row in tangled_df.iterrows():
        message: str = str(row.get("commit_message", ""))
        diffs: List[str] = json.loads(row.get("diff", "[]"))

        if include_message:
            message_tokens: List[int] = encoding.encode(message)
            remaining_tokens: int = max(context_window - len(message_tokens), 0)
            truncated_diffs: str = _truncate_diffs_equally(diffs, remaining_tokens, encoding)
            truncated_texts.append(f"- given commit message:\n {message}\n Diff: {truncated_diffs}")
        else:
            truncated_diffs: str = _truncate_diffs_equally(diffs, context_window, encoding)
            truncated_texts.append(f"- given commit diff: \n {truncated_diffs}")

    return tangled_df.assign(truncated_commit=truncated_texts)


def run_model_experiments(
    model_name: str,
    tangled_df: pd.DataFrame,
) -> None:
    commit_types: List[str] = ["with_message", "diff_only"]
    prompt_types: List[str] = ["Zero-shot", "One-shot", "Two-shot"]

    for commit_type in commit_types:
        prompt_dir: Path = Path("results") / commit_type
        prompt_dir.mkdir(parents=True, exist_ok=True)

        # Determine if commit message should be included based on prompt type
        include_message: bool = commit_type == "with_message"

        for prompt_type in prompt_types:
            system_prompt: str = prompt.get_prompt_by_type(prompt_type, include_message)
            for context_window in CONTEXT_WINDOW:
                print(f"Processing {model_name} {prompt_type} {context_window}")
                truncated_dataset: pd.DataFrame = truncate_dataset(tangled_df, context_window, include_message)
                file_name: str = (
                    f"{model_name.replace('/', '_')}_{prompt_type.replace('-', '_')}_{context_window}.csv"
                )
                csv_path: Path = prompt_dir / file_name
                if not csv_path.exists():
                    df = pd.DataFrame(columns=constant.DEFAULT_DF_COLUMNS)
                    df.to_csv(csv_path, index=False)

                print(
                    f"\n=== Model: {model_name}, Prompt Type: {commit_type}, Prompt: {prompt_type}, Context Window: {context_window} ==="
                )

                measure_performance(
                    model_name,
                    truncated_dataset,
                    system_prompt,
                    csv_path,
                )


def main() -> None:
    tangled_df: pd.DataFrame = load_dataset(DATASET_REPO_ID, "test", split="test").to_pandas()

    for model_name in MODEL_NAMES:
        print(f"\n{'='*50}")
        print(f"Processing Model: {model_name}")
        print(f"{'='*50}")

        run_model_experiments(model_name, tangled_df)

    print("\nDataset Summary:")
    print(f"Loaded tangled dataset: {len(tangled_df)} samples")
    print(f"Processed models: {', '.join(MODEL_NAMES)}")
    print("Results saved in results/ directory, organized by prompt type")


if __name__ == "__main__":
    main()
