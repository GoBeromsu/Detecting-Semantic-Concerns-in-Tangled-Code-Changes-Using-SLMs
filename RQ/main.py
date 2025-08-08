import os
import sys
import json
from pathlib import Path
from typing import Dict, List
import tiktoken
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

sys.path.append(str(Path(__file__).parent.parent))

from utils.llms import constant
import utils.llms as llms
import utils.eval as eval
import utils.prompt as prompt

# Data key constants
COMMIT_MESSAGE = "commit_message"
GIT_DIFF = "git_diff"
MASKED_COMMIT_MESSAGE_KEY = "masked_commit_message"
TYPES_KEY = "types"
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


def truncate_commits(
    commits: Dict[str, Dict[str, str]],
    context_window: int,
    include_message: bool = True,
) -> str:
    encoding = tiktoken.get_encoding(ENCODING_NAME)
    concern_count: int = len(commits)
    available_tokens_per_commit: int = context_window // concern_count

    messages: List[str] = []
    diffs: List[str] = []

    for commit_data in commits.values():
        message: str = commit_data[COMMIT_MESSAGE]
        diff: str = commit_data[GIT_DIFF]

        if include_message:
            message_tokens: List[int] = encoding.encode(message)
            remaining_tokens: int = available_tokens_per_commit - len(message_tokens)
            messages.append(message)
        else:
            remaining_tokens: int = available_tokens_per_commit

        diff_tokens: List[int] = encoding.encode(diff)
        truncated_diff: str = (
            diff
            if len(diff_tokens) <= remaining_tokens
            else encoding.decode(diff_tokens[:remaining_tokens])
        )

        diffs.append(truncated_diff)

    if include_message:
        return f"Commit Message: {' '.join(messages)}\nDiff: {' '.join(diffs)}"
    else:
        return f"Diff: {' '.join(diffs)}"


def measure_performance(
    model_name: str,
    truncated_dataset: pd.DataFrame,
    system_prompt: str,
    csv_path: Path,
) -> None:

    for idx, row in truncated_dataset.iterrows():
        commit: str = row["truncated_commit"]
        actual_types: List[str] = json.loads(row[TYPES_KEY])
        try:
            api_call = lambda: llms.api_call(
                provider=PROVIDER,
                model_name=model_name,
                commit=commit,
                system_prompt=system_prompt,
                api_key=OPENAI_KEY,
                # temperature=0.7,
            )

            predicted_types, inference_time = eval.measure_inference_time(api_call)
        except Exception as e:
            print(f"Unexpected error processing row {idx}: {e}")
            predicted_types = []
            inference_time = 0.0

        result_df = pd.DataFrame(
            {
                "predicted_types": [predicted_types],
                "actual_types": [actual_types],
                "inference_time": [inference_time],
                "shas": [row["shas"]],
            },
            columns=constant.DEFAULT_DF_COLUMNS,
        )

        result_df.to_csv(csv_path, mode="a", header=False, index=False)

        print(
            f"Row {idx}: inference_time(sec): {inference_time:.2f} - saved to {csv_path}"
        )


def create_csv_with_headers(csv_path: Path) -> None:
    """Create CSV file with headers if it doesn't exist."""
    if not csv_path.exists():
        df = pd.DataFrame(columns=constant.DEFAULT_DF_COLUMNS)
        df.to_csv(csv_path, index=False)


def truncate_dataset(
    atomic_df: pd.DataFrame,
    tangled_df: pd.DataFrame,
    context_window: int,
    include_message: bool,
) -> pd.DataFrame:
    truncated_commits: List[Dict[str, any]] = []
    for _, row in tangled_df.iterrows():
        shas: List[str] = json.loads(row["shas"])
        commits: Dict[str, Dict[str, str]] = {}
        for sha in shas:
            matching_row: pd.DataFrame = atomic_df[atomic_df["sha"] == sha]
            commit_message: str = matching_row[MASKED_COMMIT_MESSAGE_KEY].values[0]
            git_diff: str = matching_row[GIT_DIFF].values[0]
            commits[sha] = {
                COMMIT_MESSAGE: commit_message,
                GIT_DIFF: git_diff,
            }
        truncated_commit: str = truncate_commits(
            commits, context_window, include_message
        )
        truncated_commits.append(
            {
                "shas": shas,
                "truncated_commit": truncated_commit,
                TYPES_KEY: row[TYPES_KEY],
            }
        )
    return pd.DataFrame(truncated_commits)


def run_model_experiments(
    model_name: str,
    atomic_df: pd.DataFrame,
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
                truncated_dataset: pd.DataFrame = truncate_dataset(
                    atomic_df, tangled_df, context_window, include_message
                )
                file_name: str = (
                    f"{model_name.replace('/', '_')}_{prompt_type.replace('-', '_')}_{context_window}.csv"
                )
                csv_path: Path = prompt_dir / file_name
                create_csv_with_headers(csv_path)

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
    tangled_df: pd.DataFrame = pd.read_csv("datasets/data/tangled_ccs_dataset_test.csv")
    atomic_df: pd.DataFrame = pd.read_csv("datasets/data/sampled_ccs_dataset.csv")

    for model_name in MODEL_NAMES:
        print(f"\n{'='*50}")
        print(f"Processing Model: {model_name}")
        print(f"{'='*50}")

        run_model_experiments(model_name, atomic_df, tangled_df)

    print(f"\nDataset Summary:")
    print(f"Loaded tangled dataset: {len(tangled_df)} samples")
    print(f"Loaded atomic dataset: {len(atomic_df)} samples")
    print(f"Processed models: {', '.join(MODEL_NAMES)}")
    print(f"Results saved in results/ directory, organized by prompt type")


if __name__ == "__main__":
    main()
