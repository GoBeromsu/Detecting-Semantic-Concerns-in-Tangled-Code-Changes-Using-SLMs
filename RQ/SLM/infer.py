#!/usr/bin/env python3
"""
Unified Inference Script for Fine-tuned SLM Models

Usage:
    python infer.py --model phi
    python infer.py --model qwen
    python infer.py  # Default: phi
"""

import argparse
import os
import sys
import json
import pandas as pd
import yaml
import time
import torch
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from itertools import product
from datetime import datetime

# Add parent's parent directory to path (for utils/ and RQ/ access from RQ/SLM/)
sys.path.append(str(Path(__file__).resolve().parents[2]))

import utils.prompt as prompt
import utils.llms as llms
import utils.eval as eval_utils
import RQ.main as rq_main
from utils.llms import constant
from datasets import load_dataset

# Load environment variables
load_dotenv()

# Model configurations - SLM only (Phi and Qwen)
MODEL_CONFIGS = {
    "phi": {
        "repo_id": "microsoft/phi-4-gguf",
        "filename": "phi-4-bf16.gguf",
        "model_name": "Phi4",
        "chat_format": "chatml",
    },
    "phi_lora": {
        "repo_id": "Berom0227/Semantic-Concern-SLM-Phi-gguf",
        "filename": "Semantic-Concern-SLM-Phi-f16.gguf",
        "model_name": "Phi4",
        "chat_format": "chatml",
    },
    "qwen_lora": {
        "repo_id": "Berom0227/Semantic-Concern-SLM-Qwen-gguf",
        "filename": "Semantic-Concern-SLM-Qwen-f16.gguf",
        "model_name": "Qwen3-14B-LoRA",
        "chat_format": "chatml",
    },
    "qwen": {
        "repo_id": "Berom0227/Qwen3-14B-Pure-gguf",
        "filename": "Qwen3-14B-Pure-bf16.gguf",
        "model_name": "Qwen3-14B",
        "chat_format": "chatml",
    },
    "gpt_oss": {
        "repo_id": "ggml-org/gpt-oss-20b-GGUF",
        "filename": "gpt-oss-20b-F16.gguf",
        "model_name": "GPT-OSS-20B",
        "chat_format": "chatml",
    },
    "gpt_oss_lora": {
        "repo_id": "Berom0227/Semantic-Concern-SLM-GPT-OSS-gguf",
        "filename": "Semantic-Concern-SLM-GPT-OSS-f16.gguf",
        "model_name": "GPT-OSS-20B-LoRA",
        "chat_format": "chatml",
    },
}

# Experiment configuration
RESULTS_ROOT: Path = Path(__file__).resolve().parents[2] / "results"
START_TIME_STR: str = datetime.now().strftime("%Y%m%d%H%M%S")

# Default inference parameters
DEFAULT_CONTEXT_WINDOWS = [12288, 8192, 4096, 2048, 1024]
DEFAULT_MAX_TOKENS = 16384
DEFAULT_SEED = 42
DEFAULT_TEMPERATURE = 0.3
DEFAULT_INCLUDE_MESSAGE = True
DEFAULT_SHOT_TYPES = ["Zero-shot"]


def get_compute_device():
    """Determine the best available compute device"""
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple Metal GPU")
    else:
        device = "cpu"
        print("Using CPU")
    return device


def measure_performance(
    model_config: Dict[str, Any],
    truncated_dataset: pd.DataFrame,
    system_prompt: str,
    csv_path: Path,
    context_len: int,
    with_message: bool,
    temperature: float = DEFAULT_TEMPERATURE,
    seed: int = DEFAULT_SEED,
) -> None:
    """Measure model performance on dataset"""

    for row in truncated_dataset.itertuples():
        actual_types: List[str] = json.loads(row.types)
        shas: List[str] = json.loads(row.shas)

        try:
            start_time = time.time()

            predicted_types = llms.hugging_face_api_call(
                repo_id=model_config["repo_id"],
                filename=model_config["filename"],
                commit=row.truncated_commit,
                system_prompt=system_prompt,
                temperature=temperature,
                seed=seed,
                use_schema=True,
                chat_format=model_config["chat_format"],
            )

            end_time = time.time()
            inference_time = end_time - start_time

        except Exception as e:
            print(f"[{row.Index}] Error: {e}")
            predicted_types = []
            inference_time = 0.0

        # Calculate metrics
        metrics = eval_utils.calculate_metrics(predicted_types, actual_types)

        # Save results
        result_df = pd.DataFrame(
            [
                {
                    "predicted_types": json.dumps(predicted_types),
                    "actual_types": row.types,
                    "inference_time": inference_time,
                    "shas": shas,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "exact_match": metrics["exact_match"],
                    "hamming_loss": metrics["hamming_loss"],
                    "context_len": context_len,
                    "with_message": with_message,
                    "concern_count": len(actual_types),
                }
            ],
            columns=constant.DEFAULT_DF_COLUMNS,
        )

        result_df.to_csv(csv_path, mode="a", header=False, index=False)

        if row.Index % 10 == 0:
            print(f"[{row.Index}] Results appended to {csv_path}")


def run_inference(
    model_name: str,
    context_windows: List[int] = None,
    include_message: bool = DEFAULT_INCLUDE_MESSAGE,
    shot_types: List[str] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    seed: int = DEFAULT_SEED,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    revision: str = "main",
):
    """Run inference with specified model and parameters"""

    # Get model configuration
    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model: {model_name}. Choose from {list(MODEL_CONFIGS.keys())}"
        )

    model_config = MODEL_CONFIGS[model_name]

    # Set default values
    if context_windows is None:
        context_windows = DEFAULT_CONTEXT_WINDOWS
    if shot_types is None:
        shot_types = DEFAULT_SHOT_TYPES

    # Get compute device (for information only)
    device = get_compute_device()

    # Create results directory - match Phi structure
    model_display_name = model_config["model_name"]
    base_model_dir = RESULTS_ROOT / model_display_name / START_TIME_STR
    print(f"Creating base results directory: {base_model_dir}")
    base_model_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting inference for {model_display_name}")
    print(f"Results will be saved to: {base_model_dir}")

    # Pre-load GGUF model for better performance (all SLM models are GGUF)
    print(f"Pre-loading GGUF model: {model_config['filename']} (revision: {revision})")
    llms.load_model(
        repo_id=model_config["repo_id"],
        filename=model_config["filename"],
        seed=seed,
        chat_format=model_config["chat_format"],
        revision=revision,
    )

    # Load dataset
    dataset = load_dataset(
        "Berom0227/Detecting-Semantic-Concerns-in-Tangled-Code-Changes-Using-SLMs",
        split="test",
    )
    dataset_df = pd.DataFrame(dataset)

    # Run experiments for different configurations (Zero-shot only, match Phi structure)
    for shot_type, context_len, include_msg in product(
        shot_types, context_windows, (True, False)
    ):
        print(f"\n{'='*60}")
        print(
            f"Configuration: {shot_type}, Context: {context_len}, Message: {include_msg}"
        )
        print(f"{'='*60}")

        # Get system prompt
        system_prompt = prompt.get_prompt_by_type(
            shot_type=shot_type, include_message=include_msg
        )

        # Create folder structure based on message inclusion (match Phi structure)
        msg_flag = "msg1" if include_msg else "msg0"
        model_dir = base_model_dir / msg_flag
        model_dir.mkdir(parents=True, exist_ok=True)

        # Prepare output path with Phi-style naming (Zero-shot only)
        csv_filename = f"{context_len}_zs.csv"
        csv_path = model_dir / csv_filename

        # Initialize CSV with headers
        if not csv_path.exists():
            pd.DataFrame(columns=constant.DEFAULT_DF_COLUMNS).to_csv(
                csv_path, index=False
            )

        # Prepare dataset with truncation
        truncated_dataset = rq_main.add_truncated_commits(
            dataset_df,
            context_window=context_len,
            include_message=include_msg,
        )

        # Run inference
        measure_performance(
            model_config=model_config,
            truncated_dataset=truncated_dataset,
            system_prompt=system_prompt,
            csv_path=csv_path,
            context_len=context_len,
            with_message=include_msg,
            temperature=temperature,
            seed=seed,
        )

        print(f"✅ Completed: {csv_filename}")

    print(f"\n{'='*60}")
    print(f"🎉 All experiments completed!")
    print(f"📊 Results saved to: {base_model_dir}")
    print(f"{'='*60}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run inference on fine-tuned models")
    parser.add_argument(
        "--model",
        type=str,
        default="phi",
        choices=["phi", "qwen", "qwen_lora", "gpt_oss", "gpt_oss_lora"],
        help="SLM model to use for inference (default: phi)",
    )
    parser.add_argument(
        "--context-windows",
        type=int,
        nargs="+",
        default=None,
        help="Context window sizes to test (default: 12288 8192 4096 2048 1024)",
    )
    parser.add_argument(
        "--no-message", action="store_true", help="Exclude commit message from input"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Maximum tokens for generation (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--shot-types",
        type=str,
        nargs="+",
        default=None,
        choices=["Zero-shot", "Few-shot"],
        help="Shot types to test (default: Zero-shot)",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="HuggingFace revision (branch, tag, or commit) for the model (default: main)",
    )

    args = parser.parse_args()

    # Run inference
    run_inference(
        model_name=args.model,
        context_windows=args.context_windows,
        include_message=not args.no_message,
        shot_types=args.shot_types,
        temperature=args.temperature,
        seed=args.seed,
        max_tokens=args.max_tokens,
        revision=args.revision,
    )


if __name__ == "__main__":
    main()
