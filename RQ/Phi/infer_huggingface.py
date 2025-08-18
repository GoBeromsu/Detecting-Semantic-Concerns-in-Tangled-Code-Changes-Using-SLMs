import os
import sys
import json
import pandas as pd
from typing import List
from pathlib import Path
from dotenv import load_dotenv
import time
import torch
from itertools import product
from datetime import datetime

# Ensure project root on path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import utils.prompt as prompt
import utils.llms as llms
import utils.eval as eval_utils
import RQ.main as rq_main
from utils.llms import constant
from datasets import load_dataset

# Load environment variables from .env file
load_dotenv()

REPO_ID = "Berom0227/phi4-commit-gguf"
MODEL_NAME = "Phi-4"

# Paths and experiment constants (experiment script concerns)
RESULTS_ROOT: Path = Path(__file__).resolve().parents[2] / "RQ" / "results"
RESULTS_SUBDIR: str = "huggingface"
START_TIME_STR: str = datetime.now().strftime("%Y%m%d%H%M%S")

# Inference constants
CONTEXT_WINDOWS = [1024]
MAX_TOKENS = 16384
SEED = 42
TEMPERATURE = 0.3
INCLUDE_MESSAGE = True
CHAT_FORMAT = "chatml"
SHOT_TYPES = ["One-shot"]
# SHOT_TYPES = ["Zero-shot", "One-shot"]


def perform_api_call(
    repo_id: str,
    filename: str,
    commit: str,
    system_prompt: str,
) -> tuple[List[str], float]:
    """Perform single API call with error handling."""
    try:
        start_time = time.time()
        predicted_types = llms.hugging_face_api_call(
            repo_id=repo_id,
            filename=filename,
            commit=commit,
            system_prompt=system_prompt,
            temperature=TEMPERATURE,
            seed=SEED,
            use_schema=True,
            chat_format=CHAT_FORMAT,
        )
        end_time = time.time()
        return predicted_types, end_time - start_time
    except Exception as e:
        end_time = time.time()
        print(f"API call error: {e}")
        return [], end_time - start_time


def process_row(row: pd.Series, repo_id: str, filename: str, system_prompt: str, context_len: int, with_message: bool) -> dict:
    """Process single row and return result."""
    actual_types = json.loads(row.types)
    predicted_types, inference_time = perform_api_call(repo_id, filename, row.truncated_commit, system_prompt)
    metrics = eval_utils.calculate_metrics(predicted_types, actual_types)
    
    return {
        "predicted_types": json.dumps(predicted_types),
        "actual_types": row.types,
        "inference_time": inference_time,
        "shas": json.loads(row.shas),
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "exact_match": metrics["exact_match"],
        "context_len": context_len,
        "with_message": with_message,
        "concern_count": len(actual_types),
    }


def measure_performance(
    repo_id: str,
    filename: str,
    truncated_dataset: pd.DataFrame,
    system_prompt: str,
    csv_path: Path,
    context_len: int,
    with_message: bool,
) -> None:
    """Process each row immediately and save, then retry failed ones."""
    # Process and save each row immediately
    for idx, (_, row) in enumerate(truncated_dataset.iterrows()):
        result = process_row(row, repo_id, filename, system_prompt, context_len, with_message)
        pd.DataFrame([result])[constant.DEFAULT_DF_COLUMNS].to_csv(csv_path, mode="a", header=False, index=False)
        
        if idx % 10 == 0:
            print(f"[{idx}] processed and saved")
    
    # Check failures and retry for logging only
    saved_results = pd.read_csv(csv_path)
    failed_indices = saved_results[saved_results['predicted_types'] == '[]'].index
    
    for idx in failed_indices:
        row = truncated_dataset.iloc[idx]
        retry_result = process_row(row, repo_id, filename, system_prompt, context_len, with_message)
        status = "success" if retry_result['predicted_types'] != '[]' else "failed"
        print(f"Retry [{idx}]: {status}")


def get_compute_device() -> str:
    """Return a short description of the compute device (cuda/mps/cpu)."""
    if torch.cuda.is_available():
        device_name: str = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        return f"cuda ({device_name}, cc={major}.{minor})"
    mps_available: bool = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if mps_available:
        return "mps (Apple Silicon)"
    return "cpu"


def main() -> None:
    model_dir: Path = RESULTS_ROOT / f"{MODEL_NAME}_{START_TIME_STR}" / RESULTS_SUBDIR
    print(f"Creating results directory: {model_dir}")
    model_dir.mkdir(parents=True, exist_ok=True)

    tangled_df: pd.DataFrame = load_dataset(
        "Berom0227/Detecting-Semantic-Concerns-in-Tangled-Code-Changes-Using-SLMs",
        split="test",
    ).to_pandas()

    device_info: str = get_compute_device()
    print(f"Hugging Face device: {device_info}")
    
    is_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    filename = "phi-4-Q6_K.gguf" if is_mps else "phi-4-bf16.gguf"

    # Preload/caches model with chatml format for reproducibility
    llms.load_model(repo_id=REPO_ID, filename=filename, seed=SEED, chat_format=CHAT_FORMAT)

    shot_abbrev_map = {"Zero-shot": "zs", "One-shot": "os", "Two-shot": "ts"}

    for shot_type, cw, include_message in product(
        SHOT_TYPES, CONTEXT_WINDOWS, (True, False)
    ):
        print(f"Running {shot_type} {cw} {include_message}")
        system_prompt: str = prompt.get_prompt_by_type(
            shot_type=shot_type, include_message=include_message
        )
        shot_abbrev: str = shot_abbrev_map.get(shot_type, "custom")
        msg_flag: str = "msg1" if include_message else "msg0"
        file_name: str = f"{MODEL_NAME}_{cw}_{shot_abbrev}_{msg_flag}.csv"
        csv_path: Path = model_dir / file_name
        if not csv_path.exists():
            df = pd.DataFrame(columns=constant.DEFAULT_DF_COLUMNS)
            df.to_csv(csv_path, index=False)

        truncated_dataset: pd.DataFrame = rq_main.add_truncated_commits(
            tangled_df=tangled_df,
            context_window=cw,
            include_message=include_message,
        )

        measure_performance(
            repo_id=REPO_ID,
            filename=filename,
            truncated_dataset=truncated_dataset,
            system_prompt=system_prompt,
            csv_path=csv_path,
            context_len=cw,
            with_message=include_message,
        )
if __name__ == "__main__":
    main()


