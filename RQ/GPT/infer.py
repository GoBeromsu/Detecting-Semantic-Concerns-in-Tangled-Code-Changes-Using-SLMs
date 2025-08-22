import os
import sys
import json
import pandas as pd
from typing import List
from pathlib import Path
from dotenv import load_dotenv
import time
from itertools import product
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parents[2]))

import utils.prompt as prompt
import utils.llms as llms
import utils.eval as eval_utils
import RQ.main as rq_main
from utils.llms import constant
from datasets import load_dataset

# Load environment variables from .env file
load_dotenv()

MODEL_NAME = "gpt-4.1-2025-04-14"
DATASET_REPO_ID = "Berom0227/Detecting-Semantic-Concerns-in-Tangled-Code-Changes-Using-SLMs"

# Inference constants
CONTEXT_WINDOWS = [4096, 2048, 1024]
SHOT_TYPES = ["One-shot"]
MAX_TOKENS = 16384
SEED = 42
TEMPERATURE = 0.3
INCLUDE_MESSAGE = True
START_TIME_STR: str = datetime.now().strftime("%Y%m%d%H%M")


def perform_api_call(model_name: str, commit: str, system_prompt: str) -> tuple[List[str], float]:
    """Perform single API call with error handling."""
    try:
        start_time = time.time()
        predicted_types = llms.openai_api_call(
            api_key=os.getenv("OPENAI_API_KEY"),
            commit=commit,
            system_prompt=system_prompt,
            model=model_name,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            seed=SEED,
        )
        end_time = time.time()
        return predicted_types, end_time - start_time
    except Exception as e:
        end_time = time.time()
        print(f"API call error: {e}")
        return [], end_time - start_time


def process_row(row: pd.Series, model_name: str, system_prompt: str, context_len: int, with_message: bool) -> dict:
    """Process single row and return result."""
    actual_types = json.loads(row.types)
    predicted_types, inference_time = perform_api_call(model_name, row.truncated_commit, system_prompt)
    metrics = eval_utils.calculate_metrics(predicted_types, actual_types)
    
    return {
        "predicted_types": json.dumps(predicted_types),
        "actual_types": row.types,
        "inference_time": inference_time,
        "shas": row.shas,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "exact_match": metrics["exact_match"],
        "hamming_loss": metrics["hamming_loss"],
        "context_len": context_len,
        "with_message": with_message,
        "concern_count": len(actual_types),
    }


def measure_performance(
    model_name: str,
    truncated_dataset: pd.DataFrame,
    system_prompt: str,
    csv_path: Path,
    context_len: int,
    with_message: bool,
) -> None:
    """Process each row immediately and save, then retry failed ones."""
    # Process and save each row immediately
    for idx, (_, row) in enumerate(truncated_dataset.iterrows()):
        result = process_row(row, model_name, system_prompt, context_len, with_message)
        pd.DataFrame([result])[constant.DEFAULT_DF_COLUMNS].to_csv(csv_path, mode="a", header=False, index=False)
        
        if idx % 10 == 0:
            print(f"[{idx}] processed and saved")
    
    # Check failures and retry for logging only
    saved_results = pd.read_csv(csv_path)
    failed_indices = saved_results[saved_results['predicted_types'] == '[]'].index
    
    for idx in failed_indices:
        row = truncated_dataset.iloc[idx]
        retry_result = process_row(row, model_name, system_prompt, context_len, with_message)
        status = "success" if retry_result['predicted_types'] != '[]' else "failed"
        print(f"Retry [{idx}]: {status}")

def main() -> None:
    prompt_dir: Path = Path(__file__).resolve().parents[2] / "RQ" / "results" / "gpt" / f"{MODEL_NAME}_{START_TIME_STR}"
    print(f"Creating results directory: {prompt_dir}")
    prompt_dir.mkdir(parents=True, exist_ok=True)
    
    tangled_df: pd.DataFrame = load_dataset(DATASET_REPO_ID, split="test").to_pandas()

    shot_abbrev_map = {"Zero-shot": "zs", "One-shot": "os", "Two-shot": "ts"}

    for shot_type, cw, include_message in product(SHOT_TYPES, CONTEXT_WINDOWS, (True, False)):
        system_prompt: str = prompt.get_prompt_by_type(
            shot_type=shot_type, include_message=include_message
        )

        shot_abbrev: str = shot_abbrev_map.get(shot_type, "custom")
        msg_flag: str = "msg1" if include_message else "msg0"
        file_name: str = f"{MODEL_NAME}_{cw}_{shot_abbrev}_{msg_flag}.csv"
        csv_path: Path = prompt_dir / file_name

        if not csv_path.exists():
            df = pd.DataFrame(columns=constant.DEFAULT_DF_COLUMNS)
            df.to_csv(csv_path, index=False)

        truncated_dataset: pd.DataFrame = rq_main.add_truncated_commits(
            tangled_df=tangled_df,
            context_window=cw,
            include_message=include_message,
        )

        measure_performance(
            model_name=MODEL_NAME,
            truncated_dataset=truncated_dataset,
            system_prompt=system_prompt,
            csv_path=csv_path,
            context_len=cw,
            with_message=include_message,
        )


if __name__ == "__main__":
    main()
