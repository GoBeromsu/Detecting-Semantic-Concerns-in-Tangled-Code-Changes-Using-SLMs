import os
import sys
import json
import pandas as pd
from typing import List
from pathlib import Path
from dotenv import load_dotenv
import time

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

# Model and dataset configuration (mirror GPT infer structure)
MODEL_NAME = "Phi-4"  # LM Studio model key; ensure it's downloaded in LM Studio
DATASET_REPO_ID = (
    "Berom0227/Detecting-Semantic-Concerns-in-Tangled-Code-Changes-Using-SLMs"
)

# Paths and experiment constants (experiment script concerns)
RESULTS_ROOT: Path = Path("results")
RESULTS_SUBDIR: str = "lmstudio"

# Inference constants
CONTEXT_WINDOWS = [12288]
MAX_TOKENS = 16384
SEED = 42
TEMPERATURE = 0.3
INCLUDE_MESSAGE = True


def measure_performance(
    model_name: str,
    truncated_dataset: pd.DataFrame,
    system_prompt: str,
    csv_path: Path,
) -> None:
    for row in truncated_dataset.itertuples():
        actual_types: List[str] = json.loads(row.types)

        try:
            start_time = time.time()
            predicted_types = llms.lmstudio_api_call(
                model_name=model_name,
                commit=row.truncated_commit,
                system_prompt=system_prompt,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            end_time = time.time()
            inference_time = end_time - start_time
        except Exception as e:
            print(f"[{row.Index}] Error: {e}")
            predicted_types = []
            inference_time = 0.0

        metrics = eval_utils.calculate_metrics(
            predicted_types=predicted_types, actual_types=actual_types
        )

        result_df = pd.DataFrame(
            [
                {
                    "predicted_types": json.dumps(predicted_types),
                    "actual_types": row.types,
                    "inference_time": inference_time,
                    "shas": row.shas,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "exact_match": metrics["exact_match"],
                }
            ],
            columns=constant.DEFAULT_DF_COLUMNS,
        )

        result_df.to_csv(csv_path, mode="a", header=False, index=False)
        if row.Index % 10 == 0:
            print(f"[{row.Index}] appended to {csv_path}")


def main() -> None:
    model_dir: Path = RESULTS_ROOT / MODEL_NAME / RESULTS_SUBDIR
    model_dir.mkdir(parents=True, exist_ok=True)

    tangled_df: pd.DataFrame = load_dataset(DATASET_REPO_ID, split="test").to_pandas()
    system_prompt: str = prompt.get_prompt_by_type(
        shot_type="Zero-shot", include_message=INCLUDE_MESSAGE
    )

    for cw in CONTEXT_WINDOWS:
        file_name: str = f"{MODEL_NAME}_{cw}.csv"
        csv_path: Path = model_dir / file_name
        if not csv_path.exists():
            df = pd.DataFrame(columns=constant.DEFAULT_DF_COLUMNS)
            df.to_csv(csv_path, index=False)

        truncated_dataset: pd.DataFrame = rq_main.truncate_dataset(
            tangled_df=tangled_df,
            context_window=cw,
            include_message=INCLUDE_MESSAGE,
        )

        measure_performance(
            model_name=MODEL_NAME,
            truncated_dataset=truncated_dataset,
            system_prompt=system_prompt,
            csv_path=csv_path,
        )


if __name__ == "__main__":
    main()


