import os
import sys
import json
import pandas as pd
from typing import List
from pathlib import Path
from dotenv import load_dotenv
import time

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
    context_len: int,
) -> None:
    for row in truncated_dataset.itertuples():

        actual_types: List[str] = json.loads(row.types)
        try:
            start_time = time.time()
            predicted_types = llms.openai_api_call(
                api_key=os.getenv("OPENAI_API_KEY"),
                commit=row.truncated_commit,
                system_prompt=system_prompt,
                model=model_name,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                seed=SEED,
            )
            end_time = time.time()
            inference_time = end_time - start_time

        except Exception as e:
            print(f"[{row.Index}] Error: {e}")
            predicted_types = []
            inference_time = 0.0
        metrics = eval_utils.calculate_metrics(predicted_types=predicted_types, actual_types=actual_types)

        result_df = pd.DataFrame([
            {
                "predicted_types": json.dumps(predicted_types),
                "actual_types": row.types,
                "inference_time": inference_time,
                "shas": row.shas,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "exact_match": metrics["exact_match"],
                "context_len": context_len,
                "with_message": INCLUDE_MESSAGE,
                "concern_count": len(actual_types),
           } ],
            columns=constant.DEFAULT_DF_COLUMNS,
        )

        result_df.to_csv(csv_path, mode="a", header=False, index=False)
        if row.Index % 10 == 0:
            print(f"[{row.Index}] appended to {csv_path}")

def main() -> None:
    prompt_dir: Path = Path("results/gpt/")
    prompt_dir.mkdir(parents=True, exist_ok=True)
    
    tangled_df: pd.DataFrame = load_dataset(DATASET_REPO_ID, split="test").to_pandas()

    SYSTEM_PROMPT: str = prompt.get_prompt_by_type(shot_type="Zero-shot", include_message=INCLUDE_MESSAGE)

    for cw in CONTEXT_WINDOWS:
        file_name: str = MODEL_NAME+str(cw)+".csv"
        csv_path: Path = prompt_dir / file_name
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
            system_prompt=SYSTEM_PROMPT,
            csv_path=csv_path,
            context_len=cw,
        )


if __name__ == "__main__":
    main()
