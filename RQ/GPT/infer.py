import argparse
import os
import sys
import json
import pandas as pd
from typing import List, Optional
from pathlib import Path
from dotenv import load_dotenv
import time
from itertools import product
from datetime import datetime, timezone

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
DATASET_REPO_ID = (
    "Berom0227/Detecting-Semantic-Concerns-in-Tangled-Code-Changes-Using-SLMs"
)
DATASET_REVISION: Optional[str] = None
RESULTS_DIR = "results/gpt"

# Inference constants
CONTEXT_WINDOWS = [12288, 8192, 4096, 2048, 1024]
SHOT_TYPES = ["Zero-shot"]
MAX_TOKENS = 16384
SEED = 42
TEMPERATURE = 0.3
INCLUDE_MESSAGE = True
START_TIME_STR: str = datetime.now().strftime("%Y%m%d%H%M%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GPT inference sweep.")
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--dataset-repo", type=str, default=DATASET_REPO_ID)
    parser.add_argument("--dataset-revision", type=str, default=DATASET_REVISION)
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR)
    parser.add_argument(
        "--context-windows",
        type=int,
        nargs="+",
        default=CONTEXT_WINDOWS,
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def measure_performance(
    model_name: str,
    truncated_dataset: pd.DataFrame,
    system_prompt: str,
    csv_path: Path,
    context_len: int,
    with_message: bool,
) -> None:
    for row in truncated_dataset.itertuples():
        actual_types: List[str] = json.loads(row.types)
        shas: List[str] = json.loads(row.shas)
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
        metrics = eval_utils.calculate_metrics(predicted_types, actual_types)

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
            print(f"[{row.Index}] appended to {csv_path}")


def main() -> None:
    args = parse_args()

    repo_root: Path = Path(__file__).resolve().parents[2]
    base_model_dir: Path = repo_root / args.results_dir / START_TIME_STR
    print(f"Creating base results directory: {base_model_dir}")
    base_model_dir.mkdir(parents=True, exist_ok=True)

    started_at: str = datetime.now(timezone.utc).isoformat()

    load_kwargs = {"split": "test"}
    if args.dataset_revision:
        load_kwargs["revision"] = args.dataset_revision
    tangled_df: pd.DataFrame = load_dataset(
        args.dataset_repo, **load_kwargs
    ).to_pandas()

    if args.limit is not None:
        tangled_df = tangled_df.head(args.limit).reset_index(drop=True)

    run_identity = {
        "model": args.model,
        "dataset_repo": args.dataset_repo,
        "dataset_revision": args.dataset_revision,
        "context_windows": args.context_windows,
        "shot_types": SHOT_TYPES,
        "seed": SEED,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "row_count": len(tangled_df),
        "limit": args.limit,
        "started_at": started_at,
    }
    with open(base_model_dir / "run_identity.json", "w") as f:
        json.dump(run_identity, f, indent=2)

    shot_abbrev_map = {"Zero-shot": "zs", "One-shot": "os", "Two-shot": "ts"}

    for shot_type, cw, include_message in product(
        SHOT_TYPES, args.context_windows, (True, False)
    ):
        system_prompt: str = prompt.get_prompt_by_type(
            shot_type=shot_type, include_message=include_message
        )

        shot_abbrev: str = shot_abbrev_map.get(shot_type, "custom")
        msg_flag: str = "msg1" if include_message else "msg0"

        msg_dir: Path = base_model_dir / msg_flag
        msg_dir.mkdir(parents=True, exist_ok=True)

        file_name: str = f"{cw}_{shot_abbrev}.csv"
        csv_path: Path = msg_dir / file_name

        if not csv_path.exists():
            df = pd.DataFrame(columns=constant.DEFAULT_DF_COLUMNS)
            df.to_csv(csv_path, index=False)

        truncated_dataset: pd.DataFrame = rq_main.add_truncated_commits(
            tangled_df=tangled_df,
            context_window=cw,
            include_message=include_message,
        )

        measure_performance(
            model_name=args.model,
            truncated_dataset=truncated_dataset,
            system_prompt=system_prompt,
            csv_path=csv_path,
            context_len=cw,
            with_message=include_message,
        )


if __name__ == "__main__":
    main()
