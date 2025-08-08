from __future__ import annotations

import ast
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import matplotlib

# Use a non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Paths
RESULTS_DIR: Path = Path(__file__).parent / "results"
ANALYSIS_DIR: Path = RESULTS_DIR / "analysis"

# Output suffixes (kept for potential future use)
OUTPUT_SUFFIX_MACRO = "_macro.csv"
OUTPUT_SUFFIX_BY_CONCERN = "_macro_by_concern.csv"


def list_csv_files(base_dir: Path) -> List[Path]:
    """Recursively list raw results CSV files under base_dir, excluding analysis outputs."""
    if not base_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {base_dir}")

    csv_files: List[Path] = []
    for path in base_dir.rglob("*.csv"):
        # Skip anything inside the analysis directory
        try:
            path.relative_to(ANALYSIS_DIR)
            # If relative_to doesn't raise, it's inside analysis → skip
            continue
        except ValueError:
            pass

        stem = path.stem
        # Skip our generated summaries (by name suffix)
        if stem.endswith(OUTPUT_SUFFIX_MACRO.replace(".csv", "")):
            continue
        if stem.endswith(OUTPUT_SUFFIX_BY_CONCERN.replace(".csv", "")):
            continue

        csv_files.append(path)

    return csv_files


def compute_macro(df: pd.DataFrame) -> pd.DataFrame:
    """Compute macro metrics (mean across rows) from present columns.

    Requires: precision, recall, f1, exact_match
    """
    required = ["precision", "recall", "f1", "exact_match"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for macro metrics: {missing}")

    summary = {
        "macro_precision": float(df["precision"].mean()),
        "macro_recall": float(df["recall"].mean()),
        "macro_f1": float(df["f1"].mean()),
        "macro_accuracy": float(df["exact_match"].mean()),
        "num_samples": int(len(df)),
    }
    return pd.DataFrame([summary])


def compute_macro_by_concern(df: pd.DataFrame) -> pd.DataFrame:
    """Compute macro metrics grouped by 'concern_count'."""

    grouped = (
        df.groupby("concern_count")[
            ["precision", "recall", "f1", "exact_match"]
        ]
        .mean()
        .reset_index()
    )

    grouped = grouped.rename(
        columns={
            "precision": "macro_precision",
            "recall": "macro_recall",
            "f1": "macro_f1",
            "exact_match": "macro_accuracy",
        }
    )

    counts = df.groupby("concern_count").size().reset_index(name="num_samples")
    result = pd.merge(grouped, counts, on="concern_count", how="left")
    return result.sort_values("concern_count").reset_index(drop=True)


def build_summary_json(
    df: pd.DataFrame,
    macro_df: pd.DataFrame,
    by_concern_df: pd.DataFrame,
    csv_path: Path,
) -> Dict[str, Any]:
    """Build experiment summary JSON structure from inputs."""
    if "context_len" not in df.columns:
        raise ValueError("Missing 'context_len' column for JSON summary")
    if "with_message" not in df.columns:
        raise ValueError("Missing 'with_message' column for JSON summary")
    if "inference_time" not in df.columns:
        raise ValueError("Missing 'inference_time' column for JSON summary")

    # Basic metadata
    stem_parts = csv_path.stem.split("_")
    model = stem_parts[0] if len(stem_parts) > 0 else csv_path.stem
    context_length = int(df["context_len"].iloc[0])
    with_message = bool(df["with_message"].iloc[0])

    # experiment_id: <model>_<context>_<yes|no>_<YYYYMMDD>
    created_at = datetime.now(tz=timezone.utc)
    date_str = created_at.strftime("%Y%m%d")
    experiment_id = f"{model}_{context_length}_{'yes' if with_message else 'no'}_{date_str}"

    # Macro metrics mapping
    macro_row = macro_df.iloc[0]
    metrics_macro = {
        "accuracy": float(macro_row["macro_accuracy"]),
        "precision": float(macro_row["macro_precision"]),
        "recall": float(macro_row["macro_recall"]),
        "f1": float(macro_row["macro_f1"]),
        "inference_time_avg": float(df["inference_time"].mean()),
    }

    # Grouped metrics by concern_count
    metrics_by_concern: List[Dict[str, Any]] = []
    for _, row in by_concern_df.iterrows():
        metrics_by_concern.append(
            {
                "concern_count": int(row["concern_count"]),
                "accuracy": float(row["macro_accuracy"]),
                "precision": float(row["macro_precision"]),
                "recall": float(row["macro_recall"]),
                "f1": float(row["macro_f1"]),
                "num_samples": int(row["num_samples"]),
            }
        )

    summary: Dict[str, Any] = {
        "experiment_id": experiment_id,
        "model": model,
        "context_length": context_length,
        "with_message": with_message,
        "num_samples": int(len(df)),
        "metrics_macro": metrics_macro,
        "metrics_by_concern": metrics_by_concern,
        "created_at": created_at.isoformat(),
    }
    return summary


def save_json(data: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_concern_plot(by_concern_df: pd.DataFrame, csv_path: Path) -> None:
    """Save a simple plot of performance by concern_count next to the source CSV.

    Plots Macro F1, Accuracy, Precision, and Recall over concern_count.
    """
    out_plot = csv_path.parent / "plot" / f"{csv_path.stem}_by_concern.png"
    out_plot.parent.mkdir(parents=True, exist_ok=True)

    x = by_concern_df["concern_count"].astype(int).tolist()
    f1_values = by_concern_df["macro_f1"].astype(float).tolist()
    acc_values = by_concern_df["macro_accuracy"].astype(float).tolist()
    prec_values = by_concern_df["macro_precision"].astype(float).tolist()
    rec_values = by_concern_df["macro_recall"].astype(float).tolist()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, f1_values, marker="o", label="Macro F1")
    ax.plot(x, acc_values, marker="s", label="Accuracy")
    ax.plot(x, prec_values, marker="^", label="Precision")
    ax.plot(x, rec_values, marker="d", label="Recall")
    ax.set_xlabel("Concern Count")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"By-Concern Performance: {csv_path.stem}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_plot, dpi=150)
    plt.close(fig)


def process_csv(csv_path: Path) -> None:
    df = pd.read_csv(csv_path)

    macro_df = compute_macro(df)
    by_concern_df = compute_macro_by_concern(df)

    # JSON goes under the CSV's directory in a 'json' subfolder
    out_json = csv_path.parent / "json" / f"{csv_path.stem}.json"

    summary = build_summary_json(df, macro_df, by_concern_df, csv_path)
    save_json(summary, out_json)
    save_concern_plot(by_concern_df, csv_path)

    base_root = RESULTS_DIR.parent
    print(f"Saved: {out_json.relative_to(base_root)}")
    print(f"Saved: {(csv_path.parent / 'plot' / f'{csv_path.stem}_by_concern.png').relative_to(base_root)}")


def main() -> None:
    csv_files = list_csv_files(RESULTS_DIR)
    for csv_path in csv_files:
        process_csv(csv_path)


if __name__ == "__main__":
    main()
