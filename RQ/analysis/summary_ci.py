#!/usr/bin/env python3
"""
Summary tables with confidence intervals.

Emits the LaTeX-friendly CSVs behind the manuscript's summary tables (RQ1 by concern
count, RQ2 by commit-message inclusion, RQ3 by token budget). Every cell is the
commit-level mean Hamming loss with its 95% normal confidence interval, formatted as
`0.16 [0.15, 0.18]`.

Run through the per-RQ wrappers registered in config.yaml, e.g.
    python -m RQ.analysis.RQ1.performance_summary_ci
"""

from pathlib import Path

import pandas as pd
import yaml

from . import ANALYSIS_OUTPUT_BASE, CONFIG_PATH, PROJECT_ROOT
from .stats_utils import mean_ci

MODELS = ["GPT-4.1", "Qwen", "QwenFT"]
OUTPUTS = {
    "rq1": ("RQ1", "pf_summary", "performance_summary_ci_latex.csv"),
    "rq2": ("RQ2", "pf_msg_impact", "msg_impact_summary_ci_latex.csv"),
    "rq3": ("RQ3", "pf_context_length", "context_length_summary_ci_latex.csv"),
}


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_results(config: dict, model: str, msg: int, context: int) -> pd.DataFrame:
    spec = config["models"][model]
    path = PROJECT_ROOT / spec["base_path"] / spec["csv_pattern"].format(msg=msg, context=context)
    return pd.read_csv(path)


def format_row(stats: dict) -> dict:
    """Format one mean and confidence-interval cell."""
    return {
        "hl_ci": f"{stats['mean']:.2f} [{stats['ci_low']:.2f}, {stats['ci_high']:.2f}]",
        "hl_mean": round(stats["mean"], 4),
        "ci_low": round(stats["ci_low"], 4),
        "ci_high": round(stats["ci_high"], 4),
        "n_commits": stats["n_commits"],
        "n_rows": stats["n_rows"],
    }


def build_rows(rq: str, config: dict) -> list:
    default_context = config["common"]["default_context"]
    rows = []
    if rq == "rq1":
        for model in MODELS:
            df = load_results(config, model, 1, default_context)
            for n in [1, 2, 3, 4, 5, "All"]:
                sub = df if n == "All" else df[df["concern_count"] == n]
                rows.append({"concern_count": n, "model": model, **format_row(mean_ci(sub))})
    elif rq == "rq2":
        for model in MODELS:
            for msg in (0, 1):
                df = load_results(config, model, msg, default_context)
                rows.append({"with_message": msg, "model": model, **format_row(mean_ci(df))})
    elif rq == "rq3":
        for model in MODELS:
            for context in config["common"]["context_lengths"]:
                df = load_results(config, model, 1, context)
                rows.append({"context_len": context, "model": model, **format_row(mean_ci(df))})
    else:
        raise ValueError(rq)
    return rows


def main(rq: str) -> Path:
    config = load_config()
    rq_name, subdir, filename = OUTPUTS[rq]
    out_dir = ANALYSIS_OUTPUT_BASE / rq_name / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    pd.DataFrame(build_rows(rq, config)).to_csv(out_path, index=False)
    print(f"Table CSV: {out_path}")
    return out_path
