#!/usr/bin/env python3
"""
Stage 1 of 3 - repo-grouped tangled commit dataset pipeline.

    Stage 1 (this script)            CCS Dataset.csv       -> repo_grouped_pool.csv
    Stage 2 (generate_repo_tangled.py) repo_grouped_pool.csv -> repo_split.json,
                                                                 tangled_ccs_dataset_{train,test}.csv
    Stage 3 (validate_repo_dataset.py) validates the Stage 2 outputs.

Input:  data/CCS Dataset.csv (raw source, 2,000 commits)
Output: data/repo_grouped_pool.csv (eligible repo pool with repo identity,
        remapped taxonomy, and per-atomic token counts)

Reconstructs commit -> repo identity from `commit_url` (dropped by the old
atomic-sampling pipeline this replaces), applies the updated CCS taxonomy
(style/perf -> refactor; chore dropped), the model-context token filter, and
keeps only repos with >=5 distinct types so every repo can host every
concern count (1-5) without a concern-count x repo confound.

NOTE: no manual quality-exclusion filtering is applied here (spec decision,
Round 9: "no manual quality filtering" - only the token-budget filter, which
is a model-context necessity, remains).
"""

import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
import tiktoken

# File paths (run from datasets/)
CCS_SOURCE_PATH = "data/CCS Dataset.csv"
OUTPUT_POOL_PATH = "data/repo_grouped_pool.csv"

# Token limit for diff + message size (16384 - 4096 = 12288)
MAX_TOKENS = 12288
ENCODING_MODEL = "cl100k_base"

# Updated CCS taxonomy: style/perf merged into refactor, chore dropped
CONVENTIONAL_COMMIT_TYPES: List[str] = ["feat", "fix", "refactor", "test", "docs", "build", "ci"]
TYPE_REMAP: Dict[str, str] = {"style": "refactor", "perf": "refactor"}

# Repo eligibility: repos must offer >=5 distinct types after filtering
MIN_TYPES_PER_REPO = 5

REPO_URL_PATTERN = r"github\.com/([^/]+/[^/]+)/commit"

# Column name constants
COLUMN_SHA = "sha"
COLUMN_COMMIT_URL = "commit_url"
COLUMN_ANNOTATED_TYPE = "annotated_type"
COLUMN_GIT_DIFF = "git_diff"
COLUMN_MASKED_COMMIT_MESSAGE = "masked_commit_message"
COLUMN_REPO = "repo"
COLUMN_TOKEN_COUNT = "token_count"

OUTPUT_COLUMNS: List[str] = [
    COLUMN_REPO,
    COLUMN_ANNOTATED_TYPE,
    COLUMN_MASKED_COMMIT_MESSAGE,
    COLUMN_GIT_DIFF,
    COLUMN_SHA,
    COLUMN_TOKEN_COUNT,
]

# Expected results (2026-07-26 verification run); stop and report if numbers
# differ materially rather than silently proceeding on a shifted pool.
EXPECTED_REPOS = 36
EXPECTED_COMMITS = 1309
EXPECTED_TYPE_SUPPLY: Dict[str, int] = {
    "feat": 155,
    "fix": 134,
    "refactor": 449,
    "test": 176,
    "docs": 141,
    "build": 127,
    "ci": 127,
}


def load_ccs_dataset(file_path: str) -> pd.DataFrame:
    """Load the raw CCS dataset CSV."""
    df = pd.read_csv(file_path)
    logging.info(f"Loaded {len(df)} records from {file_path}")
    return df


def extract_repo(df: pd.DataFrame) -> pd.DataFrame:
    """Derive repo identity ('owner/name') from commit_url."""
    df[COLUMN_REPO] = df[COLUMN_COMMIT_URL].str.extract(REPO_URL_PATTERN)
    missing = df[COLUMN_REPO].isna().sum()
    if missing:
        logging.warning(f"Dropping {missing} commits with unparseable commit_url")
    df = df.dropna(subset=[COLUMN_REPO]).copy()
    logging.info(f"Repo extraction: {df[COLUMN_REPO].nunique()} distinct repos, {len(df)} commits")
    return df


def apply_taxonomy(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize annotated_type and remap style/perf -> refactor; drop non-CCS types."""
    # Stage 1: Taxonomy alignment - relabel style/perf as refactor (they are
    # code-improvement concerns without behavior change); drop chore entirely
    # (heterogeneous catch-all, not a semantic concern). 2,000 -> 1,800 commits.
    df[COLUMN_ANNOTATED_TYPE] = df[COLUMN_ANNOTATED_TYPE].str.lower().str.strip()
    df[COLUMN_ANNOTATED_TYPE] = df[COLUMN_ANNOTATED_TYPE].replace(TYPE_REMAP)

    before = len(df)
    df = df[df[COLUMN_ANNOTATED_TYPE].isin(CONVENTIONAL_COMMIT_TYPES)].copy()
    logging.info(f"Taxonomy filtering: kept {len(df)}/{before} commits with valid CCS types")
    return df


def apply_token_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Filter commits whose diff + masked message exceed MAX_TOKENS (cl100k_base)."""
    encoding = tiktoken.get_encoding(ENCODING_MODEL)

    def count_tokens(diff: str, message: str) -> int:
        return len(encoding.encode(str(diff), disallowed_special=())) + len(
            encoding.encode(str(message), disallowed_special=())
        )

    # Stage 2: Model-context filter - keep commits whose diff + masked message
    # fit within 12,288 cl100k tokens so every atomic fits the SLM context
    # window. 1,800 -> 1,708 commits.
    df[COLUMN_TOKEN_COUNT] = [
        count_tokens(g, m)
        for g, m in zip(df[COLUMN_GIT_DIFF], df[COLUMN_MASKED_COMMIT_MESSAGE])
    ]

    before = len(df)
    df = df[df[COLUMN_TOKEN_COUNT] <= MAX_TOKENS].copy()
    logging.info(
        f"Token filtering: kept {len(df)}/{before} commits with diff+message <= {MAX_TOKENS} tokens"
    )
    return df


def filter_eligible_repos(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only repos with >= MIN_TYPES_PER_REPO distinct types after filtering."""
    # Stage 3: Repo eligibility - keep repos with >=5 distinct concern types.
    # A 5-concern intra-repo tangled commit requires 5 distinct types from ONE
    # repo, so repos with fewer types cannot support the full concern range.
    # This is a design necessity, not quality filtering.
    # 1,708 -> 1,309 commits (125 -> 36 repos).
    type_counts_per_repo = df.groupby(COLUMN_REPO)[COLUMN_ANNOTATED_TYPE].nunique()
    eligible_repos = type_counts_per_repo[type_counts_per_repo >= MIN_TYPES_PER_REPO].index

    pool = df[df[COLUMN_REPO].isin(eligible_repos)].copy()
    logging.info(
        f"Repo eligibility (>={MIN_TYPES_PER_REPO} types): "
        f"{len(eligible_repos)} eligible repos, {len(pool)} commits"
    )
    return pool


def log_summary(pool: pd.DataFrame) -> None:
    """Log pool summary and compare against the expected verification numbers."""
    n_repos = pool[COLUMN_REPO].nunique()
    n_commits = len(pool)
    type_supply = pool.groupby(COLUMN_ANNOTATED_TYPE)[COLUMN_SHA].count().to_dict()

    logging.info("=" * 50)
    logging.info(f"Repo pool summary: {n_repos} repos, {n_commits} commits")
    logging.info(f"Per-type supply: {type_supply}")
    logging.info("=" * 50)

    mismatches = []
    if n_repos != EXPECTED_REPOS:
        mismatches.append(f"repo count {n_repos} != expected {EXPECTED_REPOS}")
    if n_commits != EXPECTED_COMMITS:
        mismatches.append(f"commit count {n_commits} != expected {EXPECTED_COMMITS}")
    for t, expected in EXPECTED_TYPE_SUPPLY.items():
        actual = type_supply.get(t, 0)
        if actual != expected:
            mismatches.append(f"type '{t}' supply {actual} != expected {expected}")

    if mismatches:
        logging.error("Pool numbers differ materially from the spec's verified baseline:")
        for m in mismatches:
            logging.error(f"  - {m}")
        raise SystemExit(
            "FATAL: repo pool does not match expected verification numbers. "
            "STOPPING per spec instruction - investigate before proceeding."
        )
    logging.info("Pool matches expected verification numbers exactly.")


def save_pool(df: pd.DataFrame, output_path: str) -> None:
    """Save the repo-grouped pool CSV."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df[OUTPUT_COLUMNS].to_csv(output_path, index=False, encoding="utf-8")
    logging.info(f"Saved {len(df)} pooled commits to {output_path}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info("Building repo-grouped atomic commit pool")
    logging.info("=" * 50)

    df = load_ccs_dataset(CCS_SOURCE_PATH)
    df = extract_repo(df)
    df = apply_taxonomy(df)
    df = apply_token_filter(df)
    pool = filter_eligible_repos(df)

    log_summary(pool)
    save_pool(pool, OUTPUT_POOL_PATH)


if __name__ == "__main__":
    main()
