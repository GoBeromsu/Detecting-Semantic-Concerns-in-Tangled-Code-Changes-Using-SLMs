#!/usr/bin/env python3
"""
Stage 3 of 3 - repo-grouped tangled commit dataset pipeline.

    Stage 1 (build_repo_pool.py)       CCS Dataset.csv       -> repo_grouped_pool.csv
    Stage 2 (generate_repo_tangled.py) repo_grouped_pool.csv -> repo_split.json,
                                                                 tangled_ccs_dataset_{train,test}.csv
    Stage 3 (this script)              validates the Stage 2 outputs.

Input:  data/repo_grouped_pool.csv, data/repo_split.json,
        data/tangled_ccs_dataset_{train,test}.csv (Stage 1/2 outputs)
Output: PASS/FAIL report to stdout (exit 1 on any failure) plus
        results/analysis/repo_dataset_validation/{type_distribution,per_repo_distribution}.png

Checks (each prints PASS/FAIL; any failure exits non-zero):
  a. Every row's shas resolve (via the pool) to the same repo as the row's `repo` column
  b. train_repos ∩ test_repos = ∅
  c. Per-type label share deviation from 1/7 is 0.00pp in both splits; train<->test gap 0.00pp
  d. Exactly 280/70 rows per concern_count (1-5); len(types) == concern_count per row;
     no duplicate SHA frozensets within or across splits
  e. Every row's combined diff (real tiktoken encoding of the joined diff text) <= 12288 tokens

Also reports reuse stats (max/mean per split), per-repo row counts, and saves a
type-distribution + per-repo bar chart to results/analysis/repo_dataset_validation/.
"""

import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import tiktoken

# Project root (three levels up from datasets/scripts/validate_repo_dataset.py)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "RQ" / "analysis"))
from plot_utils import COLORS, PLOT_STYLE, setup_plot_style  # noqa: E402

# File paths (run from datasets/, relative to that cwd)
POOL_CSV_PATH = "data/repo_grouped_pool.csv"
TRAIN_CSV_PATH = "data/tangled_ccs_dataset_train.csv"
TEST_CSV_PATH = "data/tangled_ccs_dataset_test.csv"
REPO_SPLIT_JSON_PATH = "data/repo_split.json"

OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis" / "repo_dataset_validation"

TYPES: List[str] = ["feat", "fix", "refactor", "test", "docs", "build", "ci"]
MAX_TOKENS = 12288
ENCODING_MODEL = "cl100k_base"
CONCERN_COUNTS: List[int] = [1, 2, 3, 4, 5]
TRAIN_PER_COUNT = 280
TEST_PER_COUNT = 70
SHARE_DEVIATION_EPSILON = 1e-9  # floating tolerance for "exactly 0.00pp"

failures: List[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    """Print PASS/FAIL for a single check and record failures."""
    status = "PASS" if ok else "FAIL"
    suffix = f" - {detail}" if detail else ""
    print(f"[{status}] {name}{suffix}")
    if not ok:
        failures.append(name)


def load_inputs() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    pool = pd.read_csv(POOL_CSV_PATH)
    train = pd.read_csv(TRAIN_CSV_PATH)
    test = pd.read_csv(TEST_CSV_PATH)
    with open(REPO_SPLIT_JSON_PATH, "r", encoding="utf-8") as f:
        repo_split = json.load(f)
    return pool, train, test, repo_split


def parse_rows(df: pd.DataFrame) -> List[Dict]:
    """Parse JSON-encoded columns (shas, types, diff) into Python objects."""
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "shas": json.loads(row["shas"]),
                "types": json.loads(row["types"]),
                "diffs": json.loads(row["diff"]),
                "concern_count": row["concern_count"],
                "repo": row["repo"],
            }
        )
    return rows


def check_repo_consistency(rows: List[Dict], sha_to_repo: Dict[str, str], split_name: str) -> bool:
    """(a) Every row's shas must all map to the same repo as the row's `repo` column."""
    ok = True
    for i, row in enumerate(rows):
        repos_for_shas = {sha_to_repo.get(sha) for sha in row["shas"]}
        if repos_for_shas != {row["repo"]}:
            print(
                f"  [{split_name}] row {i}: shas resolve to repos {repos_for_shas}, "
                f"expected only {{'{row['repo']}'}}"
            )
            ok = False
    return ok


def check_row_counts_and_types(rows: List[Dict], target_per_count: int, split_name: str) -> bool:
    """(d, part 1/2) Exactly N rows per concern_count; len(types) == concern_count."""
    ok = True
    counts = Counter(row["concern_count"] for row in rows)
    for k in CONCERN_COUNTS:
        if counts.get(k, 0) != target_per_count:
            print(f"  [{split_name}] concern_count={k}: got {counts.get(k, 0)}, expected {target_per_count}")
            ok = False
    for i, row in enumerate(rows):
        if len(row["types"]) != row["concern_count"]:
            print(
                f"  [{split_name}] row {i}: len(types)={len(row['types'])} != "
                f"concern_count={row['concern_count']}"
            )
            ok = False
    return ok


def check_no_duplicate_sha_sets(train_rows: List[Dict], test_rows: List[Dict]) -> bool:
    """(d, part 2/2) No duplicate SHA frozensets within or across splits."""
    seen: Dict[frozenset, str] = {}
    ok = True
    for split_name, rows in (("train", train_rows), ("test", test_rows)):
        for row in rows:
            fz = frozenset(row["shas"])
            if fz in seen:
                print(f"  duplicate SHA set {fz} in {split_name} (first seen in {seen[fz]})")
                ok = False
            else:
                seen[fz] = split_name
    return ok


def check_token_budget(rows: List[Dict], split_name: str, encoding) -> bool:
    """(e) Real combined diff (joined diff text, tiktoken-encoded) <= MAX_TOKENS."""
    ok = True
    for i, row in enumerate(rows):
        joined_diff = "".join(str(d) for d in row["diffs"])
        n_tokens = len(encoding.encode(joined_diff, disallowed_special=()))
        if n_tokens > MAX_TOKENS:
            print(f"  [{split_name}] row {i}: combined diff {n_tokens} tokens > {MAX_TOKENS}")
            ok = False
    return ok


def compute_type_shares(rows: List[Dict]) -> Dict[str, float]:
    counts: Counter = Counter()
    for row in rows:
        counts.update(row["types"])
    total = sum(counts.values())
    return {t: counts.get(t, 0) / total for t in TYPES}, counts


def check_uniform_shares(shares: Dict[str, float], split_name: str) -> bool:
    """(c) Per-type label share deviation from 1/7 is 0.00pp."""
    ok = True
    for t in TYPES:
        dev = abs(shares[t] - 1 / 7)
        if dev > SHARE_DEVIATION_EPSILON:
            print(f"  [{split_name}] type '{t}': deviation {dev * 100:.4f}pp from 1/7")
            ok = False
    return ok


def reuse_stats(rows: List[Dict]) -> Dict[str, float]:
    reuse: Counter = Counter()
    for row in rows:
        for sha in row["shas"]:
            reuse[sha] += 1
    values = sorted(reuse.values(), reverse=True)
    if not values:
        return {"max": 0, "mean": 0.0, "n_atomics": 0}
    return {"max": values[0], "mean": sum(values) / len(values), "n_atomics": len(values)}


def per_repo_row_counts(rows: List[Dict]) -> Counter:
    return Counter(row["repo"] for row in rows)


def save_charts(
    train_counts: Counter,
    test_counts: Counter,
    train_repo_counts: Counter,
    test_repo_counts: Counter,
    output_dir: Path,
) -> None:
    """Save a type-distribution bar chart and a per-repo row-count bar chart."""
    setup_plot_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_counts = train_counts + test_counts
    fig, ax = plt.subplots(figsize=PLOT_STYLE["figure_size"])
    values = [combined_counts.get(t, 0) for t in TYPES]
    ax.bar(TYPES, values, color=COLORS["primary"], width=0.6)
    ax.set_xlabel("Concern Type", fontweight="bold", color=COLORS["text"])
    ax.set_ylabel("Commit-Label Count", fontweight="bold", color=COLORS["text"])
    ax.grid(True, axis="y", alpha=PLOT_STYLE["grid_alpha"], linestyle="-", linewidth=0.5)
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)
    plt.tight_layout()
    type_dist_path = output_dir / "type_distribution.png"
    plt.savefig(type_dist_path, dpi=PLOT_STYLE["dpi"], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved type distribution chart to {type_dist_path}")

    all_repos = sorted(set(train_repo_counts) | set(test_repo_counts))
    train_vals = [train_repo_counts.get(r, 0) for r in all_repos]
    test_vals = [test_repo_counts.get(r, 0) for r in all_repos]
    fig, ax = plt.subplots(figsize=(max(10, len(all_repos) * 0.4), 6))
    x = range(len(all_repos))
    ax.bar(x, train_vals, color=COLORS["primary"], label="Train", width=0.8)
    ax.bar(x, test_vals, bottom=train_vals, color=COLORS["secondary"], label="Test", width=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(all_repos, rotation=90, fontsize=6)
    ax.set_xlabel("Repo", fontweight="bold", color=COLORS["text"])
    ax.set_ylabel("Tangled Row Count", fontweight="bold", color=COLORS["text"])
    ax.legend()
    ax.grid(True, axis="y", alpha=PLOT_STYLE["grid_alpha"], linestyle="-", linewidth=0.5)
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)
    plt.tight_layout()
    per_repo_path = output_dir / "per_repo_distribution.png"
    plt.savefig(per_repo_path, dpi=PLOT_STYLE["dpi"], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved per-repo distribution chart to {per_repo_path}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("Validating repo-grouped tangled commit dataset")
    print("=" * 50)

    pool, train_df, test_df, repo_split = load_inputs()
    sha_to_repo = dict(zip(pool["sha"].astype(str), pool["repo"]))

    train_rows = parse_rows(train_df)
    test_rows = parse_rows(test_df)
    # Normalize sha types to str for lookup consistency
    for rows in (train_rows, test_rows):
        for row in rows:
            row["shas"] = [str(s) for s in row["shas"]]

    # (a) Guarantees every tangled row is genuinely intra-repo: all its
    # constituent atomic SHAs resolve back to the single repo named in `repo`.
    ok_a_train = check_repo_consistency(train_rows, sha_to_repo, "train")
    ok_a_test = check_repo_consistency(test_rows, sha_to_repo, "test")
    check("(a) all rows' shas resolve to a single repo matching `repo` column", ok_a_train and ok_a_test)

    # (b) Guarantees no repo-identity leakage between train and test.
    train_repos_used = set(row["repo"] for row in train_rows)
    test_repos_used = set(row["repo"] for row in test_rows)
    split_train = set(repo_split["train"])
    split_test = set(repo_split["test"])
    ok_b = (
        len(split_train & split_test) == 0
        and len(train_repos_used & test_repos_used) == 0
        and train_repos_used.issubset(split_train)
        and test_repos_used.issubset(split_test)
    )
    check(
        "(b) train_repos ∩ test_repos = ∅",
        ok_b,
        f"train={len(split_train)} repos, test={len(split_test)} repos",
    )

    # (c) Guarantees exact 1/7 uniform label frequency in each split (for
    # Hamming-loss fairness across types) and that train/test share the same
    # label distribution.
    train_shares, train_counts = compute_type_shares(train_rows)
    test_shares, test_counts = compute_type_shares(test_rows)
    ok_c_train = check_uniform_shares(train_shares, "train")
    ok_c_test = check_uniform_shares(test_shares, "test")
    gap = max(abs(train_shares[t] - test_shares[t]) for t in TYPES)
    ok_c_gap = gap <= SHARE_DEVIATION_EPSILON
    check("(c) per-type share deviation from 1/7 is 0.00pp in both splits", ok_c_train and ok_c_test)
    check("(c) train<->test per-type share gap is 0.00pp", ok_c_gap, f"max gap={gap * 100:.4f}pp")

    # (d) Guarantees the generation quota was met exactly (280/70 per concern
    # count, concern_count matching len(types)) and that no tangled commit
    # duplicates another's exact atomic-commit set.
    ok_d_train = check_row_counts_and_types(train_rows, TRAIN_PER_COUNT, "train")
    ok_d_test = check_row_counts_and_types(test_rows, TEST_PER_COUNT, "test")
    ok_d_dup = check_no_duplicate_sha_sets(train_rows, test_rows)
    check("(d) exactly 280/70 rows per concern_count; len(types)==concern_count", ok_d_train and ok_d_test)
    check("(d) no duplicate SHA frozensets within or across splits", ok_d_dup)

    # (e) Guarantees every tangled commit's real combined diff (not just the
    # cheaper per-atomic token_count proxy used during generation) fits the
    # SLM context window.
    encoding = tiktoken.get_encoding(ENCODING_MODEL)
    ok_e_train = check_token_budget(train_rows, "train", encoding)
    ok_e_test = check_token_budget(test_rows, "test", encoding)
    check("(e) every row's combined diff <= 12288 tokens", ok_e_train and ok_e_test)

    # Reuse stats
    print("\nReuse stats:")
    for name, rows in (("train", train_rows), ("test", test_rows)):
        stats = reuse_stats(rows)
        print(f"  {name}: max={stats['max']}, mean={stats['mean']:.2f}, n_atomics_used={stats['n_atomics']}")

    # Per-repo row counts
    train_repo_counts = per_repo_row_counts(train_rows)
    test_repo_counts = per_repo_row_counts(test_rows)
    print("\nPer-repo row counts (train):")
    for repo, cnt in train_repo_counts.most_common():
        print(f"  {repo}: {cnt}")
    print("\nPer-repo row counts (test):")
    for repo, cnt in test_repo_counts.most_common():
        print(f"  {repo}: {cnt}")

    # Charts
    print()
    save_charts(train_counts, test_counts, train_repo_counts, test_repo_counts, OUTPUT_DIR)

    print("\n" + "=" * 50)
    if failures:
        print(f"VALIDATION FAILED: {len(failures)} check(s) failed: {failures}")
        sys.exit(1)
    print("ALL VALIDATION CHECKS PASSED")


if __name__ == "__main__":
    main()
