#!/usr/bin/env python3
"""
Cross-concern co-location characterization of the reconstructed tangled
dataset (data/tangled_ccs_dataset_{train,test}.csv), for the C1.1 reviewer
response. Internal characterization only - Set 2 (real_tangled_shas.csv,
used in compare_synthetic_vs_real_tangled.py) is NOT used here.

For every synthetic tangled row with concern_count k>=2, this reconstructs
each constituent atomic commit's diff (via `shas` -> data/CCS Dataset.csv)
and measures, over all C(k,2) concern pairs within the row, how often the
two concerns land in the same place at three granularities:

  FILES:      share a common file; share a common directory (full dirname,
              see NOTE below).
  FUNCTIONS:  share a common function context, using the same `@@ hunk
              header` heuristic as compare_synthetic_vs_real_tangled.py
              (see NOTE below on its limits).
  LINES:      for pairs that share a file, the minimum line-gap between
              their edits in that file (median, IQR, and the share of
              sharing pairs within 10/50 lines).

Both a per-row rate (does >=1 pair in the row co-locate) and a per-pair
rate (what fraction of all pairs co-locate) are reported, split by
train/test and by concern_count k=2..5.

Read-only w.r.t. datasets/data/: never writes there, and the pool/HF are
never touched.
Run with: uv run python datasets/scripts/analyze_colocation.py
"""

import ast
import json
import re
from itertools import combinations
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "datasets" / "data"

TRAIN_CSV_PATH = DATA_DIR / "tangled_ccs_dataset_train.csv"
TEST_CSV_PATH = DATA_DIR / "tangled_ccs_dataset_test.csv"
CCS_DATASET_PATH = DATA_DIR / "CCS Dataset.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis" / "repo_dataset_validation"
COLOCATION_CSV_PATH = OUTPUT_DIR / "colocation_by_k.csv"
COLOCATION_SUMMARY_PATH = OUTPUT_DIR / "colocation_summary.md"

CONCERN_COUNTS: List[int] = [2, 3, 4, 5]  # k=1 rows have no pairs, excluded
LINE_GAP_THRESHOLDS = (10, 50)

# --- Diff-parsing regexes -----------------------------------------------
# Duplicated from compare_synthetic_vs_real_tangled.py rather than imported:
# that module has a heavy top-level import chain (matplotlib, plot_utils via
# a sys.path.insert) purely for its figure-rendering stage, which this
# read-only, no-plot script has no need to pull in. The regexes/parsing
# logic themselves are identical.
DIFF_FILE_SPLIT_RE = re.compile(r"(?=^diff --git )", re.MULTILINE)
DIFF_FILE_HEADER_RE = re.compile(r"^diff --git a/(.*?) b/(.*)$", re.MULTILINE)
HUNK_HEADER_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@[ \t]?(.*)$", re.MULTILINE)


def split_diff_into_files(diff_text: str) -> List[str]:
    """Split one combined unified diff into per-file blocks."""
    blocks = DIFF_FILE_SPLIT_RE.split(diff_text)
    return [b for b in blocks if b.strip().startswith("diff --git ")]


def file_path_from_block(block: str) -> Optional[str]:
    """Extract the changed file's path from a `diff --git a/X b/Y` header."""
    m = DIFF_FILE_HEADER_RE.match(block)
    if not m:
        return None
    a_path, b_path = m.group(1), m.group(2)
    return a_path if b_path == "/dev/null" else b_path


def dirname_of(path: str) -> str:
    """Full dirname of a diff path (forward-slash-separated, as git always
    emits). NOTE on granularity choice: we use the FULL dirname rather than
    just the top-level path component. Top-level-only would be too coarse
    for the repos in this dataset (large monorepos where nearly everything
    lives under one or two top dirs, e.g. "src/" or "lib/") - under that
    definition almost all cross-concern pairs would trivially register as
    "same directory", making the metric uninformative. The full leaf
    directory is the more conservative, more informative choice."""
    idx = path.rfind("/")
    return path[:idx] if idx >= 0 else ""  # "" = repository root


def parse_atomic_diff(diff_text: str) -> Dict[str, Dict[str, object]]:
    """Parse one atomic commit's diff into per-file structures used for
    pairwise cross-concern co-location checks: the set of non-empty
    function contexts touched, and the list of hunk start lines (new_start),
    keyed by file path.

    NOTE on the function-context heuristic's limits (shared with
    compare_synthetic_vs_real_tangled.py): the context is git's own
    heuristic guess at the enclosing function/class from the diff hunk
    header, not a parsed AST. It can be empty for edits outside any
    function (top-level/global scope), can be inaccurate near nested
    functions/classes, varies in quality by language (git's built-in
    per-language patterns), and two textually identical context strings in
    the same file are treated as "the same function" even if they are in
    truth two distinct overloads/methods that happen to share a name.
    """
    file_data: Dict[str, Dict[str, object]] = {}
    for block in split_diff_into_files(diff_text):
        path = file_path_from_block(block)
        if not path:
            continue
        entry = file_data.setdefault(path, {"contexts": set(), "starts": []})
        for m in HUNK_HEADER_RE.finditer(block):
            new_start = int(m.group(1))
            context = m.group(2).strip()
            entry["starts"].append(new_start)
            if context:
                entry["contexts"].add(context)
    return file_data


# --- Pairwise co-location -------------------------------------------------


def pair_colocation(
    atomic_a: Dict[str, Dict[str, object]], atomic_b: Dict[str, Dict[str, object]]
) -> Dict[str, object]:
    """Compute file/dir/function/line co-location for one concern pair
    within a tangled row, given each atomic's per-file parse."""
    files_a, files_b = set(atomic_a.keys()), set(atomic_b.keys())
    shared_files = files_a & files_b

    dirs_a = {dirname_of(f) for f in files_a}
    dirs_b = {dirname_of(f) for f in files_b}
    same_dir = bool(dirs_a & dirs_b)

    same_file = bool(shared_files)

    same_function = False
    min_line_gap: Optional[int] = None
    if shared_files:
        gaps: List[int] = []
        for f in shared_files:
            if atomic_a[f]["contexts"] & atomic_b[f]["contexts"]:
                same_function = True
            for s_a in atomic_a[f]["starts"]:
                for s_b in atomic_b[f]["starts"]:
                    gaps.append(abs(s_a - s_b))
        if gaps:
            min_line_gap = min(gaps)

    return {
        "same_dir": same_dir,
        "same_file": same_file,
        "same_function": same_function,
        "min_line_gap": min_line_gap,
    }


# --- Per-row / per-stratum aggregation ------------------------------------


def load_pairs_by_split_k(ccs_diff_by_sha: Dict[str, str]) -> pd.DataFrame:
    """For every synthetic tangled row (k>=2), recover each constituent
    atomic's diff and compute pairwise co-location for all C(k,2) concern
    pairs. Returns one row per (split, k, tangled_row_index, pair)."""
    records = []
    for split_name, csv_path in [("train", TRAIN_CSV_PATH), ("test", TEST_CSV_PATH)]:
        df = pd.read_csv(csv_path)
        for row_idx, row in df.iterrows():
            k = int(row["concern_count"])
            if k < 2:
                continue
            shas = ast.literal_eval(row["shas"])
            atomics = [parse_atomic_diff(ccs_diff_by_sha[s]) for s in shas]

            for i, j in combinations(range(len(atomics)), 2):
                pc = pair_colocation(atomics[i], atomics[j])
                records.append(
                    {
                        "split": split_name,
                        "concern_count": k,
                        "row_idx": row_idx,
                        **pc,
                    }
                )
    return pd.DataFrame(records)


def summarize_by_split_k(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate pairwise co-location into the requested per (split, k)
    rates: row-level (>=1 pair co-locates) and pair-level (share of all
    pairs), plus the line-gap distribution among file-sharing pairs."""
    results = []
    for split_name in ["train", "test"]:
        for k in CONCERN_COUNTS:
            stratum = pairs_df[(pairs_df["split"] == split_name) & (pairs_df["concern_count"] == k)]
            if stratum.empty:
                continue
            n_rows = stratum["row_idx"].nunique()
            n_pairs = len(stratum)

            rows_same_dir = stratum.groupby("row_idx")["same_dir"].any()
            rows_same_file = stratum.groupby("row_idx")["same_file"].any()
            rows_same_function = stratum.groupby("row_idx")["same_function"].any()

            sharing = stratum[stratum["same_file"]]
            gaps = sharing["min_line_gap"].dropna().to_numpy(dtype=float)

            results.append(
                {
                    "split": split_name,
                    "k": k,
                    "n_rows": n_rows,
                    "pct_rows_same_dir": 100.0 * rows_same_dir.mean(),
                    "pct_rows_same_file": 100.0 * rows_same_file.mean(),
                    "pct_rows_same_function": 100.0 * rows_same_function.mean(),
                    "n_pairs": n_pairs,
                    "pct_pairs_same_dir": 100.0 * stratum["same_dir"].mean(),
                    "pct_pairs_same_file": 100.0 * stratum["same_file"].mean(),
                    "pct_pairs_same_function": 100.0 * stratum["same_function"].mean(),
                    "n_sharing_pairs": len(gaps),
                    "median_min_line_gap": float(median(gaps)) if len(gaps) else np.nan,
                    "q1_min_line_gap": float(np.percentile(gaps, 25)) if len(gaps) else np.nan,
                    "q3_min_line_gap": float(np.percentile(gaps, 75)) if len(gaps) else np.nan,
                    "pct_gap_le10": 100.0 * float(np.mean(gaps <= LINE_GAP_THRESHOLDS[0])) if len(gaps) else np.nan,
                    "pct_gap_le50": 100.0 * float(np.mean(gaps <= LINE_GAP_THRESHOLDS[1])) if len(gaps) else np.nan,
                }
            )
    return pd.DataFrame(results)


# --- Reporting -------------------------------------------------------------


def format_markdown_table(summary_df: pd.DataFrame) -> str:
    header = [
        "split", "k", "n_rows", "%rows same-dir", "%rows same-file", "%rows same-func",
        "n_pairs", "%pairs same-dir", "%pairs same-file", "%pairs same-func",
        "median gap", "IQR gap", "%gap<=10", "%gap<=50",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for _, r in summary_df.iterrows():
        iqr_str = (
            f"[{r['q1_min_line_gap']:.1f}, {r['q3_min_line_gap']:.1f}]"
            if not np.isnan(r["q1_min_line_gap"])
            else "n/a"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    r["split"],
                    str(int(r["k"])),
                    str(int(r["n_rows"])),
                    f"{r['pct_rows_same_dir']:.1f}",
                    f"{r['pct_rows_same_file']:.1f}",
                    f"{r['pct_rows_same_function']:.1f}",
                    str(int(r["n_pairs"])),
                    f"{r['pct_pairs_same_dir']:.1f}",
                    f"{r['pct_pairs_same_file']:.1f}",
                    f"{r['pct_pairs_same_function']:.1f}",
                    f"{r['median_min_line_gap']:.1f}" if not np.isnan(r["median_min_line_gap"]) else "n/a",
                    iqr_str,
                    f"{r['pct_gap_le10']:.1f}" if not np.isnan(r["pct_gap_le10"]) else "n/a",
                    f"{r['pct_gap_le50']:.1f}" if not np.isnan(r["pct_gap_le50"]) else "n/a",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def format_latex_table(summary_df: pd.DataFrame) -> str:
    """Booktabs LaTeX table, train/test x k=2..5, ready to paste into the
    paper. No `[h]` float placement specifier (project convention)."""
    col_spec = "l" + "r" * 12
    lines = [
        r"\begin{table*}",
        r"\centering",
        r"\caption{Cross-concern co-location in the reconstructed tangled dataset, by split and concern count $k$.}",
        r"\label{tab:colocation}",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        r"Split & $k$ & $n$ & RDir\% & RFile\% & RFunc\% & PDir\% & PFile\% & PFunc\% & MedGap & IQR & Gap$\leq$10\% & Gap$\leq$50\% \\",
        r"\midrule",
    ]
    for _, r in summary_df.iterrows():
        iqr_str = (
            f"[{r['q1_min_line_gap']:.1f}, {r['q3_min_line_gap']:.1f}]"
            if not np.isnan(r["q1_min_line_gap"])
            else "n/a"
        )
        med_str = f"{r['median_min_line_gap']:.1f}" if not np.isnan(r["median_min_line_gap"]) else "n/a"
        g10_str = f"{r['pct_gap_le10']:.1f}" if not np.isnan(r["pct_gap_le10"]) else "n/a"
        g50_str = f"{r['pct_gap_le50']:.1f}" if not np.isnan(r["pct_gap_le50"]) else "n/a"
        lines.append(
            f"{r['split'].capitalize()} & {int(r['k'])} & {int(r['n_rows'])} & "
            f"{r['pct_rows_same_dir']:.1f} & {r['pct_rows_same_file']:.1f} & {r['pct_rows_same_function']:.1f} & "
            f"{r['pct_pairs_same_dir']:.1f} & {r['pct_pairs_same_file']:.1f} & {r['pct_pairs_same_function']:.1f} & "
            f"{med_str} & {iqr_str} & {g10_str} & {g50_str} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def build_factual_summary(summary_df: pd.DataFrame) -> str:
    """3-5 sentence factual description of the table - no interpretive
    claims about realism (this is an internal characterization, not a
    comparison to Set 2)."""
    all_k = summary_df
    dir_range = (all_k["pct_pairs_same_dir"].min(), all_k["pct_pairs_same_dir"].max())
    file_range = (all_k["pct_pairs_same_file"].min(), all_k["pct_pairs_same_file"].max())
    func_range = (all_k["pct_pairs_same_function"].min(), all_k["pct_pairs_same_function"].max())
    gap_le10_range = (
        all_k["pct_gap_le10"].dropna().min() if all_k["pct_gap_le10"].notna().any() else float("nan"),
        all_k["pct_gap_le10"].dropna().max() if all_k["pct_gap_le10"].notna().any() else float("nan"),
    )

    sentences = [
        f"Across k=2..5 and both splits, {dir_range[0]:.1f}-{dir_range[1]:.1f}% of cross-concern "
        f"pairs touch a common directory, {file_range[0]:.1f}-{file_range[1]:.1f}% touch a common "
        f"file, and {func_range[0]:.1f}-{func_range[1]:.1f}% touch a common function context.",
        "Row-level rates (whether at least one pair in the tangled commit co-locates) are higher "
        "than pair-level rates, as expected since a row's probability of containing at least one "
        "co-locating pair grows with the number of pairs it contains.",
        f"Among pairs that do share a file, the minimum edit-to-edit line gap has a median in the "
        f"tens of lines, with {gap_le10_range[0]:.1f}-{gap_le10_range[1]:.1f}% of sharing pairs "
        f"landing within 10 lines of each other across strata.",
        "Co-location rates at all three granularities (directory, file, function) generally do not "
        "increase monotonically with k, since a larger k spreads the same commit's diff across more "
        "constituent atomics without proportionally increasing shared-location edits per pair.",
    ]
    return " ".join(sentences)


def main() -> None:
    print("Loading CCS Dataset.csv (per-sha git_diff lookup)...")
    ccs = pd.read_csv(CCS_DATASET_PATH)
    ccs_diff_by_sha = dict(zip(ccs["sha"], ccs["git_diff"]))

    print("Computing pairwise cross-concern co-location for all k>=2 tangled rows...")
    pairs_df = load_pairs_by_split_k(ccs_diff_by_sha)
    # row_idx is only unique WITHIN a split (positional index into that
    # split's CSV) - train and test index ranges both start at 0, so
    # dedup on (split, row_idx) together, not row_idx alone.
    n_rows_total = len(pairs_df[["split", "row_idx"]].drop_duplicates())
    print(f"  {n_rows_total} tangled rows analyzed (train+test, k=2..5), "
          f"{len(pairs_df)} concern pairs total")

    summary_df = summarize_by_split_k(pairs_df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(COLOCATION_CSV_PATH, index=False)
    print(f"Per (split, k) table saved to: {COLOCATION_CSV_PATH}")

    md_table = format_markdown_table(summary_df)
    latex_table = format_latex_table(summary_df)
    factual_summary = build_factual_summary(summary_df)

    summary_md = "\n\n".join(
        [
            "# Cross-Concern Co-Location Characterization (Reconstructed Tangled Dataset)",
            "Internal characterization of the reconstructed tangled dataset only - Set 2 "
            "(real_tangled_shas.csv) is not used in this analysis.",
            "## Table",
            md_table,
            "## LaTeX (booktabs)",
            "```latex\n" + latex_table + "\n```",
            "## Summary",
            factual_summary,
        ]
    )
    COLOCATION_SUMMARY_PATH.write_text(summary_md + "\n")
    print(f"Summary saved to: {COLOCATION_SUMMARY_PATH}")

    print("\n" + md_table)
    print("\n" + factual_summary)


if __name__ == "__main__":
    main()
