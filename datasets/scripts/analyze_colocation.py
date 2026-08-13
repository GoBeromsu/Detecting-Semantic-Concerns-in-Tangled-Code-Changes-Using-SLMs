#!/usr/bin/env python3
"""
Cross-concern co-location characterization of the reconstructed tangled
dataset (data/tangled_ccs_dataset_{train,test}.csv), for the C1.1 reviewer
response. Internal characterization of our own generated dataset only: no
externally-labelled tangled reference set is used, because the only such
set available was the first author's manual multi-concern flag, whose
unreliability is the stated reason for removing FilterByAtomicity (C2.5).

For every synthetic tangled row with concern_count k>=2, this reconstructs
each constituent atomic commit's diff (via `shas` -> data/CCS Dataset.csv)
and measures, over all C(k,2) concern pairs within the row, how often the
two concerns land in the same place at two granularities:

  FILES:    share a common file.
  FOLDERS:  share a common directory (full dirname, repository root
            included -- two root-level files count as sharing a folder).

Both a per-row rate (does >=1 pair in the row co-locate) and a per-pair
rate (what fraction of all pairs co-locate) are reported, split by
train/test and by concern_count k=2..5, plus a pooled `overall` row per
split and one combined all-splits pooled row over the 1,400 multi-concern
rows (train 1,120 + test 280).

Interpretation notes (why the numbers look the way they do):

  * Row-level rates rise with k mostly by opportunity count, not by pairs
    getting closer: a k-concern row holds C(k,2) pairs, so the per-pair
    folder rate stays roughly flat across k (train ~3-7%) while the
    "any pair co-locates" row rate climbs from 3.2% (k=2, one pair) to
    39.6% (k=5, ten pairs).
  * Per-cell test-split rates are small-sample, repo-composition effects
    (70 rows per k drawn from 15 repositories), which is how test k=5
    reaches 92.9% same-folder. Quote the pooled all/overall row (1,400
    rows: 7.1% same-file, 28.6% same-folder) as the headline, not
    individual cells.
  * Same-file is NOT monotonic in k (train: 10.7% at k=4 -> 8.6% at k=5):
    larger k spreads a commit across more atomics without proportionally
    adding shared-file edits per pair. Phrase any rises-with-k claim
    against the folder rate, not the file rate.
  * Folder identity is full-dirname equality with the repository root
    included, so these rates are not comparable to the older published
    16.7% "non-root directory" figure.

Read-only w.r.t. datasets/data/: never writes there, and the pool/HF are
never touched.
Run with: uv run python datasets/scripts/analyze_colocation.py --output-dir DIR
Default outputs under results/.../repo_dataset_validation are refused when they
already exist unless --force-default-output is set.
"""

from structural_validity.colocation_cli import (
    CCS_DATASET_PATH,
    COLOCATION_CSV_PATH,
    COLOCATION_SUMMARY_PATH,
    DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    OUTPUT_DIR,
    PROJECT_ROOT,
    TEST_CSV_PATH,
    TRAIN_CSV_PATH,
    CliOptions,
    ExistingOutputError,
    main,
    parse_cli_args,
    resolve_output_paths,
    run_analysis,
)
from structural_validity.colocation_data import (
    CONCERN_COUNTS,
    PairRow,
    SummaryRow,
    load_pairs_by_split_k,
    summarize_by_split_k,
)
from structural_validity.colocation_report import (
    build_factual_summary,
    build_summary_markdown,
    format_markdown_table,
    write_summary_csv,
)
from structural_validity.diff_metrics import (
    dirname_of,
    file_path_from_block,
    pair_colocation,
    parse_atomic_diff,
    parse_committed_diff,
    split_diff_into_files,
)

__all__ = (
    "CCS_DATASET_PATH",
    "COLOCATION_CSV_PATH",
    "COLOCATION_SUMMARY_PATH",
    "CONCERN_COUNTS",
    "CliOptions",
    "DATA_DIR",
    "DEFAULT_OUTPUT_DIR",
    "ExistingOutputError",
    "OUTPUT_DIR",
    "PROJECT_ROOT",
    "PairRow",
    "SummaryRow",
    "TEST_CSV_PATH",
    "TRAIN_CSV_PATH",
    "build_factual_summary",
    "build_summary_markdown",
    "dirname_of",
    "file_path_from_block",
    "format_markdown_table",
    "load_pairs_by_split_k",
    "main",
    "pair_colocation",
    "parse_atomic_diff",
    "parse_cli_args",
    "parse_committed_diff",
    "resolve_output_paths",
    "run_analysis",
    "split_diff_into_files",
    "summarize_by_split_k",
    "write_summary_csv",
)


if __name__ == "__main__":
    raise SystemExit(main())
