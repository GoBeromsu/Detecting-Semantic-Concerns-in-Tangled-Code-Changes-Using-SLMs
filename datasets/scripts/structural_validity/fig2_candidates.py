"""Rank compact tangled-commit candidates for the manuscript's Figure 2."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from structural_validity.colocation_data import load_diff_by_sha, load_pairs_by_split_k
from structural_validity.diff_metrics import parse_committed_diff

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "datasets" / "data"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "analysis"
    / "repo_dataset_validation"
    / "fig2_candidates.csv"
)

# Upper bound on the serialized diff size a candidate may have and still fit the
# one-page Figure 2 float. Fixed before a run and never re-tuned mid-run: lowering
# it only yields a subset of the candidates that already failed.
MAX_DIFF_CHARS = 6000


@dataclass(frozen=True)
class Candidate:
    split: str
    row_idx: int
    concern_count: int
    types: tuple[str, ...]
    commit_message: str
    diff: str
    same_file: bool
    same_dir: bool
    min_line_gap: int | None
    files_touched: int

    def rank_key(self) -> tuple[int, int, bool, int, bool, int, str, int]:
        # Figure 2 is illustrative, not evidentiary: it shows what one tangled
        # commit in the dataset looks like. It does not argue that tangling is
        # frequent or that concatenation reproduces real entanglement, so the
        # ranking optimises for a short, readable example rather than for extreme
        # co-location. Ranking by shared files first would surface the rare
        # same-file tail (7.1% of multi-concern commits, and usually a lockfile
        # collision), which reads as cherry-picking. Co-location is kept as a
        # tiebreak so the chosen sample is still spatially coherent.
        k_priority = 0 if self.concern_count == 2 else 1 if self.concern_count == 3 else 2
        return (
            k_priority,
            len(self.diff),
            not self.same_dir,
            self.min_line_gap if self.min_line_gap is not None else sys.maxsize,
            not self.same_file,
            self.files_touched,
            self.split,
            self.row_idx,
        )


def load_candidates() -> list[Candidate]:
    ccs_diffs = load_diff_by_sha(DATA_DIR / "CCS Dataset.csv")
    pairs = load_pairs_by_split_k(
        ccs_diffs,
        {
            "train": DATA_DIR / "tangled_ccs_dataset_train.csv",
            "test": DATA_DIR / "tangled_ccs_dataset_test.csv",
        },
    )
    pair_groups = defaultdict(list)
    for pair in pairs:
        pair_groups[(pair.split, pair.row_idx)].append(pair)

    candidates: list[Candidate] = []
    for split in ("train", "test"):
        with (DATA_DIR / f"tangled_ccs_dataset_{split}.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            for row_idx, row in enumerate(csv.DictReader(handle)):
                concern_count = int(row["concern_count"])
                if concern_count < 2:
                    continue
                row_pairs = pair_groups[(split, row_idx)]
                # row["diff"] is a JSON array of per-atomic unified diffs, so it must
                # be decoded before parsing. Passing the raw JSON string to
                # parse_committed_diff yields no headline paths and reports 0 files.
                headline_paths: set[str] = set()
                for atomic_diff in json.loads(row["diff"]):
                    headline_paths.update(parse_committed_diff(atomic_diff).headline_paths())
                candidates.append(
                    Candidate(
                        split=split,
                        row_idx=row_idx,
                        concern_count=concern_count,
                        types=tuple(json.loads(row["types"])),
                        commit_message=row["commit_message"],
                        diff=row["diff"],
                        same_file=any(pair.same_file for pair in row_pairs),
                        same_dir=any(pair.same_dir for pair in row_pairs),
                        min_line_gap=min(
                            (
                                pair.min_line_gap
                                for pair in row_pairs
                                if pair.min_line_gap is not None
                            ),
                            default=None,
                        ),
                        files_touched=len(headline_paths),
                    )
                )
    return candidates


def main() -> int:
    eligible = [c for c in load_candidates() if len(c.diff) <= MAX_DIFF_CHARS]
    if not eligible:
        print(
            f"No candidate fits the {MAX_DIFF_CHARS}-character budget; "
            "stop and re-approve the budget instead of lowering it here."
        )
        return 1
    candidates = sorted(eligible, key=Candidate.rank_key)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "rank",
                "split",
                "row_idx",
                "concern_count",
                "types",
                "commit_message",
                "same_file",
                "same_dir",
                "min_line_gap",
                "files_touched",
                "diff_line_count",
                "diff_character_count",
                "diff",
            ),
        )
        writer.writeheader()
        for rank, candidate in enumerate(candidates, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "split": candidate.split,
                    "row_idx": candidate.row_idx,
                    "concern_count": candidate.concern_count,
                    "types": json.dumps(candidate.types),
                    "commit_message": candidate.commit_message,
                    "same_file": candidate.same_file,
                    "same_dir": candidate.same_dir,
                    "min_line_gap": candidate.min_line_gap,
                    "files_touched": candidate.files_touched,
                    "diff_line_count": len(candidate.diff.splitlines()),
                    "diff_character_count": len(candidate.diff),
                    "diff": candidate.diff,
                }
            )
    print(OUTPUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
