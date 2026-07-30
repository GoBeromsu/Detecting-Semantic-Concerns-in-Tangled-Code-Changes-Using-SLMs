#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pandas",
#     "tiktoken",
# ]
# ///

# ─── How to run ───
# 1. Install uv (if not installed):
#      curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. From the repository root:
#      uv run python datasets/scripts/audit_combination_coverage.py --out .omo/evidence/dataset/combination_coverage.md
# ──────────────────

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations
from math import comb
from pathlib import Path
from statistics import median
from typing import Final

import pandas as pd  # noqa: PANDAS_OK - task requires pandas for CSV analysis.
import tiktoken


REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[2]
TOKEN_BUDGET: Final = 12_288
SPLITS: Final = ("train", "test")


@dataclass(frozen=True, slots=True)
class Paths:
    train_csv: Path
    test_csv: Path
    pool_csv: Path
    split_json: Path
    output: Path


@dataclass(frozen=True, slots=True)
class MissingCombination:
    types: tuple[str, ...]
    nominal_repos: tuple[str, ...]
    feasible_repos: tuple[str, ...]
    shortest_joined_tokens: int | None

    @property
    def constraint(self) -> str:
        if not self.nominal_repos:
            return "type supply: no single test repo contains every type"
        if not self.feasible_repos:
            return f"token budget: no supporting repo has a <= {TOKEN_BUDGET}-token witness"
        return "none: at least one test repo has a <= 12288-token witness"

    @property
    def verdict(self) -> str:
        if self.feasible_repos:
            return "sampler bug"
        return "sampling artifact"


def parse_paths() -> Paths:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument(
        "--train-csv",
        type=Path,
        default=REPOSITORY_ROOT / "datasets/data/tangled_ccs_dataset_train.csv",
    )
    _ = parser.add_argument(
        "--test-csv",
        type=Path,
        default=REPOSITORY_ROOT / "datasets/data/tangled_ccs_dataset_test.csv",
    )
    _ = parser.add_argument(
        "--pool-csv",
        type=Path,
        default=REPOSITORY_ROOT / "datasets/data/repo_grouped_pool.csv",
    )
    _ = parser.add_argument(
        "--split-json",
        type=Path,
        default=REPOSITORY_ROOT / "datasets/data/repo_split.json",
    )
    _ = parser.add_argument(
        "--out",
        type=Path,
        default=REPOSITORY_ROOT / ".omo/evidence/dataset/combination_coverage.md",
    )
    arguments = parser.parse_args()
    return Paths(
        train_csv=Path(str(arguments.train_csv)),
        test_csv=Path(str(arguments.test_csv)),
        pool_csv=Path(str(arguments.pool_csv)),
        split_json=Path(str(arguments.split_json)),
        output=Path(str(arguments.out)),
    )


def json_list(value: str) -> list[str]:
    decoded = json.loads(value)
    if not isinstance(decoded, list) or not all(isinstance(item, str) for item in decoded):
        msg = f"Expected a JSON list of strings, received {value!r}"
        raise TypeError(msg)
    return [str(item) for item in decoded]


def split_repositories(path: Path) -> dict[str, set[str]]:
    with path.open(encoding="utf-8") as source:
        decoded = json.load(source)
    repositories: dict[str, set[str]] = {}
    for split in SPLITS:
        values = decoded[split]
        if not isinstance(values, list) or not all(isinstance(repo, str) for repo in values):
            msg = f"repo_split.json field {split!r} must be a list of repository names"
            raise TypeError(msg)
        repositories[split] = {str(repo) for repo in values}
    return repositories


def coverage(rows: pd.DataFrame) -> dict[int, set[tuple[str, ...]]]:
    result: dict[int, set[tuple[str, ...]]] = {count: set() for count in range(1, 6)}
    for types_value, concern_count in zip(
        rows["types"], rows["concern_count"], strict=True
    ):
        result[int(concern_count)].add(tuple(sorted(json_list(str(types_value)))))
    return result


def joined_tokens(rows: pd.DataFrame, encoder: tiktoken.Encoding) -> list[int]:
    return [len(encoder.encode("".join(json_list(str(diff_value))))) for diff_value in rows["diff"]]


def missing_combinations(
    test_coverage: dict[int, set[tuple[str, ...]]],
    types: Sequence[str],
    pool: pd.DataFrame,
    test_repos: set[str],
    encoder: tiktoken.Encoding,
) -> list[MissingCombination]:
    test_pool = pool[pool["repo"].isin(sorted(test_repos))]
    evidence: list[MissingCombination] = []
    for count in range(1, 6):
        for type_combo in sorted(set(combinations(types, count)) - test_coverage[count]):
            nominal: list[str] = []
            feasible: list[str] = []
            shortest_witness: int | None = None
            for repo, repo_rows in test_pool.groupby("repo", sort=True):
                per_type = {
                    concern: repo_rows.loc[
                        repo_rows["annotated_type"] == concern, "git_diff"
                    ].tolist()
                    for concern in type_combo
                }
                if not all(per_type.values()):
                    continue
                nominal.append(str(repo))
                witness = "".join(
                    min(diffs, key=lambda diff: len(encoder.encode(str(diff))))
                    for diffs in per_type.values()
                )
                token_count = len(encoder.encode(witness))
                if shortest_witness is None or token_count < shortest_witness:
                    shortest_witness = token_count
                if token_count <= TOKEN_BUDGET:
                    feasible.append(str(repo))
            evidence.append(
                MissingCombination(
                    types=type_combo,
                    nominal_repos=tuple(nominal),
                    feasible_repos=tuple(feasible),
                    shortest_joined_tokens=shortest_witness,
                )
            )
    return evidence


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    separator = ["---" for _ in headers]
    return [
        f"| {' | '.join(headers)} |",
        f"| {' | '.join(separator)} |",
        *(f"| {' | '.join(row)} |" for row in rows),
    ]


def format_coverage(split: str, combinations_by_count: dict[int, set[tuple[str, ...]]], n_types: int) -> list[str]:
    return [f"### {split.title()}", *markdown_table(("k", "covered / C(7,k)"), [(str(count), f"{len(combinations_by_count[count])}/{comb(n_types, count)}") for count in range(1, 6)]), ""]


def reuse_rows(rows: pd.DataFrame) -> tuple[Counter[str], list[tuple[str, int]]]:
    reuse = Counter(sha for shas_value in rows["shas"] for sha in json_list(str(shas_value)))
    top_ten = sorted(reuse.items(), key=lambda item: (-item[1], item[0]))[:10]
    return reuse, top_ten


def percentile_rows(rows: pd.DataFrame, encoder: tiktoken.Encoding) -> list[tuple[str, str]]:
    values = pd.Series(joined_tokens(rows, encoder), dtype="float64")
    return [(label, f"{values.quantile(quantile):.2f}") for label, quantile in (("P50", 0.50), ("P90", 0.90), ("P95", 0.95), ("P99", 0.99))] + [("max", str(int(values.max())))]


def report(paths: Paths) -> str:
    train: pd.DataFrame = pd.read_csv(paths.train_csv)
    test: pd.DataFrame = pd.read_csv(paths.test_csv)
    pool: pd.DataFrame = pd.read_csv(paths.pool_csv)
    repositories = split_repositories(paths.split_json)
    types = tuple(sorted({str(value) for value in pool["annotated_type"].unique()}))
    encoder = tiktoken.get_encoding("cl100k_base")
    train_coverage = coverage(train)
    test_coverage = coverage(test)
    missing = missing_combinations(test_coverage, types, pool, repositories["test"], encoder)

    lines = [
        "# Tangled Dataset Combination-Coverage Audit",
        "",
        "This report is generated read-only from the committed pool, split, and tangled CSVs.",
        "Joined-token metrics re-encode `''.join(json.loads(diff))` with `cl100k_base`.",
        "",
        "## C(7,k) combination coverage",
        "",
        *format_coverage("train", train_coverage, len(types)),
        *format_coverage("test", test_coverage, len(types)),
        "## Missing test-split combinations: reachability and cause attribution",
        "",
        "A combo is a **sampling artifact** only when no single test repo can supply a <=12288-token witness. A combo with a feasible single-repo witness but no generated row is a **sampler bug**.",
        "",
        *markdown_table(
            ("Combination", "repos with all types", "feasible repos", "witness joined tokens", "binding constraint", "verdict"),
            [
                (
                    f"({', '.join(item.types)})",
                    str(len(item.nominal_repos)),
                    str(len(item.feasible_repos)),
                    "n/a" if item.shortest_joined_tokens is None else str(item.shortest_joined_tokens),
                    item.constraint,
                    item.verdict,
                )
                for item in missing
            ],
        ),
        "",
        "**Cause verdict:** " + f"{sum(item.verdict == 'sampling artifact' for item in missing)} sampling artifact; {sum(item.verdict == 'sampler bug' for item in missing)} sampler bug. Each row above supplies the per-combo single-repo and token-budget evidence.",
        "",
        "## Atomic-commit reuse",
        "",
    ]
    for split, rows in (("train", train), ("test", test)):
        reuse, top_ten = reuse_rows(rows)
        values = list(reuse.values())
        lines.extend(
            [
                f"### {split.title()}",
                f"Used atomics: {len(values)}; mean: {sum(values) / len(values):.2f}; "
                f"median: {median(values):.2f}; max: {max(values)}.",
                *markdown_table(("SHA", "reuse count"), [(sha, str(count)) for sha, count in top_ten]),
                "",
            ]
        )
    lines.extend(["## Per-type pool supply versus realized demand", ""])
    for split, rows in (("train", train), ("test", test)):
        split_pool = pool[pool["repo"].isin(sorted(repositories[split]))]
        realized = Counter(
            concern for types_value in rows["types"] for concern in json_list(str(types_value))
        )
        lines.extend(
            [
                f"### {split.title()}",
                *markdown_table(
                    ("type", "pool atomics", "realized labels"),
                    [
                        (
                            concern,
                            str((split_pool["annotated_type"] == concern).sum()),
                            str(realized[concern]),
                        )
                        for concern in types
                    ],
                ),
                "",
            ]
        )
    lines.extend(["## Joined-token distribution (cl100k_base)", ""])
    for split, rows in (("train", train), ("test", test)):
        lines.extend([f"### {split.title()}", *markdown_table(("statistic", "tokens"), percentile_rows(rows, encoder)), ""])
    return "\n".join(lines)


def main() -> None:
    paths = parse_paths()
    _ = paths.output.parent.mkdir(parents=True, exist_ok=True)
    _ = paths.output.write_text(report(paths), encoding="utf-8")
    print(f"Wrote {paths.output}")


if __name__ == "__main__":
    main()
