"""Reading the seed-43 corpus: changed paths, sampling cells, generated commits.

Split from `proximity_study` so the analysis file stays under the plan's
250-line gate, and along a real seam: everything here is I/O against
`datasets/data/`, everything there is measurement. Nothing in this module
writes.
"""

from __future__ import annotations

import ast
import collections
import csv
import sys
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from pydantic import TypeAdapter

from .contract import SyntheticCommit
from .contract_types import CommitId, Split
from .diff_metrics import parse_committed_diff

PathsBySha = Mapping[str, tuple[str, ...]]
SamplingCells = Mapping[tuple[str, str], tuple[str, ...]]
_STRING_LIST: Final[TypeAdapter[list[str]]] = TypeAdapter(list[str])

# Committed diffs run to megabytes; the stdlib default truncates them mid-field.
_ = csv.field_size_limit(sys.maxsize)


@dataclass(frozen=True, slots=True)
class ObservedCommit:
    """One generated tangled commit, with the constraints that produced it.

    `types` is the CCS type combination the generator drew against. The nulls
    need it to hold that constraint fixed while varying only the draw.
    """

    commit: SyntheticCommit
    types: tuple[str, ...]

    @property
    def is_resamplable(self) -> bool:
        """Whether the type list still lines up with the constituent count."""
        return len(self.types) == self.commit.concern_count


@dataclass(frozen=True, slots=True)
class Corpus:
    paths: PathsBySha
    cells: SamplingCells
    commits: tuple[ObservedCommit, ...]


def _rows(path: Path) -> Iterator[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            yield {key: value or "" for key, value in row.items()}


def _string_list(raw: str) -> tuple[str, ...]:
    """Parse a list column, which is JSON in some rows and a Python repr in others."""
    try:
        return tuple(_STRING_LIST.validate_json(raw))
    except ValueError:
        return tuple(_STRING_LIST.validate_python(ast.literal_eval(raw)))


def load_corpus(data: Path) -> Corpus:
    paths: dict[str, tuple[str, ...]] = {}
    cells: dict[tuple[str, str], list[str]] = collections.defaultdict(list)

    def remember(sha: str, diff: str) -> None:
        if sha not in paths:
            paths[sha] = tuple(sorted(parse_committed_diff(diff).by_headline_path()))

    for row in _rows(data / "repo_grouped_pool.csv"):
        cells[(row["repo"], row["annotated_type"])].append(row["sha"])
        remember(row["sha"], row["git_diff"])
    for row in _rows(data / "CCS Dataset.csv"):
        remember(row["sha"], row["git_diff"])

    commits: list[ObservedCommit] = []
    for split in (Split.TRAIN, Split.TEST):
        source = data / f"tangled_ccs_dataset_{split.value}.csv"
        for index, row in enumerate(_rows(source)):
            concern_count = int(row["concern_count"])
            if concern_count < 2:
                continue
            shas = _string_list(row["shas"])
            commit = SyntheticCommit(CommitId(f"{split.value}:{index}"), split, row["repo"], concern_count, shas)
            commits.append(ObservedCommit(commit, _string_list(row["types"])))
    return Corpus(paths, {key: tuple(value) for key, value in cells.items()}, tuple(commits))
