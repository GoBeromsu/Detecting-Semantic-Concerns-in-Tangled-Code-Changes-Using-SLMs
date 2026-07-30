"""Executable specification for the committed tangled-commit dataset.

Every expected value here was measured against the CSVs committed at the
dataset-reconstruction commit. They are regression guards: a future
regeneration that changes any of them must fail loudly rather than silently
shifting the distribution the paper reports.

Row counts are derived with pandas only. Line counting (``wc -l``) is invalid
for these files because the ``diff`` column embeds newlines.
"""

import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import Final

import pytest

from __test__.conftest import (
    CANONICAL_TYPES,
    MAX_DIFF_TOKENS,
    TIKTOKEN_ENCODING,
    TangledSplit,
)

CONCERN_COUNTS: Final[tuple[int, ...]] = (1, 2, 3, 4, 5)
SPLIT_NAMES: Final[tuple[str, ...]] = ("train", "test")


@dataclass(frozen=True, slots=True)
class SplitExpectation:
    rows: int
    per_type: int
    per_k: int
    per_k_type_factor: int
    coverage: Mapping[int, tuple[int, int]]


EXPECTED: Final[Mapping[str, SplitExpectation]] = {
    "train": SplitExpectation(
        rows=1400,
        per_type=600,
        per_k=280,
        per_k_type_factor=40,
        coverage={1: (7, 7), 2: (21, 21), 3: (35, 35), 4: (35, 35), 5: (21, 21)},
    ),
    "test": SplitExpectation(
        rows=350,
        per_type=150,
        per_k=70,
        per_k_type_factor=10,
        coverage={1: (7, 7), 2: (21, 21), 3: (29, 35), 4: (33, 35), 5: (20, 21)},
    ),
}

# The test split's accepted coverage gap, enumerated so the failure message can
# name exactly which combinations drifted. Documented as a paper limitation.
MISSING_TEST_COMBINATIONS: Final[frozenset[frozenset[str]]] = frozenset(
    frozenset(combo)
    for combo in (
        ("build", "ci", "test"),
        ("build", "feat", "fix"),
        ("build", "feat", "test"),
        ("ci", "docs", "test"),
        ("docs", "fix", "refactor"),
        ("feat", "refactor", "test"),
        ("build", "feat", "fix", "refactor"),
        ("build", "fix", "refactor", "test"),
        ("build", "ci", "feat", "fix", "refactor"),
    )
)


def _render(combos: frozenset[frozenset[str]]) -> str:
    return ", ".join(sorted(str(tuple(sorted(c))) for c in combos)) or "(none)"


def test_canonical_type_vocabulary_is_seven_types() -> None:
    assert CANONICAL_TYPES == (
        "feat",
        "fix",
        "refactor",
        "test",
        "docs",
        "build",
        "ci",
    )


@pytest.mark.parametrize("split_name", SPLIT_NAMES)
def test_split_row_count(
    split_name: str, tangled_splits: Mapping[str, TangledSplit]
) -> None:
    """Given the committed CSV, When counted with pandas, Then rows match spec."""
    split = tangled_splits[split_name]
    assert len(split.frame) == EXPECTED[split_name].rows


@pytest.mark.parametrize("split_name", SPLIT_NAMES)
def test_marginal_type_totals_are_uniform(
    split_name: str, tangled_splits: Mapping[str, TangledSplit]
) -> None:
    """Every canonical type appears an identical number of times per split."""
    split = tangled_splits[split_name]
    expected = EXPECTED[split_name].per_type
    counts = Counter(t for types in split.type_lists for t in types)

    assert set(counts) == set(CANONICAL_TYPES)
    assert counts == Counter({t: expected for t in CANONICAL_TYPES})


@pytest.mark.parametrize("split_name", SPLIT_NAMES)
def test_rows_per_concern_count_are_uniform(
    split_name: str, tangled_splits: Mapping[str, TangledSplit]
) -> None:
    split = tangled_splits[split_name]
    expected = EXPECTED[split_name].per_k
    counts = Counter(split.concern_counts)

    assert set(counts) == set(CONCERN_COUNTS)
    assert counts == Counter({k: expected for k in CONCERN_COUNTS})


@pytest.mark.parametrize("split_name", SPLIT_NAMES)
def test_per_concern_count_and_type_counts_scale_with_k(
    split_name: str, tangled_splits: Mapping[str, TangledSplit]
) -> None:
    """Each (k, type) cell holds factor*k rows, so type balance holds per k."""
    split = tangled_splits[split_name]
    factor = EXPECTED[split_name].per_k_type_factor
    counts = Counter(
        (count, t)
        for count, types in zip(split.concern_counts, split.type_lists)
        for t in types
    )
    expected = Counter(
        {(k, t): factor * k for k in CONCERN_COUNTS for t in CANONICAL_TYPES}
    )

    assert counts == expected


@pytest.mark.parametrize("split_name", SPLIT_NAMES)
def test_types_are_distinct_and_sized_by_concern_count(
    split_name: str, tangled_splits: Mapping[str, TangledSplit]
) -> None:
    """Row-level invariant: types has exactly concern_count DISTINCT entries."""
    split = tangled_splits[split_name]
    offenders = [
        (index, types, count)
        for index, (count, types) in enumerate(
            zip(split.concern_counts, split.type_lists)
        )
        if len(set(types)) != count or len(types) != count
    ]

    assert offenders == []


@pytest.mark.parametrize("split_name", SPLIT_NAMES)
def test_sha_sets_are_unique_within_split(
    split_name: str, tangled_splits: Mapping[str, TangledSplit]
) -> None:
    split = tangled_splits[split_name]
    sha_sets = [frozenset(shas) for shas in split.sha_lists]
    duplicates = [s for s, n in Counter(sha_sets).items() if n > 1]

    assert duplicates == [], f"{len(duplicates)} duplicated sha sets"


@pytest.mark.parametrize("split_name", SPLIT_NAMES)
def test_every_sha_resolves_to_its_rows_repo(
    split_name: str,
    tangled_splits: Mapping[str, TangledSplit],
    sha_to_repo: dict[str, str],
) -> None:
    """Tangling is repo-grouped: every atom must come from the row's own repo."""
    split = tangled_splits[split_name]
    unknown: list[str] = []
    mismatched: list[tuple[str, str, str]] = []

    for repo, shas in zip(split.repos, split.sha_lists):
        for sha in shas:
            resolved = sha_to_repo.get(sha)
            if resolved is None:
                unknown.append(sha)
            elif resolved != repo:
                mismatched.append((sha, resolved, repo))

    assert unknown == []
    assert mismatched == []


def test_repo_split_partitions_are_disjoint(
    repo_split: Mapping[str, list[str]],
    tangled_splits: Mapping[str, TangledSplit],
) -> None:
    train_repos = set(repo_split["train"])
    test_repos = set(repo_split["test"])

    assert len(train_repos) == 21
    assert len(test_repos) == 15
    assert train_repos & test_repos == set()
    assert set(tangled_splits["train"].repos) <= train_repos
    assert set(tangled_splits["test"].repos) <= test_repos


def test_no_sha_is_shared_between_splits(
    tangled_splits: Mapping[str, TangledSplit],
) -> None:
    shared = tangled_splits["train"].unique_shas() & tangled_splits["test"].unique_shas()

    assert shared == frozenset(), f"{len(shared)} shas leak across splits"


def test_train_split_covers_every_type_combination(
    train_split: TangledSplit,
) -> None:
    """Train reaches full C(7,k) coverage for every k."""
    measured = {
        k: (len(train_split.combinations_for_k(k)), comb(7, k)) for k in CONCERN_COUNTS
    }

    assert measured == dict(EXPECTED["train"].coverage)
    assert all(covered == total for covered, total in measured.values())


def test_test_split_combination_coverage_is_documented(
    test_split: TangledSplit,
) -> None:
    """The test split is knowingly INCOMPLETE at the joint level.

    Per-type totals are exactly uniform, but 9 of the 91 type combinations
    never occur. This gap is an accepted, documented paper limitation - this
    test asserts it stayed EXACTLY as measured. The nine missing combinations
    are listed in MISSING_TEST_COMBINATIONS and emitted on failure. A failure
    here means coverage DRIFTED (either direction), not that the dataset is
    broken.
    """
    measured = {
        k: (len(test_split.combinations_for_k(k)), comb(7, k)) for k in CONCERN_COUNTS
    }
    missing = frozenset(
        frozenset(combo)
        for k in CONCERN_COUNTS
        for combo in combinations(sorted(CANONICAL_TYPES), k)
        if frozenset(combo) not in test_split.combinations_for_k(k)
    )

    expected_coverage = dict(EXPECTED["test"].coverage)
    assert measured == expected_coverage, (
        f"test-split coverage drifted: measured={measured} "
        f"expected={expected_coverage}\n"
        f"  measured missing: {_render(missing)}\n"
        f"  documented missing: {_render(MISSING_TEST_COMBINATIONS)}"
    )
    assert missing == MISSING_TEST_COMBINATIONS, (
        "the documented coverage gap changed.\n"
        f"  measured missing: {_render(missing)}\n"
        f"  documented missing: {_render(MISSING_TEST_COMBINATIONS)}\n"
        f"  newly missing: {_render(missing - MISSING_TEST_COMBINATIONS)}\n"
        f"  newly covered: {_render(MISSING_TEST_COMBINATIONS - missing)}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("split_name", SPLIT_NAMES)
def test_joined_diff_fits_token_budget(
    split_name: str, tangled_splits: Mapping[str, TangledSplit]
) -> None:
    """Re-encode each row's joined diff with real tiktoken cl100k_base.

    Mirrors validate_repo_dataset.check_token_budget exactly: diffs are
    concatenated with no separator and encoded with disallowed_special=().
    The generator's own budget check sums a per-atom proxy instead, so this
    is an independent confirmation rather than a restatement of it.
    """
    import tiktoken

    split = tangled_splits[split_name]
    encoding = tiktoken.get_encoding(TIKTOKEN_ENCODING)
    over_budget: list[tuple[int, int]] = []

    for index, raw in enumerate(split.frame["diff"]):
        joined = "".join(str(d) for d in json.loads(raw))
        n_tokens = len(encoding.encode(joined, disallowed_special=()))
        if n_tokens > MAX_DIFF_TOKENS:
            over_budget.append((index, n_tokens))

    assert over_budget == [], f"rows exceeding {MAX_DIFF_TOKENS} tokens: {over_budget}"
