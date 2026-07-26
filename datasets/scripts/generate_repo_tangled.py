#!/usr/bin/env python3
"""
Stage 2 of 3 - repo-grouped tangled commit dataset pipeline.

    Stage 1 (build_repo_pool.py)     CCS Dataset.csv       -> repo_grouped_pool.csv
    Stage 2 (this script)            repo_grouped_pool.csv -> repo_split.json,
                                                               tangled_ccs_dataset_{train,test}.csv
    Stage 3 (validate_repo_dataset.py) validates the Stage 2 outputs.

Input:  data/repo_grouped_pool.csv (Stage 1 output)
Output: data/repo_split.json (repo-disjoint train/test partition + per-split
        type supply), data/tangled_ccs_dataset_train.csv, data/tangled_ccs_dataset_test.csv
        (existing files of the same name are backed up to data/legacy/ first)

Every tangled commit combines atomic commits drawn from a SINGLE repo only
(resolving the cross-repo-tangle critique that tangles were trivially
separable by repo-specific style). Train/test splits are repo-disjoint, and
per-type label frequency is driven to exact uniformity (1/7 each) in both
splits via adaptive randomization: each draw favors the currently most
under-represented types, picks a uniformly random supporting repo, and picks
atomics via reuse-weighted random choice. This is deliberately NOT a
deterministic quota assignment - draws stay random; only the resulting
totals converge to uniform (validated in the feasibility simulation,
scratchpad/feasibility_sim.py, which this script's generation logic mirrors).

Encoding conventions for the output CSVs (messages comma-joined, diff/shas/
types JSON-encoded arrays) match the legacy pipeline this replaces, plus a
new `repo` column recording which single repo each tangled commit came from.
"""

import json
import logging
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Reproducibility. Both random and numpy are seeded even though the current
# generation logic only draws from stdlib `random` - numpy is seeded too so
# any future numpy-based sampling added to this pipeline stays reproducible
# under the same single SEED, matching the rest of the pipeline's convention.
SEED = 42

# Concern-type taxonomy (post style/perf->refactor remap, chore dropped)
TYPES: List[str] = ["feat", "fix", "refactor", "test", "docs", "build", "ci"]

# Token budget for a combined tangled commit (sum of per-atomic token_count,
# a validated conservative proxy for the actual combined-diff token count -
# see feasibility_sim.py; the validation script re-checks the real joined
# diff separately).
MAX_TOKENS = 12288

CONCERN_COUNTS: List[int] = [1, 2, 3, 4, 5]
TRAIN_PER_COUNT = 280
TEST_PER_COUNT = 70
MAX_ATTEMPT_MULTIPLIER = 400  # bounded attempts per concern count = target * this

# Repo pinned to train regardless of greedy assignment (largest repo, 307 commits)
PINNED_TRAIN_REPO = "camunda/zeebe"

# Target test share of total commits (greedy stop condition allows ~20-23%)
TARGET_TEST_FRACTION = 0.20
TEST_FRACTION_OVERSHOOT_CAP = 1.1  # stop greedy once test_n >= target * this

# File paths (run from datasets/)
POOL_CSV_PATH = "data/repo_grouped_pool.csv"
REPO_SPLIT_JSON_PATH = "data/repo_split.json"
LEGACY_DIR = "data/legacy"
OUTPUT_TRAIN_PATH = "data/tangled_ccs_dataset_train.csv"
OUTPUT_TEST_PATH = "data/tangled_ccs_dataset_test.csv"

# Pool CSV column names
COLUMN_REPO = "repo"
COLUMN_TYPE = "annotated_type"
COLUMN_MESSAGE = "masked_commit_message"
COLUMN_DIFF = "git_diff"
COLUMN_SHA = "sha"
COLUMN_TOKEN_COUNT = "token_count"

# Output CSV schema (adds `repo` to the legacy schema)
OUTPUT_COLUMNS: List[str] = ["commit_message", "diff", "concern_count", "shas", "types", "repo"]

AtomRecord = Dict[str, Any]


def load_pool(path: str) -> pd.DataFrame:
    """Load the repo-grouped atomic commit pool."""
    df = pd.read_csv(path)
    logging.info(f"Loaded pool: {len(df)} commits across {df[COLUMN_REPO].nunique()} repos")
    return df


def partition_repos(pool: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Greedy, type-supply-balanced repo-disjoint partition into train/test.

    Targets ~20% of each type's total supply landing in test, stopping once
    the per-type deficit is closed or the overshoot cap is hit. The largest
    repo (camunda/zeebe) is pinned to train.

    Determinism note (verifier finding): this greedy assignment is fully
    deterministic given the input pool CSV's row order - `groupby` iterates
    repos in the order they first appear, and each step picks the single
    best-efficiency repo with no random tie-breaking. Re-running this
    function on the same repo_grouped_pool.csv always yields the same
    train/test partition; no seeding is required for this step.
    """
    # Repo-disjoint greedy partition. Purpose: no repo appears in both train
    # and test, preventing repo-identity leakage (a model could otherwise
    # learn repo-specific style cues instead of concern semantics).
    repo_stats: Dict[str, Counter] = {
        repo: Counter(sub[COLUMN_TYPE]) for repo, sub in pool.groupby(COLUMN_REPO)
    }
    repo_sizes = {repo: sum(c.values()) for repo, c in repo_stats.items()}
    total = sum(repo_sizes.values())
    target_test = TARGET_TEST_FRACTION * total

    type_supply: Counter = Counter()
    for c in repo_stats.values():
        type_supply += c
    type_target = {t: TARGET_TEST_FRACTION * type_supply[t] for t in TYPES}

    def deficit(supply: Counter) -> float:
        return sum(max(0.0, type_target[t] - supply.get(t, 0)) for t in TYPES)

    test_repos: List[str] = []
    test_supply: Counter = Counter()
    test_n = 0
    candidates = [r for r in repo_stats if r != PINNED_TRAIN_REPO]

    while True:
        cur_deficit = deficit(test_supply)
        if cur_deficit <= 0 or test_n >= target_test * TEST_FRACTION_OVERSHOOT_CAP:
            break
        best_repo, best_efficiency = None, 0.0
        for r in candidates:
            if r in test_repos:
                continue
            gain = cur_deficit - deficit(test_supply + repo_stats[r])
            efficiency = gain / repo_sizes[r]
            if efficiency > best_efficiency:
                best_repo, best_efficiency = r, efficiency
        if best_repo is None:
            break
        test_repos.append(best_repo)
        test_supply += repo_stats[best_repo]
        test_n += repo_sizes[best_repo]

    train_repos = [r for r in repo_stats if r not in test_repos]

    logging.info(
        f"Partition: train {len(train_repos)} repos / {total - test_n} commits, "
        f"test {len(test_repos)} repos / {test_n} commits ({test_n / total:.1%})"
    )
    logging.info(f"Test type supply: {dict(test_supply)}")

    return train_repos, test_repos


def save_repo_split(
    train_repos: List[str], test_repos: List[str], pool: pd.DataFrame, path: str
) -> None:
    """Persist the repo partition and per-split type supply to JSON."""
    train_supply = (
        pool[pool[COLUMN_REPO].isin(train_repos)].groupby(COLUMN_TYPE)[COLUMN_SHA].count().to_dict()
    )
    test_supply = (
        pool[pool[COLUMN_REPO].isin(test_repos)].groupby(COLUMN_TYPE)[COLUMN_SHA].count().to_dict()
    )
    payload = {
        "train": sorted(train_repos),
        "test": sorted(test_repos),
        "train_type_supply": {t: int(train_supply.get(t, 0)) for t in TYPES},
        "test_type_supply": {t: int(test_supply.get(t, 0)) for t in TYPES},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logging.info(f"Saved repo split assignment to {path}")


def build_atom_index(
    pool: pd.DataFrame, split_repos: List[str]
) -> Tuple[Dict[Tuple[str, str], List[AtomRecord]], Dict[str, set]]:
    """Index atomic commits by (repo, type) and precompute each repo's type coverage."""
    atoms: Dict[Tuple[str, str], List[AtomRecord]] = defaultdict(list)
    sub = pool[pool[COLUMN_REPO].isin(split_repos)]
    for _, row in sub.iterrows():
        atoms[(row[COLUMN_REPO], row[COLUMN_TYPE])].append(
            {
                "sha": row[COLUMN_SHA],
                "token_count": row[COLUMN_TOKEN_COUNT],
                "diff": row[COLUMN_DIFF],
                "message": row[COLUMN_MESSAGE],
            }
        )
    repo_types: Dict[str, set] = defaultdict(set)
    for (repo, t) in atoms:
        repo_types[repo].add(t)
    return atoms, repo_types


def pick_type_combination(
    k: int,
    label_counts: Counter,
    repo_types: Dict[str, set],
    split_repos: List[str],
    supporters_cache: Dict[Tuple[str, ...], List[str]],
) -> Optional[Tuple[Tuple[str, ...], List[str]]]:
    """Choose which k distinct concern types to tangle next.

    Under-represented-type-first adaptive sampling: rank types by their
    current running label count (random tie-break for stochasticity) and try
    the k most under-represented types first, falling back to random
    k-subsets. This drives exact 1/7 uniform label frequency per split, for a
    fair Hamming-loss comparison across types, without a deterministic quota
    assignment - draws stay random; only the totals converge.

    Intra-repo type-combination selection: a combo is only accepted if at
    least one repo's type coverage is a superset of it (`repo_types[r]`) -
    all atomics of one tangled commit must come from a SINGLE repo, which is
    what resolves the cross-repo heterogeneity critique that tangles were
    trivially separable by repo-specific style.

    Returns (combo, supporting_repos) for the first of 6 trials that has at
    least one supporting repo, or None if none did.
    """
    ranked = sorted(TYPES, key=lambda t: (label_counts[t], random.random()))
    for trial in range(6):
        cand = (
            tuple(sorted(ranked[:k]))
            if trial == 0
            else tuple(sorted(random.sample(TYPES, k)))
        )
        if cand not in supporters_cache:
            supporters_cache[cand] = [
                r for r in split_repos if set(cand) <= repo_types[r]
            ]
        if supporters_cache[cand]:
            return cand, supporters_cache[cand]
    return None


def pick_supporting_repo(supporters: List[str]) -> str:
    """Uniform random choice among repos that contain all k types (intra-repo constraint)."""
    return random.choice(supporters)


def sample_atomics(
    repo: str,
    combo: Tuple[str, ...],
    atoms: Dict[Tuple[str, str], List[AtomRecord]],
    reuse: Counter,
) -> List[AtomRecord]:
    """Reuse-weighted random atomic choice, one per type in combo.

    Purpose: spread usage across atomics (weight 1/(1+times_used) favors
    less-reused atomics) while keeping the pick stochastic - a deterministic
    least-used-first rule would instead loop on rejections (duplicate SHA
    sets, repeated token-budget misses).
    """
    picks: List[AtomRecord] = []
    for t in combo:
        cell = atoms[(repo, t)]
        weights = [1.0 / (1 + reuse[a["sha"]]) for a in cell]
        atom = random.choices(cell, weights=weights, k=1)[0]
        picks.append(atom)
    return picks


def exceeds_token_budget(picks: List[AtomRecord]) -> bool:
    """Combined 12,288-token check: the tangled commit's summed atomic token
    counts must still fit the SLM context window; reject and retry rather
    than truncating silently."""
    return sum(a["token_count"] for a in picks) > MAX_TOKENS


def is_duplicate(shas: List[str], seen_sha_combinations: set) -> bool:
    """SHA-frozenset dedup: no two tangled commits may draw the exact same
    set of atomic commits, within or across splits."""
    return frozenset(shas) in seen_sha_combinations


def build_tangled_row(
    k: int, repo: str, combo: Tuple[str, ...], picks: List[AtomRecord]
) -> Dict[str, Any]:
    """Assemble one tangled-commit row matching OUTPUT_COLUMNS exactly."""
    shas = [a["sha"] for a in picks]
    messages = [a["message"] for a in picks]
    diffs = [a["diff"] for a in picks]
    return {
        "commit_message": ",".join(str(m) for m in messages),
        "diff": json.dumps([str(d) for d in diffs]),
        "concern_count": k,
        "shas": json.dumps(shas),
        "types": json.dumps(list(combo)),
        "repo": repo,
    }


def generate_for_concern_count(
    k: int,
    n_target: int,
    split_name: str,
    split_repos: List[str],
    atoms: Dict[Tuple[str, str], List[AtomRecord]],
    repo_types: Dict[str, set],
    label_counts: Counter,
    reuse: Counter,
    seen_sha_combinations: set,
) -> List[Dict[str, Any]]:
    """Generate n_target tangled commits for one concern count k.

    label_counts, reuse, and seen_sha_combinations are shared, mutated
    in place across all concern counts within a split (not reset per-k),
    matching the original single-pass accumulation.
    """
    supporters_cache: Dict[Tuple[str, ...], List[str]] = {}
    rows: List[Dict[str, Any]] = []
    n_made = 0
    attempts = 0
    max_attempts = n_target * MAX_ATTEMPT_MULTIPLIER
    token_rejected = 0
    duplicate_rejected = 0
    no_repo_rejected = 0

    while n_made < n_target and attempts < max_attempts:
        attempts += 1

        combo_result = pick_type_combination(
            k, label_counts, repo_types, split_repos, supporters_cache
        )
        if combo_result is None:
            no_repo_rejected += 1
            continue
        combo, supporters = combo_result

        repo = pick_supporting_repo(supporters)
        picks = sample_atomics(repo, combo, atoms, reuse)

        if exceeds_token_budget(picks):
            token_rejected += 1
            continue

        shas = [a["sha"] for a in picks]
        if is_duplicate(shas, seen_sha_combinations):
            duplicate_rejected += 1
            continue
        seen_sha_combinations.add(frozenset(shas))

        rows.append(build_tangled_row(k, repo, combo, picks))
        for t in combo:
            label_counts[t] += 1
        for s in shas:
            reuse[s] += 1
        n_made += 1

    logging.info(
        f"[{split_name}] k={k}: generated {n_made}/{n_target} "
        f"(attempts={attempts}, token_rejected={token_rejected}, "
        f"duplicate_rejected={duplicate_rejected}, no_repo_rejected={no_repo_rejected})"
    )
    if n_made < n_target:
        raise SystemExit(
            f"FATAL: [{split_name}] k={k} only generated {n_made}/{n_target} "
            f"tangled commits after {attempts} attempts (attempt budget exhausted). "
            "Failing loudly per spec rather than silently under-delivering."
        )
    return rows


def generate_split(
    pool: pd.DataFrame, split_repos: List[str], n_per_count: int, split_name: str
) -> List[Dict[str, Any]]:
    """Generate tangled commits for one split using adaptive randomization.

    Thin orchestrator: builds the atom index once, then runs
    generate_for_concern_count for each k in CONCERN_COUNTS in order,
    accumulating label_counts/reuse/seen_sha_combinations across all k
    values in the split (same shared state as the original single loop).
    """
    atoms, repo_types = build_atom_index(pool, split_repos)

    label_counts: Counter = Counter()
    reuse: Counter = Counter()
    seen_sha_combinations: set = set()
    rows: List[Dict[str, Any]] = []

    for k in CONCERN_COUNTS:
        rows.extend(
            generate_for_concern_count(
                k,
                n_per_count,
                split_name,
                split_repos,
                atoms,
                repo_types,
                label_counts,
                reuse,
                seen_sha_combinations,
            )
        )

    reuse_values = sorted(reuse.values(), reverse=True)
    total_labels = sum(label_counts.values())
    shares = {t: label_counts[t] / total_labels for t in TYPES}
    dev = max(abs(v - 1 / 7) for v in shares.values())
    logging.info(f"[{split_name}] {len(rows)} tangled commits generated")
    logging.info(f"[{split_name}] label shares: {shares} (max deviation from 1/7: {dev * 100:.2f}pp)")
    if reuse_values:
        logging.info(
            f"[{split_name}] atomic reuse: max={reuse_values[0]}, "
            f"mean={sum(reuse_values) / len(reuse_values):.1f}, n_atomics_used={len(reuse_values)}"
        )

    return rows


def assert_exact_uniform_labels(rows: List[Dict[str, Any]], split_name: str) -> None:
    """Assert per-type label counts are EXACTLY equal within the split."""
    counts: Counter = Counter()
    for row in rows:
        counts.update(json.loads(row["types"]))

    values = set(counts[t] for t in TYPES)
    if len(values) != 1:
        raise SystemExit(
            f"FATAL: [{split_name}] per-type label counts are not exactly equal: "
            f"{dict(counts)}. Expected all 7 types to have identical counts."
        )
    per_type = next(iter(values))
    logging.info(f"[{split_name}] exact uniform check PASSED: {per_type} labels per type")


def backup_legacy_csvs() -> None:
    """Back up the old cross-repo tangled CSVs before overwriting them."""
    Path(LEGACY_DIR).mkdir(parents=True, exist_ok=True)
    for path in (OUTPUT_TRAIN_PATH, OUTPUT_TEST_PATH):
        src = Path(path)
        if src.exists():
            dst = Path(LEGACY_DIR) / src.name
            shutil.copy2(src, dst)
            logging.info(f"Backed up {src} -> {dst}")
        else:
            logging.info(f"No existing file at {src}, nothing to back up")


def save_split_csv(rows: List[Dict[str, Any]], path: str) -> None:
    """Save generated tangled commits to CSV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    df.to_csv(path, index=False, encoding="utf-8")
    logging.info(f"Saved {len(df)} rows to {path}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info("Generating repo-grouped tangled commit dataset")
    logging.info("=" * 50)

    random.seed(SEED)
    np.random.seed(SEED)

    pool = load_pool(POOL_CSV_PATH)

    train_repos, test_repos = partition_repos(pool)
    assert not (set(train_repos) & set(test_repos)), "train/test repo overlap detected"
    save_repo_split(train_repos, test_repos, pool, REPO_SPLIT_JSON_PATH)

    train_rows = generate_split(pool, train_repos, TRAIN_PER_COUNT, "TRAIN")
    test_rows = generate_split(pool, test_repos, TEST_PER_COUNT, "TEST")

    assert_exact_uniform_labels(train_rows, "TRAIN")
    assert_exact_uniform_labels(test_rows, "TEST")

    backup_legacy_csvs()

    save_split_csv(train_rows, OUTPUT_TRAIN_PATH)
    save_split_csv(test_rows, OUTPUT_TEST_PATH)

    logging.info("=" * 50)
    logging.info("Repo-grouped tangled dataset generation completed successfully!")


if __name__ == "__main__":
    main()
