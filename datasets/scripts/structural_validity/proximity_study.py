"""One runnable study of how close a tangled commit's concerns sit in the tree.

Replaces three ad-hoc probes with a single entry point that measures the
observed corpus and both permutation nulls under one definition, so the numbers
can no longer drift apart between scripts.

The nulls are not decoration. "7% of commits share a file" is uninterpretable
alone; it only means something against what the generator would have produced
by chance:

  null_repo_free  keeps each commit's CCS type combination but draws atomics
                  from any repository -- proximity with the generator's
                  repository constraint removed.
  null_same_repo  keeps (repository, type combination, k) exactly as generated
                  and resamples only WHICH atomic is drawn per type. Observed
                  values inside this interval mean the particular atomics chosen
                  carry no structural coupling of their own.

Read-only with respect to `datasets/data/`; writes only the artifact it is told
to write.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import random
import statistics
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from .contract import (
    CommitMetrics,
    PairPathMetrics,
    SyntheticCommit,
    aggregate_commit_metrics,
)
from .contract_types import PRIMARY_COMMIT_COUNT, SECONDARY_PAIR_COUNT, ReasonCode
from .path_proximity import shared_directory_count, shared_file_count, shared_path_depth
from .proximity_corpus import (
    Corpus,
    ObservedCommit,
    PathsBySha,
    SamplingCells,
    load_corpus,
)

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR: Final[Path] = PROJECT_ROOT / "datasets" / "data"
DEFAULT_ARTIFACT: Final[Path] = PROJECT_ROOT / "results" / "structural_validity" / "proximity_seed43.json"
SEED = 43
DRAWS = 200
HEADLINE_KEYS = ("commit_file_any", "commit_directory_any", "mean_max_depth", "pair_file_rate",
                 "pair_directory_rate", "mean_file_share", "mean_directory_share")


@dataclass(frozen=True, slots=True)
class Summary:
    """One corpus collapsed to comparable numbers, percentages except the depth.

    The estimands sit in a dict keyed by `HEADLINE_KEYS` rather than in seven
    named fields, so an observed row and a null row can be averaged and compared
    by name without reflecting over attributes.
    """

    label: str
    commits: int
    pairs: int
    headline: dict[str, float]
    depth_histogram: tuple[float, ...]

    def payload(self) -> dict[str, object]:
        """Flat JSON, so a consumer reads `summary["pair_file_rate"]` directly."""
        return {"label": self.label, "commits": self.commits, "pairs": self.pairs,
                **self.headline, "depth_histogram": list(self.depth_histogram)}


@dataclass(frozen=True, slots=True)
class NullResult:
    summary: Summary
    interval: dict[str, tuple[float, float]]
    unresolved_slots: int

    def payload(self) -> dict[str, object]:
        return {"summary": self.summary.payload(), "unresolved_slots": self.unresolved_slots,
                "interval": {key: list(bounds) for key, bounds in self.interval.items()}}


def measure_commit(commit: SyntheticCommit, paths: PathsBySha) -> tuple[CommitMetrics, list[PairPathMetrics]]:
    """Measure one commit, keeping unresolved sides in the denominator."""
    def measure(a: int, b: int) -> PairPathMetrics:
        left, right = paths.get(commit.shas[a], ()), paths.get(commit.shas[b], ())
        if not left or not right:
            return PairPathMetrics(a, b, 0, 0, 0, (ReasonCode.UNRESOLVED_PATH,))
        return PairPathMetrics(a, b, shared_file_count(left, right),
                               shared_directory_count(left, right), shared_path_depth(left, right))

    pairs = [measure(a, b) for a, b in commit.pair_indices()]
    return aggregate_commit_metrics(commit, pairs), pairs


def summarise(label: str, commits: Sequence[SyntheticCommit], paths: PathsBySha) -> Summary:
    """Collapse a whole corpus into one comparable row."""
    measured = [measure_commit(commit, paths) for commit in commits]
    metrics = [metric for metric, _ in measured]
    every_pair = [pair for _, pairs in measured for pair in pairs]
    depths = collections.Counter(pair.shared_path_depth for pair in every_pair)
    return Summary(
        label=label,
        commits=len(metrics),
        pairs=len(every_pair),
        headline={
            "commit_file_any": 100 * statistics.fmean([m.any_pair_shares_file for m in metrics]),
            "commit_directory_any": 100 * statistics.fmean([m.any_pair_shares_directory for m in metrics]),
            "mean_max_depth": statistics.fmean([m.max_shared_path_depth for m in metrics]),
            "pair_file_rate": 100 * statistics.fmean([p.shares_file for p in every_pair]),
            "pair_directory_rate": 100 * statistics.fmean([p.shares_directory for p in every_pair]),
            "mean_file_share": 100 * statistics.fmean([m.file_share for m in metrics]),
            "mean_directory_share": 100 * statistics.fmean([m.directory_share for m in metrics]),
        },
        depth_histogram=tuple(100 * depths[depth] / len(every_pair) for depth in range(max(depths, default=0) + 1)),
    )


def _resample(entry: ObservedCommit, cells: SamplingCells, rng: random.Random) -> tuple[SyntheticCommit, int]:
    """Redraw one commit's constituents, holding its type combination and k.

    Already-drawn SHAs are excluded so the redrawn commit stays as distinct as a
    generated one. A slot whose cell cannot supply a fresh atomic keeps its
    observed SHA and is counted; that count is reported rather than absorbed,
    because keeping observed atomics pulls the null toward the observed corpus.
    """
    if not entry.is_resamplable:
        return entry.commit, entry.commit.concern_count
    chosen: list[str] = []
    unresolved = 0
    for index, annotated in enumerate(entry.types):
        options = [sha for sha in cells.get((entry.commit.repo, annotated), ()) if sha not in chosen]
        if not options:
            chosen.append(entry.commit.shas[index])
            unresolved += 1
            continue
        chosen.append(rng.choice(options))
    if len(frozenset(chosen)) != len(chosen):
        return entry.commit, entry.commit.concern_count
    return SyntheticCommit(entry.commit.commit_id, entry.commit.split, entry.commit.repo,
                           entry.commit.concern_count, tuple(chosen)), unresolved


def permutation_null(label: str, corpus: Corpus, *, repository_free: bool, draws: int) -> NullResult:
    """Mean and 95% percentile interval over `draws` resamplings of the corpus."""
    cells = corpus.cells
    if repository_free:
        # Dropping the repository constraint is expressed as data rather than as
        # a branch inside the resampler: every cell of a type sees one pool.
        pooled: dict[str, list[str]] = collections.defaultdict(list)
        for (_repo, annotated), shas in corpus.cells.items():
            pooled[annotated].extend(shas)
        by_type = {annotated: tuple(shas) for annotated, shas in pooled.items()}
        cells = {key: by_type[key[1]] for key in corpus.cells}

    rng = random.Random(SEED)
    samples: list[Summary] = []
    unresolved = 0
    for _draw in range(draws):
        redrawn: list[SyntheticCommit] = []
        for entry in corpus.commits:
            commit, missing = _resample(entry, cells, rng)
            redrawn.append(commit)
            unresolved += missing
        samples.append(summarise(label, redrawn, corpus.paths))

    sorted_values = {key: sorted(sample.headline[key] for sample in samples) for key in HEADLINE_KEYS}
    averaged = {key: statistics.fmean(values) for key, values in sorted_values.items()}
    # Depth is a distribution, not a scalar, so it is averaged bin by bin. Short
    # histograms are padded: a draw with no depth-9 pair contributes 0% there.
    histogram = tuple(statistics.fmean(bins) for bins in itertools.zip_longest(
        *(sample.depth_histogram for sample in samples), fillvalue=0.0))
    return NullResult(
        summary=Summary(label, samples[0].commits, samples[0].pairs, averaged, histogram),
        interval={key: (values[int(0.025 * draws)], values[min(int(0.975 * draws), draws - 1)])
                  for key, values in sorted_values.items()},
        unresolved_slots=unresolved // draws,
    )


class _CliNamespace(argparse.Namespace):
    """Names, types and defaults for the parsed arguments.

    argparse leaves an attribute it already finds alone, so these class-body
    values are what an unpassed flag resolves to; `main` declares them to match.
    """

    data: Path = DEFAULT_DATA_DIR
    out: Path = DEFAULT_ARTIFACT
    draws: int = DRAWS
    repo_free_draws: int = DRAWS // 4


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seed-43 structural proximity study")
    _ = parser.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR)
    _ = parser.add_argument("--out", type=Path, default=DEFAULT_ARTIFACT, help="JSON artifact to write")
    _ = parser.add_argument("--draws", type=int, default=DRAWS)
    _ = parser.add_argument("--repo-free-draws", type=int, default=DRAWS // 4,
                            help="the repository-free null sits far from observed, so it needs fewer draws")
    arguments = parser.parse_args(list(argv) if argv is not None else None, _CliNamespace())
    data, out, draws = arguments.data, arguments.out, arguments.draws

    corpus = load_corpus(data)
    observed = summarise("observed", [entry.commit for entry in corpus.commits], corpus.paths)
    if (observed.commits, observed.pairs) != (PRIMARY_COMMIT_COUNT, SECONDARY_PAIR_COUNT):
        print(f"conservation failed: {observed.commits} commits, {observed.pairs} pairs", file=sys.stderr)
        return 1

    same_repo = permutation_null("null_same_repo", corpus, repository_free=False, draws=draws)
    repo_free = permutation_null("null_repo_free", corpus, repository_free=True, draws=arguments.repo_free_draws)
    by_k = {k: summarise(f"k={k}", [e.commit for e in corpus.commits if e.commit.concern_count == k], corpus.paths)
            for k in sorted({entry.commit.concern_count for entry in corpus.commits})}

    payload = {
        "seed": SEED,
        "draws": draws,
        "observed": observed.payload(),
        "null_same_repo": same_repo.payload(),
        "null_repo_free": repo_free.payload(),
        "by_concern_count": {str(k): value.payload() for k, value in by_k.items()},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    _ = out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"observed: {observed.commits} commits / {observed.pairs} pairs (conservation holds)")
    print(f"unresolved slots per draw: same-repo {same_repo.unresolved_slots}, repo-free {repo_free.unresolved_slots}")
    for key in HEADLINE_KEYS:
        value, free = observed.headline[key], repo_free.summary.headline[key]
        low, high = same_repo.interval[key]
        verdict = "inside null" if low <= value <= high else "OUTSIDE NULL"
        row = f"  {key:22s} observed {value:7.3f}  same-repo null [{low:7.3f}, {high:7.3f}]"
        print(f"{row} {verdict:12s}  repo-free null {free:7.3f}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
