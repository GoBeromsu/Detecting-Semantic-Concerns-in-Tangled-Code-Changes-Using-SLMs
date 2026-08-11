"""Immutable metric and claim contracts for the C1.1 structural-validity study."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import TypedDict

from .contract_types import (
    PRIMARY_K_VALUES,
    CommitId,
    CoverageStatus,
    ReasonCode,
    Split,
    StructuralContractError,
)


@dataclass(frozen=True, slots=True)
class Stratum:
    split: Split
    concern_count: int

    def __post_init__(self) -> None:
        if self.concern_count not in PRIMARY_K_VALUES:
            raise StructuralContractError(CommitId(f"{self.split.value}:stratum"), ReasonCode.INVALID_CONCERN_COUNT, f"expected k in {PRIMARY_K_VALUES}, got {self.concern_count}")


@dataclass(frozen=True, slots=True)
class SyntheticCommit:
    commit_id: CommitId
    split: Split
    repo: str
    concern_count: int
    shas: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.concern_count not in PRIMARY_K_VALUES:
            raise StructuralContractError(self.commit_id, ReasonCode.INVALID_CONCERN_COUNT, f"expected k in {PRIMARY_K_VALUES}, got {self.concern_count}")
        if len(self.shas) != self.concern_count:
            raise StructuralContractError(self.commit_id, ReasonCode.SHA_COUNT_MISMATCH, f"concern_count={self.concern_count}, len(shas)={len(self.shas)}")
        if len(frozenset(self.shas)) != self.concern_count:
            raise StructuralContractError(self.commit_id, ReasonCode.DUPLICATE_SHA, "constituent SHAs must be distinct")

    def pair_indices(self) -> tuple[tuple[int, int], ...]:
        return tuple(combinations(range(self.concern_count), 2))


@dataclass(frozen=True, slots=True)
class PairPathMetrics:
    """How close the changed paths of two constituent concerns are.

    Counts, not flags. `shares_file` used to be the only file observable, which
    made "one file in common out of twenty" and "every file in common" the same
    measurement. The booleans are derived so they can never disagree with the
    counts they summarise.
    """

    concern_a: int
    concern_b: int
    shared_file_count: int
    shared_directory_count: int
    shared_path_depth: int
    reason_codes: tuple[ReasonCode, ...] = ()

    @property
    def shares_file(self) -> bool:
        return self.shared_file_count > 0

    @property
    def shares_directory(self) -> bool:
        return self.shared_directory_count > 0

    def __post_init__(self) -> None:
        if self.concern_a < 0 or self.concern_a >= self.concern_b:
            raise StructuralContractError(CommitId("pair"), ReasonCode.PAIR_CONSERVATION, f"expected 0 <= concern_a < concern_b, got {(self.concern_a, self.concern_b)}")
        if self.shared_file_count < 0 or self.shared_directory_count < 0 or self.shared_path_depth < 0:
            raise StructuralContractError(CommitId("pair"), ReasonCode.PAIR_CONSERVATION, "shared counts and depth cannot be negative")
        if self.shared_file_count > 0 and self.shared_directory_count == 0:
            raise StructuralContractError(CommitId("pair"), ReasonCode.PAIR_CONSERVATION, f"a shared file implies a shared directory, got {self.shared_file_count} files in 0 directories")
        if self.reason_codes and (self.shared_file_count or self.shared_directory_count or self.shared_path_depth):
            raise StructuralContractError(CommitId("pair"), ReasonCode.PAIR_CONSERVATION, "reason-coded path failures must remain zero")


@dataclass(frozen=True, slots=True)
class HunkLineEvidence:
    hunk_context_overlap: bool
    min_line_gap: int | None
    diff_local_normalized_gap: float | None
    legacy_hunk_start_gap: int | None
    reason_codes: tuple[ReasonCode, ...] = ()


@dataclass(frozen=True, slots=True)
class AstEvidence:
    """Whether two concerns' edits landed on the same named entity.

    `shares_function` reports that both edits fell inside an entity with the
    same normalised qualified name, role, and arity. It is a name match, not
    function identity across history, and not evidence that either edit depends
    on the other. `shared_identifier_ratio` is a string comparison kept for
    robustness only and stays out of the headline.
    """

    coverage_status: CoverageStatus
    shares_function: bool | None
    shared_function_count: int | None
    shared_identifier_ratio: float | None
    reason_codes: tuple[ReasonCode, ...] = ()


class PairRecordJson(TypedDict):
    commit_id: str
    split: str
    concern_count: int
    concern_a: int
    concern_b: int
    sha_a: str
    sha_b: str
    shares_file: bool
    shared_file_count: int
    shares_directory: bool
    shared_directory_count: int
    shared_path_depth: int
    reason_codes: list[str]


@dataclass(frozen=True, slots=True)
class PairRecord:
    commit_id: CommitId
    split: Split
    concern_count: int
    sha_a: str
    sha_b: str
    path: PairPathMetrics
    hunk_line: HunkLineEvidence | None = None
    ast: AstEvidence | None = None

    @classmethod
    def from_commit(cls, commit: SyntheticCommit, path: PairPathMetrics) -> PairRecord:
        if path.concern_b >= commit.concern_count:
            raise StructuralContractError(commit.commit_id, ReasonCode.PAIR_CONSERVATION, f"pair {(path.concern_a, path.concern_b)} exceeds k={commit.concern_count}")
        return cls(commit.commit_id, commit.split, commit.concern_count, commit.shas[path.concern_a], commit.shas[path.concern_b], path)

    def as_json(self) -> PairRecordJson:
        return PairRecordJson(
            commit_id=self.commit_id,
            split=self.split.value,
            concern_count=self.concern_count,
            concern_a=self.path.concern_a,
            concern_b=self.path.concern_b,
            sha_a=self.sha_a,
            sha_b=self.sha_b,
            shares_file=self.path.shares_file,
            shared_file_count=self.path.shared_file_count,
            shares_directory=self.path.shares_directory,
            shared_directory_count=self.path.shared_directory_count,
            shared_path_depth=self.path.shared_path_depth,
            reason_codes=[reason.value for reason in self.path.reason_codes],
        )


@dataclass(frozen=True, slots=True)
class CommitMetrics:
    """One tangled commit's proximity, counted over concerns rather than pairs.

    `concerns_sharing_file` is how many of the commit's k concerns land on
    ground some other concern also touched -- the "3 of the 5" reading. Pooling
    over pairs instead would let a k=5 commit contribute ten observations to a
    k=2 commit's one; here every commit contributes exactly one, the way
    Hamming loss normalises within an instance before averaging across them.

    Normalising the denominator does not make the measure k-neutral: a concern
    has more potential partners as k grows, so `file_share` still rises with k
    and must be reported k-stratified.
    """

    commit_id: CommitId
    concern_count: int
    concerns_sharing_file: int
    concerns_sharing_directory: int
    all_pairs_share_file: bool
    all_pairs_share_directory: bool
    max_shared_path_depth: int
    min_shared_path_depth: int
    reason_codes: tuple[ReasonCode, ...]

    @property
    def any_pair_shares_file(self) -> bool:
        return self.concerns_sharing_file > 0

    @property
    def any_pair_shares_directory(self) -> bool:
        return self.concerns_sharing_directory > 0

    @property
    def all_concerns_share_file(self) -> bool:
        return self.concerns_sharing_file == self.concern_count

    @property
    def all_concerns_share_directory(self) -> bool:
        return self.concerns_sharing_directory == self.concern_count

    @property
    def file_share(self) -> float:
        return self.concerns_sharing_file / self.concern_count

    @property
    def directory_share(self) -> float:
        return self.concerns_sharing_directory / self.concern_count


def _connected_concerns(concern_count: int, edges: Sequence[tuple[int, int]]) -> int:
    """How many concerns share with at least one other concern.

    Sharing is symmetric, so this is 0 or at least 2 -- never exactly 1.
    """
    return sum(1 for vertex in range(concern_count) if any(vertex in edge for edge in edges))


def aggregate_commit_metrics(commit: SyntheticCommit, pairs: Sequence[PairPathMetrics]) -> CommitMetrics:
    """Aggregate conserved unordered pair metrics into one commit observation."""
    expected_pairs = frozenset(commit.pair_indices())
    measured_pairs = tuple((pair.concern_a, pair.concern_b) for pair in pairs)
    if len(measured_pairs) != len(expected_pairs) or frozenset(measured_pairs) != expected_pairs:
        raise StructuralContractError(commit.commit_id, ReasonCode.PAIR_CONSERVATION, f"expected pairs {sorted(expected_pairs)}, got {sorted(measured_pairs)}")
    file_edges = tuple(edge for edge, pair in zip(measured_pairs, pairs) if pair.shares_file)
    directory_edges = tuple(edge for edge, pair in zip(measured_pairs, pairs) if pair.shares_directory)
    nearest_depths = tuple(max(pair.shared_path_depth for pair in pairs if vertex in (pair.concern_a, pair.concern_b)) for vertex in range(commit.concern_count))
    reasons = tuple(dict.fromkeys(reason for pair in pairs for reason in pair.reason_codes))
    return CommitMetrics(
        commit_id=commit.commit_id,
        concern_count=commit.concern_count,
        concerns_sharing_file=_connected_concerns(commit.concern_count, file_edges),
        concerns_sharing_directory=_connected_concerns(commit.concern_count, directory_edges),
        all_pairs_share_file=all(pair.shares_file for pair in pairs),
        all_pairs_share_directory=all(pair.shares_directory for pair in pairs),
        max_shared_path_depth=max(pair.shared_path_depth for pair in pairs),
        min_shared_path_depth=min(nearest_depths),
        reason_codes=reasons,
    )
