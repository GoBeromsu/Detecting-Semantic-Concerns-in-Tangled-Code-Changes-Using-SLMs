"""Immutable metric and claim contracts for the C1.1 structural-validity study."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from typing import Final, NewType, TypedDict

CommitId = NewType("CommitId", str)
PRIMARY_COMMIT_COUNT: Final[int] = 1_400
SECONDARY_PAIR_COUNT: Final[int] = 7_000
PRIMARY_K_VALUES: Final[tuple[int, ...]] = (2, 3, 4, 5)


class Split(str, Enum):
    TRAIN = "train"
    TEST = "test"


class ObservationUnit(str, Enum):
    COMMIT = "commit"
    PAIR = "pair"
    SAME_FILE_MEASURABLE_PAIR = "same_file_measurable_pair"
    PATH_MATCHED_RESOLVED_AST_PAIR = "path_matched_resolved_ast_pair"
    SOURCE_CLASSIFIED_FILE = "source_classified_file"


class CoverageStatus(str, Enum):
    SOURCE_AST = "source_ast"
    STRUCTURED_PARSE_TREE = "structured_parse_tree"
    TEXT_ONLY = "text_only"
    BINARY_GENERATED = "binary_generated"
    AMBIGUOUS = "ambiguous"
    UNRESOLVED_SOURCE = "unresolved_source"


class ReasonCode(str, Enum):
    INVALID_CONCERN_COUNT = "invalid_concern_count"
    SHA_COUNT_MISMATCH = "sha_count_mismatch"
    DUPLICATE_SHA = "duplicate_sha"
    PAIR_CONSERVATION = "pair_conservation"
    MISSING_SHA = "missing_sha"
    REPO_MISMATCH = "repo_mismatch"
    SPLIT_MISMATCH = "split_mismatch"
    MISSING_OBJECT = "missing_object"
    MISSING_REVISION = "missing_revision"
    MISSING_BLOB = "missing_blob"
    UNAVAILABLE_PARENT = "unavailable_parent"
    MERGE_PARENT = "merge_parent"
    RENAME = "rename"
    DELETION = "deletion"
    COPY = "copy"
    BINARY_GENERATED = "binary_generated"
    SUBMODULE = "submodule"
    UNRESOLVED_PATH = "unresolved_path"
    AMBIGUOUS_PATH = "ambiguous_path"
    AMBIGUOUS = "ambiguous"
    UNSUPPORTED_SOURCE = "unsupported_source"
    PARSE_FAILED_QUALITY = "parse_failed_quality"


@dataclass(frozen=True, slots=True)
class StructuralContractError(Exception):
    commit_id: CommitId
    reason: ReasonCode
    detail: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", (f"{self.commit_id}: {self.reason.value}: {self.detail}",))


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
    concern_a: int
    concern_b: int
    same_file: bool
    same_leaf_directory: bool
    common_parent_prefix_depth: int
    reason_codes: tuple[ReasonCode, ...] = ()

    def __post_init__(self) -> None:
        if self.concern_a < 0 or self.concern_a >= self.concern_b:
            raise StructuralContractError(CommitId("pair"), ReasonCode.PAIR_CONSERVATION, f"expected 0 <= concern_a < concern_b, got {(self.concern_a, self.concern_b)}")
        if self.common_parent_prefix_depth < 0:
            raise StructuralContractError(CommitId("pair"), ReasonCode.PAIR_CONSERVATION, "prefix depth cannot be negative")
        if self.reason_codes and (self.same_file or self.same_leaf_directory or self.common_parent_prefix_depth != 0):
            raise StructuralContractError(CommitId("pair"), ReasonCode.PAIR_CONSERVATION, "reason-coded path failures must remain false with depth 0")


@dataclass(frozen=True, slots=True)
class HunkLineEvidence:
    hunk_context_overlap: bool
    min_line_gap: int | None
    diff_local_normalized_gap: float | None
    legacy_hunk_start_gap: int | None
    reason_codes: tuple[ReasonCode, ...] = ()


@dataclass(frozen=True, slots=True)
class AstEvidence:
    coverage_status: CoverageStatus
    name_matched_entity_co_touch: bool | None
    syntax_identifier_string_overlap: float | None
    entity_span_overlap: bool | None
    reason_codes: tuple[ReasonCode, ...] = ()


class PairRecordJson(TypedDict):
    commit_id: str
    split: str
    concern_count: int
    concern_a: int
    concern_b: int
    sha_a: str
    sha_b: str
    same_file: bool
    same_leaf_directory: bool
    common_parent_prefix_depth: int
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
        return PairRecordJson(commit_id=self.commit_id, split=self.split.value, concern_count=self.concern_count, concern_a=self.path.concern_a, concern_b=self.path.concern_b, sha_a=self.sha_a, sha_b=self.sha_b, same_file=self.path.same_file, same_leaf_directory=self.path.same_leaf_directory, common_parent_prefix_depth=self.path.common_parent_prefix_depth, reason_codes=[reason.value for reason in self.path.reason_codes])


@dataclass(frozen=True, slots=True)
class CommitMetrics:
    commit_id: CommitId
    same_file_any: bool
    same_file_all_concerns: bool
    same_file_all_pairs: bool
    same_leaf_directory_any: bool
    same_leaf_directory_all_concerns: bool
    same_leaf_directory_all_pairs: bool
    common_parent_prefix_depth: int
    strict_common_parent_prefix_depth: int
    reason_codes: tuple[ReasonCode, ...]


def common_parent_prefix_depth(paths_a: Sequence[str], paths_b: Sequence[str]) -> int:
    """Return the maximum shared parent-component depth across two path sets."""
    maximum = 0
    for path_a in paths_a:
        parent_a = tuple(part for part in path_a.split("/")[:-1] if part and part != ".")
        for path_b in paths_b:
            parent_b = tuple(part for part in path_b.split("/")[:-1] if part and part != ".")
            depth = 0
            for left, right in zip(parent_a, parent_b):
                if left != right:
                    break
                depth += 1
            maximum = max(maximum, depth)
    return maximum


def _all_concerns(concern_count: int, edges: Sequence[tuple[int, int]]) -> bool:
    degrees = tuple(sum(vertex in edge for edge in edges) for vertex in range(concern_count))
    return all(degree >= 1 for degree in degrees)


def aggregate_commit_metrics(commit: SyntheticCommit, pairs: Sequence[PairPathMetrics]) -> CommitMetrics:
    """Aggregate conserved unordered pair metrics into one commit observation."""
    expected_pairs = frozenset(commit.pair_indices())
    measured_pairs = tuple((pair.concern_a, pair.concern_b) for pair in pairs)
    if len(measured_pairs) != len(expected_pairs) or frozenset(measured_pairs) != expected_pairs:
        raise StructuralContractError(commit.commit_id, ReasonCode.PAIR_CONSERVATION, f"expected pairs {sorted(expected_pairs)}, got {sorted(measured_pairs)}")
    file_edges = tuple(edge for edge, pair in zip(measured_pairs, pairs) if pair.same_file)
    directory_edges = tuple(edge for edge, pair in zip(measured_pairs, pairs) if pair.same_leaf_directory)
    nearest_depths = tuple(max(pair.common_parent_prefix_depth for pair in pairs if vertex in (pair.concern_a, pair.concern_b)) for vertex in range(commit.concern_count))
    reasons = tuple(dict.fromkeys(reason for pair in pairs for reason in pair.reason_codes))
    return CommitMetrics(
        commit_id=commit.commit_id,
        same_file_any=bool(file_edges),
        same_file_all_concerns=_all_concerns(commit.concern_count, file_edges),
        same_file_all_pairs=all(pair.same_file for pair in pairs),
        same_leaf_directory_any=bool(directory_edges),
        same_leaf_directory_all_concerns=_all_concerns(commit.concern_count, directory_edges),
        same_leaf_directory_all_pairs=all(pair.same_leaf_directory for pair in pairs),
        common_parent_prefix_depth=max(pair.common_parent_prefix_depth for pair in pairs),
        strict_common_parent_prefix_depth=min(nearest_depths),
        reason_codes=reasons,
    )


class HeadlineJson(TypedDict):
    wording: str
    commit_denominator: int
    pair_denominator: int
    concern_counts: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class HeadlineContract:
    wording: str
    commit_denominator: int
    pair_denominator: int
    concern_counts: tuple[int, ...]

    def as_json(self) -> HeadlineJson:
        return HeadlineJson(wording=self.wording, commit_denominator=self.commit_denominator, pair_denominator=self.pair_denominator, concern_counts=self.concern_counts)


class DenominatorJson(TypedDict):
    estimand: str
    unit: str
    expected_count: int | None
    inclusion_rule: str
    failure_policy: str
    headline: bool


@dataclass(frozen=True, slots=True)
class DenominatorContract:
    estimand: str
    unit: ObservationUnit
    expected_count: int | None
    inclusion_rule: str
    failure_policy: str
    headline: bool

    def as_json(self) -> DenominatorJson:
        return DenominatorJson(estimand=self.estimand, unit=self.unit.value, expected_count=self.expected_count, inclusion_rule=self.inclusion_rule, failure_policy=self.failure_policy, headline=self.headline)


class ArtifactSchemaJson(TypedDict):
    name: str
    unit: str
    required_fields: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ArtifactSchema:
    name: str
    unit: ObservationUnit
    required_fields: tuple[str, ...]

    def as_json(self) -> ArtifactSchemaJson:
        return ArtifactSchemaJson(name=self.name, unit=self.unit.value, required_fields=self.required_fields)


HEADLINE_CONTRACT: Final[HeadlineContract] = HeadlineContract(
    wording="Across N = 1,400 multi-concern synthetic commits with k=2–5, X% contain at least one constituent-concern pair that shares a changed file path, Y% contain at least one pair that shares a leaf directory, and mean commit-level maximum common-prefix depth is Z.",
    commit_denominator=PRIMARY_COMMIT_COUNT,
    pair_denominator=SECONDARY_PAIR_COUNT,
    concern_counts=PRIMARY_K_VALUES,
)
DENOMINATOR_CONTRACTS: Final[tuple[DenominatorContract, ...]] = (
    DenominatorContract("same_file_any", ObservationUnit.COMMIT, PRIMARY_COMMIT_COUNT, "all multi-concern commits with k=2..5", "retain_false_or_zero", True),
    DenominatorContract("same_leaf_directory_any", ObservationUnit.COMMIT, PRIMARY_COMMIT_COUNT, "all multi-concern commits with k=2..5", "retain_false_or_zero", True),
    DenominatorContract("mean_commit_max_common_parent_prefix_depth", ObservationUnit.COMMIT, PRIMARY_COMMIT_COUNT, "all multi-concern commits with k=2..5", "retain_false_or_zero", True),
    DenominatorContract("pair_same_file_rate", ObservationUnit.PAIR, SECONDARY_PAIR_COUNT, "all unordered constituent-concern pairs", "retain_false_or_zero", False),
    DenominatorContract("pair_same_leaf_directory_rate", ObservationUnit.PAIR, SECONDARY_PAIR_COUNT, "all unordered constituent-concern pairs", "retain_false_or_zero", False),
    DenominatorContract("min_line_gap", ObservationUnit.SAME_FILE_MEASURABLE_PAIR, None, "same-file pairs with measurable new-side coordinates", "exclude_with_reason", False),
    DenominatorContract("name_matched_entity_co_touch", ObservationUnit.PATH_MATCHED_RESOLVED_AST_PAIR, None, "path-matched pairs where both concerns resolve source entities", "exclude_with_reason", False),
)
COMMIT_ARTIFACT_SCHEMA: Final[ArtifactSchema] = ArtifactSchema("commit_metrics.csv", ObservationUnit.COMMIT, ("commit_id", "split", "repo", "concern_count", "same_file_any", "same_file_all_concerns", "same_file_all_pairs", "same_leaf_directory_any", "same_leaf_directory_all_concerns", "same_leaf_directory_all_pairs", "common_parent_prefix_depth", "strict_common_parent_prefix_depth", "reason_codes"))
PAIR_ARTIFACT_SCHEMA: Final[ArtifactSchema] = ArtifactSchema("pair_metrics.csv", ObservationUnit.PAIR, ("commit_id", "split", "concern_count", "concern_a", "concern_b", "sha_a", "sha_b", "same_file", "same_leaf_directory", "common_parent_prefix_depth", "reason_codes"))
