"""Frozen declarations: the headline wording, every denominator, every artifact.

These are statements about what the study promises to report, not code that
measures anything. They live apart from `contract` so the measurement types stay
readable and so a change to a denominator shows up in review as a change to a
declaration file rather than as a diff buried in aggregation logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, TypedDict

from .contract_types import (
    PRIMARY_COMMIT_COUNT,
    PRIMARY_K_VALUES,
    SECONDARY_PAIR_COUNT,
    ObservationUnit,
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
        return HeadlineJson(
            wording=self.wording,
            commit_denominator=self.commit_denominator,
            pair_denominator=self.pair_denominator,
            concern_counts=self.concern_counts,
        )


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
        return DenominatorJson(
            estimand=self.estimand,
            unit=self.unit.value,
            expected_count=self.expected_count,
            inclusion_rule=self.inclusion_rule,
            failure_policy=self.failure_policy,
            headline=self.headline,
        )


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


_ALL_COMMITS: Final[str] = "all multi-concern commits with k=2..5"
_ALL_PAIRS: Final[str] = "all unordered constituent-concern pairs"
_RETAIN: Final[str] = "retain_false_or_zero"

HEADLINE_CONTRACT: Final[HeadlineContract] = HeadlineContract(
    wording="Across N = 1,400 multi-concern synthetic commits with k=2–5, X% contain at least one constituent-concern pair that shares a changed file path, Y% contain at least one pair that shares a non-root leaf directory, and mean commit-level maximum common-prefix depth is Z.",
    commit_denominator=PRIMARY_COMMIT_COUNT,
    pair_denominator=SECONDARY_PAIR_COUNT,
    concern_counts=PRIMARY_K_VALUES,
)

DENOMINATOR_CONTRACTS: Final[tuple[DenominatorContract, ...]] = (
    DenominatorContract("any_pair_shares_file", ObservationUnit.COMMIT, PRIMARY_COMMIT_COUNT, _ALL_COMMITS, _RETAIN, True),
    # Directory sharing excludes the repository root: every top-level file
    # shares it by construction, which made "one concern touched README.md, the
    # other touched .gitignore" score the same as two concerns inside one
    # module. That case was 51% of all observed directory sharing.
    DenominatorContract("any_pair_shares_directory", ObservationUnit.COMMIT, PRIMARY_COMMIT_COUNT, _ALL_COMMITS, _RETAIN, True),
    DenominatorContract("mean_max_shared_path_depth", ObservationUnit.COMMIT, PRIMARY_COMMIT_COUNT, _ALL_COMMITS, _RETAIN, True),
    # Concern-share companions. Each commit contributes one observation whatever
    # its k, so a k=5 commit no longer carries ten times a k=2 commit's weight
    # the way pair pooling makes it. They remain k-stratified: "shares with at
    # least one other concern" still grows with k because there are more
    # concerns to share with, which normalisation does not remove.
    DenominatorContract("mean_file_share_of_concerns", ObservationUnit.COMMIT, PRIMARY_COMMIT_COUNT, _ALL_COMMITS, _RETAIN, False),
    DenominatorContract("mean_directory_share_of_concerns", ObservationUnit.COMMIT, PRIMARY_COMMIT_COUNT, _ALL_COMMITS, _RETAIN, False),
    DenominatorContract("pair_shares_file_rate", ObservationUnit.PAIR, SECONDARY_PAIR_COUNT, _ALL_PAIRS, _RETAIN, False),
    DenominatorContract("pair_shares_directory_rate", ObservationUnit.PAIR, SECONDARY_PAIR_COUNT, _ALL_PAIRS, _RETAIN, False),
    DenominatorContract("mean_shared_file_count", ObservationUnit.PAIR, SECONDARY_PAIR_COUNT, _ALL_PAIRS, _RETAIN, False),
    DenominatorContract("min_line_gap", ObservationUnit.SAME_FILE_MEASURABLE_PAIR, None, "same-file pairs with measurable new-side coordinates", "exclude_with_reason", False),
    DenominatorContract("shares_function", ObservationUnit.PATH_MATCHED_RESOLVED_AST_PAIR, None, "path-matched pairs where both concerns resolve source entities", "exclude_with_reason", False),
)

COMMIT_ARTIFACT_SCHEMA: Final[ArtifactSchema] = ArtifactSchema(
    "commit_metrics.csv",
    ObservationUnit.COMMIT,
    (
        "commit_id", "split", "repo", "concern_count",
        "concerns_sharing_file", "file_share", "any_pair_shares_file", "all_pairs_share_file",
        "concerns_sharing_directory", "directory_share", "any_pair_shares_directory", "all_pairs_share_directory",
        "max_shared_path_depth", "min_shared_path_depth", "reason_codes",
    ),
)

PAIR_ARTIFACT_SCHEMA: Final[ArtifactSchema] = ArtifactSchema(
    "pair_metrics.csv",
    ObservationUnit.PAIR,
    (
        "commit_id", "split", "concern_count", "concern_a", "concern_b", "sha_a", "sha_b",
        "shares_file", "shared_file_count", "shares_directory", "shared_directory_count",
        "shared_path_depth", "reason_codes",
    ),
)

__all__ = [
    "COMMIT_ARTIFACT_SCHEMA",
    "DENOMINATOR_CONTRACTS",
    "HEADLINE_CONTRACT",
    "PAIR_ARTIFACT_SCHEMA",
    "ArtifactSchema",
    "ArtifactSchemaJson",
    "DenominatorContract",
    "DenominatorJson",
    "HeadlineContract",
    "HeadlineJson",
]
