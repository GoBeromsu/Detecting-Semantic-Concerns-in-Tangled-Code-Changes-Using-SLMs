"""Public contract surface for the structural-validity study."""

from .claim_guard import FORBIDDEN_CLAIM_PHRASES, ForbiddenClaimError, validate_claim
from .contract import (
    AstEvidence,
    CommitMetrics,
    HunkLineEvidence,
    PairPathMetrics,
    PairRecord,
    Stratum,
    SyntheticCommit,
    aggregate_commit_metrics,
)
from .contract_schema import (
    COMMIT_ARTIFACT_SCHEMA,
    DENOMINATOR_CONTRACTS,
    HEADLINE_CONTRACT,
    PAIR_ARTIFACT_SCHEMA,
    ArtifactSchema,
    DenominatorContract,
    HeadlineContract,
)
from .contract_types import (
    PRIMARY_COMMIT_COUNT,
    PRIMARY_K_VALUES,
    SECONDARY_PAIR_COUNT,
    CommitId,
    CoverageStatus,
    ObservationUnit,
    ReasonCode,
    Split,
    StructuralContractError,
)
from .path_proximity import shared_directory_count, shared_file_count, shared_path_depth

__all__ = (
    "COMMIT_ARTIFACT_SCHEMA",
    "DENOMINATOR_CONTRACTS",
    "FORBIDDEN_CLAIM_PHRASES",
    "HEADLINE_CONTRACT",
    "PAIR_ARTIFACT_SCHEMA",
    "PRIMARY_COMMIT_COUNT",
    "PRIMARY_K_VALUES",
    "SECONDARY_PAIR_COUNT",
    "ArtifactSchema",
    "AstEvidence",
    "CommitId",
    "CommitMetrics",
    "CoverageStatus",
    "DenominatorContract",
    "ForbiddenClaimError",
    "HeadlineContract",
    "HunkLineEvidence",
    "ObservationUnit",
    "PairPathMetrics",
    "PairRecord",
    "ReasonCode",
    "Split",
    "Stratum",
    "StructuralContractError",
    "SyntheticCommit",
    "aggregate_commit_metrics",
    "shared_directory_count",
    "shared_file_count",
    "shared_path_depth",
    "validate_claim",
)
