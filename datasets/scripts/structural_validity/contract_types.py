"""Identifiers, enumerations, and the one error every contract check raises.

Split out of `contract` so the measurement types stay inside the plan's
per-file size gate, and so the schema declarations can name an observation unit
without importing the measurement code.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final, NewType

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
