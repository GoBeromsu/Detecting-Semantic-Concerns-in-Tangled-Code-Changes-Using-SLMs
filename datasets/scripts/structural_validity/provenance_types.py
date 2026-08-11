from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import override


class MetadataReason(str, Enum):
    OK = "ok"
    MALFORMED_JSON = "malformed_json"
    UNKNOWN_SHA = "unknown_sha"
    RAW_SHA_MISSING = "raw_sha_missing"
    DUPLICATE_POOL_SHA = "duplicate_pool_sha"
    DUPLICATE_RAW_SHA = "duplicate_raw_sha"
    INVALID_REMOTE = "invalid_remote"
    SOURCE_REPO_MISMATCH = "source_repo_mismatch"
    REPO_MISMATCH = "repo_mismatch"
    SPLIT_MISMATCH = "split_mismatch"
    CROSS_SPLIT_SHA = "cross_split_sha"


@dataclass(frozen=True, slots=True)
class MetadataPaths:
    train_csv: Path
    test_csv: Path
    pool_csv: Path
    raw_csv: Path
    repo_split_json: Path


@dataclass(frozen=True, slots=True)
class MetadataCounts:
    total_rows: int
    multiconcern_rows: int
    sha_references: int
    unique_shas: int
    repositories: int


@dataclass(frozen=True, slots=True)
class MetadataRecord:
    sha: str
    repo: str
    split: str
    remote_url: str
    reference_count: int
    reason: MetadataReason

    def with_sha(self, sha: str) -> MetadataRecord:
        return MetadataRecord(
            sha, self.repo, self.split, self.remote_url, self.reference_count, self.reason
        )


@dataclass(frozen=True, slots=True)
class MetadataManifest:
    records: tuple[MetadataRecord, ...]
    counts: MetadataCounts


@dataclass(frozen=True, slots=True)
class MetadataValidationError(Exception):
    reason: MetadataReason
    location: str
    detail: str

    @override
    def __str__(self) -> str:
        return f"{self.reason.value} at {self.location}: {self.detail}"
