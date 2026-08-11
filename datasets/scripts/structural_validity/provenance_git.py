from __future__ import annotations

import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Final

from structural_validity.provenance_metadata import canonical_github_remote
from structural_validity.provenance_types import MetadataReason, MetadataRecord


class ResolutionStatus(str, Enum):
    METADATA_ONLY = "metadata_only"
    RESOLVED = "resolved"
    MISSING_REPOSITORY = "missing_repository"
    INVALID_REPOSITORY = "invalid_repository"
    INVALID_REMOTE = "invalid_remote"
    REPO_MISMATCH = "repo_mismatch"
    SPLIT_MISMATCH = "split_mismatch"
    DIRTY_WORKTREE = "dirty_worktree"
    MISSING_OBJECT = "missing_object"
    STALE_STATE = "stale_state"
    UNAVAILABLE_PARENT = "unavailable_parent"
    GIT_ERROR = "git_error"
    METADATA_INVALID = "metadata_invalid"


class ParentPolicy(str, Enum):
    NONE = "none"
    SINGLE_PARENT = "single_parent"
    FIRST_PARENT = "first_parent"


class ChangeKind(str, Enum):
    ADD = "add"
    MODIFY = "modify"
    DELETE = "delete"
    RENAME = "rename"
    COPY = "copy"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class ResolutionRow:
    sha: str
    repo: str
    remote_url: str
    status: ResolutionStatus
    parent_count: int | None = None
    parent_sha: str | None = None
    parent_policy: ParentPolicy = ParentPolicy.NONE
    change_kinds: tuple[ChangeKind, ...] = ()
    cache_path: None = None


@dataclass(frozen=True, slots=True)
class _GitResult:
    returncode: int
    stdout: str
    stderr: str


_CHANGE_CODES: Final[Mapping[str, ChangeKind]] = {
    "A": ChangeKind.ADD,
    "M": ChangeKind.MODIFY,
    "D": ChangeKind.DELETE,
    "R": ChangeKind.RENAME,
    "C": ChangeKind.COPY,
}

_FAILED_METADATA_STATUS: Final[Mapping[MetadataReason, ResolutionStatus]] = {
    MetadataReason.UNKNOWN_SHA: ResolutionStatus.MISSING_OBJECT,
    MetadataReason.RAW_SHA_MISSING: ResolutionStatus.MISSING_OBJECT,
    MetadataReason.REPO_MISMATCH: ResolutionStatus.REPO_MISMATCH,
    MetadataReason.SOURCE_REPO_MISMATCH: ResolutionStatus.REPO_MISMATCH,
    MetadataReason.CROSS_SPLIT_SHA: ResolutionStatus.REPO_MISMATCH,
    MetadataReason.SPLIT_MISMATCH: ResolutionStatus.SPLIT_MISMATCH,
    MetadataReason.MALFORMED_JSON: ResolutionStatus.METADATA_INVALID,
    MetadataReason.DUPLICATE_POOL_SHA: ResolutionStatus.METADATA_INVALID,
    MetadataReason.DUPLICATE_RAW_SHA: ResolutionStatus.METADATA_INVALID,
    MetadataReason.INVALID_REMOTE: ResolutionStatus.METADATA_INVALID,
}


def _git(repository: Path, *arguments: str) -> _GitResult:
    completed = subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=False,
        capture_output=True,
        text=True,
    )
    return _GitResult(completed.returncode, completed.stdout.strip(), completed.stderr.strip())


def _resolution(record: MetadataRecord, status: ResolutionStatus) -> ResolutionRow:
    return ResolutionRow(record.sha, record.repo, record.remote_url, status)


def _change_kind(line: str) -> ChangeKind:
    code = re.split(r"\s+", line, maxsplit=1)[0][:1]
    return _CHANGE_CODES.get(code, ChangeKind.OTHER)


def resolve_repository_record(record: MetadataRecord, repository: Path) -> ResolutionRow:
    if not repository.exists():
        return _resolution(record, ResolutionStatus.MISSING_REPOSITORY)
    bare = _git(repository, "rev-parse", "--is-bare-repository")
    if bare.returncode != 0:
        return _resolution(record, ResolutionStatus.INVALID_REPOSITORY)
    remote = _git(repository, "remote", "get-url", "origin")
    canonical_remote = canonical_github_remote(remote.stdout) if remote.returncode == 0 else None
    if canonical_remote is None:
        return _resolution(record, ResolutionStatus.INVALID_REMOTE)
    if canonical_remote != record.remote_url:
        return _resolution(record, ResolutionStatus.REPO_MISMATCH)
    if bare.stdout == "false":
        worktree = _git(repository, "status", "--porcelain", "--untracked-files=all")
        if worktree.returncode != 0:
            return _resolution(record, ResolutionStatus.GIT_ERROR)
        if worktree.stdout:
            return _resolution(record, ResolutionStatus.DIRTY_WORKTREE)
    if _git(repository, "cat-file", "-e", f"{record.sha}^{{commit}}").returncode != 0:
        return _resolution(record, ResolutionStatus.MISSING_OBJECT)
    parents_result = _git(repository, "rev-list", "--parents", "-n", "1", record.sha)
    tokens = parents_result.stdout.split()
    if parents_result.returncode != 0 or not tokens:
        return _resolution(record, ResolutionStatus.GIT_ERROR)
    if tokens[0] != record.sha:
        return _resolution(record, ResolutionStatus.STALE_STATE)
    parents = tokens[1:]
    if not parents:
        return ResolutionRow(
            record.sha, record.repo, record.remote_url, ResolutionStatus.UNAVAILABLE_PARENT, 0
        )
    policy = ParentPolicy.FIRST_PARENT if len(parents) > 1 else ParentPolicy.SINGLE_PARENT
    parent = parents[0]
    if _git(repository, "cat-file", "-e", f"{parent}^{{commit}}").returncode != 0:
        return ResolutionRow(
            record.sha,
            record.repo,
            record.remote_url,
            ResolutionStatus.UNAVAILABLE_PARENT,
            len(parents),
            parent,
            policy,
        )
    changes = _git(
        repository,
        "diff-tree",
        "--no-commit-id",
        "--name-status",
        "-r",
        "-M",
        parent,
        record.sha,
    )
    if changes.returncode != 0:
        return _resolution(record, ResolutionStatus.GIT_ERROR)
    kinds = tuple(
        sorted(
            {_change_kind(line) for line in changes.stdout.splitlines() if line},
            key=lambda kind: kind.value,
        )
    )
    return ResolutionRow(
        record.sha,
        record.repo,
        record.remote_url,
        ResolutionStatus.RESOLVED,
        len(parents),
        parent,
        policy,
        kinds,
    )


def resolve_cached_sources(
    records: Sequence[MetadataRecord],
    cache_root: Path,
) -> tuple[ResolutionRow, ...]:
    rows: list[ResolutionRow] = []
    for record in records:
        if record.reason is MetadataReason.OK:
            owner, name = record.repo.split("/", maxsplit=1)
            rows.append(resolve_repository_record(record, cache_root / owner / f"{name}.git"))
            continue
        status = _FAILED_METADATA_STATUS.get(record.reason, ResolutionStatus.METADATA_INVALID)
        rows.append(_resolution(record, status))
    return tuple(rows)
