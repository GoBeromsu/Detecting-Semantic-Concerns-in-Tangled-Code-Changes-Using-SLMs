"""Typed vocabulary for version-correct before/after file snapshots.

Each changed path yields two independently reason-coded sides. An added file
legitimately has no before side and a deleted file legitimately has no after
side, so ``ABSENT`` is a normal outcome and is kept distinct from the failure
states (``MISSING_BLOB``, ``SUBMODULE``, ``BINARY_GENERATED``). No side is ever
dropped, so a later AST denominator can always partition every request.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

from .grammar_types import FileRole

SUBMODULE_MODE: Final = "160000"
ABSENT_MODE: Final = "000000"


class SideStatus(str, Enum):
    PRESENT = "present"
    ABSENT = "absent"
    MISSING_BLOB = "missing_blob"
    SUBMODULE = "submodule"
    BINARY_GENERATED = "binary_generated"


class SnapshotStatus(str, Enum):
    RESOLVED = "resolved"
    MISSING_REVISION = "missing_revision"
    MISSING_BLOB = "missing_blob"
    SUBMODULE = "submodule"
    BINARY_GENERATED = "binary_generated"
    UNRESOLVED_PATH = "unresolved_path"
    GIT_ERROR = "git_error"


class ChangeKind(str, Enum):
    ADD = "add"
    MODIFY = "modify"
    DELETE = "delete"
    RENAME = "rename"
    COPY = "copy"
    TYPE_CHANGE = "type_change"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class RawChange:
    """One `diff-tree --raw` record, before any blob is read."""

    old_mode: str
    new_mode: str
    old_blob: str
    new_blob: str
    status_code: str
    old_path: str | None
    new_path: str | None


@dataclass(frozen=True, slots=True)
class SnapshotSide:
    status: SideStatus
    blob_id: str | None = None
    mode: str | None = None
    size: int | None = None
    sha256: str | None = None
    role: FileRole | None = None


@dataclass(frozen=True, slots=True)
class SnapshotRow:
    repo: str
    sha: str
    parent_sha: str
    change_kind: ChangeKind
    path: str
    old_path: str | None
    new_path: str | None
    before: SnapshotSide
    after: SnapshotSide
    status: SnapshotStatus


@dataclass(frozen=True, slots=True)
class CommitSnapshot:
    repo: str
    sha: str
    parent_sha: str
    status: SnapshotStatus
    rows: tuple[SnapshotRow, ...]
    from_cache: bool = False


@dataclass(frozen=True, slots=True)
class SnapshotCacheError(Exception):
    path: str
    detail: str


CHANGE_CODES: Final[dict[str, ChangeKind]] = {
    "A": ChangeKind.ADD,
    "M": ChangeKind.MODIFY,
    "D": ChangeKind.DELETE,
    "R": ChangeKind.RENAME,
    "C": ChangeKind.COPY,
    "T": ChangeKind.TYPE_CHANGE,
}
RENAME_CODES: Final = frozenset({"R", "C"})
