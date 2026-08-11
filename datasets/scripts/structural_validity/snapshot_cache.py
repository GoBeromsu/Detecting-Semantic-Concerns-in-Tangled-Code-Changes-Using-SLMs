"""Content-addressed cache for resolved commit snapshots.

Two stores share one root. Blob bytes live under ``blobs/<oid[:2]>/<oid>``, so
identical content across repositories and revisions is written once, and the
per-commit metadata lives under ``commits/<repo>/<sha>__<parent>.json``. A warm
run answers entirely from these files, which is what lets the caller assert that
zero Git commands were issued.

The on-disk document is declared once as a TypedDict and both written and read
through the same `TypeAdapter`, so a shape drift is a validation error rather
than a silently mistyped field. Writes are atomic (temp file, fsync, replace)
and JSON is emitted with sorted keys and a fixed indent, so regenerating a
manifest is byte-identical.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypedDict

from pydantic import TypeAdapter, ValidationError

from .grammar_types import FileRole
from .snapshot_types import (
    ChangeKind,
    CommitSnapshot,
    SideStatus,
    SnapshotCacheError,
    SnapshotRow,
    SnapshotSide,
    SnapshotStatus,
)

SCHEMA_VERSION: Final = 1
AUTO_PARENT: Final = "auto"
"""Filename key used when the caller let the first-parent policy pick the parent.

Keeping the key distinct from a pinned parent SHA means a warm lookup needs no
Git call to learn which parent was chosen; the resolved SHA is recorded inside
the document.
"""


class SideDocument(TypedDict):
    status: str
    blob_id: str | None
    mode: str | None
    size: int | None
    sha256: str | None
    role: str | None


class RowDocument(TypedDict):
    change_kind: str
    path: str
    old_path: str | None
    new_path: str | None
    before: SideDocument
    after: SideDocument
    status: str


class SnapshotDocument(TypedDict):
    schema_version: int
    repo: str
    sha: str
    parent_sha: str
    status: str
    rows: list[RowDocument]


_SNAPSHOT_ADAPTER: Final[TypeAdapter[SnapshotDocument]] = TypeAdapter(SnapshotDocument)


def _slug(repo: str) -> str:
    return repo.replace("/", "__")


def _side_json(side: SnapshotSide) -> SideDocument:
    return {
        "status": side.status.value,
        "blob_id": side.blob_id,
        "mode": side.mode,
        "size": side.size,
        "sha256": side.sha256,
        "role": side.role.value if side.role is not None else None,
    }


def _side_from(document: SideDocument) -> SnapshotSide:
    role = document["role"]
    return SnapshotSide(
        SideStatus(document["status"]),
        document["blob_id"],
        document["mode"],
        document["size"],
        document["sha256"],
        FileRole(role) if role is not None else None,
    )


def _row_json(row: SnapshotRow) -> RowDocument:
    return {
        "change_kind": row.change_kind.value,
        "path": row.path,
        "old_path": row.old_path,
        "new_path": row.new_path,
        "before": _side_json(row.before),
        "after": _side_json(row.after),
        "status": row.status.value,
    }


def _row_from(repo: str, sha: str, parent: str, document: RowDocument) -> SnapshotRow:
    return SnapshotRow(
        repo,
        sha,
        parent,
        ChangeKind(document["change_kind"]),
        document["path"],
        document["old_path"],
        document["new_path"],
        _side_from(document["before"]),
        _side_from(document["after"]),
        SnapshotStatus(document["status"]),
    )


def snapshot_document(snapshot: CommitSnapshot) -> SnapshotDocument:
    return {
        "schema_version": SCHEMA_VERSION,
        "repo": snapshot.repo,
        "sha": snapshot.sha,
        "parent_sha": snapshot.parent_sha,
        "status": snapshot.status.value,
        "rows": [_row_json(row) for row in snapshot.rows],
    }


def snapshot_json(snapshot: CommitSnapshot) -> str:
    """Render a snapshot as the exact text stored on disk."""
    return json.dumps(snapshot_document(snapshot), indent=2, sort_keys=True) + "\n"


@dataclass(frozen=True, slots=True)
class SnapshotCache:
    root: Path

    def manifest_path(self, repo: str, sha: str, parent_key: str) -> Path:
        return self.root / "commits" / _slug(repo) / f"{sha}__{parent_key or AUTO_PARENT}.json"

    def blob_path(self, blob_id: str) -> Path:
        return self.root / "blobs" / blob_id[:2] / blob_id

    def has_blob(self, blob_id: str) -> bool:
        return self.blob_path(blob_id).exists()

    def store_blob(self, blob_id: str, payload: bytes) -> Path:
        destination = self.blob_path(blob_id)
        if not destination.exists():
            _write_atomic(destination, payload)
        return destination

    def load_blob(self, blob_id: str) -> bytes | None:
        destination = self.blob_path(blob_id)
        return destination.read_bytes() if destination.exists() else None

    def read(self, repo: str, sha: str, parent_key: str) -> CommitSnapshot | None:
        """Return a cached snapshot, or None when this commit was never resolved."""
        path = self.manifest_path(repo, sha, parent_key)
        if not path.exists():
            return None
        try:
            document = _SNAPSHOT_ADAPTER.validate_json(path.read_bytes())
        except ValidationError as error:
            raise SnapshotCacheError(str(path), f"{error.error_count()} invalid field(s)") from error
        if document["schema_version"] != SCHEMA_VERSION:
            raise SnapshotCacheError(str(path), f"unrecognized schema_version {document['schema_version']}")
        repo_name, commit, parent = document["repo"], document["sha"], document["parent_sha"]
        return CommitSnapshot(
            repo_name,
            commit,
            parent,
            SnapshotStatus(document["status"]),
            tuple(_row_from(repo_name, commit, parent, row) for row in document["rows"]),
            from_cache=True,
        )

    def write(self, snapshot: CommitSnapshot, parent_key: str) -> Path:
        path = self.manifest_path(snapshot.repo, snapshot.sha, parent_key)
        _write_atomic(path, snapshot_json(snapshot).encode("utf-8"))
        return path


def _write_atomic(destination: Path, payload: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        _ = handle.write(payload)
        handle.flush()
        _ = os.fsync(handle.fileno())
    try:
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
