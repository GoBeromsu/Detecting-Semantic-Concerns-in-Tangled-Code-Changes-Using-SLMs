"""Version-correct before/after file snapshots for each changed path.

Both sides of every changed path are reconstructed from the object store of a
bare mirror -- never from unified diff text, which carries only hunks and would
silently truncate any file a later AST pass tries to parse. No working tree is
checked out; rename and copy status comes from `diff-tree -M -C`, so a renamed
file keeps its before side under the old path.

Every request produces a row. `ABSENT` marks the legitimately missing side of an
added or deleted file and is kept apart from the failure codes
(`missing_revision`, `missing_blob`, `submodule`, `binary_generated`,
`unresolved_path`, `git_error`), so a later denominator can partition the whole
set without dropping anything.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final

from .grammar_detect import classify_file
from .grammar_runtime import TreeSitterRuntime
from .grammar_types import FileRole, GrammarRuntime
from .snapshot_cache import AUTO_PARENT, SnapshotCache
from .snapshot_git import (
    GitRunner,
    SubprocessGitRunner,
    change_kind_of,
    first_parent,
    list_changes,
    read_blob,
    revision_exists,
)
from .snapshot_types import (
    ABSENT_MODE,
    SUBMODULE_MODE,
    ChangeKind,
    CommitSnapshot,
    RawChange,
    SideStatus,
    SnapshotRow,
    SnapshotSide,
    SnapshotStatus,
)

EMPTY_TREE: Final = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
"""Git's well-known empty tree, used to diff a root commit that has no parent."""

_FAILING_SIDES: Final = {
    SideStatus.SUBMODULE: SnapshotStatus.SUBMODULE,
    SideStatus.MISSING_BLOB: SnapshotStatus.MISSING_BLOB,
    SideStatus.BINARY_GENERATED: SnapshotStatus.BINARY_GENERATED,
}


def mirror_path(mirror_root: Path, repo: str) -> Path:
    """Locate the bare mirror, matching the provenance lane's `owner/name.git`."""
    owner, _, name = repo.partition("/")
    return mirror_root / owner / f"{name}.git"


def _side(
    runner: GitRunner,
    repository: Path,
    cache: SnapshotCache,
    runtime: GrammarRuntime,
    path: str | None,
    mode: str,
    blob_id: str,
) -> SnapshotSide:
    if mode == ABSENT_MODE or path is None:
        return SnapshotSide(SideStatus.ABSENT)
    if mode == SUBMODULE_MODE:
        return SnapshotSide(SideStatus.SUBMODULE, blob_id, mode)
    payload = cache.load_blob(blob_id)
    if payload is None:
        payload = read_blob(runner, repository, blob_id)
        if payload is None:
            return SnapshotSide(SideStatus.MISSING_BLOB, blob_id, mode)
        _ = cache.store_blob(blob_id, payload)
    role = classify_file(path, payload, runtime=runtime).role
    digest = hashlib.sha256(payload).hexdigest()
    status = SideStatus.BINARY_GENERATED if role is FileRole.BINARY_GENERATED else SideStatus.PRESENT
    return SnapshotSide(status, blob_id, mode, len(payload), digest, role)


def _row_status(path: str | None, before: SnapshotSide, after: SnapshotSide) -> SnapshotStatus:
    if path is None:
        return SnapshotStatus.UNRESOLVED_PATH
    for side in (before, after):
        failure = _FAILING_SIDES.get(side.status)
        if failure is not None:
            return failure
    return SnapshotStatus.RESOLVED


def _row(
    runner: GitRunner,
    repository: Path,
    cache: SnapshotCache,
    runtime: GrammarRuntime,
    repo: str,
    sha: str,
    parent: str,
    change: RawChange,
) -> SnapshotRow:
    path = change.new_path or change.old_path
    before = _side(runner, repository, cache, runtime, change.old_path, change.old_mode, change.old_blob)
    after = _side(runner, repository, cache, runtime, change.new_path, change.new_mode, change.new_blob)
    return SnapshotRow(
        repo,
        sha,
        parent,
        change_kind_of(change),
        path or "",
        change.old_path,
        change.new_path,
        before,
        after,
        _row_status(path, before, after),
    )


def _failed(repo: str, sha: str, parent: str, status: SnapshotStatus) -> CommitSnapshot:
    return CommitSnapshot(repo, sha, parent, status, ())


def resolve_commit_snapshot(
    repo: str,
    sha: str,
    *,
    mirror_root: Path,
    cache: SnapshotCache,
    runner: GitRunner | None = None,
    runtime: GrammarRuntime | None = None,
    parent: str | None = None,
) -> CommitSnapshot:
    """Resolve both sides of every path changed by `sha`, caching the result.

    Passing `parent` pins the comparison; leaving it None applies the
    first-parent policy, so a merge commit is compared against its mainline.
    """
    parent_key = parent or AUTO_PARENT
    cached = cache.read(repo, sha, parent_key)
    if cached is not None:
        return cached
    active_runner = runner if runner is not None else SubprocessGitRunner()
    active_runtime = runtime if runtime is not None else TreeSitterRuntime()
    repository = mirror_path(mirror_root, repo)
    if not revision_exists(active_runner, repository, sha):
        return _failed(repo, sha, parent or "", SnapshotStatus.MISSING_REVISION)
    resolved_parent = parent if parent is not None else first_parent(active_runner, repository, sha)[0]
    if parent is not None and not revision_exists(active_runner, repository, parent):
        return _failed(repo, sha, parent, SnapshotStatus.MISSING_REVISION)
    changes = list_changes(active_runner, repository, resolved_parent or EMPTY_TREE, sha)
    if changes is None:
        return _failed(repo, sha, resolved_parent or "", SnapshotStatus.GIT_ERROR)
    rows = tuple(
        _row(active_runner, repository, cache, active_runtime, repo, sha, resolved_parent or "", change)
        for change in changes
    )
    snapshot = CommitSnapshot(repo, sha, resolved_parent or "", SnapshotStatus.RESOLVED, rows)
    _ = cache.write(snapshot, parent_key)
    return snapshot


def resolve_commit_snapshots(
    commits: tuple[tuple[str, str], ...],
    *,
    mirror_root: Path,
    cache: SnapshotCache,
    runner: GitRunner | None = None,
    runtime: GrammarRuntime | None = None,
) -> tuple[CommitSnapshot, ...]:
    """Resolve many commits, reason-coding failures instead of aborting the run."""
    active_runner = runner if runner is not None else SubprocessGitRunner()
    active_runtime = runtime if runtime is not None else TreeSitterRuntime()
    return tuple(
        resolve_commit_snapshot(
            repo,
            sha,
            mirror_root=mirror_root,
            cache=cache,
            runner=active_runner,
            runtime=active_runtime,
        )
        for repo, sha in commits
    )


__all__ = [
    "EMPTY_TREE",
    "ChangeKind",
    "CommitSnapshot",
    "SideStatus",
    "SnapshotCache",
    "SnapshotRow",
    "SnapshotSide",
    "SnapshotStatus",
    "SubprocessGitRunner",
    "mirror_path",
    "resolve_commit_snapshot",
    "resolve_commit_snapshots",
]
