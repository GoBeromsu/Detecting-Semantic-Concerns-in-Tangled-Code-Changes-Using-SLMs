"""Read changed-path blobs straight out of a bare mirror.

Every read goes through `diff-tree --raw -z` and `cat-file blob`, so no working
tree is ever checked out and no unified diff text is mistaken for complete
source. Paths arrive NUL-separated, which sidesteps the octal/quoted-path
decoding that the committed-diff lane had to handle by hand.

All Git access is funnelled through an injectable `GitRunner` so a warm-cache
run can assert that it invoked Git zero times.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .snapshot_types import (
    ABSENT_MODE,
    CHANGE_CODES,
    RENAME_CODES,
    ChangeKind,
    RawChange,
)


@dataclass(frozen=True, slots=True)
class GitOutput:
    returncode: int
    stdout: bytes
    stderr: str


class GitRunner(Protocol):
    def run(self, repository: Path, *arguments: str) -> GitOutput: ...


@dataclass(slots=True)
class SubprocessGitRunner:
    """Counts invocations so callers can prove a warm cache touched no Git."""

    invocations: int = 0

    def run(self, repository: Path, *arguments: str) -> GitOutput:
        self.invocations += 1
        completed = subprocess.run(
            ("git", "-C", str(repository), *arguments),
            check=False,
            capture_output=True,
        )
        return GitOutput(
            completed.returncode,
            completed.stdout,
            completed.stderr.decode("utf-8", errors="replace").strip(),
        )


def revision_exists(runner: GitRunner, repository: Path, revision: str) -> bool:
    if not repository.exists():
        return False
    return runner.run(repository, "cat-file", "-e", f"{revision}^{{commit}}").returncode == 0


def _change_kind(status_code: str) -> ChangeKind:
    return CHANGE_CODES.get(status_code[:1], ChangeKind.OTHER)


def parse_raw_changes(payload: bytes) -> tuple[RawChange, ...]:
    """Parse `diff-tree --raw -z` output.

    Records look like ``:<oldmode> <newmode> <oldblob> <newblob> <status>`` in
    one NUL-terminated field, followed by one path field, or two when the status
    is a rename or copy.
    """
    fields = payload.split(b"\x00")
    if fields and fields[-1] == b"":
        # Git terminates the final field, so the split leaves one empty tail.
        # Dropping it is what makes a truncated rename record detectable rather
        # than silently completing itself with an empty second path.
        fields = fields[:-1]
    changes: list[RawChange] = []
    index = 0
    while index < len(fields):
        field = fields[index]
        index += 1
        if not field.startswith(b":"):
            continue
        parts = field[1:].decode("utf-8", errors="replace").split()
        if len(parts) != 5:
            continue
        old_mode, new_mode, old_blob, new_blob, status_code = parts
        wanted = 2 if status_code[:1] in RENAME_CODES else 1
        if index + wanted > len(fields):
            break
        paths = [fields[position].decode("utf-8", errors="surrogateescape") for position in range(index, index + wanted)]
        index += wanted
        old_path = paths[0] if wanted == 2 else (paths[0] if old_mode != ABSENT_MODE else None)
        new_path = paths[1] if wanted == 2 else (paths[0] if new_mode != ABSENT_MODE else None)
        changes.append(RawChange(old_mode, new_mode, old_blob, new_blob, status_code, old_path, new_path))
    return tuple(changes)


def list_changes(
    runner: GitRunner,
    repository: Path,
    parent: str,
    sha: str,
) -> tuple[RawChange, ...] | None:
    """Return the parent->child change records, or None on a Git failure."""
    result = runner.run(
        repository,
        "diff-tree",
        "--no-commit-id",
        "-r",
        "-M",
        "-C",
        "-z",
        "--raw",
        parent,
        sha,
    )
    if result.returncode != 0:
        return None
    return parse_raw_changes(result.stdout)


def read_blob(runner: GitRunner, repository: Path, blob_id: str) -> bytes | None:
    """Return raw blob bytes, or None when the object is absent."""
    result = runner.run(repository, "cat-file", "blob", blob_id)
    if result.returncode != 0:
        return None
    return result.stdout


def first_parent(runner: GitRunner, repository: Path, sha: str) -> tuple[str | None, int] | None:
    """Return (first parent sha, parent count) for a commit, or None on git failure.

    Merge commits keep the first parent, matching the provenance lane's
    recorded `parent_policy=first_parent`; a root commit yields (None, 0).

    A root commit and a failed lookup must not share a return value. Both would
    otherwise fall back to the empty tree, and diffing against the empty tree
    succeeds, so a broken mirror would be reported as a resolved snapshot in
    which every path is freshly added. `None` keeps the failure distinguishable
    the way the provenance lane's `GIT_ERROR` does.
    """
    result = runner.run(repository, "rev-list", "--parents", "-n", "1", sha)
    if result.returncode != 0:
        return None
    tokens = result.stdout.decode("utf-8", errors="replace").split()
    if not tokens or tokens[0] != sha:
        return None
    parents = tokens[1:]
    return ((parents[0], len(parents)) if parents else (None, 0))


def change_kind_of(change: RawChange) -> ChangeKind:
    return _change_kind(change.status_code)
