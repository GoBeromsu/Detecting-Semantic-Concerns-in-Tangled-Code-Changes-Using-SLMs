from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from structural_validity.grammar_types import FileRole, ParseScore, RuntimeMetadata
from structural_validity.snapshot_cache import AUTO_PARENT, SnapshotCache, snapshot_json
from structural_validity.snapshot_git import (
    GitOutput,
    GitRunner,
    SubprocessGitRunner,
    parse_raw_changes,
)
from structural_validity.snapshot_types import SnapshotCacheError
from structural_validity.source_snapshots import (
    ChangeKind,
    CommitSnapshot,
    SideStatus,
    SnapshotStatus,
    mirror_path,
    resolve_commit_snapshot,
    resolve_commit_snapshots,
)

REPO = "owner/repo"
ALPHA_V1 = "def alpha():\n    return 1\n\n\ndef helper():\n    return 2\n"
ALPHA_V2 = "def alpha():\n    return 11\n\n\ndef helper():\n    return 2\n"
BETA_V2 = "def alpha():\n    return 11\n\n\ndef helper():\n    return 22\n"
GITLINK_SHA = "1" * 40


class _StubRuntime:
    """Keeps classification hermetic; roles come from the rule table, not a parse."""

    def metadata(self) -> RuntimeMetadata:
        return RuntimeMetadata("stub", "stub", 15, 13)

    def grammar_abi(self, grammar: str) -> int | None:
        del grammar
        return 15

    def score(self, grammar: str, source: bytes) -> ParseScore | None:
        del grammar, source
        return None


@dataclass(slots=True)
class _CountingRunner:
    inner: SubprocessGitRunner = field(default_factory=SubprocessGitRunner)
    calls: list[tuple[str, ...]] = field(default_factory=list)

    def run(self, repository: Path, *arguments: str) -> GitOutput:
        self.calls.append(arguments)
        return self.inner.run(repository, *arguments)


@dataclass(slots=True)
class _ForbiddenRunner:
    def run(self, repository: Path, *arguments: str) -> GitOutput:
        del repository
        raise AssertionError(f"warm cache must not invoke git: {arguments}")


@dataclass(slots=True)
class _BlobLosingRunner:
    inner: SubprocessGitRunner = field(default_factory=SubprocessGitRunner)

    def run(self, repository: Path, *arguments: str) -> GitOutput:
        if arguments[:2] == ("cat-file", "blob"):
            return GitOutput(128, b"", "fatal: Not a valid object name")
        return self.inner.run(repository, *arguments)


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _commit(repository: Path, message: str) -> str:
    _ = _git(repository, "add", "-A")
    _ = _git(repository, "commit", "-m", message)
    return _git(repository, "rev-parse", "HEAD")


def _history(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """Build one work tree covering every change kind, then mirror it bare."""
    work = tmp_path / "work"
    _ = work.mkdir()
    _ = _git(work, "init", "-b", "main")
    _ = _git(work, "config", "user.email", "fixture@example.com")
    _ = _git(work, "config", "user.name", "Fixture")
    _ = (work / "alpha.py").write_text(ALPHA_V1, encoding="utf-8")
    _ = (work / "notes.txt").write_text("notes\n", encoding="utf-8")
    root = _commit(work, "root")
    _ = (work / "alpha.py").write_text(ALPHA_V2, encoding="utf-8")
    modified = _commit(work, "modify")
    _ = _git(work, "checkout", "-b", "side", modified)
    _ = (work / "side.py").write_text("def side():\n    return 3\n", encoding="utf-8")
    side = _commit(work, "side")
    _ = _git(work, "checkout", "main")
    _ = _git(work, "mv", "alpha.py", "beta.py")
    renamed = _commit(work, "rename")
    _ = (work / "beta.py").write_text(BETA_V2, encoding="utf-8")
    _ = (work / "gamma.py").write_text(ALPHA_V2, encoding="utf-8")
    copied = _commit(work, "copy")
    _ = (work / "gamma.py").unlink()
    deleted = _commit(work, "delete")
    _ = (work / "assets").mkdir()
    _ = (work / "assets" / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00binary")
    binary = _commit(work, "binary")
    _ = _git(work, "merge", "--no-ff", "side", "-m", "merge")
    merged = _git(work, "rev-parse", "HEAD")
    _ = _git(work, "update-index", "--add", "--cacheinfo", f"160000,{GITLINK_SHA},vendor/lib")
    _ = _git(work, "commit", "-m", "submodule")
    submodule = _git(work, "rev-parse", "HEAD")
    mirror = tmp_path / "mirrors" / "owner" / "repo.git"
    _ = mirror.parent.mkdir(parents=True)
    _ = subprocess.run(("git", "clone", "--bare", str(work), str(mirror)), check=True, capture_output=True)
    return tmp_path / "mirrors", {
        "root": root,
        "modified": modified,
        "side": side,
        "renamed": renamed,
        "copied": copied,
        "deleted": deleted,
        "binary": binary,
        "merged": merged,
        "submodule": submodule,
    }


@pytest.fixture(name="mirror", scope="session")
def mirror_fixture(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, str]]:
    """Built once: nothing under test ever writes to a mirror, only reads from it."""
    return _history(tmp_path_factory.mktemp("fixture"))


@pytest.fixture(name="history")
def history_fixture(
    mirror: tuple[Path, dict[str, str]],
    tmp_path: Path,
) -> tuple[Path, dict[str, str], SnapshotCache]:
    mirror_root, shas = mirror
    return mirror_root, shas, SnapshotCache(tmp_path / "cache")


def _resolve(
    history: tuple[Path, dict[str, str], SnapshotCache],
    key: str,
    runner: GitRunner | None = None,
) -> CommitSnapshot:
    mirror_root, shas, cache = history
    return resolve_commit_snapshot(
        REPO,
        shas[key],
        mirror_root=mirror_root,
        cache=cache,
        runner=runner if runner is not None else SubprocessGitRunner(),
        runtime=_StubRuntime(),
    )


def _row(snapshot: CommitSnapshot, path: str):
    matches = [row for row in snapshot.rows if row.path == path]
    assert matches, f"{path} missing from {[row.path for row in snapshot.rows]}"
    return matches[0]


def test_mirror_path_when_repo_is_owner_slash_name_matches_provenance_layout(tmp_path: Path) -> None:
    assert mirror_path(tmp_path, "owner/repo") == tmp_path / "owner" / "repo.git"


def test_snapshot_when_commit_is_a_root_has_absent_before_for_every_added_file(
    history: tuple[Path, dict[str, str], SnapshotCache],
) -> None:
    snapshot = _resolve(history, "root")
    assert snapshot.status is SnapshotStatus.RESOLVED
    alpha = _row(snapshot, "alpha.py")
    assert alpha.change_kind is ChangeKind.ADD
    assert alpha.before.status is SideStatus.ABSENT
    assert alpha.before.blob_id is None
    assert alpha.after.status is SideStatus.PRESENT
    assert alpha.status is SnapshotStatus.RESOLVED


def test_snapshot_when_file_is_modified_reconstructs_whole_files_not_diff_hunks(
    history: tuple[Path, dict[str, str], SnapshotCache],
) -> None:
    _, _, cache = history
    snapshot = _resolve(history, "modified")
    row = _row(snapshot, "alpha.py")
    assert row.change_kind is ChangeKind.MODIFY
    assert row.before.blob_id is not None and row.after.blob_id is not None
    before = cache.load_blob(row.before.blob_id)
    after = cache.load_blob(row.after.blob_id)
    # A unified diff would carry only the changed line; both sides are complete.
    assert before is not None and before.decode() == ALPHA_V1
    assert after is not None and after.decode() == ALPHA_V2
    assert row.before.size == len(ALPHA_V1.encode())
    assert row.after.role is FileRole.SOURCE_AST


def test_snapshot_when_file_is_renamed_keeps_before_side_under_the_old_path(
    history: tuple[Path, dict[str, str], SnapshotCache],
) -> None:
    _, _, cache = history
    snapshot = _resolve(history, "renamed")
    row = _row(snapshot, "beta.py")
    assert row.change_kind is ChangeKind.RENAME
    assert row.old_path == "alpha.py"
    assert row.new_path == "beta.py"
    assert row.before.status is SideStatus.PRESENT
    assert row.before.blob_id is not None
    payload = cache.load_blob(row.before.blob_id)
    assert payload is not None and payload.decode() == ALPHA_V2


def test_snapshot_when_file_is_copied_keeps_the_source_before_side(
    history: tuple[Path, dict[str, str], SnapshotCache],
) -> None:
    _, _, cache = history
    snapshot = _resolve(history, "copied")
    row = _row(snapshot, "gamma.py")
    assert row.change_kind is ChangeKind.COPY
    assert row.old_path == "beta.py"
    assert row.before.status is SideStatus.PRESENT
    assert row.before.blob_id is not None
    payload = cache.load_blob(row.before.blob_id)
    assert payload is not None and payload.decode() == ALPHA_V2


def test_snapshot_when_file_is_deleted_has_present_before_and_absent_after(
    history: tuple[Path, dict[str, str], SnapshotCache],
) -> None:
    snapshot = _resolve(history, "deleted")
    row = _row(snapshot, "gamma.py")
    assert row.change_kind is ChangeKind.DELETE
    assert row.before.status is SideStatus.PRESENT
    assert row.after.status is SideStatus.ABSENT
    assert row.after.blob_id is None
    assert row.status is SnapshotStatus.RESOLVED


def test_snapshot_when_commit_is_a_merge_compares_against_the_first_parent(
    history: tuple[Path, dict[str, str], SnapshotCache],
) -> None:
    _, shas, _ = history
    snapshot = _resolve(history, "merged")
    assert snapshot.parent_sha == shas["binary"]
    # Only the side branch's file arrives through the mainline comparison.
    assert [row.path for row in snapshot.rows] == ["side.py"]


def test_snapshot_when_path_is_a_submodule_is_reason_coded_and_never_read(
    history: tuple[Path, dict[str, str], SnapshotCache],
) -> None:
    _, _, cache = history
    snapshot = _resolve(history, "submodule")
    row = _row(snapshot, "vendor/lib")
    assert row.after.status is SideStatus.SUBMODULE
    assert row.after.mode == "160000"
    assert row.status is SnapshotStatus.SUBMODULE
    assert cache.load_blob(GITLINK_SHA) is None


def test_snapshot_when_content_is_binary_is_reason_coded_not_treated_as_source(
    history: tuple[Path, dict[str, str], SnapshotCache],
) -> None:
    snapshot = _resolve(history, "binary")
    row = _row(snapshot, "assets/logo.png")
    assert row.after.status is SideStatus.BINARY_GENERATED
    assert row.after.role is FileRole.BINARY_GENERATED
    assert row.status is SnapshotStatus.BINARY_GENERATED
    assert row.after.sha256 is not None


def test_snapshot_when_cache_is_warm_invokes_no_git_and_is_byte_identical(
    history: tuple[Path, dict[str, str], SnapshotCache],
) -> None:
    mirror_root, shas, cache = history
    counting = _CountingRunner()
    cold = _resolve(history, "copied", runner=counting)
    assert counting.calls, "cold resolution must consult git"
    warm = resolve_commit_snapshot(
        REPO,
        shas["copied"],
        mirror_root=mirror_root,
        cache=cache,
        runner=_ForbiddenRunner(),
        runtime=_StubRuntime(),
    )
    assert warm.from_cache is True
    assert cold.from_cache is False
    assert snapshot_json(warm) == snapshot_json(cold)
    assert cache.manifest_path(REPO, shas["copied"], AUTO_PARENT).exists()


def test_snapshot_when_repository_is_absent_reports_missing_revision(tmp_path: Path) -> None:
    snapshot = resolve_commit_snapshot(
        REPO,
        "0" * 40,
        mirror_root=tmp_path / "mirrors",
        cache=SnapshotCache(tmp_path / "cache"),
        runner=SubprocessGitRunner(),
        runtime=_StubRuntime(),
    )
    assert snapshot.status is SnapshotStatus.MISSING_REVISION
    assert snapshot.rows == ()


def test_snapshot_when_revision_is_unknown_reports_missing_revision(
    history: tuple[Path, dict[str, str], SnapshotCache],
) -> None:
    mirror_root, _, cache = history
    snapshot = resolve_commit_snapshot(
        REPO,
        "0" * 40,
        mirror_root=mirror_root,
        cache=cache,
        runner=SubprocessGitRunner(),
        runtime=_StubRuntime(),
    )
    assert snapshot.status is SnapshotStatus.MISSING_REVISION


def test_snapshot_when_blob_cannot_be_read_reason_codes_the_row_and_keeps_the_rest(
    history: tuple[Path, dict[str, str], SnapshotCache],
) -> None:
    snapshot = _resolve(history, "copied", runner=_BlobLosingRunner())
    assert snapshot.status is SnapshotStatus.RESOLVED
    assert snapshot.rows, "a git object failure must not empty the manifest"
    for row in snapshot.rows:
        assert row.status is SnapshotStatus.MISSING_BLOB
        assert row.after.status is SideStatus.MISSING_BLOB
        assert row.after.blob_id is not None


def test_resolve_many_when_one_repository_is_missing_others_still_resolve(
    history: tuple[Path, dict[str, str], SnapshotCache],
) -> None:
    mirror_root, shas, cache = history
    snapshots = resolve_commit_snapshots(
        (("owner/gone", "0" * 40), (REPO, shas["modified"])),
        mirror_root=mirror_root,
        cache=cache,
        runner=SubprocessGitRunner(),
        runtime=_StubRuntime(),
    )
    assert [snapshot.status for snapshot in snapshots] == [
        SnapshotStatus.MISSING_REVISION,
        SnapshotStatus.RESOLVED,
    ]


def test_snapshot_when_regenerated_into_a_fresh_cache_is_byte_identical(
    mirror: tuple[Path, dict[str, str]],
    tmp_path: Path,
) -> None:
    mirror_root, shas = mirror
    caches = (SnapshotCache(tmp_path / "first"), SnapshotCache(tmp_path / "second"))
    for cache in caches:
        _ = resolve_commit_snapshot(
            REPO,
            shas["copied"],
            mirror_root=mirror_root,
            cache=cache,
            runner=SubprocessGitRunner(),
            runtime=_StubRuntime(),
        )
    digests = {
        hashlib.sha256(cache.manifest_path(REPO, shas["copied"], AUTO_PARENT).read_bytes()).hexdigest()
        for cache in caches
    }
    assert len(digests) == 1


def test_snapshot_when_cached_document_is_corrupt_is_rejected_not_silently_trusted(
    history: tuple[Path, dict[str, str], SnapshotCache],
) -> None:
    _, shas, cache = history
    _ = _resolve(history, "modified")
    path = cache.manifest_path(REPO, shas["modified"], AUTO_PARENT)
    _ = path.write_text('{"schema_version": 1, "repo": "owner/repo"}', encoding="utf-8")
    with pytest.raises(SnapshotCacheError):
        _ = _resolve(history, "modified")


def test_parse_raw_changes_when_path_holds_non_ascii_bytes_survives_nul_framing() -> None:
    payload = (
        b":100644 100644 aaaa bbbb M\x00src/caf\xc3\xa9 space.py\x00"
        b":100644 100644 cccc dddd R100\x00old \xed\x95\x9c.py\x00new \xed\x95\x9c.py\x00"
    )
    changes = parse_raw_changes(payload)
    assert len(changes) == 2
    assert changes[0].new_path == "src/café space.py"
    assert changes[1].old_path == "old 한.py"
    assert changes[1].new_path == "new 한.py"


def test_parse_raw_changes_when_record_is_truncated_stops_without_inventing_paths() -> None:
    assert parse_raw_changes(b":100644 100644 aaaa bbbb R100\x00only-one-path.py\x00") == ()
