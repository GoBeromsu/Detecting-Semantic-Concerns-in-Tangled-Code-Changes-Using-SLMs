from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import pytest

from __test__.conftest import DATASET_DIR
from structural_validity.provenance import (
    ChangeKind,
    MetadataPaths,
    MetadataReason,
    MetadataRecord,
    MetadataValidationError,
    ParentPolicy,
    ResolutionRow,
    ResolutionStatus,
    canonical_github_remote,
    load_metadata_manifest,
    resolve_cached_sources,
    resolve_repository_record,
)
from structural_validity.provenance_cli import CSV_FIELDS, main as provenance_cli_main


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _metadata_paths(tmp_path: Path, shas_json: str = '["a"]') -> MetadataPaths:
    _ = tmp_path.mkdir(parents=True, exist_ok=True)
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    pool = tmp_path / "pool.csv"
    raw = tmp_path / "raw.csv"
    split = tmp_path / "split.json"
    tangled_fields = ("commit_message", "diff", "concern_count", "shas", "types", "repo")
    _write_csv(
        train,
        tangled_fields,
        [
            {
                "commit_message": "m",
                "diff": "[]",
                "concern_count": "1",
                "shas": shas_json,
                "types": '["fix"]',
                "repo": "owner/repo",
            },
            {
                "commit_message": "u",
                "diff": "[]",
                "concern_count": "1",
                "shas": '["unknown"]',
                "types": '["fix"]',
                "repo": "owner/repo",
            },
            {
                "commit_message": "x",
                "diff": "[]",
                "concern_count": "1",
                "shas": '["b"]',
                "types": '["fix"]',
                "repo": "owner/repo",
            },
        ],
    )
    _write_csv(test, tangled_fields, [])
    _write_csv(
        pool,
        ("repo", "annotated_type", "masked_commit_message", "git_diff", "sha", "token_count"),
        [
            {
                "repo": "owner/repo",
                "annotated_type": "fix",
                "masked_commit_message": "m",
                "git_diff": "d",
                "sha": "a",
                "token_count": "1",
            },
            {
                "repo": "other/repo",
                "annotated_type": "fix",
                "masked_commit_message": "x",
                "git_diff": "d",
                "sha": "b",
                "token_count": "1",
            },
        ],
    )
    _write_csv(
        raw,
        (
            "commit_message",
            "sha",
            "type",
            "commit_url",
            "annotated_type",
            "masked_commit_message",
            "git_diff",
        ),
        [
            {
                "commit_message": "m",
                "sha": "a",
                "type": "fix",
                "commit_url": "https://github.com/owner/repo/commit/a",
                "annotated_type": "fix",
                "masked_commit_message": "m",
                "git_diff": "d",
            },
            {
                "commit_message": "x",
                "sha": "b",
                "type": "fix",
                "commit_url": "https://github.com/other/repo/commit/b",
                "annotated_type": "fix",
                "masked_commit_message": "x",
                "git_diff": "d",
            },
        ],
    )
    _ = split.write_text(
        json.dumps({"train": ["owner/repo"], "test": ["test/repo"]}),
        encoding="utf-8",
    )
    return MetadataPaths(train, test, pool, raw, split)


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
    repository = tmp_path / "work"
    _ = repository.mkdir()
    _ = _git(repository, "init", "-b", "main")
    _ = _git(repository, "config", "user.email", "fixture@example.com")
    _ = _git(repository, "config", "user.name", "Fixture")
    _ = _git(repository, "remote", "add", "origin", "git@github.com:owner/repo.git")
    tracked = repository / "old.py"
    _ = tracked.write_text("one\n", encoding="utf-8")
    root = _commit(repository, "parent deadbeef and resolved according to fake logs")
    _ = tracked.write_text("one\ntwo\n", encoding="utf-8")
    ordinary = _commit(repository, "ordinary")
    _ = _git(repository, "mv", "old.py", "new.py")
    renamed = _commit(repository, "rename")
    _ = (repository / "new.py").unlink()
    deleted = _commit(repository, "delete")
    _ = _git(repository, "checkout", "-b", "side", ordinary)
    _ = (repository / "side.py").write_text("side\n", encoding="utf-8")
    _ = _commit(repository, "side")
    _ = _git(repository, "checkout", "main")
    _ = (repository / "main.py").write_text("main\n", encoding="utf-8")
    _ = _commit(repository, "main")
    _ = _git(repository, "merge", "--no-ff", "side", "-m", "merge")
    merged = _git(repository, "rev-parse", "HEAD")
    return repository, {
        "root": root,
        "ordinary": ordinary,
        "renamed": renamed,
        "deleted": deleted,
        "merged": merged,
    }


def _cli_arguments(paths: MetadataPaths, output: Path) -> list[str]:
    return [
        "--train-csv",
        str(paths.train_csv),
        "--test-csv",
        str(paths.test_csv),
        "--pool-csv",
        str(paths.pool_csv),
        "--raw-csv",
        str(paths.raw_csv),
        "--repo-split-json",
        str(paths.repo_split_json),
        "--output",
        str(output),
    ]


def test_load_metadata_manifest_when_rows_include_failures_preserves_reason_codes(
    tmp_path: Path,
) -> None:
    manifest = load_metadata_manifest(_metadata_paths(tmp_path))

    records = {record.sha: record for record in manifest.records}
    assert records["a"].reason is MetadataReason.OK
    assert records["a"].remote_url == "https://github.com/owner/repo.git"
    assert records["unknown"].reason is MetadataReason.UNKNOWN_SHA
    assert records["b"].reason is MetadataReason.REPO_MISMATCH


def test_load_metadata_manifest_when_shas_are_malformed_rejects_at_boundary(
    tmp_path: Path,
) -> None:
    paths = _metadata_paths(tmp_path, shas_json="not-json")

    with pytest.raises(MetadataValidationError) as captured:
        _ = load_metadata_manifest(paths)

    assert captured.value.reason is MetadataReason.MALFORMED_JSON


def test_committed_metadata_manifest_reports_seed43_counts() -> None:
    paths = MetadataPaths(
        DATASET_DIR / "tangled_ccs_dataset_train.csv",
        DATASET_DIR / "tangled_ccs_dataset_test.csv",
        DATASET_DIR / "repo_grouped_pool.csv",
        DATASET_DIR / "CCS Dataset.csv",
        DATASET_DIR / "repo_split.json",
    )

    manifest = load_metadata_manifest(paths)

    assert manifest.counts.total_rows == 1750
    assert manifest.counts.multiconcern_rows == 1400
    assert manifest.counts.sha_references == 5250
    assert manifest.counts.unique_shas == 1081
    assert manifest.counts.repositories == 36
    assert {record.reason for record in manifest.records} == {MetadataReason.OK}


@pytest.mark.parametrize(
    ("remote", "canonical"),
    [
        ("git@github.com:Owner/Repo.git", "https://github.com/Owner/Repo.git"),
        ("ssh://git@github.com/Owner/Repo.git", "https://github.com/Owner/Repo.git"),
        ("https://github.com/Owner/Repo/commit/abc", "https://github.com/Owner/Repo.git"),
    ],
)
def test_canonical_github_remote_when_supported_form_is_given_normalizes(
    remote: str,
    canonical: str,
) -> None:
    assert canonical_github_remote(remote) == canonical


def test_resolve_cached_sources_when_local_history_varies_records_parent_and_change_states(
    tmp_path: Path,
) -> None:
    repository, shas = _history(tmp_path)
    cache = tmp_path / "cache"
    mirror = cache / "owner" / "repo.git"
    _ = mirror.parent.mkdir(parents=True)
    _ = subprocess.run(
        ("git", "clone", "--bare", str(repository), str(mirror)),
        check=True,
        capture_output=True,
    )
    _ = _git(mirror, "remote", "set-url", "origin", "https://github.com/owner/repo.git")
    paths = _metadata_paths(tmp_path / "metadata")
    base_record = load_metadata_manifest(paths).records[0]
    rows: list[ResolutionRow] = []
    for name in ("root", "ordinary", "renamed", "deleted", "merged"):
        rows.extend(resolve_cached_sources((base_record.with_sha(shas[name]),), cache))

    resolved: dict[str, ResolutionRow] = {row.sha: row for row in rows}
    assert resolved[shas["root"]].status is ResolutionStatus.UNAVAILABLE_PARENT
    assert resolved[shas["ordinary"]].parent_policy is ParentPolicy.SINGLE_PARENT
    assert ChangeKind.RENAME in resolved[shas["renamed"]].change_kinds
    assert ChangeKind.DELETE in resolved[shas["deleted"]].change_kinds
    assert resolved[shas["merged"]].parent_count == 2
    assert resolved[shas["merged"]].parent_policy is ParentPolicy.FIRST_PARENT
    assert all(row.cache_path is None for row in rows)


def test_resolve_repository_record_when_state_is_adversarial_uses_git_objects(
    tmp_path: Path,
) -> None:
    repository, shas = _history(tmp_path)
    record: MetadataRecord = load_metadata_manifest(
        _metadata_paths(tmp_path / "metadata")
    ).records[0]
    resolved = resolve_repository_record(record.with_sha(shas["ordinary"]), repository)
    missing = resolve_repository_record(record.with_sha("f" * 40), repository)
    stale = resolve_repository_record(record.with_sha(shas["ordinary"][:12]), repository)
    _ = (repository / "dirty.txt").write_text("dirty\n", encoding="utf-8")
    dirty = resolve_repository_record(record.with_sha(shas["ordinary"]), repository)
    _ = _git(repository, "remote", "set-url", "origin", "https://github.com/wrong/repo.git")
    mismatch = resolve_repository_record(record.with_sha(shas["ordinary"]), repository)

    assert resolved.status is ResolutionStatus.RESOLVED
    assert missing.status is ResolutionStatus.MISSING_OBJECT
    assert stale.status is ResolutionStatus.STALE_STATE
    assert dirty.status is ResolutionStatus.DIRTY_WORKTREE
    assert mismatch.status is ResolutionStatus.REPO_MISMATCH


def test_provenance_cli_when_metadata_only_writes_deterministic_cache_neutral_csv(
    tmp_path: Path,
) -> None:
    paths = _metadata_paths(tmp_path / "metadata")
    output = tmp_path / "source_resolution.csv"
    arguments = _cli_arguments(paths, output)

    assert provenance_cli_main(arguments) == 0
    first_bytes = output.read_bytes()
    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert tuple(rows[0]) == CSV_FIELDS
    assert [row["sha"] for row in rows] == ["a", "b", "unknown"]
    assert {row["resolution_status"] for row in rows} == {"metadata_only"}
    assert all("cache" not in column for column in rows[0])

    assert provenance_cli_main([*arguments, "--force"]) == 0
    assert output.read_bytes() == first_bytes


def test_provenance_cli_when_output_exists_refuses_without_partial_file_and_cache_is_reason_coded(
    tmp_path: Path,
) -> None:
    paths = _metadata_paths(tmp_path / "metadata")
    output = tmp_path / "source_resolution.csv"
    _ = output.write_text("existing\n", encoding="utf-8")
    arguments = _cli_arguments(paths, output)

    with pytest.raises(SystemExit) as captured:
        _ = provenance_cli_main(arguments)
    assert captured.value.code == 2
    assert output.read_text(encoding="utf-8") == "existing\n"
    assert tuple(tmp_path.glob(f".{output.name}.*.tmp")) == ()

    cache = tmp_path / "empty-cache"
    _ = cache.mkdir()
    assert provenance_cli_main([*arguments, "--cache", str(cache), "--force"]) == 0
    with output.open(encoding="utf-8", newline="") as handle:
        statuses = {
            row["sha"]: row["resolution_status"] for row in csv.DictReader(handle)
        }
    assert statuses == {
        "a": "missing_repository",
        "b": "repo_mismatch",
        "unknown": "missing_object",
    }
