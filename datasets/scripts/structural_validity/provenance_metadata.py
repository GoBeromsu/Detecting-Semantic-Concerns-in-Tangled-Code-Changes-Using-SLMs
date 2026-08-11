from __future__ import annotations

import csv
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

from structural_validity.provenance_json import parse_repo_splits, parse_string_list
from structural_validity.provenance_types import (
    MetadataCounts,
    MetadataManifest,
    MetadataPaths,
    MetadataReason,
    MetadataRecord,
    MetadataValidationError,
)

_GITHUB_PREFIXES: Final[tuple[str, ...]] = (
    "https://github.com/",
    "http://github.com/",
    "ssh://git@github.com/",
    "git@github.com:",
)


def _configure_csv_field_size_limit() -> int:
    try:
        return csv.field_size_limit(sys.maxsize)
    except OverflowError:
        return csv.field_size_limit(2**31 - 1)


_CSV_FIELD_SIZE_LIMIT_PREVIOUS: Final[int] = _configure_csv_field_size_limit()


def canonical_github_remote(remote: str) -> str | None:
    value = remote.strip()
    prefix = next((item for item in _GITHUB_PREFIXES if value.startswith(item)), None)
    if prefix is None:
        return None
    suffix = value[len(prefix) :].split("/commit/", maxsplit=1)[0].rstrip("/")
    suffix = suffix.removesuffix(".git")
    parts = suffix.split("/")
    if len(parts) != 2 or not all(parts):
        return None
    return f"https://github.com/{parts[0]}/{parts[1]}.git"


def _required(row: Mapping[str, str | None], column: str, location: str) -> str:
    value = row.get(column)
    if value is None or value == "":
        raise MetadataValidationError(MetadataReason.MALFORMED_JSON, location, f"missing {column}")
    return value


def _repo_from_remote(remote: str) -> str | None:
    canonical = canonical_github_remote(remote)
    if canonical is None:
        return None
    return canonical.removeprefix("https://github.com/").removesuffix(".git")


def _index_column(path: Path, key: str, value: str) -> dict[str, list[str]]:
    indexed: dict[str, list[str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle), start=2):
            location = f"{path}:{index}"
            indexed.setdefault(_required(row, key, location), []).append(
                _required(row, value, location)
            )
    return indexed


def _index_raw_repos(path: Path) -> dict[str, list[str | None]]:
    indexed: dict[str, list[str | None]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle), start=2):
            location = f"{path}:{index}"
            indexed.setdefault(_required(row, "sha", location), []).append(
                _repo_from_remote(_required(row, "commit_url", location))
            )
    return indexed


def _metadata_reason(
    sha: str,
    splits: set[str],
    expected_repos: set[str],
    pool_repos: Mapping[str, Sequence[str]],
    raw_repos: Mapping[str, Sequence[str | None]],
    split_repos: Mapping[str, frozenset[str]],
) -> MetadataReason:
    pooled = pool_repos.get(sha, ())
    raw = raw_repos.get(sha, ())
    if not pooled:
        return MetadataReason.UNKNOWN_SHA
    if len(pooled) != 1:
        return MetadataReason.DUPLICATE_POOL_SHA
    if not raw:
        return MetadataReason.RAW_SHA_MISSING
    if len(raw) != 1:
        return MetadataReason.DUPLICATE_RAW_SHA
    if raw[0] is None:
        return MetadataReason.INVALID_REMOTE
    if raw[0] != pooled[0]:
        return MetadataReason.SOURCE_REPO_MISMATCH
    if expected_repos != {pooled[0]}:
        return MetadataReason.REPO_MISMATCH
    if len(splits) != 1:
        return MetadataReason.CROSS_SPLIT_SHA
    if pooled[0] not in split_repos[next(iter(splits))]:
        return MetadataReason.SPLIT_MISMATCH
    return MetadataReason.OK


def load_metadata_manifest(paths: MetadataPaths) -> MetadataManifest:
    pool_repos = _index_column(paths.pool_csv, "sha", "repo")
    raw_repos = _index_raw_repos(paths.raw_csv)
    split_repos = parse_repo_splits(paths.repo_split_json)
    references: dict[str, list[tuple[str, str]]] = {}
    total_rows = multiconcern_rows = sha_references = 0
    repositories: set[str] = set()
    for split, path in (("train", paths.train_csv), ("test", paths.test_csv)):
        with path.open(encoding="utf-8", newline="") as handle:
            for index, row in enumerate(csv.DictReader(handle), start=2):
                location = f"{path}:{index}"
                repo = _required(row, "repo", location)
                shas = parse_string_list(_required(row, "shas", location), f"{location}:shas")
                concern_count = int(_required(row, "concern_count", location))
                total_rows += 1
                multiconcern_rows += int(concern_count >= 2)
                sha_references += len(shas)
                repositories.add(repo)
                for sha in shas:
                    references.setdefault(sha, []).append((split, repo))
    records: list[MetadataRecord] = []
    for sha, usages in sorted(references.items()):
        usage_splits = {split for split, _ in usages}
        usage_repos = {repo for _, repo in usages}
        repo = sorted(usage_repos)[0]
        records.append(
            MetadataRecord(
                sha,
                repo,
                sorted(usage_splits)[0],
                f"https://github.com/{repo}.git",
                len(usages),
                _metadata_reason(sha, usage_splits, usage_repos, pool_repos, raw_repos, split_repos),
            )
        )
    return MetadataManifest(
        tuple(records),
        MetadataCounts(total_rows, multiconcern_rows, sha_references, len(records), len(repositories)),
    )
