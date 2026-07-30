#!/usr/bin/env python3
"""Verify that the HuggingFace dataset repo matches this local repo after a sync.

MAINTAINER-ONLY. This script is never uploaded to HuggingFace and must not appear
in any upload manifest.

The module is split into two halves:

* pure logic (constants, ``forbidden_paths``, ``diff_trees``, ``sha256_of``) which is
  fully offline and is what the test suite targets;
* a thin network shell (``main`` and the ``_remote_*`` helpers) which imports
  ``huggingface_hub`` lazily so that importing this module has no side effects.

Authentication relies on the ambient cached HuggingFace login; no token is read,
printed, or logged here.

Usage:
    python datasets/scripts/verify_hf_sync.py
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final


REPO_ID: Final[str] = "Berom0227/tangled-ccs-commits"
REPO_TYPE: Final[str] = "dataset"
REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

_CHUNK_SIZE: Final[int] = 1 << 20

# Remote -> local pairs that were already in sync and MUST stay byte-identical.
IDENTITY_MAP: Final[Mapping[str, str]] = {
    "data/tangled_ccs_dataset_train.csv": "datasets/data/tangled_ccs_dataset_train.csv",
    "data/tangled_ccs_dataset_test.csv": "datasets/data/tangled_ccs_dataset_test.csv",
    "data/repo_grouped_pool.csv": "datasets/data/repo_grouped_pool.csv",
    "data/repo_split.json": "datasets/data/repo_split.json",
    "data/CCS Dataset.csv": "datasets/data/CCS Dataset.csv",
    "dataset_info.yaml": "datasets/dataset_info.yaml",
    "scripts/generate_repo_tangled.py": "datasets/scripts/generate_repo_tangled.py",
    "scripts/show_tokens_distribution.py": "datasets/scripts/show_tokens_distribution.py",
}

# Remote -> local pairs that were STALE on HuggingFace and must match after the push.
MUST_MATCH_MAP: Final[Mapping[str, str]] = {
    "README.md": "datasets/README.md",
    "scripts/build_repo_pool.py": "datasets/scripts/build_repo_pool.py",
    "scripts/validate_repo_dataset.py": "datasets/scripts/validate_repo_dataset.py",
    "scripts/upload_to_huggingface.py": "datasets/scripts/upload_to_huggingface.py",
}

# Superseded pipeline scripts that must no longer exist on HuggingFace.
DELETED_SCRIPTS: Final[frozenset[str]] = frozenset(
    {
        "scripts/clean_ccs_dataset.py",
        "scripts/generate_tangled_commites.py",
        "scripts/sample_atomic_commites.py",
    }
)

# The exact remote listing expected after the sync (13 entries, .gitattributes included).
EXPECTED_REMOTE_TREE: Final[frozenset[str]] = frozenset(
    {
        ".gitattributes",
        "README.md",
        "dataset_info.yaml",
        "data/tangled_ccs_dataset_train.csv",
        "data/tangled_ccs_dataset_test.csv",
        "data/repo_grouped_pool.csv",
        "data/repo_split.json",
        "data/CCS Dataset.csv",
        "scripts/build_repo_pool.py",
        "scripts/generate_repo_tangled.py",
        "scripts/validate_repo_dataset.py",
        "scripts/show_tokens_distribution.py",
        "scripts/upload_to_huggingface.py",
    }
)

ALLOWED_DOT_GIT: Final[str] = ".gitattributes"
LEGACY_PREFIX: Final[str] = "data/legacy/"
OMC_MARKER: Final[str] = ".omc"
MACOS_CRUFT: Final[str] = ".DS_Store"


@dataclass(frozen=True, slots=True)
class TreeDiff:
    """Difference between the remote listing and ``EXPECTED_REMOTE_TREE``."""

    missing: tuple[str, ...]
    unexpected: tuple[str, ...]

    @property
    def is_clean(self) -> bool:
        return not self.missing and not self.unexpected


def _is_forbidden(path: str) -> bool:
    """True when a remote path must never exist in the dataset repo."""
    if path.startswith(LEGACY_PREFIX):
        return True
    if OMC_MARKER in path:
        return True
    if path.rsplit("/", maxsplit=1)[-1] == MACOS_CRUFT:
        return True
    return path.startswith(".git") and path != ALLOWED_DOT_GIT


def forbidden_paths(remote_files: Iterable[str]) -> list[str]:
    """Return the remote paths that must never be published, in input order."""
    return [path for path in remote_files if _is_forbidden(path)]


def diff_trees(remote_files: Iterable[str]) -> TreeDiff:
    """Compare a remote listing against ``EXPECTED_REMOTE_TREE``."""
    remote = frozenset(remote_files)
    return TreeDiff(
        missing=tuple(sorted(EXPECTED_REMOTE_TREE - remote)),
        unexpected=tuple(sorted(remote - EXPECTED_REMOTE_TREE)),
    )


def sha256_of(path: Path) -> str:
    """Hex sha256 digest of a file, read in streaming chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _remote_listing() -> list[str]:
    """Fetch the remote file listing. Imports huggingface_hub lazily."""
    from huggingface_hub import HfApi

    return list(HfApi().list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE))


def _remote_digest(remote_path: str) -> str:
    """Download one remote file and hash it. Imports huggingface_hub lazily."""
    from huggingface_hub import hf_hub_download

    local_copy: str = hf_hub_download(
        repo_id=REPO_ID, repo_type=REPO_TYPE, filename=remote_path
    )
    return sha256_of(Path(local_copy))


def _compare_digests(pairs: Mapping[str, str], label: str) -> list[str]:
    """Print a sha256 comparison table for one mapping and return the mismatches."""
    print(f"\n{label} ({len(pairs)} files)")
    mismatches: list[str] = []
    for remote_path, local_path in pairs.items():
        local_digest = sha256_of(REPO_ROOT / local_path)
        remote_hex = _remote_digest(remote_path)
        matched = remote_hex == local_digest
        marker = "OK  " if matched else "DIFF"
        print(f"  {marker} {remote_path}  {remote_hex[:12]} vs {local_digest[:12]}")
        if not matched:
            mismatches.append(remote_path)
    return mismatches


def _report_tree(remote_files: list[str]) -> list[str]:
    """Print tree/forbidden/deletion checks and return the failure descriptions."""
    failures: list[str] = []

    still_present = sorted(DELETED_SCRIPTS.intersection(remote_files))
    print(f"\nRemote listing: {len(remote_files)} files")
    for path in still_present:
        failures.append(f"deleted script still on HF: {path}")

    diff = diff_trees(remote_files)
    for path in diff.missing:
        failures.append(f"missing from HF: {path}")
    for path in diff.unexpected:
        failures.append(f"unexpected on HF: {path}")

    forbidden = forbidden_paths(remote_files)
    for path in forbidden:
        failures.append(f"forbidden path on HF: {path}")

    print(f"  deleted-scripts absent : {'FAIL' if still_present else 'OK'}")
    print(f"  tree matches expected  : {'OK' if diff.is_clean else 'FAIL'}")
    print(f"  forbidden paths        : {'FAIL' if forbidden else 'OK'}")
    return failures


def main() -> int:
    """Run every check and return the process exit code."""
    parser = argparse.ArgumentParser(description=f"Verify HF sync for {REPO_ID}.")
    _ = parser.parse_args()

    remote_files = _remote_listing()
    failures = _report_tree(remote_files)

    for remote_path in _compare_digests(IDENTITY_MAP, "Identity files"):
        failures.append(f"identity file differs: {remote_path}")
    for remote_path in _compare_digests(MUST_MATCH_MAP, "Refreshed files"):
        failures.append(f"refreshed file differs: {remote_path}")

    if failures:
        print(f"\nFAILED ({len(failures)} problems):")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("\nPASSED: HuggingFace repo is in sync with the local repo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
