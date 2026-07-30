"""Offline tests for the maintainer-only HF sync verifier.

These tests exercise ONLY the pure logic of ``datasets/scripts/verify_hf_sync.py``.
No network, no HuggingFace API, no ``datasets`` library import.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Final


REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
MODULE_PATH: Final[Path] = REPO_ROOT / "datasets" / "scripts" / "verify_hf_sync.py"
MODULE_NAME: Final[str] = "verify_hf_sync"

EXPECTED_TREE_LITERAL: Final[frozenset[str]] = frozenset(
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


def _load_module_fresh() -> ModuleType:
    """Import the verifier from its file path as a brand-new module object."""
    assert MODULE_PATH.is_file(), f"verifier module does not exist yet: {MODULE_PATH}"
    spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot build import spec for {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    # Required before exec_module: @dataclass(slots=True) resolves annotations via
    # sys.modules[cls.__module__], which is None if the module is not registered.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@functools.cache
def _verifier() -> ModuleType:
    """Load the verifier once per test session."""
    return _load_module_fresh()


def test_identity_map_has_eight_entries_and_covers_ccs_dataset_with_space() -> None:
    # Given: the identity map of byte-identical remote->local pairs
    identity = _verifier().IDENTITY_MAP

    # When / Then: it has exactly 8 entries and keeps the spaced filename intact
    assert len(identity) == 8, f"expected 8 identity pairs, got {sorted(identity)}"
    assert identity["data/CCS Dataset.csv"] == "datasets/data/CCS Dataset.csv"
    assert identity["dataset_info.yaml"] == "datasets/dataset_info.yaml"
    assert identity["data/repo_split.json"] == "datasets/data/repo_split.json"


def test_deleted_scripts_are_exactly_the_three_old_pipeline_files() -> None:
    # Given / When: the set of scripts that must be absent from the HF repo
    # Then: it is exactly the three superseded pipeline files
    assert _verifier().DELETED_SCRIPTS == frozenset(
        {
            "scripts/clean_ccs_dataset.py",
            "scripts/generate_tangled_commites.py",
            "scripts/sample_atomic_commites.py",
        }
    )


def test_surviving_scripts_are_not_in_deleted_set() -> None:
    # Given: two scripts that survived the pipeline rewrite
    verifier = _verifier()
    survivors = (
        "scripts/show_tokens_distribution.py",
        "scripts/upload_to_huggingface.py",
    )

    # When / Then: neither is marked for deletion and both are expected remotely
    for survivor in survivors:
        assert survivor not in verifier.DELETED_SCRIPTS
        assert survivor in verifier.EXPECTED_REMOTE_TREE


def test_forbidden_paths_flags_legacy_and_omc_but_allows_gitattributes() -> None:
    # Given: a remote listing polluted with local-only artefacts
    remote = [
        "README.md",
        ".gitattributes",
        "data/legacy/tangled_ccs_dataset.csv",
        ".omc/state/session.json",
        "data/.DS_Store",
        ".gitignore",
        "data/repo_split.json",
    ]

    # When: forbidden paths are computed
    flagged = _verifier().forbidden_paths(remote)

    # Then: only the pollution is flagged, in input order
    assert flagged == [
        "data/legacy/tangled_ccs_dataset.csv",
        ".omc/state/session.json",
        "data/.DS_Store",
        ".gitignore",
    ]


def test_diff_trees_reports_missing_and_unexpected() -> None:
    # Given: a remote tree with one file dropped and two strangers added
    verifier = _verifier()
    dropped = "scripts/build_repo_pool.py"
    remote = sorted(verifier.EXPECTED_REMOTE_TREE - {dropped}) + [
        "scripts/zzz_stray.py",
        "notes.txt",
    ]

    # When: the tree is diffed
    diff = verifier.diff_trees(remote)

    # Then: both sides are reported and sorted
    assert diff.missing == (dropped,)
    assert diff.unexpected == ("notes.txt", "scripts/zzz_stray.py")


def test_diff_trees_is_clean_for_expected_tree() -> None:
    # Given: the expected tree is exactly the agreed literal set
    verifier = _verifier()
    assert verifier.EXPECTED_REMOTE_TREE == EXPECTED_TREE_LITERAL

    # When: the remote tree is exactly what we expect
    diff = verifier.diff_trees(sorted(verifier.EXPECTED_REMOTE_TREE))

    # Then: nothing is missing and nothing is unexpected
    assert diff.missing == ()
    assert diff.unexpected == ()


def test_sha256_of_matches_hashlib_on_a_tmp_file(tmp_path: Path) -> None:
    # Given: a file with non-trivial binary content
    payload = b"concern-is-all-you-need\n" * 4096
    target = tmp_path / "blob.bin"
    _ = target.write_bytes(payload)

    # When / Then: the streaming digest equals the one-shot digest
    assert _verifier().sha256_of(target) == hashlib.sha256(payload).hexdigest()


def test_module_import_is_side_effect_free() -> None:
    # Given: no cached `datasets` library module
    _ = sys.modules.pop("datasets", None)

    # When: the verifier is imported fresh
    _ = _load_module_fresh()

    # Then: importing it pulled in neither the datasets library nor a HF call
    assert "datasets" not in sys.modules


def test_all_mapped_local_paths_exist_on_disk() -> None:
    # Given: every local counterpart declared in both maps
    verifier = _verifier()
    local_paths = [
        *verifier.IDENTITY_MAP.values(),
        *verifier.MUST_MATCH_MAP.values(),
    ]

    # When / Then: each one resolves to a real file in the repo
    missing = [p for p in local_paths if not (REPO_ROOT / p).is_file()]
    assert missing == [], f"local paths declared but absent: {missing}"
