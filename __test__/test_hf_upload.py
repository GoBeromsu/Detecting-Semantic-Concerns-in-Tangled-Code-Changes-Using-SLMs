"""Offline tests for datasets/scripts/upload_to_huggingface.py.

Every test here is offline by construction: __test__/conftest.py replaces
``huggingface_hub`` in sys.modules with a MagicMock for the whole session, and
no test issues a mutating Hub call.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType
from typing import Final
from unittest.mock import MagicMock

import pytest


REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
MODULE_PATH: Final[Path] = REPO_ROOT / "datasets" / "scripts" / "upload_to_huggingface.py"
MODULE_NAME: Final[str] = "hf_upload_under_test"


@dataclass(frozen=True, slots=True)
class _DeleteOperation:
    path_in_repo: str


class _DatasetsStubLoader(Loader):
    def create_module(self, spec: ModuleSpec) -> ModuleType | None:
        return None

    def exec_module(self, module: ModuleType) -> None:
        module.__dict__["load_dataset"] = MagicMock()


class _StubFinder(MetaPathFinder):
    def __init__(self, module_name: str) -> None:
        self._module_name = module_name

    def find_spec(
        self,
        fullname: str,
        path: object = None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        if fullname != self._module_name:
            return None
        return ModuleSpec(fullname, _DatasetsStubLoader())


def _import_module(name: str = MODULE_NAME) -> ModuleType:
    """Import the upload script by path under a private name."""
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def stub_datasets(patch_heavy_modules: None) -> Iterator[None]:
    """Stub ``datasets`` so importing the script never pulls pyarrow/pandas.

    conftest mocks ``huggingface_hub`` as a plain module, which the real
    ``datasets`` package cannot import from. Stubbing it here keeps every test
    in this file offline and fast, exactly like conftest's HEAVY_MODULES.
    """
    original = sys.modules.get("datasets")
    stub = MagicMock()
    stub.__spec__ = MagicMock()
    stub.__name__ = "datasets"
    sys.modules["datasets"] = stub
    yield
    if original is None:
        sys.modules.pop("datasets", None)
    else:
        sys.modules["datasets"] = original


@pytest.fixture
def hf_upload(
    stub_datasets: None, monkeypatch: pytest.MonkeyPatch
) -> ModuleType:
    """The upload script, imported once with heavy modules already mocked."""
    monkeypatch.delenv("DATASET_REPO_ID", raising=False)
    sys.modules.pop(MODULE_NAME, None)
    return _import_module()


def test_default_repo_id_is_canonical(hf_upload: ModuleType) -> None:
    # Given the module as shipped / When reading the default repo id
    # Then it is the real HF dataset repo, not the redirect-only legacy name
    assert hf_upload.DATASET_REPO_ID == "Berom0227/tangled-ccs-commits"


def test_old_pipeline_scripts_are_exactly_three(hf_upload: ModuleType) -> None:
    assert hf_upload.OLD_PIPELINE_SCRIPTS == (
        "scripts/clean_ccs_dataset.py",
        "scripts/generate_tangled_commites.py",
        "scripts/sample_atomic_commites.py",
    )


def test_surviving_scripts_not_in_delete_set(hf_upload: ModuleType) -> None:
    survivors = (
        "scripts/show_tokens_distribution.py",
        "scripts/upload_to_huggingface.py",
    )
    for survivor in survivors:
        assert survivor not in hf_upload.OLD_PIPELINE_SCRIPTS


def test_plan_deletions_returns_only_present_targets(hf_upload: ModuleType) -> None:
    remote = [
        "README.md",
        "scripts/clean_ccs_dataset.py",
        "scripts/sample_atomic_commites.py",
        "scripts/upload_to_huggingface.py",
    ]
    assert hf_upload.plan_deletions(remote) == [
        "scripts/clean_ccs_dataset.py",
        "scripts/sample_atomic_commites.py",
    ]


def test_plan_deletions_empty_when_none_present(hf_upload: ModuleType) -> None:
    remote = ["README.md", "scripts/upload_to_huggingface.py", "data/repo_split.json"]
    assert hf_upload.plan_deletions(remote) == []


def test_plan_deletions_is_order_stable(hf_upload: ModuleType) -> None:
    reversed_remote = list(reversed(hf_upload.OLD_PIPELINE_SCRIPTS))
    assert hf_upload.plan_deletions(reversed_remote) == list(
        hf_upload.OLD_PIPELINE_SCRIPTS
    )


def test_module_import_is_side_effect_free(stub_datasets: None) -> None:
    # Given "datasets" absent from sys.modules, but importable as a stub if
    # anything asks for it (so a real import failure can never masquerade as
    # the behaviour under test)
    stashed = sys.modules.pop("datasets", None)
    sys.modules.pop("fresh_hf_upload", None)
    finder = _StubFinder("datasets")
    sys.meta_path.insert(0, finder)
    try:
        # When importing the upload script
        _import_module("fresh_hf_upload")
        # Then it pulled in no heavy dataset machinery
        assert "datasets" not in sys.modules
    finally:
        sys.meta_path.remove(finder)
        sys.modules.pop("fresh_hf_upload", None)
        sys.modules.pop("datasets", None)
        if stashed is not None:
            sys.modules["datasets"] = stashed


def test_script_manifest_has_five_entries_all_on_disk(hf_upload: ModuleType) -> None:
    # Given the scripts that ship with the dataset / When reading the manifest
    # Then only the five synchronization scripts are selected and all exist
    expected_manifest = (
        "build_repo_pool.py",
        "generate_repo_tangled.py",
        "validate_repo_dataset.py",
        "show_tokens_distribution.py",
        "upload_to_huggingface.py",
    )
    assert hf_upload.SCRIPT_MANIFEST == expected_manifest
    assert all((hf_upload.SCRIPTS_PATH / script).is_file() for script in expected_manifest)


def test_expected_remote_tree_is_thirteen_paths(hf_upload: ModuleType) -> None:
    # Given the post-sync public dataset layout / When comparing expected paths
    # Then it contains precisely the thirteen allowed remote files
    expected_tree = frozenset(
        {
            ".gitattributes",
            "README.md",
            "dataset_info.yaml",
            "data/CCS Dataset.csv",
            "data/repo_grouped_pool.csv",
            "data/repo_split.json",
            "data/tangled_ccs_dataset_test.csv",
            "data/tangled_ccs_dataset_train.csv",
            "scripts/build_repo_pool.py",
            "scripts/generate_repo_tangled.py",
            "scripts/show_tokens_distribution.py",
            "scripts/upload_to_huggingface.py",
            "scripts/validate_repo_dataset.py",
        }
    )
    assert hf_upload.EXPECTED_REMOTE_TREE == expected_tree
    assert len(hf_upload.EXPECTED_REMOTE_TREE) == 13


def test_expected_remote_tree_excludes_deleted_scripts(hf_upload: ModuleType) -> None:
    # Given the removal plan / When comparing it with the desired remote tree
    # Then no deprecated pipeline script can remain in the resulting tree
    assert not set(hf_upload.OLD_PIPELINE_SCRIPTS) & hf_upload.EXPECTED_REMOTE_TREE


def test_upload_scripts_raises_on_failure(
    hf_upload: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given an API that rejects every script / When the manifest is uploaded
    # Then every failure is named together after all script attempts finish
    api = MagicMock()
    api.upload_file.side_effect = OSError("offline upload failure")
    monkeypatch.setattr(hf_upload, "HfApi", lambda: api)

    with pytest.raises(RuntimeError) as error:
        hf_upload.upload_scripts("test/repo")

    assert all(f"scripts/{name}" in str(error.value) for name in hf_upload.SCRIPT_MANIFEST)
    assert api.upload_file.call_count == len(hf_upload.SCRIPT_MANIFEST)


def test_upload_metadata_raises_on_failure(
    hf_upload: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given an API that rejects every metadata file / When metadata uploads run
    # Then the aggregate error names each failed repository path
    api = MagicMock()
    api.upload_file.side_effect = OSError("offline upload failure")
    monkeypatch.setattr(hf_upload, "HfApi", lambda: api)

    with pytest.raises(RuntimeError) as error:
        hf_upload.upload_metadata_files("test/repo")

    assert "README.md" in str(error.value)
    assert "dataset_info.yaml" in str(error.value)
    assert api.upload_file.call_count == 2


def test_upload_scripts_raises_on_missing_local_file(
    hf_upload: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given a manifest entry absent from disk / When uploading scripts
    # Then the missing path is a hard error and no upload is attempted
    api = MagicMock()
    monkeypatch.setattr(hf_upload, "HfApi", lambda: api)
    monkeypatch.setattr(hf_upload, "SCRIPT_MANIFEST", ("missing_script.py",))

    with pytest.raises(RuntimeError, match="scripts/missing_script.py"):
        hf_upload.upload_scripts("test/repo")

    api.upload_file.assert_not_called()


def test_resolve_token_prefers_argv(hf_upload: ModuleType) -> None:
    # Given explicit CLI credentials and configured environment values
    # When resolving authentication credentials
    # Then the explicit CLI argument wins without inspecting environment order
    token = hf_upload.resolve_token(
        ("upload_to_huggingface.py", "cli-token"),
        {"HUGGINGFACE_HUB_TOKEN": "environment-token"},
    )
    assert token == "cli-token"


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"HUGGINGFACE_HUB_TOKEN": "hub-token"}, "hub-token"),
        ({"HF_TOKEN": "hf-token"}, "hf-token"),
        ({"HUGGINGFACE_TOKEN": "legacy-token"}, "legacy-token"),
    ],
)
def test_resolve_token_falls_back_through_env_names(
    hf_upload: ModuleType, env: dict[str, str], expected: str
) -> None:
    # Given no positional token and one configured environment alias
    # When resolving authentication credentials
    # Then aliases are considered in their defined fallback order
    assert hf_upload.resolve_token(("upload_to_huggingface.py",), env) == expected


def test_resolve_token_returns_none_when_nothing_set(hf_upload: ModuleType) -> None:
    # Given no positional token and empty credential environment
    # When resolving authentication credentials
    # Then the ambient-login caller receives None instead of terminating
    assert hf_upload.resolve_token(("upload_to_huggingface.py",), {}) is None


def test_authenticate_uses_cached_login_when_token_is_absent(
    hf_upload: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given no resolved token and an authenticated ambient hub client
    # When authenticating the synchronization script
    # Then cached credentials are confirmed with a read-only whoami call
    api = MagicMock()
    monkeypatch.setattr(hf_upload, "HfApi", lambda: api)
    monkeypatch.setattr(hf_upload, "resolve_token", lambda _argv, _env: None)

    hf_upload.authenticate_huggingface()

    api.whoami.assert_called_once_with()


def test_delete_skips_network_when_nothing_to_delete(hf_upload: ModuleType) -> None:
    # Given a remote tree without stale pipeline scripts / When deletion runs
    # Then no commit mutation is issued for the idempotent re-run
    api = MagicMock()

    hf_upload.delete_old_pipeline_scripts(api, "test/repo", ["README.md"])

    api.create_commit.assert_not_called()


def test_delete_issues_single_commit_for_present_targets(
    hf_upload: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given every deprecated script in the remote tree / When deletion runs
    # Then one atomic commit contains all three exact delete operations
    api = MagicMock()
    monkeypatch.setattr(hf_upload, "CommitOperationDelete", _DeleteOperation)

    hf_upload.delete_old_pipeline_scripts(
        api, "test/repo", hf_upload.OLD_PIPELINE_SCRIPTS
    )

    api.create_commit.assert_called_once()
    create_kwargs = api.create_commit.call_args.kwargs
    assert create_kwargs["operations"] == [
        _DeleteOperation(path) for path in hf_upload.OLD_PIPELINE_SCRIPTS
    ]
    assert create_kwargs["repo_id"] == "test/repo"
    assert create_kwargs["repo_type"] == "dataset"


def test_dry_run_reads_remote_tree_without_mutating(
    hf_upload: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Given a remote tree with all deprecated scripts / When --dry-run executes
    # Then it reports the planned state without invoking any sync mutation
    api = MagicMock()
    api.list_repo_files.return_value = tuple(
        hf_upload.EXPECTED_REMOTE_TREE | set(hf_upload.OLD_PIPELINE_SCRIPTS)
    )
    authenticate = MagicMock()
    upload_data = MagicMock()
    upload_scripts = MagicMock()
    upload_metadata = MagicMock()
    verify = MagicMock()
    monkeypatch.setattr(hf_upload, "HfApi", lambda: api)
    monkeypatch.setattr(hf_upload, "load_dotenv", lambda: None)
    monkeypatch.setattr(hf_upload.sys, "argv", ["upload_to_huggingface.py", "--dry-run"])
    monkeypatch.setattr(hf_upload, "check_required_files", MagicMock())
    monkeypatch.setattr(hf_upload, "authenticate_huggingface", authenticate)
    monkeypatch.setattr(hf_upload, "upload_data_folder", upload_data)
    monkeypatch.setattr(hf_upload, "upload_scripts", upload_scripts)
    monkeypatch.setattr(hf_upload, "upload_metadata_files", upload_metadata)
    monkeypatch.setattr(hf_upload, "verify_upload", verify)

    hf_upload.main()

    api.list_repo_files.assert_called_once_with(
        repo_id=hf_upload.DATASET_REPO_ID, repo_type="dataset"
    )
    api.create_commit.assert_not_called()
    authenticate.assert_not_called()
    upload_data.assert_not_called()
    upload_scripts.assert_not_called()
    upload_metadata.assert_not_called()
    verify.assert_not_called()
    output = capsys.readouterr().out
    assert "scripts/clean_ccs_dataset.py" in output
    assert "show_tokens_distribution.py" in output
