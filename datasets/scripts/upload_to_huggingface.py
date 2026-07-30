#!/usr/bin/env python3
"""Upload dataset to HuggingFace Hub with overwrite functionality."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterable, Mapping, Sequence, Sized
from pathlib import Path
from typing import Final

from dotenv import load_dotenv
from huggingface_hub import CommitOperationDelete, HfApi, login, upload_folder

# noqa: SIZE_OK — this self-contained artifact is uploaded to Hugging Face.
DATASET_REPO_ID: Final[str] = os.getenv(
    "DATASET_REPO_ID",
    "Berom0227/tangled-ccs-commits",
)
DATASETS_PATH: Final[Path] = Path(__file__).parent.parent
DATA_PATH: Final[Path] = DATASETS_PATH / "data"
SCRIPTS_PATH: Final[Path] = DATASETS_PATH / "scripts"

# This script always updates the existing HF dataset repo as a new commit on
# top of its history (upload_folder/upload_file, no delete_repo or wholesale
# delete_patterns). Override via COMMIT_MESSAGE env var for future revisions.
COMMIT_MESSAGE: Final[str] = os.getenv(
    "COMMIT_MESSAGE",
    "Rebuild: intra-repo tangled commits, repo-disjoint split, "
    "7-type taxonomy (chore removed, style/perf merged into refactor)",
)

REQUIRED_FILES: Final[tuple[Path, ...]] = (
    DATA_PATH / "repo_grouped_pool.csv",
    DATA_PATH / "repo_split.json",
    DATA_PATH / "tangled_ccs_dataset_train.csv",
    DATA_PATH / "tangled_ccs_dataset_test.csv",
    DATA_PATH / "CCS Dataset.csv",
)

OLD_PIPELINE_SCRIPTS: Final[tuple[str, ...]] = (
    "scripts/clean_ccs_dataset.py",
    "scripts/generate_tangled_commites.py",
    "scripts/sample_atomic_commites.py",
)
TOKEN_ENVIRONMENT_VARIABLES: Final[tuple[str, ...]] = (
    "HUGGINGFACE_HUB_TOKEN",
    "HF_TOKEN",
    "HUGGINGFACE_TOKEN",
)
SCRIPT_MANIFEST: Final[tuple[str, ...]] = (
    "build_repo_pool.py",
    "generate_repo_tangled.py",
    "validate_repo_dataset.py",
    "show_tokens_distribution.py",
    "upload_to_huggingface.py",
)
# The Hub reads configs ONLY from the card's YAML frontmatter; dataset_info.yaml is
# ignored. test_dataset_card.py pins these pairs to that frontmatter block.
VERIFICATION_TARGETS: Final[tuple[tuple[str, str], ...]] = (
    ("default", "train"),
    ("default", "test"),
    ("original", "train"),
)
EXPECTED_REMOTE_TREE: Final[frozenset[str]] = frozenset(
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

# Helper function for file checking
def get_data_files() -> list[Path]:
    """Get list of data files for verification."""
    data_files: list[Path] = []
    for data_file in DATA_PATH.glob("*"):
        if data_file.is_file() and data_file.name != ".DS_Store":
            data_files.append(data_file)
    return data_files


def plan_deletions(
    remote_files: Iterable[str], targets: Iterable[str] = OLD_PIPELINE_SCRIPTS
) -> list[str]:
    """Return configured deletion targets that currently exist remotely."""
    remote_file_set = frozenset(remote_files)
    return [target for target in targets if target in remote_file_set]


def resolve_token(argv: Sequence[str], env: Mapping[str, str]) -> str | None:
    """Resolve an explicit Hugging Face token without mutating process state."""
    if len(argv) > 1:
        return argv[1]

    for variable_name in TOKEN_ENVIRONMENT_VARIABLES:
        token = env.get(variable_name)
        if token:
            return token
    return None


def authenticate_huggingface() -> None:
    """Authenticate with HuggingFace Hub."""
    token = resolve_token(sys.argv, os.environ)
    try:
        if token is not None:
            login(token=token)
            print("✓ Successfully authenticated with HuggingFace Hub")
            return

        HfApi().whoami()
        print("✓ Successfully authenticated with cached HuggingFace credentials")
    except Exception as error:  # noqa: BROAD_EXCEPT_OK
        raise RuntimeError("Unable to authenticate with HuggingFace Hub") from error


def upload_data_folder(repo_id: str) -> None:
    """Upload entire data folder to HuggingFace Hub."""
    print("Uploading data folder to HuggingFace Hub...")
    
    # Calculate total size of data folder
    total_size = sum(f.stat().st_size for f in DATA_PATH.glob("*") if f.is_file() and f.name != ".DS_Store")
    total_size_mb = total_size / (1024 * 1024)
    print(f"Data folder size: {total_size_mb:.1f} MB")
    
    try:
        upload_folder(
            folder_path=str(DATA_PATH),
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo="data",
            # "legacy/*" excludes data/legacy/ (old cross-repo dataset backup):
            # that prior version already lives in the HF repo's commit
            # history, so re-uploading it as new files would just clutter the
            # dataset.
            # ".omc/*" and ".git*" defensively exclude internal tool/orchestration
            # state and VCS metadata that must never end up inside a public
            # dataset folder, even if such a directory is accidentally present
            # locally at upload time (this is how internal state files were
            # once uploaded by mistake - see the cleanup commit in repo history).
            ignore_patterns=[
                ".DS_Store",
                "*.tmp",
                "__pycache__",
                "legacy/*",
                "legacy/**",
                ".omc/*",
                ".git*",
            ],
            commit_message=COMMIT_MESSAGE,
            # No delete_patterns: this call only adds/overwrites files present
            # locally, it never removes files that already exist in the repo.
        )
        print("✓ Data folder uploaded successfully")
    except Exception as e:
        print(f"✗ Failed to upload data folder: {e}")
        raise


def upload_scripts(repo_id: str) -> None:
    """Upload selected scripts to HuggingFace Hub."""
    api = HfApi()
    failed_paths: list[str] = []

    print("Uploading selected scripts...")

    for script_name in SCRIPT_MANIFEST:
        script_path = SCRIPTS_PATH / script_name
        repo_path = f"scripts/{script_name}"
        if not script_path.is_file():
            failed_paths.append(repo_path)
            print(f"✗ Missing script: {repo_path}")
            continue

        try:
            api.upload_file(
                path_or_fileobj=str(script_path),
                path_in_repo=repo_path,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=COMMIT_MESSAGE,
            )
            print(f"✓ Uploaded {repo_path}")
        except Exception as error:  # noqa: BROAD_EXCEPT_OK
            failed_paths.append(repo_path)
            print(f"✗ Failed to upload {repo_path}: {error}")

    if failed_paths:
        raise RuntimeError(f"Failed to upload scripts: {', '.join(failed_paths)}")


def upload_metadata_files(repo_id: str) -> None:
    """Upload README and dataset_info.yaml files."""
    api = HfApi()
    metadata_files: tuple[tuple[str, Path], ...] = (
        ("README.md", DATASETS_PATH / "README.md"),
        ("dataset_info.yaml", DATASETS_PATH / "dataset_info.yaml"),
    )
    failed_paths: list[str] = []

    print("Uploading metadata files...")

    for repo_path, local_path in metadata_files:
        if not local_path.is_file():
            failed_paths.append(repo_path)
            print(f"✗ Missing metadata file: {repo_path}")
            continue

        try:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=repo_path,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=COMMIT_MESSAGE,
            )
            print(f"✓ Uploaded {repo_path}")
        except Exception as error:  # noqa: BROAD_EXCEPT_OK
            failed_paths.append(repo_path)
            print(f"✗ Failed to upload {repo_path}: {error}")

    if failed_paths:
        raise RuntimeError(f"Failed to upload metadata files: {', '.join(failed_paths)}")


def verify_upload(repo_id: str) -> None:
    """Verify dataset upload by loading all configurations."""
    from datasets import load_dataset

    print("\nVerifying dataset upload...")

    try:
        for config_name, split in VERIFICATION_TARGETS:
            dataset = load_dataset(repo_id, config_name, split=split)
            if not isinstance(dataset, Sized):
                raise RuntimeError(f"{config_name}/{split} dataset is not sized")
            print(f"✓ {config_name}/{split} loaded: {len(dataset)} samples")
            print(f"  Columns: {dataset.column_names}")

        print("\n✓ Dataset upload verification successful!")

    except Exception as e:
        print(f"✗ Dataset verification failed: {e}")
        print("Dataset may still be processing. Try again in a few minutes.")
        raise RuntimeError("Dataset upload verification failed") from e


def delete_old_pipeline_scripts(
    api: HfApi, repo_id: str, remote_files: Iterable[str]
) -> None:
    """Delete present deprecated pipeline scripts in one atomic commit."""
    deletion_plan = plan_deletions(remote_files)
    if not deletion_plan:
        print("✓ nothing to delete")
        return

    api.create_commit(
        repo_id=repo_id,
        operations=[
            CommitOperationDelete(path_in_repo=path) for path in deletion_plan
        ],
        repo_type="dataset",
        commit_message="Remove deprecated dataset pipeline scripts",
    )
    print(f"✓ Deleted {len(deletion_plan)} deprecated pipeline scripts")


def parse_cli_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse the optional token and read-only dry-run switch."""
    parser = argparse.ArgumentParser(
        description="Synchronize the tangled commits dataset with Hugging Face."
    )
    parser.add_argument("token", nargs="?", help="Optional Hugging Face access token")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the planned synchronization without changing the remote repository",
    )
    return parser.parse_args(argv[1:])


def report_dry_run(remote_files: Iterable[str]) -> None:
    """Print the read-only deletion plan and post-deletion tree diff."""
    remote_tree = frozenset(remote_files)
    deletion_plan = plan_deletions(remote_tree)
    resulting_tree = remote_tree.difference(deletion_plan)
    missing_paths = sorted(EXPECTED_REMOTE_TREE.difference(resulting_tree))
    unexpected_paths = sorted(resulting_tree.difference(EXPECTED_REMOTE_TREE))

    print("\nDry-run synchronization plan")
    print(f"Deletion plan: {deletion_plan}")
    print(f"Upload manifest: {list(SCRIPT_MANIFEST)}")
    print(f"Missing paths after synchronization: {missing_paths}")
    print(f"Unexpected paths after synchronization: {unexpected_paths}")


def check_required_files() -> None:
    """Check if required dataset files exist."""
    missing_files = []
    for file_path in REQUIRED_FILES:
        if not file_path.exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("✗ Required files not found:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        sys.exit(1)
    
    print("✓ All required files found")
    
    # Show summary of files to upload
    data_files = get_data_files()
    print(f"\nData files to upload ({len(data_files)}):")
    for data_file in data_files:
        file_size = data_file.stat().st_size / (1024 * 1024)
        print(f"  - data/{data_file.name} ({file_size:.1f} MB)")
    
    print(f"\nScripts to upload ({len(SCRIPT_MANIFEST)}):")
    for script_name in SCRIPT_MANIFEST:
        script_path = SCRIPTS_PATH / script_name
        if script_path.exists():
            print(f"  - scripts/{script_name}")
        else:
            print(f"  - scripts/{script_name} (NOT FOUND)")


def main() -> None:
    """Main execution function."""
    cli_args = parse_cli_args(sys.argv)
    load_dotenv()
    repo_id = os.getenv("DATASET_REPO_ID", DATASET_REPO_ID)

    print("🚀 Starting HuggingFace dataset upload...")
    print(f"Repository: {repo_id}")
    print(f"Dataset path: {DATASETS_PATH}")
    print(f"Data path: {DATA_PATH}")
    print(f"Commit message: {COMMIT_MESSAGE}")

    check_required_files()

    if cli_args.dry_run:
        remote_files = HfApi().list_repo_files(repo_id=repo_id, repo_type="dataset")
        report_dry_run(remote_files)
        return

    try:
        authenticate_huggingface()

        # Upload in separate steps for better control
        api = HfApi()
        remote_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        delete_old_pipeline_scripts(api, repo_id, remote_files)
        upload_data_folder(repo_id)
        upload_scripts(repo_id)
        upload_metadata_files(repo_id)
        
        verify_upload(repo_id)

        print(
            f"\n🎉 Dataset successfully updated at: https://huggingface.co/datasets/{repo_id}"
        )
        print("\nDataset configurations available:")
        print("  - train: Tangled commits for training")
        print("  - test: Tangled commits for testing") 
        print("  - original: Original atomic commits")

    except Exception as e:
        print(f"✗ Upload failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
