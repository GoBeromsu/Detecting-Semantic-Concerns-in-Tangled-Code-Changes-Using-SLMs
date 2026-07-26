#!/usr/bin/env python3
"""Upload dataset to HuggingFace Hub with overwrite functionality."""

import sys
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from datasets import load_dataset
from huggingface_hub import HfApi, login, create_repo, upload_folder

# Load environment variables from .env file
load_dotenv()

DATASET_REPO_ID = os.getenv(
    "DATASET_REPO_ID",
    "Berom0227/Detecting-Semantic-Concerns-in-Tangled-Code-Changes-Using-SLMs",
)
DATASETS_PATH = Path(__file__).parent.parent
DATA_PATH = DATASETS_PATH / "data"
SCRIPTS_PATH = DATASETS_PATH / "scripts"

# This script always updates the existing HF dataset repo as a new commit on
# top of its history (upload_folder/upload_file, no delete_repo or wholesale
# delete_patterns). Override via COMMIT_MESSAGE env var for future revisions.
COMMIT_MESSAGE = os.getenv(
    "COMMIT_MESSAGE",
    "Rebuild: intra-repo tangled commits, repo-disjoint split, "
    "7-type taxonomy (chore removed, style/perf merged into refactor)",
)

REQUIRED_FILES = [
    DATA_PATH / "repo_grouped_pool.csv",
    DATA_PATH / "repo_split.json",
    DATA_PATH / "tangled_ccs_dataset_train.csv",
    DATA_PATH / "tangled_ccs_dataset_test.csv",
    DATA_PATH / "CCS Dataset.csv",
]

# Helper function for file checking
def get_data_files() -> list:
    """Get list of data files for verification."""
    data_files = []
    for data_file in DATA_PATH.glob("*"):
        if data_file.is_file() and data_file.name != ".DS_Store":
            data_files.append(data_file)
    return data_files


def get_hf_token() -> Optional[str]:
    """Get HuggingFace token from command line args or environment."""
    if len(sys.argv) > 1:
        return sys.argv[1]
    return os.getenv("HUGGINGFACE_HUB_TOKEN")


def authenticate_huggingface(token: Optional[str] = None) -> None:
    """Authenticate with HuggingFace Hub."""
    if not token:
        token = get_hf_token()

    if not token:
        print("✗ No HuggingFace token provided")
        print("Usage: python upload_to_huggingface.py <token>")
        print("Or set HUGGINGFACE_HUB_TOKEN in .env file")
        sys.exit(1)

    try:
        login(token=token)
        print("✓ Successfully authenticated with HuggingFace Hub")
    except Exception as e:
        print(f"✗ Authentication failed: {e}")
        sys.exit(1)


def ensure_repo_exists(repo_id: str) -> None:
    """Ensure HuggingFace repository exists, create if it doesn't."""
    api = HfApi()

    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset")
        print(f"✓ Repository {repo_id} exists, will update existing files")
    except Exception:
        print(f"Repository {repo_id} doesn't exist, creating new one...")
        try:
            create_repo(repo_id=repo_id, repo_type="dataset", private=False)
            print(f"✓ Created new repository: {repo_id}")
        except Exception as create_error:
            print(f"Error creating repository: {create_error}")
            raise


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
    
    specified_scripts = [
        "build_repo_pool.py",
        "generate_repo_tangled.py",
        "validate_repo_dataset.py",
    ]
    
    print("Uploading selected scripts...")
    
    for script_name in specified_scripts:
        script_path = SCRIPTS_PATH / script_name
        if script_path.exists():
            try:
                api.upload_file(
                    path_or_fileobj=str(script_path),
                    path_in_repo=f"scripts/{script_name}",
                    repo_id=repo_id,
                    repo_type="dataset",
                    commit_message=COMMIT_MESSAGE,
                )
                print(f"✓ Uploaded scripts/{script_name}")
            except Exception as e:
                print(f"✗ Failed to upload {script_name}: {e}")
        else:
            print(f"⚠ Script not found: {script_path}")


def upload_metadata_files(repo_id: str) -> None:
    """Upload README and dataset_info.yaml files."""
    api = HfApi()
    
    metadata_files = [
        ("README.md", DATASETS_PATH / "README.md"),
        ("dataset_info.yaml", DATASETS_PATH / "dataset_info.yaml"),
    ]
    
    print("Uploading metadata files...")
    
    for repo_path, local_path in metadata_files:
        if local_path.exists():
            try:
                api.upload_file(
                    path_or_fileobj=str(local_path),
                    path_in_repo=repo_path,
                    repo_id=repo_id,
                    repo_type="dataset",
                    commit_message=COMMIT_MESSAGE,
                )
                print(f"✓ Uploaded {repo_path}")
            except Exception as e:
                print(f"✗ Failed to upload {repo_path}: {e}")
        else:
            print(f"⚠ File not found: {local_path}")


def verify_upload(repo_id: str) -> None:
    """Verify dataset upload by loading all configurations."""
    print("\nVerifying dataset upload...")

    try:
        # Verify train dataset
        train_dataset = load_dataset(repo_id, "train", split="train")
        print(f"✓ Train dataset loaded: {len(train_dataset)} samples")
        print(f"  Columns: {train_dataset.column_names}")

        # Verify test dataset
        test_dataset = load_dataset(repo_id, "test", split="train")
        print(f"✓ Test dataset loaded: {len(test_dataset)} samples")
        print(f"  Columns: {test_dataset.column_names}")
        
        # Verify original dataset
        original_dataset = load_dataset(repo_id, "original", split="train")
        print(f"✓ Original dataset loaded: {len(original_dataset)} samples")
        print(f"  Columns: {original_dataset.column_names}")

        print("\n✓ Dataset upload verification successful!")

    except Exception as e:
        print(f"✗ Dataset verification failed: {e}")
        print("Dataset may still be processing. Try again in a few minutes.")


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
    
    specified_scripts = ["build_repo_pool.py", "generate_repo_tangled.py", "validate_repo_dataset.py"]
    print(f"\nScripts to upload ({len(specified_scripts)}):")
    for script_name in specified_scripts:
        script_path = SCRIPTS_PATH / script_name
        if script_path.exists():
            print(f"  - scripts/{script_name}")
        else:
            print(f"  - scripts/{script_name} (NOT FOUND)")


def main() -> None:
    """Main execution function."""
    print("🚀 Starting HuggingFace dataset upload...")
    print(f"Repository: {DATASET_REPO_ID}")
    print(f"Dataset path: {DATASETS_PATH}")
    print(f"Data path: {DATA_PATH}")
    print(f"Commit message: {COMMIT_MESSAGE}")

    check_required_files()

    try:
        authenticate_huggingface()
        ensure_repo_exists(DATASET_REPO_ID)
        
        # Upload in separate steps for better control
        upload_data_folder(DATASET_REPO_ID)
        upload_scripts(DATASET_REPO_ID)
        upload_metadata_files(DATASET_REPO_ID)
        
        verify_upload(DATASET_REPO_ID)

        print(
            f"\n🎉 Dataset successfully updated at: https://huggingface.co/datasets/{DATASET_REPO_ID}"
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
