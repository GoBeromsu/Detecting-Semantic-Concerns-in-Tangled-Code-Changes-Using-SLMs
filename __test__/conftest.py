"""Test configuration and fixtures for all tests."""

import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd  # noqa: PANDAS_OK - D2 requires pandas row-count semantics.
import pytest


REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DATASET_DIR: Final[Path] = REPO_ROOT / "datasets" / "data"

CANONICAL_TYPES: Final[tuple[str, ...]] = (
    "feat",
    "fix",
    "refactor",
    "test",
    "docs",
    "build",
    "ci",
)

MAX_DIFF_TOKENS: Final[int] = 12288
TIKTOKEN_ENCODING: Final[str] = "cl100k_base"


@dataclass(frozen=True, slots=True)
class TangledSplit:
    """A tangled-commit CSV with its JSON columns already parsed."""

    name: str
    frame: pd.DataFrame
    type_lists: tuple[tuple[str, ...], ...]
    sha_lists: tuple[tuple[str, ...], ...]

    @property
    def concern_counts(self) -> tuple[int, ...]:
        return tuple(int(k) for k in self.frame["concern_count"])

    @property
    def repos(self) -> tuple[str, ...]:
        return tuple(str(r) for r in self.frame["repo"])

    def unique_shas(self) -> frozenset[str]:
        return frozenset(sha for shas in self.sha_lists for sha in shas)

    def combinations_for_k(self, k: int) -> set[frozenset[str]]:
        return {
            frozenset(types)
            for count, types in zip(self.concern_counts, self.type_lists)
            if count == k
        }


def _load_split(name: str) -> TangledSplit:
    frame = pd.read_csv(DATASET_DIR / f"tangled_ccs_dataset_{name}.csv")
    return TangledSplit(
        name=name,
        frame=frame,
        type_lists=tuple(tuple(json.loads(raw)) for raw in frame["types"]),
        sha_lists=tuple(tuple(json.loads(raw)) for raw in frame["shas"]),
    )


@pytest.fixture(scope="session")
def train_split() -> TangledSplit:
    return _load_split("train")


@pytest.fixture(scope="session")
def test_split() -> TangledSplit:
    return _load_split("test")


@pytest.fixture(scope="session")
def tangled_splits(
    train_split: TangledSplit, test_split: TangledSplit
) -> Mapping[str, TangledSplit]:
    return {"train": train_split, "test": test_split}


@pytest.fixture(scope="session")
def pool_frame() -> pd.DataFrame:
    return pd.read_csv(DATASET_DIR / "repo_grouped_pool.csv")


@pytest.fixture(scope="session")
def sha_to_repo(pool_frame: pd.DataFrame) -> dict[str, str]:
    return {str(s): str(r) for s, r in zip(pool_frame["sha"], pool_frame["repo"])}


@pytest.fixture(scope="session")
def repo_split() -> Mapping[str, list[str]]:
    with (DATASET_DIR / "repo_split.json").open(encoding="utf-8") as handle:
        return json.load(handle)


# Heavy external libraries that we don't need for domain logic testing
HEAVY_MODULES = [
    "outlines",
    "outlines.inputs", 
    "lmstudio",
    "torch",
    "transformers",
    "huggingface_hub",
]


@pytest.fixture(autouse=True, scope="session") 
def patch_heavy_modules():
    """
    Replace heavy external libraries with smart mocks for the entire test session.
    This allows imports to work without actually loading heavy dependencies.
    """
    from unittest.mock import MagicMock
    
    original = {}
    
    for name in HEAVY_MODULES:
        original[name] = sys.modules.get(name)
        
        # Create a smart mock that can handle attribute access
        mock_module = MagicMock()
        mock_module.__spec__ = MagicMock()
        mock_module.__name__ = name
        
        # For transformers, add specific attributes that are commonly imported
        if name == "transformers":
            mock_module.AutoTokenizer = MagicMock()
            mock_module.AutoModelForCausalLM = MagicMock()
            
        sys.modules[name] = mock_module
    
    yield
    
    # Restore original modules after tests
    for name, mod in original.items():
        if mod is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = mod
