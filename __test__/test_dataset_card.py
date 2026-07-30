"""Dataset-card invariants.

The Hub reads config definitions ONLY from the YAML frontmatter of README.md.
`dataset_info.yaml` is ignored by the Hub, so a config declared solely there is
invisible to `load_dataset`. These tests pin the frontmatter contract so the
published card and the documented configs cannot drift apart again.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import yaml

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[1]
CARD_PATH: Final = REPOSITORY_ROOT / "datasets" / "README.md"
FRONTMATTER_DELIMITER: Final = "---"


def read_frontmatter(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith(FRONTMATTER_DELIMITER):
        raise AssertionError(f"{path} does not start with a YAML frontmatter block")
    _, _, remainder = text.partition(FRONTMATTER_DELIMITER)
    block, delimiter, _ = remainder.partition(f"\n{FRONTMATTER_DELIMITER}")
    if not delimiter:
        raise AssertionError(f"{path} frontmatter block is not terminated")
    parsed = yaml.safe_load(block)
    if not isinstance(parsed, dict):
        raise AssertionError(f"{path} frontmatter is not a mapping")
    return parsed


def config_by_name(frontmatter: dict[str, Any], name: str) -> dict[str, Any]:
    configs = frontmatter.get("configs")
    assert isinstance(configs, list), "frontmatter `configs` must be a list"
    for entry in configs:
        if isinstance(entry, dict) and entry.get("config_name") == name:
            return entry
    raise AssertionError(f"config {name!r} not found in frontmatter configs")


def split_paths(config: dict[str, Any]) -> dict[str, str]:
    data_files = config.get("data_files")
    assert isinstance(data_files, list), "config `data_files` must be a list"
    mapping: dict[str, str] = {}
    for entry in data_files:
        assert isinstance(entry, dict), "each data_files entry must be a mapping"
        mapping[str(entry["split"])] = str(entry["path"])
    return mapping


def test_card_frontmatter_declares_configs() -> None:
    frontmatter = read_frontmatter(CARD_PATH)
    assert "configs" in frontmatter, (
        "the Hub reads configs from README.md frontmatter only; without this key "
        "load_dataset sees a single 'default' config and the atomic pool is unreachable"
    )


def test_default_config_exposes_train_and_test_splits() -> None:
    default_config = config_by_name(read_frontmatter(CARD_PATH), "default")
    assert split_paths(default_config) == {
        "train": "data/tangled_ccs_dataset_train.csv",
        "test": "data/tangled_ccs_dataset_test.csv",
    }


def test_original_config_exposes_the_atomic_pool() -> None:
    original_config = config_by_name(read_frontmatter(CARD_PATH), "original")
    assert split_paths(original_config) == {"train": "data/repo_grouped_pool.csv"}


def test_every_declared_data_file_exists_locally() -> None:
    frontmatter = read_frontmatter(CARD_PATH)
    configs = frontmatter["configs"]
    assert isinstance(configs, list)
    declared: list[str] = []
    for entry in configs:
        declared.extend(split_paths(entry).values())
    assert declared, "no data files declared in frontmatter"
    missing = [
        path for path in declared if not (REPOSITORY_ROOT / "datasets" / path).is_file()
    ]
    assert missing == [], f"frontmatter references nonexistent files: {missing}"


def test_verification_targets_match_the_published_card() -> None:
    """The uploader's post-upload check must probe exactly what the card declares.

    These two live in different files, so without this test a card edit silently
    breaks verify_upload (or vice versa) and only surfaces after a real upload.
    """
    import importlib
    import sys

    sys.path.insert(0, str(REPOSITORY_ROOT / "datasets" / "scripts"))
    uploader = importlib.import_module("upload_to_huggingface")

    frontmatter = read_frontmatter(CARD_PATH)
    configs = frontmatter["configs"]
    assert isinstance(configs, list)
    declared = {
        (str(entry["config_name"]), split)
        for entry in configs
        for split in split_paths(entry)
    }
    assert set(uploader.VERIFICATION_TARGETS) == declared


def test_declared_configs_are_exactly_default_and_original() -> None:
    frontmatter = read_frontmatter(CARD_PATH)
    configs = frontmatter["configs"]
    assert isinstance(configs, list)
    names = [entry["config_name"] for entry in configs]
    assert names == ["default", "original"], (
        "'default' must stay first and keep its name so existing "
        "load_dataset(repo, split=...) callers are unaffected"
    )
