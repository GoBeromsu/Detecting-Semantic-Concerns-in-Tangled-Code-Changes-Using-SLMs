from __future__ import annotations

import re
import tomllib
from collections.abc import Mapping
from datetime import date, datetime, time
from pathlib import Path
from typing import Final, TypeAlias


REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
PYPROJECT_PATH: Final[Path] = REPO_ROOT / "pyproject.toml"
GITIGNORE_PATH: Final[Path] = REPO_ROOT / ".gitignore"
LINUX_X86_64_MARKER: Final[str] = (
    "sys_platform == 'linux' and platform_machine == 'x86_64'"
)
REQUIRED_LOCAL_GPU_PACKAGES: Final[frozenset[str]] = frozenset(
    {
        "torch",
        "transformers",
        "unsloth",
        "unsloth-zoo",
        "triton",
        "bitsandbytes",
        "outlines",
        "ninja",
    }
)
TomlValue: TypeAlias = (
    str
    | int
    | float
    | bool
    | datetime
    | date
    | time
    | list["TomlValue"]
    | dict[str, "TomlValue"]
)


def _load_pyproject() -> dict[str, TomlValue]:
    with PYPROJECT_PATH.open("rb") as handle:
        document: dict[str, TomlValue] = tomllib.load(handle)
    return document


def _mapping(value: TomlValue) -> Mapping[str, TomlValue]:
    assert isinstance(value, dict)
    return value


def _mapping_list(value: TomlValue) -> tuple[Mapping[str, TomlValue], ...]:
    values = value if isinstance(value, list) else [value]
    mappings: list[Mapping[str, TomlValue]] = []
    for item in values:
        assert isinstance(item, dict)
        mappings.append(item)
    return tuple(mappings)


def _local_gpu_requirements() -> tuple[str, ...]:
    document = _load_pyproject()
    project = _mapping(document["project"])
    optional_dependencies = _mapping(project["optional-dependencies"])
    requirements = optional_dependencies["local-gpu"]
    assert isinstance(requirements, list)
    parsed: list[str] = []
    for requirement in requirements:
        assert isinstance(requirement, str)
        parsed.append(requirement)
    return tuple(parsed)


def _dependency_name(requirement: str) -> str:
    raw_name = re.split(r"[\s\[<>=!~;]", requirement, maxsplit=1)[0]
    return raw_name.casefold().replace("_", "-")


def _minimum_version(requirement: str) -> tuple[int, ...]:
    match = re.search(r">=\s*([0-9]+(?:\.[0-9]+)*)", requirement)
    assert match is not None, f"missing >= floor in {requirement!r}"
    return tuple(int(part) for part in match.group(1).split("."))


def test_forbidden_attention_packages_are_absent() -> None:
    # Given
    normalized = PYPROJECT_PATH.read_text(encoding="utf-8").casefold().replace("_", "-")

    # When
    forbidden = {name for name in ("flash-attn", "xformers") if name in normalized}

    # Then
    assert forbidden == set()


def test_local_gpu_extra_contains_one_linux_only_transaction() -> None:
    # Given
    requirements = _local_gpu_requirements()

    # When
    by_name = {_dependency_name(requirement): requirement for requirement in requirements}

    # Then
    assert frozenset(by_name) == REQUIRED_LOCAL_GPU_PACKAGES
    assert {"unsloth", "unsloth-zoo"} <= set(by_name)
    assert all(
        requirement.partition(";")[2].strip() == LINUX_X86_64_MARKER
        for requirement in requirements
    )


def test_local_gpu_binary_dependencies_keep_required_floors() -> None:
    # Given
    by_name = {
        _dependency_name(requirement): requirement
        for requirement in _local_gpu_requirements()
    }

    # When
    triton_floor = _minimum_version(by_name["triton"])
    bitsandbytes_floor = _minimum_version(by_name["bitsandbytes"])
    transformers_floor = _minimum_version(by_name["transformers"])

    # Then
    assert triton_floor >= (3, 3, 1)
    assert bitsandbytes_floor >= (0, 45, 1)
    assert transformers_floor >= (5, 2, 0)


def test_linux_torch_is_exclusively_bound_to_explicit_cu128_index() -> None:
    # Given
    document = _load_pyproject()
    tool_config = _mapping(document["tool"])
    uv_config = _mapping(tool_config["uv"])
    sources = _mapping(uv_config["sources"])
    indexes = _mapping_list(uv_config["index"])

    # When
    torch_sources = _mapping_list(sources["torch"])
    packages_using_cu128 = {
        package
        for package, package_sources in sources.items()
        for source in _mapping_list(package_sources)
        if source.get("index") == "pytorch-cu128"
    }
    cu128_indexes = tuple(
        index for index in indexes if index.get("name") == "pytorch-cu128"
    )

    # Then
    assert torch_sources == (
        {
            "index": "pytorch-cu128",
            "extra": "local-gpu",
            "marker": LINUX_X86_64_MARKER,
        },
    )
    assert packages_using_cu128 == {"torch"}
    assert cu128_indexes == (
        {
            "name": "pytorch-cu128",
            "url": "https://download.pytorch.org/whl/cu128",
            "explicit": True,
        },
    )


def test_local_gpu_artifacts_are_ignored_without_ignoring_uv_lock() -> None:
    # Given
    rules = {
        line.strip()
        for line in GITIGNORE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    # When
    required_rules = {
        "outputs/",
        "*.safetensors",
        "*.bin",
        "*.gguf",
        "**/llama.cpp/build/",
        "**/llama.cpp/build-*/",
    }

    # Then
    assert required_rules <= rules
    assert "uv.lock" not in rules
