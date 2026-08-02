import inspect
import subprocess
from pathlib import Path
from textwrap import dedent
from typing import Final

import pytest


REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
BLOCK_LLAMA_CPP: Final[str] = dedent(
    """
    import importlib.abc
    import sys

    class BlockLlamaCpp(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "llama_cpp" or fullname.startswith("llama_cpp."):
                raise ImportError(f"blocked heavy module: {fullname}")
            return None

    sys.meta_path.insert(0, BlockLlamaCpp())
    """
)
EXPECTED_UTILS_EXPORTS: Final[list[str]] = [
    "load_model_and_tokenizer",
    "get_prediction",
    "get_models",
    "load_model",
    "api_call",
    "calculate_metrics",
]
EXPECTED_LLMS_EXPORTS: Final[list[str]] = [
    "api_call",
    "get_models",
    "load_model",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_MAX_TOKENS",
    "COMMIT_TYPES",
    "RESPONSE_SCHEMA",
    "OPENAI_STRUCTURED_OUTPUT_FORMAT",
    "LMSTUDIO_STRUCTURED_OUTPUT_FORMAT",
]
PROVIDER_MODULES: Final[tuple[str, ...]] = (
    "utils.llms.openai",
    "utils.llms.hugging_face",
    "utils.llms.lmstudio",
)
EXPECTED_API_CALL_SIGNATURE: Final[str] = (
    "(provider: str, model_name: str, commit: str, system_prompt: str, "
    "temperature: float = 0.3, max_tokens: int = 16384, api_key: str = '') "
    "-> list[str]"
)


def _run_python(source: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["uv", "run", "python", "-c", source],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def test_prompt_and_eval_import_without_llama_cpp() -> None:
    result = _run_python(
        BLOCK_LLAMA_CPP
        + "\nimport utils.prompt\nimport utils.eval\n"
    )

    assert result.returncode == 0, result.stderr


def test_importing_utils_does_not_load_llama_cpp() -> None:
    result = _run_python(
        BLOCK_LLAMA_CPP
        + '\nimport utils\nassert "llama_cpp" not in sys.modules\n'
    )

    assert result.returncode == 0, result.stderr


def test_importing_llms_eagerly_loads_only_constants() -> None:
    provider_assertions = "\n".join(
        f'assert "{module_name}" not in sys.modules'
        for module_name in PROVIDER_MODULES
    )
    result = _run_python(
        BLOCK_LLAMA_CPP
        + "\nimport utils.llms\n"
        + 'assert "utils.llms.constant" in sys.modules\n'
        + provider_assertions
        + "\n"
    )

    assert result.returncode == 0, result.stderr


def test_utils_public_exports_are_unchanged() -> None:
    import utils

    assert utils.__all__ == EXPECTED_UTILS_EXPORTS


def test_llms_public_exports_are_unchanged() -> None:
    import utils.llms as llms

    assert llms.__all__ == EXPECTED_LLMS_EXPORTS


def test_utils_public_callables_still_resolve() -> None:
    import utils

    public_callables = (
        utils.load_model_and_tokenizer,
        utils.get_prediction,
        utils.get_models,
        utils.load_model,
        utils.api_call,
        utils.calculate_metrics,
    )

    assert all(callable(public_callable) for public_callable in public_callables)


def test_unified_api_call_signature_is_unchanged() -> None:
    import utils.llms as llms

    signature = inspect.signature(llms.api_call)

    assert str(signature) == EXPECTED_API_CALL_SIGNATURE


def test_unified_api_call_invalid_provider_message_is_unchanged() -> None:
    import utils.llms as llms

    with pytest.raises(ValueError) as raised:
        _ = llms.api_call(
            provider="invalid",
            model_name="model",
            commit="commit",
            system_prompt="prompt",
        )

    assert str(raised.value) == "Invalid provider: invalid"


def test_legacy_provider_call_aliases_still_resolve() -> None:
    import utils.llms as llms

    assert callable(llms.openai_api_call)
    assert callable(llms.hugging_face_api_call)
    assert callable(llms.lmstudio_api_call)
