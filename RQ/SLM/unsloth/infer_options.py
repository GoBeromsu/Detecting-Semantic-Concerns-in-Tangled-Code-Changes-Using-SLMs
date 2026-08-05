"""Typed, GPU-free command-line options for Transformers evaluation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, TypeAlias, runtime_checkable

from RQ.SLM.unsloth.results import CONTEXT_SWEEP, MESSAGE_CONDITIONS
from RQ.SLM.unsloth.data import DatasetSource, LOCAL_DATA_DIR
from RQ.SLM.unsloth.generation import MAX_NEW_TOKENS


DEFAULT_CONFIG_PATH = Path("RQ/SLM/unsloth/configs/qwen3_6_27b.yml")
DEFAULT_RESULTS_ROOT = Path("results")
MessageSelection: TypeAlias = Literal["both", "msg0", "msg1"]


@dataclass(frozen=True, slots=True)
class InferenceRunError(Exception):
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", (self.reason,))


@dataclass(frozen=True, slots=True)
class EvaluationOptions:
    config_path: Path
    adapter_path: Path | None
    data_source: DatasetSource
    data_dir: Path
    contexts: tuple[int, ...]
    message_conditions: tuple[bool, ...]
    seed: int
    temperature: float
    max_new_tokens: int
    limit: int | None
    resume: bool
    output_root: Path
    run_directory: Path | None


@dataclass(frozen=True, slots=True)
class VerifyOptions:
    run_directory: Path


@runtime_checkable
class ParsedArguments(Protocol):
    config: Path
    adapter: Path | None
    base: bool
    data_source: DatasetSource
    resume: bool
    verify_only: bool
    output: Path
    run_directory: Path | None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Qwen3.6 base or PEFT adapter via Transformers")
    _ = parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    _ = parser.add_argument("--adapter", type=Path)
    _ = parser.add_argument(
        "--base",
        action="store_true",
        help="evaluate the unadapted base model; mutually exclusive with --adapter",
    )
    _ = parser.add_argument("--data-source", "--data", choices=("local", "hub"), default="local")
    _ = parser.add_argument("--resume", action="store_true")
    _ = parser.add_argument("--run-directory", type=Path)
    _ = parser.add_argument("--verify-only", action="store_true")
    _ = parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="results root, or the complete run directory with --verify-only",
    )
    return parser


def parse_command(arguments: Sequence[str] | None = None) -> EvaluationOptions | VerifyOptions:
    """Parse CLI input into immutable evaluation or verification options."""
    parser = _parser()
    parsed = parser.parse_args(arguments)
    if not isinstance(parsed, ParsedArguments):
        raise InferenceRunError("argparse did not return the configured option shape")
    if parsed.verify_only:
        return VerifyOptions(run_directory=parsed.output)
    # Require an explicit mode: a mistyped --adapter path must never degrade into a base run.
    if (parsed.adapter is not None) == parsed.base:
        raise InferenceRunError("exactly one of --adapter or --base is required unless --verify-only is used")
    if parsed.resume and parsed.run_directory is None:
        parser.error("--resume requires --run-directory")
    return EvaluationOptions(
        config_path=parsed.config,
        adapter_path=parsed.adapter,
        data_source=parsed.data_source,
        data_dir=LOCAL_DATA_DIR,
        contexts=tuple(CONTEXT_SWEEP),
        message_conditions=MESSAGE_CONDITIONS,
        seed=42,
        temperature=0.3,
        max_new_tokens=MAX_NEW_TOKENS,
        limit=None,
        resume=parsed.resume,
        output_root=parsed.output,
        run_directory=parsed.run_directory,
    )
