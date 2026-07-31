"""CPU-only validation for the saved content-addressed LoRA adapter."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Never, Protocol

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "RQ.SLM.unsloth"

from ._types import ContractError, JsonValue, RunMode

BASE_MODEL: Final[str] = "Qwen/Qwen3.6-27B"
MODEL_REVISION: Final[str] = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
DATASET_REVISION: Final[str] = "234e8cb034bace7f6fa2a87e73e8c86bc0b04a7d"
SCHEMA_VERSION: Final[int] = 2
EVIDENCE_DIR: Final[str] = "evidence"
COMMON_EVIDENCE: Final[tuple[str, ...]] = ("config.yml", "template.txt", "training_identity.json", "provenance.json")
FULL_EVIDENCE: Final[tuple[str, ...]] = ("host_profile.yml", "template-inspection.json", "overflow-rows.json", "measurements.jsonl", "qualification.json", "preflight.json")
DTYPE_SIZES: Final[dict[str, int]] = {"BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "U16": 2, "F16": 2, "BF16": 2, "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8}


class _JsonDecoder(Protocol):
    def decode(self, s: str) -> JsonValue: ...


def _json_decoder() -> _JsonDecoder:
    return json.JSONDecoder()


JSON_DECODER: Final[_JsonDecoder] = _json_decoder()


@dataclass(frozen=True, slots=True)
class AdapterReport:
    base_model: str
    model_revision: str
    dataset_revision: str
    training_rows: int
    adapter_files: tuple[str, ...]
    tokenizer_files: tuple[str, ...]
    manifest_file: str

    def as_json(self) -> dict[str, str | int | list[str]]:
        return {"base_model": self.base_model, "model_revision": self.model_revision, "dataset_revision": self.dataset_revision, "training_rows": self.training_rows, "adapter_files": list(self.adapter_files), "tokenizer_files": list(self.tokenizer_files), "manifest_file": self.manifest_file}


def _fail(field: str, reason: str) -> Never:
    raise ContractError(f"{field}: {reason}")


def _json(path: Path, field: str) -> Mapping[str, JsonValue]:
    try:
        value = JSON_DECODER.decode(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        _fail(field, f"invalid JSON ({error})")
    if not isinstance(value, dict):
        _fail(field, "must be a JSON object")
    return value


def _mapping(value: JsonValue | None, field: str) -> Mapping[str, JsonValue]:
    if not isinstance(value, Mapping):
        _fail(field, "must be a JSON object")
    return value


def _regular(path: Path, field: str) -> None:
    if path.is_symlink() or not path.is_file():
        _fail(field, "must be a regular file")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_safetensors(path: Path) -> None:
    raw = path.read_bytes()
    if len(raw) < 8:
        _fail(path.name, "is not a safetensors file")
    header_size = int.from_bytes(raw[:8], "little")
    if header_size == 0 or header_size > 16 * 1024 * 1024 or 8 + header_size > len(raw):
        _fail(path.name, "has an invalid safetensors header")
    try:
        header = JSON_DECODER.decode(raw[8 : 8 + header_size].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail(path.name, f"has invalid safetensors JSON ({error})")
    tensors = [(name, value) for name, value in _mapping(header, path.name).items() if name != "__metadata__"]
    if not tensors or not all("lora_" in name.casefold() for name, _ in tensors):
        _fail(path.name, "must contain only LoRA tensor names")
    ranges: list[tuple[str, str, list[JsonValue], int, int]] = []
    for name, value in tensors:
        descriptor = _mapping(value, f"{path.name}.{name}")
        dtype, shape, offsets = descriptor.get("dtype"), descriptor.get("shape"), descriptor.get("data_offsets")
        if not isinstance(dtype, str) or dtype not in DTYPE_SIZES:
            _fail(path.name, f"tensor {name!r} has an unsupported dtype")
        if not isinstance(shape, list) or not shape or not isinstance(offsets, list) or len(offsets) != 2:
            _fail(path.name, f"tensor {name!r} has invalid shape or offsets")
        start, end = offsets
        if not isinstance(start, int) or not isinstance(end, int):
            _fail(path.name, f"tensor {name!r} has invalid offsets")
        ranges.append((name, dtype, shape, start, end))
    expected_start = 0
    for name, dtype, shape, start, end in sorted(ranges, key=lambda item: item[3]):
        count = 1
        for size in shape:
            if not isinstance(size, int) or size <= 0:
                _fail(path.name, f"tensor {name!r} has an invalid shape")
            count *= size
        if start != expected_start or end - start != count * DTYPE_SIZES[dtype]:
            _fail(path.name, f"tensor {name!r} has inconsistent data offsets")
        expected_start = end
    if expected_start != len(raw) - 8 - header_size:
        _fail(path.name, "payload length does not match tensor metadata")


def _manifest(root: Path) -> tuple[RunMode, Mapping[str, JsonValue], Mapping[str, JsonValue], Mapping[str, str], str]:
    manifest = _json(root / "run_manifest.json", "run_manifest")
    if set(manifest) != {"schema_version", "run_mode", "git", "identity", "hashes"} or manifest.get("schema_version") != SCHEMA_VERSION:
        _fail("run_manifest", "has an unsupported shape")
    match manifest.get("run_mode"):
        case "full": run_mode: RunMode = "full"
        case "smoke": run_mode = "smoke"
        case _: _fail("run_mode", "must be full or smoke")
    git, identity, hashes = _mapping(manifest.get("git"), "git"), _mapping(manifest.get("identity"), "identity"), _mapping(manifest.get("hashes"), "hashes")
    git_sha, git_clean = git.get("sha"), git.get("clean")
    if set(git) != {"sha", "clean"} or not isinstance(git_sha, str) or re.fullmatch(r"[0-9a-f]{40}", git_sha) is None or not isinstance(git_clean, bool):
        _fail("git", "has an invalid provenance record")
    if run_mode == "full" and git_clean is not True:
        _fail("git.clean", "full publication requires clean provenance")
    artifacts = _mapping(hashes.get("artifacts"), "hashes.artifacts")
    parsed = {name: value for name, value in artifacts.items() if isinstance(value, str) and len(value) == 64}
    template_hash = hashes.get("template_sha256")
    if len(parsed) != len(artifacts) or set(hashes) != {"artifacts", "template_sha256"} or not isinstance(template_hash, str) or len(template_hash) != 64:
        _fail("hashes", "must contain SHA-256 values")
    return run_mode, git, identity, parsed, template_hash


def _safe_files(root: Path, run_mode: RunMode) -> tuple[Path, ...]:
    if root.is_symlink() or not root.is_dir():
        _fail("adapter_dir", "must be a regular directory")
    evidence = root / EVIDENCE_DIR
    names = COMMON_EVIDENCE + (FULL_EVIDENCE if run_mode == "full" else ())
    if evidence.is_symlink() or not evidence.is_dir() or {path.name for path in evidence.iterdir()} != set(names):
        _fail(EVIDENCE_DIR, "contains missing or unexpected files")
    files: list[Path] = []
    for path in root.iterdir():
        if path.name == EVIDENCE_DIR:
            continue
        _regular(path, path.name)
        if path.name == "run_manifest.json":
            continue
        if (path.name.startswith("model") and path.suffix == ".safetensors") or path.suffix in {".gguf", ".bin"}:
            _fail("files", "full-model, GGUF, and pickle artifacts are forbidden")
        if path.stat().st_size > (4 * 1024 * 1024 * 1024 if path.name == "adapter_model.safetensors" else 64 * 1024 * 1024):
            _fail(path.name, "exceeds the evidence size limit")
        files.append(path)
    for name in ("adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json", *names):
        path = evidence / name if name in names else root / name
        _regular(path, name)
        if name in names:
            files.append(path)
    return tuple(sorted(files, key=lambda path: path.relative_to(root).as_posix()))


def _validate_identity(root: Path, identity: Mapping[str, JsonValue]) -> None:
    if set(identity) != {"effective_config", "final_row_count", "excluded_rows", "objective", "packing", "merged_model"} or _json(root / EVIDENCE_DIR / "training_identity.json", "training_identity") != identity:
        _fail("identity", "does not match captured training identity")
    effective = _mapping(identity.get("effective_config"), "identity.effective_config")
    model, lora, training = (_mapping(effective.get(name), f"identity.effective_config.{name}") for name in ("model", "lora", "training"))
    adapter = _json(root / "adapter_config.json", "adapter_config")
    expected = {"base_model_name_or_path": model.get("id"), "revision": model.get("revision"), "r": lora.get("rank"), "lora_alpha": lora.get("alpha"), "lora_dropout": lora.get("dropout"), "bias": lora.get("bias"), "target_modules": lora.get("target_modules"), "exclude_modules": lora.get("exclude_modules")}
    for field, value in expected.items():
        if adapter.get(field) != value:
            _fail(f"adapter_config.{field}", "does not match effective training identity")
    if adapter.get("peft_type") != "LORA" or adapter.get("task_type") != "CAUSAL_LM":
        _fail("adapter_config", "must be an unmerged causal LoRA adapter")
    excluded = [{"index": 1326, "hash": "fc26be9a7f99e5f0d7db53cc53714dfd0c512a269fb3bd22ea3e7bdc12b742ba", "token_length": 16816}]
    if model.get("id") != BASE_MODEL or model.get("revision") != MODEL_REVISION or identity.get("final_row_count") != 1399 or identity.get("excluded_rows") != excluded or training.get("packing") is not False or identity.get("objective") != "response_only_json_eos" or identity.get("packing") is not False or identity.get("merged_model") is not False:
        _fail("identity", "does not match the pinned unmerged training contract")


def validate_adapter(root: Path, config_path: Path | None = None) -> AdapterReport:
    run_mode, git, identity, artifacts, template_hash = _manifest(root)
    files = _safe_files(root, run_mode)
    if artifacts != {path.relative_to(root).as_posix(): _sha256(path) for path in files}:
        _fail("hashes.artifacts", "do not match the saved evidence bytes")
    provenance = dict(git); provenance["run_mode"] = run_mode
    if _json(root / EVIDENCE_DIR / "provenance.json", "provenance") != provenance:
        _fail("git", "does not match captured provenance evidence")
    _validate_safetensors(root / "adapter_model.safetensors")
    if not (root / "tokenizer.json").is_file() and not ((root / "vocab.json").is_file() and (root / "merges.txt").is_file()):
        _fail("tokenizer", "tokenizer.json or vocab.json+merges.txt is required")
    template = _json(root / "tokenizer_config.json", "tokenizer_config").get("chat_template")
    if not isinstance(template, str) and (root / "chat_template.jinja").is_file():
        template = (root / "chat_template.jinja").read_text(encoding="utf-8")
    if not isinstance(template, str) or _sha256(root / EVIDENCE_DIR / "template.txt") != hashlib.sha256(template.encode()).hexdigest() or hashlib.sha256(template.encode()).hexdigest() != template_hash:
        _fail("template", "does not match the tokenizer template")
    _validate_identity(root, identity)
    if config_path is not None:
        from .config import ConfigurationError, load_unsloth_config

        try:
            _ = load_unsloth_config(root / EVIDENCE_DIR / "config.yml")
            _ = load_unsloth_config(config_path)
        except ConfigurationError as error:
            _fail("config", str(error))
        if _sha256(config_path) != _sha256(root / EVIDENCE_DIR / "config.yml"):
            _fail("config_sha256", "selected config bytes differ from captured config")
    tokenizer_files = ("tokenizer.json", "tokenizer_config.json") if (root / "tokenizer.json").is_file() else ("vocab.json", "merges.txt", "tokenizer_config.json")
    return AdapterReport(BASE_MODEL, MODEL_REVISION, DATASET_REVISION, 1399, ("adapter_config.json", "adapter_model.safetensors"), tokenizer_files, "run_manifest.json")


def _write_report(path: Path, report: AdapterReport) -> None:
    _ = path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", encoding="utf-8", delete=False) as handle:
        _ = handle.write(json.dumps(report.as_json(), indent=2, sort_keys=True) + "\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    if arguments in (("-h",), ("--help",)):
        print("usage: adapter.py --adapter-dir DIR --config FILE --output FILE")
        return 0
    if len(arguments) != 6 or set(arguments[::2]) != {"--adapter-dir", "--config", "--output"}:
        _fail("cli", "expected --adapter-dir DIR --config FILE --output FILE")
    values = dict(zip(arguments[::2], arguments[1::2], strict=True))
    _write_report(Path(values["--output"]), validate_adapter(Path(values["--adapter-dir"]), Path(values["--config"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
