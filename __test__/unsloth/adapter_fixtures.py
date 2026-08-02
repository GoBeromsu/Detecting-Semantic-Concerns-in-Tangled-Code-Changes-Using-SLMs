from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Final, Literal

from RQ.SLM.unsloth.train import build_manifest, write_manifest
from RQ.SLM.unsloth._types import (
    ExcludedRow,
    JsonValue,
    ManifestEvidence,
    RunProvenance,
)
from RQ.SLM.unsloth.adapter import JSON_DECODER
from RQ.SLM.unsloth.config import load_unsloth_config

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
CONFIG_PATH: Final[Path] = REPO_ROOT / "RQ/SLM/unsloth/configs/qwen3_6_27b.yml"
BASE: Final[str] = "Qwen/Qwen3.6-27B"
MODEL_REVISION: Final[str] = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
DATASET_REVISION: Final[str] = "234e8cb034bace7f6fa2a87e73e8c86bc0b04a7d"
TARGETS: Final[tuple[str, ...]] = (
    "q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z",
    "in_proj_b", "in_proj_a", "out_proj", "gate_proj", "up_proj", "down_proj",
)
EXCLUSION: Final[str] = r"(?:.*\.)?(?:mtp|visual)(?:\..*)?"
EXCLUDED_HASH: Final[str] = "fc26be9a7f99e5f0d7db53cc53714dfd0c512a269fb3bd22ea3e7bdc12b742ba"
TokenizerForm = Literal["json", "bpe"]
TOKENIZER_FILES: Final[dict[TokenizerForm, tuple[tuple[str, str], ...]]] = {
    "json": (("tokenizer.json", "{}"),),
    "bpe": (("vocab.json", "{}"), ("merges.txt", "#version: 0.2\n")),
}


def load_json(path: Path) -> dict[str, JsonValue]:
    value = JSON_DECODER.decode(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def safetensors_payload() -> bytes:
    header = json.dumps({
        "base_model.model.lora_A.weight": {
            "dtype": "F32", "shape": [1, 1], "data_offsets": [0, 4],
        },
    }, separators=(",", ":")).encode()
    return struct.pack("<Q", len(header)) + header + b"\x00\x00\x00\x00"


def _write_tokenizer(root: Path, form: TokenizerForm) -> None:
    for name, contents in TOKENIZER_FILES[form]:
        _ = (root / name).write_text(contents, encoding="utf-8")
    _ = (root / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ messages }}"}), encoding="utf-8"
    )


def _write_adapter_config(root: Path) -> None:
    _ = (root / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": BASE,
        "revision": MODEL_REVISION,
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": 32,
        "lora_alpha": 48,
        "lora_dropout": 0.05,
        "bias": "none",
        "target_modules": list(TARGETS),
        "exclude_modules": EXCLUSION,
    }), encoding="utf-8")


def write_adapter_files(root: Path, *, tokenizer_form: TokenizerForm = "json") -> None:
    _ = (root / "adapter_model.safetensors").write_bytes(safetensors_payload())
    _write_tokenizer(root, tokenizer_form)
    _write_adapter_config(root)


def write_contract(root: Path, *, tokenizer_form: TokenizerForm = "json") -> dict[str, JsonValue]:
    write_adapter_files(root, tokenizer_form=tokenizer_form)
    manifest = build_manifest(
        load_unsloth_config(CONFIG_PATH),
        RunProvenance("a" * 40, False, "smoke"),
        ManifestEvidence(
            root, CONFIG_PATH, "{{ messages }}", None, None,
            (ExcludedRow(1326, EXCLUDED_HASH, 16816),), 1399,
        ),
    )
    _ = write_manifest(root, manifest)
    return load_json(root / "run_manifest.json")


def artifact_hashes(root: Path) -> dict[str, str]:
    paths = [
        path for path in root.iterdir()
        if path.is_file() and path.name != "run_manifest.json"
    ]
    paths.extend(path for path in (root / "evidence").iterdir() if path.is_file())
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix())
    }


def contract_manifest(root: Path, artifacts: dict[str, str]) -> dict[str, JsonValue]:
    manifest = load_json(root / "run_manifest.json")
    hashes = manifest["hashes"]
    assert isinstance(hashes, dict)
    hashes["artifacts"] = artifacts
    _ = (root / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest
