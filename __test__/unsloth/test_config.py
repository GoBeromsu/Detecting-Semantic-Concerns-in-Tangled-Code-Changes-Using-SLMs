"""Contract tests for the local Unsloth BF16 LoRA configuration boundary."""

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Final

import pytest

from RQ.SLM.unsloth.config import (
    ApprovedDeviation,
    ConfigurationError,
    budget_vram_mib,
    load_config,
    load_host_profile,
)
from utils.llms.constant import COMMIT_TYPES

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
CONFIG_PATH: Final[Path] = REPO_ROOT / "RQ/SLM/unsloth/configs/qwen3_6_27b.yml"
HOST_PATH: Final[Path] = REPO_ROOT / "RQ/SLM/configs/hosts/blackwell-rtx-pro-6000.yml"
TARGET_MODULES: Final[tuple[str, ...]] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def _write_variant(tmp_path: Path, old: str, new: str) -> Path:
    source = CONFIG_PATH.read_text(encoding="utf-8")
    assert source.count(old) == 1
    variant = tmp_path / "variant.yml"
    _ = variant.write_text(source.replace(old, new), encoding="utf-8")
    return variant


def test_load_config_when_canonical_parses_locked_model_and_precision() -> None:
    # Given/When: the research-locked local training configuration is loaded.
    config = load_config(CONFIG_PATH)

    # Then: model identity, dataset pin, precision, and single-device attention match.
    assert config.model.id == "Qwen/Qwen3.6-27B"
    assert config.model.revision == "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
    assert config.model.text_only is True
    assert config.model.dataset_id == "Berom0227/tangled-ccs-commits"
    assert config.model.dataset_revision == "234e8cb034bace7f6fa2a87e73e8c86bc0b04a7d"
    assert config.precision.load_in_16bit is True
    assert config.precision.load_in_4bit is False
    assert config.precision.load_in_8bit is False
    assert config.technical.attn_implementation == "sdpa"
    assert config.technical.device_map == "cuda:0"
    assert config.labels == tuple(COMMIT_TYPES)


def test_load_config_when_canonical_parses_locked_lora_values() -> None:
    # Given/When: the canonical LoRA section is loaded.
    lora = load_config(CONFIG_PATH).lora

    # Then: legacy hyperparameters and Qwen3.6 exclusion/checkpointing are preserved.
    assert lora.rank == 32
    assert lora.alpha == 48
    assert lora.dropout == 0.05
    assert lora.bias == "none"
    assert lora.exclude_modules == r"(?:.*\.)?(?:mtp|visual)(?:\..*)?"
    assert lora.use_gradient_checkpointing == "unsloth"


def test_target_modules_when_canonical_are_exact_and_cover_gated_delta_net() -> None:
    # Given/When: target modules are parsed from the canonical configuration.
    target_modules = load_config(CONFIG_PATH).lora.target_modules

    # Then: order and all five GatedDeltaNet projections are regression-locked.
    assert target_modules == TARGET_MODULES
    assert frozenset(("in_proj_a", "in_proj_b", "in_proj_qkv", "in_proj_z", "out_proj")).issubset(target_modules)


def test_load_config_when_canonical_parses_locked_training_values() -> None:
    # Given/When: the canonical training section is loaded.
    training = load_config(CONFIG_PATH).training

    # Then: every locked optimizer, schedule, batching, and sequence value matches.
    assert training.optim == "adamw_torch"
    assert training.learning_rate == 5e-5
    assert training.num_train_epochs == 5
    assert training.per_device_train_batch_size == 1
    assert training.gradient_accumulation_steps == 8
    assert training.warmup_ratio == 0.1
    assert training.lr_scheduler_type == "linear"
    assert training.seed == 42
    assert training.logging_steps == 10
    assert training.save_strategy == "epoch"
    assert training.save_total_limit == 2
    assert training.eval_strategy == "no"
    assert training.max_seq_length == 16384
    assert training.packing is False
    assert training.padding_side == "right"


def test_load_config_when_canonical_parses_data_and_masking_markers() -> None:
    # Given/When: data and assistant-only masking sections are loaded.
    config = load_config(CONFIG_PATH)

    # Then: oversized rows are dropped and the empty-think scaffold is unsupervised.
    assert config.data.drop_rows_over_max_seq_length is True
    assert config.masking.instruction_part == "<|im_start|>user\n"
    assert config.masking.response_part == "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def test_load_config_when_canonical_parses_locked_destinations_and_reserve() -> None:
    # Given/When: output integration sections are loaded.
    config = load_config(CONFIG_PATH)

    # Then: local, Hub, W&B, and disk destinations retain their locked identities.
    assert config.output.adapter_dir_root == "outputs/unsloth/Qwen3.6-27B-LoRA"
    assert config.hub.adapter_repo == "Berom0227/Semantic-Concern-SLM-Qwen3.6-27B-adapter"
    assert config.hub.merged_repo == "Berom0227/Semantic-Concern-SLM-Qwen3.6-27B"
    assert config.hub.gguf_repo == "Berom0227/Semantic-Concern-SLM-Qwen3.6-27B-gguf"
    assert config.wandb.project == "Untangling-Multi-Concern-Commits-with-Small-Language-Models"
    assert config.wandb.experiment_name == "qwen3.6-27b-semantic-concern-slm-unsloth-lora"
    assert config.disk.min_free_reserve_gib == 15


def test_load_config_when_created_returns_deeply_frozen_values() -> None:
    # Given: a parsed configuration and host profile.
    config = load_config(CONFIG_PATH)
    host = load_host_profile(HOST_PATH)

    # When/Then: both top-level and nested values reject mutation.
    with pytest.raises(FrozenInstanceError):
        attribute = "learning_rate"
        setattr(config.training, attribute, 1.0)
    with pytest.raises(FrozenInstanceError):
        attribute = "vram_free_observed_mib"
        setattr(host, attribute, 0)


def test_load_host_profile_when_canonical_parses_all_observed_facts() -> None:
    # Given/When: the read-only 2026-07-31 host observation is loaded.
    host = load_host_profile(HOST_PATH)

    # Then: every observed host, GPU, CPU, memory, disk, and software fact matches.
    assert host.hostname == "dcs33979"
    assert host.ssh_target == "beomsu@blackwell.tailee178.ts.net"
    assert host.gpu_name == "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"
    assert host.gpu_count == 1
    assert host.compute_capability == "12.0"
    assert host.vram_total_mib == 97887
    assert host.vram_free_observed_mib == 96793
    assert host.known_display_processes == ("Xorg", "gnome-shell", "rustdesk", "obs")
    assert host.power_cap_w == 300
    assert host.ecc == "disabled"
    assert host.persistence_mode == "disabled"
    assert host.driver_version == "595.71.05"
    assert host.cuda_toolkit == "12.8"
    assert host.cuda_path == Path("/usr/local/cuda-12.8")
    assert host.ram_gib == 31
    assert host.swap_mib == 980
    assert host.cpu == "i9-12900K"
    assert host.cpu_threads == 24
    assert host.disk_free_gib == 108
    assert host.os == "Ubuntu 22.04.5"
    assert host.kernel == "6.8.0"
    assert host.python_system == "3.10.12"
    assert host.notes == "budget VRAM against vram_free_observed_mib not total; tolerate the 4 display processes; do not change ECC/persistence/power cap."
    assert budget_vram_mib(host) == 96793
    assert host.budget_vram_mib() == 96793


@pytest.mark.parametrize(
    ("old", "new", "expected_field"),
    (
        ("  load_in_4bit: false", "  load_in_4bit: true", "precision"),
        ("  load_in_8bit: false", "  load_in_8bit: true", "precision"),
        ("  load_in_16bit: true", "  load_in_16bit: false", "precision"),
        ('  attn_implementation: "sdpa"', '  attn_implementation: "flash_attention_2"', "technical.attn_implementation"),
        ('  padding_side: "right"', '  padding_side: "left"', "training.padding_side"),
        ("  packing: false", "  packing: true", "training.packing"),
        ('  optim: "adamw_torch"', '  optim: "adamw_8bit"', "training.optim"),
        ('  device_map: "cuda:0"', '  device_map: "auto"', "technical.device_map"),
        ('    - "in_proj_a"\n', "", "lora.target_modules"),
        (r'  exclude_modules: "(?:.*\\.)?(?:mtp|visual)(?:\\..*)?"', r'  exclude_modules: "(?:.*\\.)?visual(?:\\..*)?"', "lora.exclude_modules"),
        (r'  exclude_modules: "(?:.*\\.)?(?:mtp|visual)(?:\\..*)?"', r'  exclude_modules: "(?:.*\\.)?mtp(?:\\..*)?"', "lora.exclude_modules"),
        ("  max_seq_length: 16384", "  max_seq_length: 8192", "training.max_seq_length"),
    ),
    ids=("4bit", "8bit", "16bit-disabled", "flash-attention-2", "left-padding", "packing", "8bit-optimizer", "automatic-device-map", "missing-in-proj-a", "missing-mtp-exclusion", "missing-visual-exclusion", "unapproved-sequence-length"),
)
def test_load_config_when_forbidden_value_is_present_raises_typed_error(tmp_path: Path, old: str, new: str, expected_field: str) -> None:
    # Given: one isolated YAML variant violating a locked safety invariant.
    path = _write_variant(tmp_path, old, new)

    # When/Then: parsing rejects it with the structured configuration error.
    with pytest.raises(ConfigurationError) as captured:
        _ = load_config(path)
    assert captured.value.field == expected_field


def test_load_config_when_sequence_deviation_matches_explicit_approval_passes(tmp_path: Path) -> None:
    # Given: an 8192-token variant with approval scoped to that exact value.
    path = _write_variant(tmp_path, "  max_seq_length: 16384", "  max_seq_length: 8192")
    deviation = ApprovedDeviation(max_seq_length=8192)

    # When: the variant is loaded with the explicit deviation.
    config = load_config(path, approved_deviation=deviation)

    # Then: the approved value is accepted without weakening the default guard.
    assert config.training.max_seq_length == 8192
