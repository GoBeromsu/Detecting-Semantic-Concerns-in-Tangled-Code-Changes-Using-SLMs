from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, NamedTuple, Never, Protocol, override

import yaml

from utils.llms.constant import COMMIT_TYPES

from .data import JsonValue


@dataclass(frozen=True, slots=True)
class ConfigurationError(Exception):
    reason: str
    field: str | None = None

    @override
    def __str__(self) -> str:
        return self.reason if self.field is None else f"{self.field}: {self.reason}"


@dataclass(frozen=True, slots=True)
class ApprovedDeviation:
    max_seq_length: int


class ModelConfig(NamedTuple):
    id: str
    revision: str
    text_only: bool
    dataset_id: str
    dataset_revision: str


class PrecisionConfig(NamedTuple):
    load_in_16bit: bool
    load_in_4bit: bool
    load_in_8bit: bool


@dataclass(frozen=True, slots=True)
class LoraConfig:
    rank: int
    alpha: int
    dropout: float
    bias: str
    target_modules: tuple[str, ...]
    exclude_modules: str
    use_gradient_checkpointing: str


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    optim: str
    learning_rate: float
    num_train_epochs: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    warmup_ratio: float
    lr_scheduler_type: str
    seed: int
    logging_steps: int
    save_strategy: str
    save_total_limit: int
    eval_strategy: str
    max_seq_length: int
    packing: bool
    padding_side: str


class TechnicalConfig(NamedTuple):
    attn_implementation: str
    device_map: str


class DataConfig(NamedTuple):
    drop_rows_over_max_seq_length: bool


class MaskingConfig(NamedTuple):
    instruction_part: str
    response_part: str


class OutputConfig(NamedTuple):
    adapter_dir_root: str


class HubConfig(NamedTuple):
    adapter_repo: str
    merged_repo: str
    gguf_repo: str


class WandbConfig(NamedTuple):
    project: str
    experiment_name: str


class DiskConfig(NamedTuple):
    min_free_reserve_gib: int


@dataclass(frozen=True, slots=True)
class UnslothConfig:
    model: ModelConfig
    precision: PrecisionConfig
    lora: LoraConfig
    training: TrainingConfig
    technical: TechnicalConfig
    data: DataConfig
    masking: MaskingConfig
    output: OutputConfig
    hub: HubConfig
    wandb: WandbConfig
    disk: DiskConfig
    tags: tuple[str, ...]
    labels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HostProfile:
    hostname: str
    ssh_target: str
    gpu_name: str
    gpu_count: int
    compute_capability: str
    vram_total_mib: int
    vram_free_observed_mib: int
    known_display_processes: tuple[str, ...]
    power_cap_w: int
    ecc: str
    persistence_mode: str
    driver_version: str
    cuda_toolkit: str
    cuda_path: Path
    ram_gib: int
    swap_mib: int
    cpu: str
    cpu_threads: int
    disk_free_gib: int
    os: str
    kernel: str
    python_system: str
    notes: str

    def budget_vram_mib(self) -> int:
        return self.vram_free_observed_mib


DEFAULT_MAX_SEQ_LENGTH: Final[int] = 16384
REQUIRED_GATED_DELTA_MODULES: Final[frozenset[str]] = frozenset(("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"))
LABELS: Final[tuple[str, ...]] = tuple(COMMIT_TYPES)


class _SafeLoad(Protocol):
    def __call__(self, stream: str) -> JsonValue: ...


def _safe_loader() -> _SafeLoad:
    return yaml.safe_load


SAFE_LOAD: Final[_SafeLoad] = _safe_loader()


def _fail(reason: str, field: str | None = None) -> Never:
    raise ConfigurationError(reason=reason, field=field)


def _mapping(value: JsonValue, field: str) -> Mapping[str, JsonValue]:
    if isinstance(value, dict):
        return value
    _fail("must be a mapping", field)


def _value(values: Mapping[str, JsonValue], key: str, field: str) -> JsonValue:
    if key in values:
        return values[key]
    _fail("is required", field)


def _text(value: JsonValue, field: str) -> str:
    if isinstance(value, str):
        return value
    _fail("must be a string", field)


def _integer(value: JsonValue, field: str) -> int:
    if isinstance(value, bool):
        _fail("must be an integer", field)
    if isinstance(value, int):
        return value
    _fail("must be an integer", field)


def _number(value: JsonValue, field: str) -> float:
    if isinstance(value, bool):
        _fail("must be numeric", field)
    if isinstance(value, int | float):
        return float(value)
    _fail("must be numeric", field)


def _boolean(value: JsonValue, field: str) -> bool:
    if isinstance(value, bool):
        return value
    _fail("must be a boolean", field)


def _strings(value: JsonValue, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        _fail("must be a list of strings", field)
    parsed: list[str] = []
    for item in value:
        if not isinstance(item, str):
            _fail("must be a list of strings", field)
        parsed.append(item)
    return tuple(parsed)


@dataclass(frozen=True, slots=True)
class _Section:
    values: Mapping[str, JsonValue]
    prefix: str

    def _field(self, key: str) -> str:
        return key if not self.prefix else f"{self.prefix}.{key}"

    def text(self, key: str) -> str:
        return _text(_value(self.values, key, self._field(key)), self._field(key))

    def integer(self, key: str) -> int:
        return _integer(_value(self.values, key, self._field(key)), self._field(key))

    def number(self, key: str) -> float:
        return _number(_value(self.values, key, self._field(key)), self._field(key))

    def boolean(self, key: str) -> bool:
        return _boolean(_value(self.values, key, self._field(key)), self._field(key))

    def strings(self, key: str) -> tuple[str, ...]:
        return _strings(_value(self.values, key, self._field(key)), self._field(key))


def _section(root: Mapping[str, JsonValue], key: str) -> _Section:
    return _Section(_mapping(_value(root, key, key), key), key)


def _load_yaml(path: Path) -> Mapping[str, JsonValue]:
    try:
        raw = SAFE_LOAD(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConfigurationError(reason=str(error), field=str(path)) from error
    return _mapping(raw, "root")


def _parse_config(root: Mapping[str, JsonValue]) -> UnslothConfig:
    model, precision, lora, training = (_section(root, name) for name in ("model", "precision", "lora", "training"))
    technical, data, masking, output = (_section(root, name) for name in ("technical", "data", "masking", "output"))
    hub, wandb, disk = (_section(root, name) for name in ("hub", "wandb", "disk"))
    return UnslothConfig(
        ModelConfig(model.text("id"), model.text("revision"), model.boolean("text_only"), model.text("dataset_id"), model.text("dataset_revision")),
        PrecisionConfig(precision.boolean("load_in_16bit"), precision.boolean("load_in_4bit"), precision.boolean("load_in_8bit")),
        LoraConfig(lora.integer("rank"), lora.integer("alpha"), lora.number("dropout"), lora.text("bias"), lora.strings("target_modules"), lora.text("exclude_modules"), lora.text("use_gradient_checkpointing")),
        TrainingConfig(training.text("optim"), training.number("learning_rate"), training.integer("num_train_epochs"), training.integer("per_device_train_batch_size"), training.integer("gradient_accumulation_steps"), training.number("warmup_ratio"), training.text("lr_scheduler_type"), training.integer("seed"), training.integer("logging_steps"), training.text("save_strategy"), training.integer("save_total_limit"), training.text("eval_strategy"), training.integer("max_seq_length"), training.boolean("packing"), training.text("padding_side")),
        TechnicalConfig(technical.text("attn_implementation"), technical.text("device_map")),
        DataConfig(data.boolean("drop_rows_over_max_seq_length")),
        MaskingConfig(masking.text("instruction_part"), masking.text("response_part")),
        OutputConfig(output.text("adapter_dir_root")),
        HubConfig(hub.text("adapter_repo"), hub.text("merged_repo"), hub.text("gguf_repo")),
        WandbConfig(wandb.text("project"), wandb.text("experiment_name")),
        DiskConfig(disk.integer("min_free_reserve_gib")),
        _Section(root, "").strings("tags"), LABELS,
    )


def _validate(config: UnslothConfig, deviation: ApprovedDeviation | None) -> None:
    violations = (
        (not config.precision.load_in_16bit or config.precision.load_in_4bit or config.precision.load_in_8bit, "precision", "BF16 LoRA requires 16-bit loading only"),
        (config.technical.attn_implementation == "flash_attention_2", "technical.attn_implementation", "flash_attention_2 is forbidden on sm_120"),
        (config.training.padding_side == "left", "training.padding_side", "left padding is forbidden for causal SFT"),
        (config.training.packing, "training.packing", "packing is unsupported for GatedDeltaNet"),
        (config.training.optim == "adamw_8bit", "training.optim", "adamw_8bit is forbidden for BF16 LoRA"),
        (config.technical.device_map == "auto", "technical.device_map", "device_map must select one device"),
    )
    for invalid, field, reason in violations:
        if invalid:
            _fail(reason, field)
    if not REQUIRED_GATED_DELTA_MODULES.issubset(config.lora.target_modules):
        _fail("missing a required GatedDeltaNet projection", "lora.target_modules")
    try:
        exclusion = re.compile(config.lora.exclude_modules)
    except re.error as error:
        raise ConfigurationError("must be a valid regex", "lora.exclude_modules") from error
    # mtp (multi-token-prediction head) and visual (vision branch) receive no gradient
    # under response-only text supervision, so LoRA must exclude both.
    if exclusion.fullmatch("model.mtp.q_proj") is None or exclusion.fullmatch("model.visual.q_proj") is None:
        _fail("must exclude both mtp and visual modules", "lora.exclude_modules")
    approved_length = None if deviation is None else deviation.max_seq_length
    if config.training.max_seq_length != DEFAULT_MAX_SEQ_LENGTH and config.training.max_seq_length != approved_length:
        _fail("requires an exact ApprovedDeviation", "training.max_seq_length")


def load_config(path: Path, approved_deviation: ApprovedDeviation | None = None) -> UnslothConfig:
    configured = _parse_config(_load_yaml(path))
    _validate(configured, approved_deviation)
    return configured


load_unsloth_config: Final = load_config


def load_host_profile(path: Path) -> HostProfile:
    values = _Section(_load_yaml(path), "")
    return HostProfile(values.text("hostname"), values.text("ssh_target"), values.text("gpu_name"), values.integer("gpu_count"), values.text("compute_capability"), values.integer("vram_total_mib"), values.integer("vram_free_observed_mib"), values.strings("known_display_processes"), values.integer("power_cap_w"), values.text("ecc"), values.text("persistence_mode"), values.text("driver_version"), values.text("cuda_toolkit"), Path(values.text("cuda_path")), values.integer("ram_gib"), values.integer("swap_mib"), values.text("cpu"), values.integer("cpu_threads"), values.integer("disk_free_gib"), values.text("os"), values.text("kernel"), values.text("python_system"), values.text("notes"))


def budget_vram_mib(profile: HostProfile) -> int:
    return profile.vram_free_observed_mib
