from __future__ import annotations

from argparse import Namespace
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Final,
    Literal,
    NotRequired,
    Protocol,
    TypeAlias,
    TypedDict,
    Unpack,
    override,
    runtime_checkable,
)

from .data import ChatMessage


class DtypeLike(Protocol): ...


class ModelLoadKwargs(TypedDict):
    model_name: str
    revision: str
    max_seq_length: int
    dtype: DtypeLike
    load_in_16bit: bool
    load_in_4bit: bool
    load_in_8bit: bool
    full_finetuning: bool
    text_only: bool
    trust_remote_code: bool
    attn_implementation: str
    use_gradient_checkpointing: str
    device_map: dict[str, int]


class PeftKwargs(TypedDict):
    r: int
    target_modules: tuple[str, ...]
    lora_alpha: int
    lora_dropout: float
    bias: str
    use_gradient_checkpointing: str
    random_state: int
    max_seq_length: int
    exclude_modules: str
    autocast_adapter_dtype: bool
    finetune_vision_layers: bool


class TokenIds(TypedDict):
    input_ids: Sequence[int]


class TokenizerLike(Protocol):
    padding_side: str
    chat_template: str
    eos_token_id: int

    def apply_chat_template(
        self,
        conversation: Sequence[ChatMessage],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str: ...

    def __call__(self, text: str, *, add_special_tokens: bool) -> TokenIds: ...

    def save_pretrained(self, save_directory: str) -> None: ...


class ModelConfigLike(Protocol):
    use_cache: bool


class TrainableParameter(Protocol):
    requires_grad: bool
    dtype: DtypeLike


class TrainableAdapter(Protocol):
    def named_parameters(self) -> Iterator[tuple[str, TrainableParameter]]: ...


class ModelLike(Protocol):
    @property
    def config(self) -> ModelConfigLike: ...

    def named_parameters(self) -> Iterator[tuple[str, TrainableParameter]]: ...


class TrainerLike(Protocol):
    def train(self) -> None: ...

    def save_model(self, output_dir: str) -> None: ...


class SftConfigLike(Protocol): ...


class TrainingDataset(Protocol):
    @property
    def column_names(self) -> Sequence[str]: ...


class DatasetFactory(Protocol):
    def from_list(self, rows: list[RenderedExample]) -> TrainingDataset: ...


@runtime_checkable
class DatasetsModule(Protocol):
    Dataset: DatasetFactory


class RenderedExample(TypedDict):
    text: str


class SftKwargs(TypedDict):
    output_dir: str
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    num_train_epochs: int
    warmup_ratio: float
    lr_scheduler_type: str
    optim: str
    bf16: bool
    fp16: bool
    max_length: int
    dataset_text_field: str
    packing: bool
    save_strategy: str
    save_total_limit: int
    eval_strategy: str
    seed: int
    report_to: str
    gradient_checkpointing: bool
    use_cache: bool
    run_name: NotRequired[str]
    max_steps: NotRequired[int]


class FastLanguageModelFactory(Protocol):
    def from_pretrained(
        self, **kwargs: Unpack[ModelLoadKwargs]
    ) -> tuple[ModelLike, TokenizerLike]: ...

    def get_peft_model(self, model: ModelLike, **kwargs: Unpack[PeftKwargs]) -> ModelLike: ...


@runtime_checkable
class TorchModule(Protocol):
    bfloat16: DtypeLike
    float32: DtypeLike


@runtime_checkable
class UnslothModule(Protocol):
    FastLanguageModel: FastLanguageModelFactory


class SftConfigFactory(Protocol):
    def __call__(self, **kwargs: Unpack[SftKwargs]) -> SftConfigLike: ...


class SftTrainerFactory(Protocol):
    def __call__(
        self,
        *,
        model: ModelLike,
        args: SftConfigLike,
        train_dataset: TrainingDataset,
        processing_class: TokenizerLike,
    ) -> TrainerLike: ...


class ResponseMasker(Protocol):
    def __call__(
        self, trainer: TrainerLike, *, instruction_part: str, response_part: str
    ) -> TrainerLike: ...


@runtime_checkable
class TrlModule(Protocol):
    SFTConfig: SftConfigFactory
    SFTTrainer: SftTrainerFactory


@runtime_checkable
class TemplateModule(Protocol):
    train_on_responses_only: ResponseMasker


@dataclass(frozen=True, slots=True)
class UnslothRuntime:
    model: ModelLike
    tokenizer: TokenizerLike


if TYPE_CHECKING:
    from .config import HostProfile, UnslothConfig


GIB_BYTES: Final[int] = 1024**3
MODEL_STORAGE_BYTES: Final[int] = 55_562_855_904
PROTECTED_CACHE_NAMES: Final[tuple[str, str]] = (
    "models--Qwen--Qwen3-14B",
    "models--Berom0227--Semantic-Concern-SLM-Qwen-gguf",
)

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | Sequence["JsonValue"] | Mapping[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]
RunMode: TypeAlias = Literal["full", "smoke"]


class JsonDecoder(Protocol):
    def decode(self, s: str) -> JsonValue: ...




@dataclass(frozen=True, slots=True)
class ContractError(Exception):
    """An adapter evidence contract violation."""

    reason: str

    @override
    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class ExcludedRow:
    index: int
    row_hash: str
    token_length: int


@dataclass(frozen=True, slots=True)
class RunProvenance:
    git_sha: str
    git_clean: bool
    run_mode: RunMode = "full"


@dataclass(frozen=True, slots=True)
class ManifestEvidence:
    adapter_dir: Path
    config_path: Path
    chat_template: str
    host_profile_path: Path | None
    qualification_dir: Path | None
    excluded_rows: tuple[ExcludedRow, ...]
    final_count: int


class ExcludedRowManifest(TypedDict):
    index: int
    hash: str
    token_length: int


class GitManifest(TypedDict):
    sha: str
    clean: bool


class HashManifest(TypedDict):
    artifacts: dict[str, str]
    template_sha256: str


class RunManifest(TypedDict):
    schema_version: int
    run_mode: RunMode
    git: GitManifest
    identity: dict[str, JsonValue]
    hashes: HashManifest


class CsvResultRow(TypedDict):
    predicted_types: str
    actual_types: str
    inference_time: float
    shas: str
    precision: float
    recall: float
    f1: float
    exact_match: bool
    hamming_loss: float
    context_len: int
    with_message: bool
    concern_count: int


@dataclass(frozen=True, slots=True)
class PreflightError(Exception):
    field: str
    reason: str

    @override
    def __str__(self) -> str:
        return f"{self.field}: {self.reason}"


@dataclass(frozen=True, slots=True)
class ProbeFacts:
    hostname: str
    gpu_name: str
    gpu_count: int
    compute_capability: str
    vram_total_mib: int
    vram_free_mib: int
    power_limit_w: float
    ecc_mode: str
    persistence_mode: str
    cuda_runtime: str
    bf16_supported: bool
    compute_processes: tuple[str | None, ...]
    graphics_processes: tuple[str | None, ...]
    ram_total_bytes: int
    ram_free_bytes: int
    swap_total_bytes: int
    swap_free_bytes: int
    disk_free_bytes: int


@dataclass(frozen=True, slots=True)
class PreflightInputs:
    config: UnslothConfig
    profile: HostProfile
    probe: ProbeFacts
    cached_bytes: int


@dataclass(frozen=True, slots=True)
class PreflightResult:
    valid: bool
    observed_vram_free_mib: int
    required_vram_free_mib: int
    disk_free_bytes: int
    cached_bytes: int
    missing_model_bytes: int
    reserve_bytes: int
    remaining_disk_bytes: int
    warnings: tuple[str, ...]
    require_peak_rss_measurement: bool
    protected_cache_names: tuple[str, ...]

    def as_json(self) -> JsonObject:
        return {
            "valid": self.valid,
            "observed_vram_free_mib": self.observed_vram_free_mib,
            "required_vram_free_mib": self.required_vram_free_mib,
            "disk_free_bytes": self.disk_free_bytes,
            "cached_bytes": self.cached_bytes,
            "missing_model_bytes": self.missing_model_bytes,
            "reserve_bytes": self.reserve_bytes,
            "remaining_disk_bytes": self.remaining_disk_bytes,
            "warnings": list(self.warnings),
            "require_peak_rss_measurement": self.require_peak_rss_measurement,
            "protected_cache_names": list(self.protected_cache_names),
        }


@dataclass(frozen=True, slots=True)
class PreflightEvidence:
    config_sha256: str
    host_profile_sha256: str
    hostname: str
    cached_bytes: int
    probe: JsonObject
    result: JsonObject

    def as_json(self) -> JsonObject:
        return {
            "config_sha256": self.config_sha256,
            "host_profile_sha256": self.host_profile_sha256,
            "hostname": self.hostname,
            "cached_bytes": self.cached_bytes,
            "probe": self.probe,
            "result": self.result,
        }


class PreflightArguments(Namespace):
    config: Path
    host_profile: Path
    probe_json: Path
    cached_bytes: int
    output: Path

    def __init__(self) -> None:
        super().__init__()
        self.config = Path()
        self.host_profile = Path()
        self.probe_json = Path()
        self.cached_bytes = 0
        self.output = Path()
