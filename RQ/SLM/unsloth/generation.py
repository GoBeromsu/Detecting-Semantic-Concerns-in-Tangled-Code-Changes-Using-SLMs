"""Lazy BF16 PEFT backend using the supported Outlines 1.3.2 API."""

from __future__ import annotations

import importlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import (
    Final,
    Literal,
    Protocol,
    TypeAlias,
    TypedDict,
    override,
    runtime_checkable,
)

from RQ.SLM.unsloth._types import JsonDecoder, JsonValue
from utils.llms.constant import COMMIT_TYPES

STRICT_RESPONSE_SCHEMA: Final[Mapping[str, JsonValue]] = {"type": "object", "properties": {"types": {"type": "array", "items": {"type": "string", "enum": COMMIT_TYPES}, "minItems": 1, "maxItems": 7, "uniqueItems": True}}, "required": ["types"], "additionalProperties": False}
def _json_decoder() -> JsonDecoder:
    return json.JSONDecoder()


JSON_DECODER: Final[JsonDecoder] = _json_decoder()


@dataclass(frozen=True, slots=True)
class ModelOutputError(Exception):
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", (self.reason,))


def _parse_model_output(raw_text: str) -> tuple[str, ...]:
    try:
        payload: JsonValue = JSON_DECODER.decode(raw_text)
    except json.JSONDecodeError as error:
        raise ModelOutputError("invalid JSON object") from error
    if not isinstance(payload, dict) or set(payload) != {"types"}:
        raise ModelOutputError("invalid JSON object")
    raw_labels = payload["types"]
    if not isinstance(raw_labels, Sequence) or isinstance(raw_labels, str):
        raise ModelOutputError("types must be an array of strings")
    labels = tuple(label for label in raw_labels if isinstance(label, str))
    if len(labels) != len(raw_labels):
        raise ModelOutputError("types must be an array of strings")
    if not labels:
        raise ModelOutputError("types must contain at least one label")
    if len(labels) > len(COMMIT_TYPES):
        raise ModelOutputError("types exceeds the seven-label maximum")
    if unknown := next((label for label in labels if label not in COMMIT_TYPES), None):
        raise ModelOutputError(f"unknown label {unknown!r}")
    if len(labels) != len(set(labels)):
        raise ModelOutputError("duplicate label")
    return labels

GPU_DEVICE: Literal["cuda:0"] = "cuda:0"
MAX_NEW_TOKENS = 128
NON_THINKING_TAIL = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


class ChatMessage(TypedDict):
    role: Literal["system", "user"]
    content: str


class RuntimeDType(Protocol): ...


class RuntimeParameter(Protocol):
    device: str
    dtype: RuntimeDType

    def is_floating_point(self) -> bool: ...


class RuntimeConfig(Protocol):
    use_cache: bool


class RuntimeModel(Protocol):
    config: RuntimeConfig

    def named_parameters(self) -> Iterable[tuple[str, RuntimeParameter]]: ...
    def eval(self) -> None: ...


class TokenIds(Protocol):
    shape: tuple[int, ...]


class TokenBatch(Protocol):
    def to(self, device: str) -> TokenBatch: ...
    def __getitem__(self, key: str) -> TokenIds: ...


RuntimeArgument: TypeAlias = str | int | float | bool | Path | TokenIds | TokenBatch


class RuntimeTokenizer(Protocol):
    pad_token_id: int | None
    eos_token_id: int

    def apply_chat_template(self, conversation: Sequence[ChatMessage], *, tokenize: bool, add_generation_prompt: bool, enable_thinking: bool) -> str: ...
    def __call__(self, text: str, *, return_tensors: Literal["pt"]) -> TokenBatch: ...


class FastLanguageModelFactory(Protocol):
    def from_pretrained(self, *, model_name: str, revision: str, max_seq_length: int, dtype: RuntimeDType, load_in_4bit: bool, load_in_8bit: bool, load_in_16bit: bool, device_map: Mapping[str, int], attn_implementation: str, text_only: bool) -> tuple[RuntimeModel, RuntimeTokenizer]: ...


class PeftModelFactory(Protocol):
    def from_pretrained(self, model: RuntimeModel, adapter_path: Path, *, is_trainable: bool, autocast_adapter_dtype: bool) -> RuntimeModel: ...


class OutlinesModel(Protocol): ...


class JsonSchemaFactory(Protocol):
    def __call__(self, schema: str) -> JsonValue: ...


class Generator(Protocol):
    def __call__(self, prompt: str, *, max_new_tokens: int, temperature: float, seed: int) -> str: ...


class GeneratorFactory(Protocol):
    def __call__(self, model: OutlinesModel, output_type: JsonValue) -> Generator: ...


@runtime_checkable
class TorchModule(Protocol):
    bfloat16: RuntimeDType


@runtime_checkable
class UnslothModule(Protocol):
    FastLanguageModel: FastLanguageModelFactory


@runtime_checkable
class PeftModule(Protocol):
    PeftModel: PeftModelFactory


@runtime_checkable
class OutlinesModule(Protocol):
    Generator: GeneratorFactory
    def from_transformers(self, model: RuntimeModel, tokenizer: RuntimeTokenizer) -> OutlinesModel: ...


@runtime_checkable
class OutlinesTypesModule(Protocol):
    JsonSchema: JsonSchemaFactory


@dataclass(frozen=True, slots=True)
class GenerationError(Exception):
    phase: str
    reason: str
    raw_output: str | None = None

    @override
    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class PeftLoadRequest:
    model_id: str
    revision: str
    adapter_path: Path
    max_seq_length: int


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    system_prompt: str
    commit: str
    seed: int
    temperature: float = 0.3
    max_new_tokens: int = MAX_NEW_TOKENS

    def __post_init__(self) -> None:
        if self.temperature < 0 or self.max_new_tokens < 1 or self.max_new_tokens > MAX_NEW_TOKENS:
            raise GenerationError("configuration", "invalid generation configuration")


@dataclass(frozen=True, slots=True)
class PeftBackend:
    model: RuntimeModel
    tokenizer: RuntimeTokenizer
    model_max_tokens: int

    def generate(self, request: GenerationRequest) -> tuple[str, ...]:
        return generate_labels(self, request)


def _import_module(name: str) -> ModuleType:
    return importlib.import_module(name)


def _assert_text_bf16_cuda(model: RuntimeModel, dtype: RuntimeDType) -> None:
    for name, parameter in model.named_parameters():
        if str(parameter.device) != GPU_DEVICE:
            raise GenerationError("model", f"parameter is not on CUDA: {name}")
        if parameter.is_floating_point() and str(parameter.dtype) != str(dtype):
            raise GenerationError("model", f"floating parameter is not BF16: {name}")


def load_backend(request: PeftLoadRequest) -> PeftBackend:
    torch = _import_module("torch")
    unsloth = _import_module("unsloth")
    peft = _import_module("peft")
    if not isinstance(torch, TorchModule) or not isinstance(unsloth, UnslothModule) or not isinstance(peft, PeftModule):
        raise GenerationError("dependencies", "missing Unsloth or PEFT runtime API")
    base_model, tokenizer = unsloth.FastLanguageModel.from_pretrained(model_name=request.model_id, revision=request.revision, max_seq_length=request.max_seq_length, dtype=torch.bfloat16, load_in_4bit=False, load_in_8bit=False, load_in_16bit=True, device_map={"": 0}, attn_implementation="sdpa", text_only=True)
    model = peft.PeftModel.from_pretrained(base_model, request.adapter_path, is_trainable=False, autocast_adapter_dtype=False)
    model.eval()
    model.config.use_cache = True
    _assert_text_bf16_cuda(model, torch.bfloat16)
    return PeftBackend(model, tokenizer, request.max_seq_length)


def _render_prompt(tokenizer: RuntimeTokenizer, request: GenerationRequest) -> str:
    rendered = tokenizer.apply_chat_template(({"role": "system", "content": request.system_prompt}, {"role": "user", "content": request.commit}), tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if not rendered.endswith(NON_THINKING_TAIL):
        raise GenerationError("template", "missing Qwen non-thinking scaffold")
    return rendered


def validate_prompt_budget(input_tokens: int, *, max_new_tokens: int, model_max_tokens: int) -> None:
    if input_tokens + max_new_tokens > model_max_tokens:
        raise GenerationError("context", "rendered prompt plus output exceeds model context")


def generate_labels(backend: PeftBackend, request: GenerationRequest) -> tuple[str, ...]:
    rendered = _render_prompt(backend.tokenizer, request)
    tokenized = backend.tokenizer(rendered, return_tensors="pt").to(GPU_DEVICE)
    validate_prompt_budget(tokenized["input_ids"].shape[-1], max_new_tokens=request.max_new_tokens, model_max_tokens=backend.model_max_tokens)
    if backend.tokenizer.pad_token_id is None or backend.tokenizer.pad_token_id < 0:
        raise GenerationError("tokenizer", "tokenizer pad token is missing or invalid")
    if backend.tokenizer.pad_token_id == backend.tokenizer.eos_token_id:
        raise GenerationError("tokenizer", "tokenizer pad token must differ from eos token")
    outlines = _import_module("outlines")
    outlines_types = _import_module("outlines.types")
    if not isinstance(outlines, OutlinesModule) or not isinstance(outlines_types, OutlinesTypesModule):
        raise GenerationError("dependencies", "missing Outlines 1.3.2 public API")
    schema = outlines_types.JsonSchema(json.dumps(STRICT_RESPONSE_SCHEMA, separators=(",", ":"), sort_keys=True))
    generator = outlines.Generator(outlines.from_transformers(backend.model, backend.tokenizer), schema)
    raw_output = generator(rendered, max_new_tokens=request.max_new_tokens, temperature=request.temperature, seed=request.seed)
    try:
        return _parse_model_output(raw_output)
    except ModelOutputError as error:
        raise GenerationError("structured_output", str(error), raw_output) from error
