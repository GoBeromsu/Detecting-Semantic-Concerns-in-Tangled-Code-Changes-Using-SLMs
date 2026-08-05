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


class RuntimeConfig(Protocol):
    use_cache: bool
    architectures: list[str] | None


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
    def __call__(self, prompt: str, *, max_new_tokens: int, temperature: float) -> str: ...


class SeedFunction(Protocol):
    def __call__(self, seed: int) -> object: ...


class GeneratorFactory(Protocol):
    def __call__(self, model: OutlinesModel, output_type: JsonValue) -> Generator: ...


@runtime_checkable
class TorchModule(Protocol):
    bfloat16: RuntimeDType
    float32: RuntimeDType

    def manual_seed(self, seed: int) -> object: ...


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
    adapter_path: Path | None
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
    generator: Generator
    seed_rng: SeedFunction

    def generate(self, request: GenerationRequest) -> tuple[str, ...]:
        return generate_labels(self, request)


def _import_module(name: str) -> ModuleType:
    return importlib.import_module(name)


def _assert_runtime_precision(
    model: RuntimeModel, allowed_dtypes: tuple[RuntimeDType, ...]
) -> None:
    """Fail closed on a tower loaded in a precision we did not ask for.

    Demanding BF16 of everything was wrong, for the same reason it was wrong on the training
    side: mixed precision is the intended arrangement, not a defect. Unsloth deliberately
    keeps every normalisation weight in FP32 above a BF16 base — measured on this model,
    209 of 1843 parameters, all of them norms (64 input_layernorm, 64 post_attention_layernorm,
    49 norm, 16 q_norm, 16 k_norm) — and PEFT keeps LoRA master weights in FP32 whenever it
    holds them for training. A check that rejects FP32 rejects every model this project can
    build, which is what it did.

    So the allowed set is passed in explicitly rather than inferred from what the loader
    happened to return. What still cannot pass: FP16, which runs without complaint and
    degrades silently, and a quantised load, whose packed weights are not floating point at
    all — hence no is_floating_point() escape hatch, since that is precisely how a 4-bit
    tower would have slipped through unnoticed.
    """
    allowed = tuple(str(dtype) for dtype in allowed_dtypes)
    for name, parameter in model.named_parameters():
        if str(parameter.device) != GPU_DEVICE:
            raise GenerationError("model", f"parameter is not on CUDA: {name}")
        if str(parameter.dtype) not in allowed:
            raise GenerationError(
                "model", f"parameter has an unsupported dtype: {name} is {parameter.dtype}"
            )


def _name_architecture(base_model: RuntimeModel) -> None:
    """Record which class was built, because a text-only load leaves the config unnamed.

    Unsloth's patched generate decides whether it is driving a vision model by scanning
    config.architectures for ForConditionalGeneration/ForVisionText2Text suffixes
    (unsloth/models/vision.py:491). For Qwen3.6 that field is None and the scan dies on
    'NoneType' object is not iterable before a single token is produced: text_only=True
    replaces the composite config with its nested text_config, and the released config.json
    names architectures only at the top level, never inside that nested block. Unsloth's own
    republished copy of the model ships the same shape, so this is how the model is, not a
    stale cache.

    Filling it in from the class is what Transformers itself does whenever it saves a model,
    so the answer is right by construction rather than by luck — whatever class the loader
    actually built gets named, and the vision test then reads the truth about it. This must
    happen before the adapter is attached: afterwards the outermost class is PeftModel, whose
    name says nothing about the tower underneath.
    """
    config = base_model.config
    if config.architectures is None:
        config.architectures = [type(base_model).__name__]


def _attach_adapter(base_model: RuntimeModel, adapter_path: Path) -> RuntimeModel:
    peft = _import_module("peft")
    if not isinstance(peft, PeftModule):
        raise GenerationError("dependencies", "missing PEFT runtime API")
    return peft.PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False, autocast_adapter_dtype=False)


def _build_generator(model: RuntimeModel, tokenizer: RuntimeTokenizer) -> Generator:
    """Compile the constrained-JSON generator once, at load time, not per row."""
    outlines = _import_module("outlines")
    outlines_types = _import_module("outlines.types")
    if not isinstance(outlines, OutlinesModule) or not isinstance(outlines_types, OutlinesTypesModule):
        raise GenerationError("dependencies", "missing Outlines 1.3.2 public API")
    schema = outlines_types.JsonSchema(json.dumps(STRICT_RESPONSE_SCHEMA, separators=(",", ":"), sort_keys=True))
    return outlines.Generator(outlines.from_transformers(model, tokenizer), schema)


def load_backend(request: PeftLoadRequest) -> PeftBackend:
    """Load the base tower, optionally attach the LoRA adapter, and fail closed early."""
    torch = _import_module("torch")
    unsloth = _import_module("unsloth")
    if not isinstance(torch, TorchModule) or not isinstance(unsloth, UnslothModule):
        raise GenerationError("dependencies", "missing Unsloth runtime API")
    base_model, tokenizer = unsloth.FastLanguageModel.from_pretrained(model_name=request.model_id, revision=request.revision, max_seq_length=request.max_seq_length, dtype=torch.bfloat16, load_in_4bit=False, load_in_8bit=False, load_in_16bit=True, device_map={"": 0}, attn_implementation="sdpa", text_only=True)
    _name_architecture(base_model)
    model = base_model if request.adapter_path is None else _attach_adapter(base_model, request.adapter_path)
    model.eval()
    model.config.use_cache = True
    _assert_runtime_precision(model, (torch.bfloat16, torch.float32))
    return PeftBackend(model, tokenizer, request.max_seq_length, _build_generator(model, tokenizer), torch.manual_seed)


def _render_prompt(tokenizer: RuntimeTokenizer, request: GenerationRequest) -> str:
    # Invariant: thinking is always disabled; the response must be pure JSON+EOS.
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
    # Seed the global RNG here rather than handing the seed to Outlines. Its Transformers
    # backend forwards every inference kwarg straight into model.generate
    # (outlines/models/transformers.py:349), and generate rejects anything it does not
    # recognise — 'seed' is honoured by the vLLM and llama.cpp backends, not this one, so
    # passing it raised before a single token was produced. Sampling is live here
    # (Qwen3.6's generation_config sets do_sample=true, so temperature=0.3 is real), and
    # torch.manual_seed covers the CUDA generator the sampler actually draws from. Seeding
    # per row, not once per run, is what makes a row reproducible on its own: a resumed run
    # skips completed rows, so a run-level seed would put every resumed row on a different
    # RNG stream than the original pass.
    _ = backend.seed_rng(request.seed)
    raw_output = backend.generator(rendered, max_new_tokens=request.max_new_tokens, temperature=request.temperature)
    try:
        return _parse_model_output(raw_output)
    except ModelOutputError as error:
        raise GenerationError("structured_output", str(error), raw_output) from error
