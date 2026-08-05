from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Literal, TypeAlias

import pytest

from RQ.SLM.unsloth._types import JsonValue
from RQ.SLM.unsloth.generation import (
    ChatMessage,
    GenerationRequest,
    PeftBackend,
    PeftLoadRequest,
    RuntimeArgument,
    GenerationError,
    generate_labels,
    load_backend,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPT_TAIL = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
LoadArgument: TypeAlias = str | int | bool | dict[str, int]


class FakeParameter:
    device: str = "cuda:0"
    dtype: str = "bfloat16"


class FakeNormParameter(FakeParameter):
    """A normalisation weight as Unsloth actually returns one: FP32 above a BF16 base.

    Measured on the real 27B load: 209 of 1843 parameters, every one of them a norm.
    """

    dtype: str = "float32"


class FakeConfig:
    use_cache: bool = False
    # None is what a text-only load actually hands back: the nested text_config never names
    # its architecture. A fake that pre-filled this could not have caught the crash it caused.
    architectures: list[str] | None = None


class FakeCausalLM:
    def __init__(self) -> None:
        self.config: FakeConfig = FakeConfig()

    def named_parameters(self) -> tuple[tuple[str, FakeParameter], ...]:
        # All three kinds a real load returns. The norm is the one that matters: a fake made of
        # BF16 weights alone let a load-time check ship that no real model could have passed,
        # and it took a GPU run to find out. Adapter weights come back BF16 at inference —
        # PEFT only holds them in FP32 while it is training them.
        return (
            ("model.layers.0.self_attn.q_proj.weight", FakeParameter()),
            ("model.layers.0.input_layernorm.weight", FakeNormParameter()),
            ("base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight", FakeParameter()),
        )


class FakePeftModelForCausalLM(FakeCausalLM):
    def __init__(self) -> None:
        super().__init__()
        self.generation_calls: list[dict[str, RuntimeArgument]] = []
        self.evaluated: bool = False

    def eval(self) -> None:
        self.evaluated = True

    def generate(self, **kwargs: RuntimeArgument) -> FakeGeneratedIds:
        self.generation_calls.append(kwargs)
        return FakeGeneratedIds()


class FakeFastLanguageModel:
    calls: list[dict[str, LoadArgument]] = []
    last_tokenizer: FakeTokenizer | None = None
    last_model: FakeCausalLM | None = None

    @classmethod
    def from_pretrained(
        cls, **kwargs: LoadArgument
    ) -> tuple[FakeCausalLM, FakeTokenizer]:
        cls.calls.append(kwargs)
        tokenizer = FakeTokenizer('{"types": ["fix"]}')
        cls.last_tokenizer = tokenizer
        cls.last_model = FakeCausalLM()
        return cls.last_model, tokenizer


class FakePeftModel:
    model: FakePeftModelForCausalLM = FakePeftModelForCausalLM()
    calls: list[dict[str, FakeCausalLM | Path | bool]] = []

    @classmethod
    def from_pretrained(
        cls, model: FakeCausalLM, adapter_path: Path, **kwargs: bool
    ) -> FakePeftModelForCausalLM:
        cls.calls.append({"model": model, "adapter_path": adapter_path, **kwargs})
        return cls.model


class FakeIds:
    shape: tuple[int, ...] = (1, 2)


class FakeBatch(dict[str, FakeIds]):
    def to(self, device: str) -> FakeBatch:
        assert device == "cuda:0"
        return self


class FakeGeneratedIds:
    def __getitem__(self, index: tuple[int, slice]) -> tuple[int, int]:
        assert index == (0, slice(2, None))
        return (101, 102)


class FakeTokenizer:
    def __init__(self, decoded: str) -> None:
        self.decoded: str = decoded
        self.rendered_messages: list[Sequence[ChatMessage]] = []
        self.template_kwargs: list[dict[str, bool]] = []
        self.tokenized_prompts: list[str] = []
        self.decoded_ids: list[tuple[int, ...]] = []
        self.pad_token_id: int | None = 248044
        self.eos_token_id: int = 248046

    def apply_chat_template(
        self, messages: Sequence[ChatMessage], **kwargs: bool
    ) -> str:
        self.rendered_messages.append(messages)
        self.template_kwargs.append(kwargs)
        return "rendered " + PROMPT_TAIL

    def __call__(self, prompt: str, *, return_tensors: Literal["pt"]) -> FakeBatch:
        assert return_tensors == "pt"
        self.tokenized_prompts.append(prompt)
        return FakeBatch(input_ids=FakeIds())

    def decode(self, ids: Iterable[int], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        self.decoded_ids.append(tuple(ids))
        return self.decoded


class FakeGenerator:
    def __init__(self, *, device: str) -> None:
        assert device == "cuda:0"
        self.seed: int | None = None

    def manual_seed(self, seed: int) -> FakeGenerator:
        self.seed = seed
        return self


class FakeOutlinesGenerator:
    calls: list[tuple[str, int, float, int]] = []

    def __init__(self, _: FakePeftModelForCausalLM, __: str) -> None:
        pass

    def __call__(self, prompt: str, *, max_new_tokens: int, temperature: float, seed: int) -> str:
        self.calls.append((prompt, max_new_tokens, temperature, seed))
        tokenizer = FakeFastLanguageModel.last_tokenizer
        assert tokenizer is not None
        return tokenizer.decoded


class FakeTorch:
    Generator: type[FakeGenerator] = FakeGenerator
    bfloat16: str = "bfloat16"
    float32: str = "float32"


class FakeJsonLogitsProcessor:
    calls: list[tuple[Mapping[str, JsonValue], FakeTokenizer, str | None]] = []

    def __init__(
        self, schema: Mapping[str, JsonValue], tokenizer: FakeTokenizer, *,
        tensor_library_name: str | None = None,
    ) -> None:
        self.calls.append((schema, tokenizer, tensor_library_name))


class FakeLogitsProcessorList(list[FakeJsonLogitsProcessor]):
    pass


class FakeModules:
    FastLanguageModel: type[FakeFastLanguageModel] = FakeFastLanguageModel
    PeftModel: type[FakePeftModel] = FakePeftModel
    JSONLogitsProcessor: type[FakeJsonLogitsProcessor] = FakeJsonLogitsProcessor
    LogitsProcessorList: type[FakeLogitsProcessorList] = FakeLogitsProcessorList
    Generator: type[FakeOutlinesGenerator] = FakeOutlinesGenerator
    JsonSchema: type[str] = str

    @staticmethod
    def from_transformers(model: FakePeftModelForCausalLM, _: FakeTokenizer) -> FakePeftModelForCausalLM:
        return model


def _patch_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    from RQ.SLM.unsloth import generation

    modules = {
        "torch": FakeTorch,
        "unsloth": FakeModules,
        "peft": FakeModules,
        "outlines": FakeModules,
        "outlines.types": FakeModules,
    }
    monkeypatch.setattr(generation, "_import_module", modules.__getitem__)


def _request() -> PeftLoadRequest:
    return PeftLoadRequest(
        model_id="Qwen/Qwen3.6-27B", revision="pinned-revision",
        adapter_path=Path("outputs/adapter"), max_seq_length=16384,
    )


def _generate(
    backend: PeftBackend, *, seed: int = 42, temperature: float = 0.3
) -> tuple[str, ...]:
    request = GenerationRequest(
        system_prompt="system", commit="commit", seed=seed, temperature=temperature
    )
    return generate_labels(backend, request)


def test_load_backend_when_requested_uses_text_only_unsloth_and_unmerged_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: optional GPU packages are replaced by narrow no-GPU fakes.
    _patch_modules(monkeypatch)

    # When: the BF16 adapter backend is loaded.
    _ = load_backend(_request())

    # Then: Unsloth remaps the official text tower and PEFT attaches without merging.
    assert FakeFastLanguageModel.calls[-1] == {
        "model_name": "Qwen/Qwen3.6-27B",
        "revision": "pinned-revision",
        "max_seq_length": 16384,
        "dtype": FakeTorch.bfloat16,
        "load_in_4bit": False,
        "load_in_8bit": False,
        "load_in_16bit": True,
        "device_map": {"": 0},
        "attn_implementation": "sdpa",
        "text_only": True,
    }
    assert FakePeftModel.calls[-1]["adapter_path"] == Path("outputs/adapter")
    assert FakePeftModel.calls[-1]["is_trainable"] is False
    assert FakePeftModel.calls[-1]["autocast_adapter_dtype"] is False
    assert FakePeftModel.model.config.use_cache is True
    assert FakePeftModel.model.evaluated is True


def test_load_backend_when_the_text_only_config_is_unnamed_records_the_built_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a text-only load, whose nested config carries no architecture name.
    _patch_modules(monkeypatch)

    # When: the backend loads.
    _ = load_backend(_request())

    # Then: the class that was actually built is named, so Unsloth's patched generate can ask
    # whether this is a vision model instead of iterating None and dying before the first token.
    base_model = FakeFastLanguageModel.last_model
    assert base_model is not None
    assert base_model.config.architectures == ["FakeCausalLM"]


def test_load_backend_when_the_config_already_names_its_architecture_leaves_it_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a config that already knows what it is.
    _patch_modules(monkeypatch)
    monkeypatch.setattr(FakeConfig, "architectures", ["Qwen3_5ForConditionalGeneration"])

    # When: the backend loads.
    _ = load_backend(_request())

    # Then: the loader's own answer stands — we fill a gap, we do not overrule it.
    base_model = FakeFastLanguageModel.last_model
    assert base_model is not None
    assert base_model.config.architectures == ["Qwen3_5ForConditionalGeneration"]


def test_load_backend_when_unsloth_keeps_the_norms_in_fp32_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: the precision split a real load returns — BF16 weights, FP32 normalisation.
    _patch_modules(monkeypatch)

    # When: the backend loads.
    backend = load_backend(_request())

    # Then: it is accepted, because that split is what Unsloth builds on purpose.
    assert backend.model_max_tokens == 16384


def test_load_backend_when_a_parameter_is_fp16_rejects_before_returning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a tower in FP16 — the precision that trains and infers without ever complaining.
    _patch_modules(monkeypatch)
    monkeypatch.setattr(FakeParameter, "dtype", "float16")

    # When/Then: admitting FP32 did not widen the check into admitting anything.
    with pytest.raises(GenerationError, match="unsupported dtype"):
        _ = load_backend(_request())


def test_load_backend_when_a_parameter_is_quantised_rejects_before_returning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: packed 4-bit weights, which are uint8 and not floating point at all.
    _patch_modules(monkeypatch)
    monkeypatch.setattr(FakeParameter, "dtype", "uint8")

    # When/Then: rejected rather than skipped — the run must be BF16, not silently quantised.
    with pytest.raises(GenerationError, match="unsupported dtype"):
        _ = load_backend(_request())


def test_load_backend_when_a_parameter_is_off_gpu_rejects_before_returning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a parameter left on the CPU, as an offloading device_map would produce.
    _patch_modules(monkeypatch)
    monkeypatch.setattr(FakeNormParameter, "device", "cpu")

    # When/Then: rejected, rather than paying host-transfer latency on every generated token.
    with pytest.raises(GenerationError, match="not on CUDA"):
        _ = load_backend(_request())


def test_backend_source_when_examined_uses_supported_outlines_public_api_only() -> None:
    # Given: the specialized backend source that supplies constrained inference.
    source = Path(__file__).resolve().parents[2] / "RQ/SLM/unsloth/generation.py"

    # When: its public constrained-generation boundary is inspected.
    contents = source.read_text(encoding="utf-8")

    # Then: it does not depend on removed processor internals.
    assert "JSONLogitsProcessor" not in contents


def test_validate_prompt_budget_when_input_plus_output_exceeds_model_limit_rejects() -> None:
    # Given: a fully rendered Qwen input with no room for requested output.
    from RQ.SLM.unsloth import generation

    # When/Then: the pre-generation budget guard fails before model generation.
    with pytest.raises(GenerationError, match="context"):
        _ = generation.validate_prompt_budget(
            16300, max_new_tokens=128, model_max_tokens=16384
        )


def test_generate_labels_when_rendered_uses_one_qwen_template_and_generated_tokens_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a loaded fake backend with a valid constrained JSON continuation.
    _patch_modules(monkeypatch)
    backend = load_backend(_request())
    tokenizer = FakeFastLanguageModel.last_tokenizer
    assert tokenizer is not None

    # When: one sampled row is generated through raw Transformers.
    labels = _generate(backend, seed=43)

    # Then: Qwen renders once with the non-thinking scaffold and Outlines only constrains logits.
    assert labels == ("fix",)
    assert tokenizer.template_kwargs == [
        {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
    ]
    assert tokenizer.tokenized_prompts == ["rendered " + PROMPT_TAIL]
    assert FakeOutlinesGenerator.calls[-1] == ("rendered " + PROMPT_TAIL, 128, 0.3, 43)


def test_generate_labels_when_tokenizer_has_distinct_padding_accepts_its_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: Qwen exposes different real padding and end-of-sequence token IDs.
    _patch_modules(monkeypatch)
    backend = load_backend(_request())

    # When: raw Transformers generation is requested.
    _ = _generate(backend)

    # Then: constrained generation accepts the real padding metadata.
    assert FakeOutlinesGenerator.calls


@pytest.mark.parametrize(
    ("pad_token_id", "reason"),
    [
        (None, "missing or invalid"),
        (-1, "missing or invalid"),
        (248046, "must differ"),
    ],
    ids=("missing", "negative", "equal-to-eos"),
)
def test_generate_labels_when_padding_is_unusable_rejects_tokenizer_metadata(
    monkeypatch: pytest.MonkeyPatch,
    pad_token_id: int | None,
    reason: str,
) -> None:
    # Given: tokenizer padding metadata cannot identify a distinct valid token.
    _patch_modules(monkeypatch)
    backend = load_backend(_request())
    backend.tokenizer.pad_token_id = pad_token_id

    # When/Then: generation rejects the metadata instead of inventing a fallback.
    with pytest.raises(GenerationError, match=reason):
        _ = _generate(backend)


def test_generate_labels_when_outlines_generator_is_built_uses_rendered_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a loaded PyTorch-backed Transformers model.
    _patch_modules(monkeypatch)
    backend = load_backend(_request())

    # When: Outlines builds the JSON generator.
    _ = _generate(backend)

    # Then: it receives the already-rendered prompt once.
    assert FakeOutlinesGenerator.calls[-1][0] == "rendered " + PROMPT_TAIL


def test_generate_labels_when_greedy_crosscheck_has_zero_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: the same raw backend configured for a controlled greedy comparison.
    _patch_modules(monkeypatch)
    backend = load_backend(_request())

    # When: generation is requested with zero temperature.
    _ = _generate(backend, temperature=0)

    # Then: public Outlines generation receives the requested deterministic temperature.
    assert FakeOutlinesGenerator.calls[-1][2] == 0


def test_generate_labels_when_json_is_invalid_raises_typed_failure_without_empty_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: constrained generation still produces an invalid duplicate-label object.
    _patch_modules(monkeypatch)
    backend = load_backend(_request())
    tokenizer = FakeFastLanguageModel.last_tokenizer
    assert tokenizer is not None
    tokenizer.decoded = '{"types": ["fix", "fix"]}'

    # When/Then: parsing failure retains raw output and never degrades to an empty prediction.
    with pytest.raises(GenerationError, match="duplicate label") as raised:
        _ = _generate(backend)
    assert raised.value.raw_output == '{"types": ["fix", "fix"]}'


def test_module_import_when_optional_ml_packages_are_blocked_stays_gpu_free() -> None:
    # Given: a fresh interpreter rejects every optional model package.
    code = """
import builtins
real_import = builtins.__import__
blocked = {"torch", "transformers", "unsloth", "peft", "outlines", "datasets", "pandas"}
def guarded_import(name, *args, **kwargs):
    if name.split(".")[0] in blocked:
        raise RuntimeError(f"heavy import: {name}")
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import RQ.SLM.unsloth.generation
"""

    # When: only the backend module is imported.
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )

    # Then: no GPU/data import is required before an actual run.
    assert completed.returncode == 0, completed.stderr
