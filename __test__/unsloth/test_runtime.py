from __future__ import annotations

import ast
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import ModuleType
from typing import Final, Unpack, final

import pytest
from pytest import MonkeyPatch

from RQ.SLM.unsloth._types import (
    DtypeLike,
    ModelLike,
    ModelLoadKwargs,
    PeftKwargs,
    RenderedExample,
    TokenIds,
    TokenizerLike,
    TrainerLike,
    TrainableParameter,
    TrainingDataset,
)
from RQ.SLM.unsloth.config import load_unsloth_config
from RQ.SLM.unsloth.data import ChatMessage


REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
CONFIG_PATH: Final[Path] = REPO_ROOT / "RQ/SLM/unsloth/configs/qwen3_6_27b.yml"
RUNTIME_PATH: Final[Path] = REPO_ROOT / "RQ/SLM/unsloth/runtime.py"


def test_runtime_builders_when_using_pinned_config_preserve_required_settings() -> None:
    # Given: the approved Qwen3.6 LoRA configuration.
    from RQ.SLM.unsloth.runtime import (
        build_model_load_kwargs,
        build_peft_kwargs,
        build_sft_kwargs,
        device_map_for,
    )

    config = load_unsloth_config(CONFIG_PATH)

    # When: pure constructor arguments are derived without ML imports.
    model_kwargs = build_model_load_kwargs(config, device_index=0, bf16_dtype="bf16")
    peft_kwargs = build_peft_kwargs(config)
    sft_kwargs = build_sft_kwargs(config, output_dir=Path("adapter"), max_steps=None)

    # Then: all pinned loading, LoRA, and SFT policy is explicit.
    assert model_kwargs == {
        "model_name": "Qwen/Qwen3.6-27B",
        "revision": "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
        "max_seq_length": 16384,
        "dtype": "bf16",
        "load_in_16bit": True,
        "load_in_4bit": False,
        "load_in_8bit": False,
        "full_finetuning": False,
        "text_only": True,
        "trust_remote_code": False,
        "attn_implementation": "sdpa",
        "use_gradient_checkpointing": "unsloth",
        "device_map": {"": 0},
    }
    assert device_map_for(7) == {"": 7}
    assert peft_kwargs["r"] == 32
    assert peft_kwargs["lora_alpha"] == 48
    assert peft_kwargs["lora_dropout"] == 0.05
    assert peft_kwargs["bias"] == "none"
    assert peft_kwargs["target_modules"] == config.lora.target_modules
    assert peft_kwargs["exclude_modules"] == "(?:.*\\.)?(?:mtp|visual)(?:\\..*)?"
    assert peft_kwargs["random_state"] == 42
    assert peft_kwargs["autocast_adapter_dtype"] is False
    assert sft_kwargs["max_length"] == 16384
    assert sft_kwargs["dataset_text_field"] == "text"
    assert sft_kwargs["packing"] is False


def test_runtime_module_when_imported_has_no_heavy_module_imports() -> None:
    # Given: the runtime source before a CUDA run is requested.
    module = ast.parse(RUNTIME_PATH.read_text(encoding="utf-8"))

    # When: its top-level imports are examined.
    imported = {
        alias.name.split(".")[0]
        for node in module.body
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".")[0]
        for node in module.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    # Then: ML packages remain inside runtime constructors.
    assert {"torch", "unsloth", "trl", "peft"}.isdisjoint(imported)


def test_runtime_factories_when_injected_with_modules_use_response_only_markers(
    monkeypatch: MonkeyPatch,
) -> None:
    # Given: small in-process stand-ins for the Linux-only runtime modules.
    from RQ.SLM.unsloth.runtime import create_runtime, create_trainer

    config = load_unsloth_config(CONFIG_PATH)
    load_device_maps: list[dict[str, int]] = []
    peft_targets: list[tuple[str, ...]] = []
    sft_configurations: list[dict[str, str | int | float | bool]] = []
    dataset_rows: list[list[RenderedExample]] = []
    trainer_datasets: list[TrainingDataset] = []
    trainer_tokenizers: list[TokenizerLike] = []
    response_markers: list[tuple[str, str]] = []

    @final
    class FakeTokenizer:
        __slots__ = ("padding_side", "chat_template", "eos_token_id")

        def __init__(self) -> None:
            self.padding_side = "left"
            self.chat_template = "{{ messages }}"
            self.eos_token_id = 0

        def apply_chat_template(
            self,
            conversation: Sequence[ChatMessage],
            *,
            tokenize: bool,
            add_generation_prompt: bool,
            enable_thinking: bool,
        ) -> str:
            _ = (conversation, tokenize, add_generation_prompt, enable_thinking)
            return ""

        def __call__(self, text: str, *, add_special_tokens: bool) -> TokenIds:
            return {"input_ids": ()}

        def save_pretrained(self, save_directory: str) -> None:
            _ = save_directory
            return None

    tokenizer = FakeTokenizer()

    @final
    class FakeModelConfig:
        use_cache: bool = True

    @final
    class FakeModel:
        config: FakeModelConfig

        def __init__(self) -> None:
            self.config = FakeModelConfig()

        def named_parameters(self) -> Iterator[tuple[str, TrainableParameter]]:
            return iter((("adapter", FakeParameter(True, "bf16")),))

    @final
    class FakeParameter:
        requires_grad: bool
        dtype: DtypeLike

        def __init__(self, requires_grad: bool, dtype: DtypeLike) -> None:
            self.requires_grad = requires_grad
            self.dtype = dtype

    base_model = FakeModel()
    adapter = FakeModel()

    @final
    class FakeLanguageModel:
        @staticmethod
        def from_pretrained(
            **kwargs: Unpack[ModelLoadKwargs],
        ) -> tuple[ModelLike, TokenizerLike]:
            load_device_maps.append(kwargs["device_map"])
            return base_model, tokenizer

        @staticmethod
        def get_peft_model(
            model: ModelLike, **kwargs: Unpack[PeftKwargs]
        ) -> ModelLike:
            _ = model
            peft_targets.append(kwargs["target_modules"])
            return adapter

    @final
    class FakeSftConfig:
        def __init__(
            self,
            *,
            max_length: int,
            dataset_text_field: str,
            packing: bool,
            **kwargs: str | int | float | bool,
        ) -> None:
            sft_configurations.append(
                {
                    "max_length": max_length,
                    "dataset_text_field": dataset_text_field,
                    "packing": packing,
                    **kwargs,
                }
            )

    @final
    class FakeDataset:
        def __init__(self, rows: list[RenderedExample]) -> None:
            self.column_names = tuple(rows[0])

        @classmethod
        def from_list(cls, rows: list[RenderedExample]) -> TrainingDataset:
            dataset_rows.append(rows)
            return cls(rows)

    @final
    class FakeTrainer:
        def __init__(
            self,
            *,
            model: ModelLike,
            args: FakeSftConfig,
            train_dataset: TrainingDataset,
            processing_class: TokenizerLike,
        ) -> None:
            _ = (model, args)
            trainer_datasets.append(train_dataset)
            trainer_tokenizers.append(processing_class)

        def train(self) -> None:
            return None

        def save_model(self, output_dir: str) -> None:
            _ = output_dir
            return None

    def mask_responses(
        trainer: TrainerLike, *, instruction_part: str, response_part: str
    ) -> TrainerLike:
        response_markers.append((instruction_part, response_part))
        return trainer

    torch = ModuleType("torch")
    setattr(torch, "bfloat16", "bf16")
    unsloth = ModuleType("unsloth")
    setattr(unsloth, "FastLanguageModel", FakeLanguageModel)
    trl = ModuleType("trl")
    setattr(trl, "SFTConfig", FakeSftConfig)
    setattr(trl, "SFTTrainer", FakeTrainer)
    templates = ModuleType("unsloth.chat_templates")
    setattr(templates, "train_on_responses_only", mask_responses)
    datasets = ModuleType("datasets")
    setattr(datasets, "Dataset", FakeDataset)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "unsloth", unsloth)
    monkeypatch.setitem(sys.modules, "trl", trl)
    monkeypatch.setitem(sys.modules, "unsloth.chat_templates", templates)
    monkeypatch.setitem(sys.modules, "datasets", datasets)

    # When: a runtime and trainer are constructed.
    runtime = create_runtime(config, device_index=0)
    _ = create_trainer(
        runtime,
        train_dataset=({"text": "row"},),
        config=config,
        output_dir=Path("out"),
    )

    # Then: SFTConfig owns data formatting; SFTTrainer receives only its upstream contract.
    assert runtime.model is adapter
    assert runtime.tokenizer.padding_side == "right"
    assert adapter.config.use_cache is False
    assert not hasattr(tokenizer, "use_cache")
    assert load_device_maps == [{"": 0}]
    assert peft_targets == [config.lora.target_modules]
    assert sft_configurations[0]["max_length"] == 16384
    assert sft_configurations[0]["dataset_text_field"] == "text"
    assert sft_configurations[0]["packing"] is False
    assert dataset_rows == [[{"text": "row"}]]
    assert tuple(trainer_datasets[0].column_names) == ("text",)
    assert trainer_tokenizers == [tokenizer]
    assert response_markers == [
        ("<|im_start|>user\n", "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    ]


def test_bf16_adapter_validation_when_trainable_parameter_is_fp32_fails() -> None:
    # Given: PEFT exposes a trainable adapter that escaped BF16 conversion.
    from RQ.SLM.unsloth.runtime import RuntimeDependencyError, validate_bf16_trainable_adapters

    @final
    class Parameter:
        requires_grad: bool
        dtype: DtypeLike

        def __init__(self) -> None:
            self.requires_grad = True
            self.dtype = "fp32"

    @final
    class Adapter:
        def named_parameters(self) -> Iterator[tuple[str, TrainableParameter]]:
            return iter((("adapter", Parameter()),))

    # When/Then: runtime construction rejects a non-BF16 trainable adapter.
    with pytest.raises(RuntimeDependencyError, match="adapter"):
        validate_bf16_trainable_adapters(Adapter(), "bf16")
