"""Lazy Unsloth constructors for the pinned Qwen3.6 local LoRA run."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ._types import (
    DatasetsModule,
    DtypeLike,
    ModelLoadKwargs,
    PeftKwargs,
    RenderedExample,
    SftKwargs,
    TemplateModule,
    TorchModule,
    TrainableAdapter,
    TrainerLike,
    TrainingDataset,
    TrlModule,
    UnslothModule,
    UnslothRuntime,
)
from .config import UnslothConfig


@dataclass(frozen=True, slots=True)
class RuntimeDependencyError(Exception):
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", (self.reason,))


def device_map_for(device_index: int) -> dict[str, int]:
    return {"": device_index}


def build_model_load_kwargs(
    config: UnslothConfig, device_index: int, bf16_dtype: DtypeLike
) -> ModelLoadKwargs:
    return {
        "model_name": config.model.id,
        "revision": config.model.revision,
        "max_seq_length": config.training.max_seq_length,
        "dtype": bf16_dtype,
        "load_in_16bit": config.precision.load_in_16bit,
        "load_in_4bit": config.precision.load_in_4bit,
        "load_in_8bit": config.precision.load_in_8bit,
        "full_finetuning": False,
        "text_only": config.model.text_only,
        "trust_remote_code": False,
        "attn_implementation": config.technical.attn_implementation,
        "use_gradient_checkpointing": config.lora.use_gradient_checkpointing,
        "device_map": device_map_for(device_index),
    }


def build_peft_kwargs(config: UnslothConfig) -> PeftKwargs:
    return {
        "r": config.lora.rank,
        "target_modules": config.lora.target_modules,
        "lora_alpha": config.lora.alpha,
        "lora_dropout": config.lora.dropout,
        "bias": config.lora.bias,
        "use_gradient_checkpointing": config.lora.use_gradient_checkpointing,
        "random_state": config.training.seed,
        "max_seq_length": config.training.max_seq_length,
        "exclude_modules": config.lora.exclude_modules,
        "autocast_adapter_dtype": False,
        # Unsloth routes an explicit target_modules list through get_peft_regex whenever any
        # finetune_(vision|language|attention|mlp) family is switched off, and FastLanguageModel
        # switches vision off for us. That regex only knows the dense-attention leaf names, so it
        # silently drops every linear_attn.in_proj_*/out_proj target — on Qwen3.6-27B that is the
        # sequence mixer of 48 of the 64 layers, leaving them adapted in the MLP only. Keeping
        # vision on makes the list reach PEFT verbatim; the model is loaded text_only, so there is
        # no vision tower to attach to, and exclude_modules (inert under the regex path) is what
        # keeps the MTP head out.
        "finetune_vision_layers": True,
    }


def build_sft_kwargs(
    config: UnslothConfig,
    output_dir: Path,
    max_steps: int | None,
    report_to: str,
    run_name: str | None,
) -> SftKwargs:
    kwargs: SftKwargs = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": config.training.per_device_train_batch_size,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "learning_rate": config.training.learning_rate,
        "num_train_epochs": config.training.num_train_epochs,
        "warmup_ratio": config.training.warmup_ratio,
        "lr_scheduler_type": config.training.lr_scheduler_type,
        "optim": config.training.optim,
        "bf16": True,
        "fp16": False,
        "max_length": config.training.max_seq_length,
        "dataset_text_field": "text",
        "packing": False,
        "save_strategy": config.training.save_strategy,
        "save_total_limit": config.training.save_total_limit,
        "eval_strategy": config.training.eval_strategy,
        "seed": config.training.seed,
        "report_to": report_to,
        "gradient_checkpointing": False,
        "use_cache": False,
    }
    if run_name is not None:
        kwargs["run_name"] = run_name
    if max_steps is not None:
        kwargs["max_steps"] = max_steps
    return kwargs


def validate_trainable_adapter_precision(
    model: TrainableAdapter, allowed_dtypes: tuple[DtypeLike, ...]
) -> None:
    """Fail fast on an adapter whose trainable precision is not one we intend to train in.

    Demanding BF16 here was wrong and cost a ladder run: Unsloth keeps LoRA master weights in
    FP32 above a BF16 base on purpose — the standard mixed-precision arrangement, and one
    `autocast_adapter_dtype=False` does not opt out of — so the check rejected every adapter
    the runtime is capable of building. The hazard actually worth stopping for is FP16, which
    trains without complaint and degrades silently, so the allowed set is passed in explicitly
    rather than inferred from whatever the loader happened to produce.
    """
    trainable = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    if not trainable:
        raise RuntimeDependencyError("PEFT adapter has no trainable parameters")
    rejected = tuple(name for name, parameter in trainable if parameter.dtype not in allowed_dtypes)
    if rejected:
        raise RuntimeDependencyError(
            f"trainable adapter parameters are not BF16 or FP32: {', '.join(rejected)}"
        )


def validate_target_module_coverage(
    model: TrainableAdapter, target_modules: Sequence[str]
) -> None:
    """Fail fast when a configured target module received no adapter.

    A LoRA target that matches nothing costs nothing to ignore and everything to discover
    late: the loader emits at most a warning, training runs to completion, and the resulting
    adapter is quietly narrower than the one the config describes. That is how the whole
    linear-attention sequence mixer went unadapted here — a config naming twelve targets
    produced adapters for seven of them.
    """
    adapted = tuple(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    missing = tuple(
        target for target in target_modules if not any(f".{target}." in name for name in adapted)
    )
    if missing:
        raise RuntimeDependencyError(
            f"configured LoRA targets received no adapter: {', '.join(missing)}"
        )


def create_training_dataset(examples: Sequence[RenderedExample]) -> TrainingDataset:
    datasets = importlib.import_module("datasets")
    if not isinstance(datasets, DatasetsModule):
        raise RuntimeDependencyError("datasets does not expose Dataset.from_list")
    return datasets.Dataset.from_list(list(examples))


def create_runtime(config: UnslothConfig, device_index: int) -> UnslothRuntime:
    torch = importlib.import_module("torch")
    unsloth = importlib.import_module("unsloth")
    if not isinstance(torch, TorchModule) or not isinstance(unsloth, UnslothModule):
        raise RuntimeDependencyError("local torch or unsloth runtime is unavailable")
    model, tokenizer = unsloth.FastLanguageModel.from_pretrained(
        **build_model_load_kwargs(config, device_index, torch.bfloat16)
    )
    tokenizer.padding_side = "right"
    adapter = unsloth.FastLanguageModel.get_peft_model(model, **build_peft_kwargs(config))
    validate_trainable_adapter_precision(adapter, (torch.bfloat16, torch.float32))
    validate_target_module_coverage(adapter, config.lora.target_modules)
    adapter.config.use_cache = False
    return UnslothRuntime(model=adapter, tokenizer=tokenizer)


def create_trainer(
    runtime: UnslothRuntime,
    train_dataset: Sequence[RenderedExample],
    config: UnslothConfig,
    output_dir: Path,
    report_to: str,
    run_name: str | None,
    max_steps: int | None = None,
) -> TrainerLike:
    """Lazily construct TRL and mask loss to the assistant response only."""
    trl = importlib.import_module("trl")
    templates = importlib.import_module("unsloth.chat_templates")
    if not isinstance(trl, TrlModule) or not isinstance(templates, TemplateModule):
        raise RuntimeDependencyError("local TRL or Unsloth template runtime is unavailable")
    arguments = trl.SFTConfig(
        **build_sft_kwargs(config, output_dir, max_steps, report_to, run_name)
    )
    trainer = trl.SFTTrainer(
        model=runtime.model,
        args=arguments,
        train_dataset=create_training_dataset(train_dataset),
        processing_class=runtime.tokenizer,
    )
    return templates.train_on_responses_only(
        trainer,
        instruction_part=config.masking.instruction_part,
        response_part=config.masking.response_part,
    )
