from __future__ import annotations

import importlib
import math
import resource
import sys
import time
from collections.abc import Callable, Collection, Iterator, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Final, Protocol, cast, override, runtime_checkable

from RQ.SLM.unsloth._memory_types import (
    ChildRequest,
    Measurement,
    PaddingTokenizer,
    ParentOptions,
    ProbeBatch,
    ProbeError,
)
from RQ.SLM.unsloth.data import ResponseOnlyTokenizer, render_response_only

if TYPE_CHECKING:
    from RQ.SLM.unsloth.data import ChatExample, DatasetRow


IGNORE_LABEL: Final[int] = -100


class TorchToken(Protocol):
    @override
    def __str__(self) -> str: ...


class TensorLike(Protocol):
    device: TorchToken
    shape: Sequence[int]

    def all(self) -> TensorLike: ...

    def backward(self) -> None: ...

    def detach(self) -> TensorLike: ...

    def float(self) -> TensorLike: ...

    def item(self) -> int | float | bool: ...


class ParameterLike(TensorLike, Protocol):
    grad: TensorLike | None
    requires_grad: bool


class ModelOutput(Protocol):
    loss: TensorLike


@runtime_checkable
class ProbeModel(Protocol):
    def parameters(self) -> Iterator[ParameterLike]: ...

    def named_parameters(self) -> Iterator[tuple[str, ParameterLike]]: ...

    def train(self, mode: bool = True) -> ProbeModel: ...

    def __call__(
        self, *, input_ids: TensorLike, attention_mask: TensorLike, labels: TensorLike
    ) -> ModelOutput: ...


class CudaModule(Protocol):
    def is_available(self) -> bool: ...

    def max_memory_allocated(self, device: int) -> int: ...

    def max_memory_reserved(self, device: int) -> int: ...

    def mem_get_info(self, device: int) -> tuple[int, int]: ...

    def reset_peak_memory_stats(self, device: int) -> None: ...

    def set_device(self, device: int) -> None: ...

    def synchronize(self, device: int) -> None: ...


class FunctionalModule(Protocol):
    def pad(self, tensor: TensorLike, padding: tuple[int, int], *, value: int) -> TensorLike: ...


class NNModule(Protocol):
    functional: FunctionalModule


class OptimizerLike(Protocol):
    state: Collection[ParameterLike]

    def step(self) -> None: ...

    def zero_grad(self, *, set_to_none: bool) -> None: ...


class OptimizerFactory(Protocol):
    def __call__(
        self, parameters: Sequence[ParameterLike], *, lr: float
    ) -> OptimizerLike: ...


class ResponseProbeTokenizer(ResponseOnlyTokenizer, PaddingTokenizer, Protocol):
    """Static type only — validate instances with `require_probe_tokenizer`, never isinstance.

    Deliberately not runtime_checkable: two of its members are data attributes, and no
    structural instance check can see those on a tokenizer that serves them lazily.
    """


class OptimModule(Protocol):
    AdamW: OptimizerFactory


class LinalgModule(Protocol):
    def vector_norm(self, tensor: TensorLike) -> TensorLike: ...


@runtime_checkable
class TorchModule(Protocol):
    cuda: CudaModule
    linalg: LinalgModule
    long: TorchToken
    nn: NNModule
    optim: OptimModule

    def isfinite(self, tensor: TensorLike) -> TensorLike: ...

    def ones_like(self, tensor: TensorLike) -> TensorLike: ...

    def tensor(
        self, data: Sequence[Sequence[int]], *, dtype: TorchToken, device: TorchToken
    ) -> TensorLike: ...


def _load_torch() -> TorchModule:
    torch = importlib.import_module("torch")
    if not isinstance(torch, TorchModule):
        raise ProbeError("torch does not expose the memory-probe interface")
    return torch


def require_probe_tokenizer(tokenizer: object) -> ResponseProbeTokenizer:
    """Check the probe's tokenizer requirements by reading them the way the probe will.

    `isinstance` against a runtime_checkable Protocol resolves members through
    `inspect.getattr_static`, which deliberately bypasses `__getattr__`. Transformers 5.5
    hands back a `TokenizersBackend` that serves `pad_token_id` and `eos_token_id` from
    exactly that hook, so the gate declared a perfectly usable tokenizer to be missing the
    interface and stopped the memory ladder on its first rung — the second time a
    structural check has cost this run for that reason. Live access is also the stricter
    test: a `pad_token_id` of None satisfies any structural check and then pads with
    nothing, which is the failure worth refusing to start on.
    """
    for name in ("apply_chat_template", "__call__"):
        if not callable(getattr(tokenizer, name, None)):
            raise ProbeError(f"loaded tokenizer has no callable {name}")
    for name in ("eos_token_id", "pad_token_id"):
        value = getattr(tokenizer, name, None)
        if not isinstance(value, int):
            raise ProbeError(f"loaded tokenizer has no integer {name}: {value!r}")
    return cast(ResponseProbeTokenizer, tokenizer)


def run_probe_child(options: ParentOptions, request: ChildRequest) -> Measurement:
    torch = _load_torch()
    if not torch.cuda.is_available():
        return Measurement(request.length, "terminal_failure", "cuda_unavailable", packing_enabled=False)
    from RQ.SLM.unsloth.config import (
        ApprovedDeviation,
        load_host_profile,
        load_unsloth_config,
    )
    from RQ.SLM.unsloth.data import build_chat_messages, load_split
    from RQ.SLM.unsloth.runtime import create_runtime

    _ = load_host_profile(options.host_profile_path)
    configured = load_unsloth_config(
        options.config_path, approved_deviation=ApprovedDeviation(request.length)
    )
    config = replace(
        configured,
        training=replace(configured.training, max_seq_length=request.length, packing=False),
    )
    torch.cuda.set_device(0)
    runtime = create_runtime(config, 0)
    if not isinstance(runtime.model, ProbeModel):
        raise ProbeError("loaded model does not expose the memory-probe interface")
    tokenizer = require_probe_tokenizer(runtime.tokenizer)
    batch = select_supervised_json_row(
        load_split("local", "train"), tokenizer, request.length, build_chat_messages
    )
    return measure_step(runtime.model, tokenizer, config.training.learning_rate, batch)


def select_supervised_json_row(
    rows: Sequence[DatasetRow],
    tokenizer: ResponseProbeTokenizer,
    requested_length: int,
    build_messages: Callable[[DatasetRow], ChatExample],
) -> ProbeBatch:
    for row in rows:
        rendered = render_response_only(tokenizer, build_messages(row)["messages"])
        if len(rendered.input_ids) <= requested_length:
            return ProbeBatch(requested_length, rendered.input_ids, rendered.labels)
    raise ProbeError("no eligible train row fits the requested length with supervised JSON tokens")


def measure_step(
    model: ProbeModel | None,
    tokenizer: PaddingTokenizer,
    learning_rate: float,
    batch: ProbeBatch,
) -> Measurement:
    pad_token = tokenizer.pad_token_id
    if pad_token is None:
        return Measurement(
            requested_length=batch.requested_length,
            status="terminal_failure",
            failure_class="pad_token_unavailable",
            unpadded_length=len(batch.input_ids),
            supervised_token_count=sum(label != IGNORE_LABEL for label in batch.labels),
            packing_enabled=False,
        )
    if model is None:
        raise ProbeError("model is required when the tokenizer has a pad token")

    torch = _load_torch()
    device = next(model.parameters()).device
    _ = model.train()
    input_tensor = torch.tensor([batch.input_ids], dtype=torch.long, device=device)
    label_tensor = torch.tensor([batch.labels], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_tensor)
    padding = batch.requested_length - len(batch.input_ids)
    input_tensor = torch.nn.functional.pad(input_tensor, (0, padding), value=pad_token)
    label_tensor = torch.nn.functional.pad(label_tensor, (0, padding), value=IGNORE_LABEL)
    attention_mask = torch.nn.functional.pad(attention_mask, (0, padding), value=0)
    trainable_parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    optimizer = torch.optim.AdamW(trainable_parameters, lr=learning_rate)
    torch.cuda.synchronize(0)
    torch.cuda.reset_peak_memory_stats(0)
    free_before, _ = torch.cuda.mem_get_info(0)
    started_at = time.perf_counter()
    output = model(input_ids=input_tensor, attention_mask=attention_mask, labels=label_tensor)
    loss = output.loss
    loss.backward()
    named_parameters = tuple(model.named_parameters())
    global_norm, global_finite = _gradient_norm(
        torch, tuple(parameter.grad for _, parameter in named_parameters if parameter.requires_grad)
    )
    a_norm, a_finite = _gradient_norm(
        torch,
        tuple(parameter.grad for name, parameter in named_parameters if "in_proj_a" in name and parameter.requires_grad),
    )
    b_norm, b_finite = _gradient_norm(
        torch,
        tuple(parameter.grad for name, parameter in named_parameters if "in_proj_b" in name and parameter.requires_grad),
    )
    loss_finite = bool(torch.isfinite(loss).item())
    stepped = loss_finite and global_finite and a_finite and b_finite
    if stepped:
        optimizer.step()
    state_allocated = bool(optimizer.state)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(0)
    free_after, _ = torch.cuda.mem_get_info(0)
    wall_time = time.perf_counter() - started_at
    return Measurement(
        requested_length=int(input_tensor.shape[1]),
        status="completed",
        failure_class=_failure_class(loss_finite, global_finite, a_finite, b_finite, stepped),
        input_shape=(int(input_tensor.shape[0]), int(input_tensor.shape[1])),
        unpadded_length=len(batch.input_ids),
        supervised_token_count=sum(label != IGNORE_LABEL for label in batch.labels),
        loss_is_finite=loss_finite,
        global_grad_norm=global_norm,
        in_proj_a_grad_norm=a_norm,
        in_proj_a_grad_is_finite=a_finite,
        in_proj_b_grad_norm=b_norm,
        in_proj_b_grad_is_finite=b_finite,
        optimizer_state_allocated=state_allocated,
        optimizer_step_succeeded=stepped,
        packing_enabled=False,
        max_memory_allocated_bytes=int(torch.cuda.max_memory_allocated(0)),
        max_memory_reserved_bytes=int(torch.cuda.max_memory_reserved(0)),
        pre_step_free_vram_bytes=int(free_before),
        post_step_free_vram_bytes=int(free_after),
        host_peak_rss_bytes=_peak_rss_bytes(),
        wall_time_seconds=wall_time,
    )


_measure_step = measure_step


def _gradient_norm(
    torch: TorchModule, gradients: Sequence[TensorLike | None]
) -> tuple[float | None, bool]:
    finite = bool(gradients) and all(
        gradient is not None and bool(torch.isfinite(gradient).all().item()) for gradient in gradients
    )
    if not finite:
        return None, False
    norm = math.sqrt(
        sum(
            float(torch.linalg.vector_norm(gradient.detach().float()).item()) ** 2
            for gradient in gradients
            if gradient is not None
        )
    )
    return (norm, True) if math.isfinite(norm) else (None, False)


def _failure_class(
    loss_finite: bool, global_finite: bool, a_finite: bool, b_finite: bool, stepped: bool
) -> str | None:
    if not loss_finite:
        return "non_finite_loss"
    if not global_finite:
        return "non_finite_gradient"
    if not a_finite:
        return "non_finite_in_proj_a_gradient"
    if not b_finite:
        return "non_finite_in_proj_b_gradient"
    return None if stepped else "optimizer_step_not_run"


def _peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024
