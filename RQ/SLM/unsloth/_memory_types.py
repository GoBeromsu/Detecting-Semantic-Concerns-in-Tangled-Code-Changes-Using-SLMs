from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Final,
    Literal,
    Protocol,
    TypedDict,
    override,
    runtime_checkable,
)

from pydantic import BaseModel, ConfigDict, ValidationError

from RQ.SLM.unsloth._types import PreflightError
from RQ.SLM.unsloth.preflight import verify_preflight_evidence

if TYPE_CHECKING:
    from RQ.SLM.unsloth.data import ChatMessage


DEFAULT_LENGTHS: Final[tuple[int, ...]] = (2048, 4096, 8192, 12288, 16384)
ProbeStatus = Literal["completed", "terminal_failure", "not_attempted_after_boundary"]
QualificationStatus = Literal["approved_16384", "requires_owner_decision"]
JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject = dict[str, JsonValue]


class JsonDecoder(Protocol):
    def decode(self, s: str) -> JsonValue: ...


class TokenIds(TypedDict):
    input_ids: Sequence[int]


class PaddingTokenizer(Protocol):
    @property
    def pad_token_id(self) -> int | None: ...


@runtime_checkable
class ProbeTokenizer(PaddingTokenizer, Protocol):
    def apply_chat_template(
        self,
        conversation: Sequence[ChatMessage],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str: ...

    def __call__(self, text: str, *, add_special_tokens: bool) -> TokenIds: ...


def _json_decoder() -> JsonDecoder:
    return json.JSONDecoder()


JSON_DECODER: Final[JsonDecoder] = _json_decoder()


@dataclass(frozen=True, slots=True)
class ProbeError(Exception):
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", (self.reason,))


@dataclass(frozen=True, slots=True)
class ParentOptions:
    config_path: Path
    host_profile_path: Path
    lengths: tuple[int, ...]
    output_directory: Path
    preflight_path: Path | None = None


@dataclass(frozen=True, slots=True)
class ChildRequest:
    length: int
    result_path: Path


@dataclass(frozen=True, slots=True)
class ProbeBatch:
    requested_length: int
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Measurement:
    requested_length: int
    status: ProbeStatus
    failure_class: str | None
    input_shape: tuple[int, int] | None = None
    unpadded_length: int | None = None
    supervised_token_count: int | None = None
    loss_is_finite: bool | None = None
    global_grad_norm: float | None = None
    in_proj_a_grad_norm: float | None = None
    in_proj_a_grad_is_finite: bool | None = None
    in_proj_b_grad_norm: float | None = None
    in_proj_b_grad_is_finite: bool | None = None
    optimizer_state_allocated: bool | None = None
    optimizer_step_succeeded: bool | None = None
    packing_enabled: bool | None = None
    max_memory_allocated_bytes: int | None = None
    max_memory_reserved_bytes: int | None = None
    pre_step_free_vram_bytes: int | None = None
    post_step_free_vram_bytes: int | None = None
    host_peak_rss_bytes: int | None = None
    wall_time_seconds: float | None = None

    def as_json(self) -> JsonObject:
        return {
            "requested_length": self.requested_length,
            "status": self.status,
            "failure_class": self.failure_class,
            "input_shape": list(self.input_shape) if self.input_shape is not None else None,
            "unpadded_length": self.unpadded_length,
            "supervised_token_count": self.supervised_token_count,
            "loss_is_finite": self.loss_is_finite,
            "global_grad_norm": self.global_grad_norm,
            "in_proj_a_grad_norm": self.in_proj_a_grad_norm,
            "in_proj_a_grad_is_finite": self.in_proj_a_grad_is_finite,
            "in_proj_b_grad_norm": self.in_proj_b_grad_norm,
            "in_proj_b_grad_is_finite": self.in_proj_b_grad_is_finite,
            "optimizer_state_allocated": self.optimizer_state_allocated,
            "optimizer_step_succeeded": self.optimizer_step_succeeded,
            "packing_enabled": self.packing_enabled,
            "max_memory_allocated_bytes": self.max_memory_allocated_bytes,
            "max_memory_reserved_bytes": self.max_memory_reserved_bytes,
            "pre_step_free_vram_bytes": self.pre_step_free_vram_bytes,
            "post_step_free_vram_bytes": self.post_step_free_vram_bytes,
            "host_peak_rss_bytes": self.host_peak_rss_bytes,
            "wall_time_seconds": self.wall_time_seconds,
        }


@dataclass(frozen=True, slots=True)
class Qualification:
    status: QualificationStatus
    approved_max_seq_length: int | None
    first_failure_boundary: int | None
    lengths: tuple[int, ...]
    config_sha256: str
    host_profile_sha256: str
    evidence_sha256: str
    preflight_hostname: str | None
    preflight_sha256: str | None

    def as_json(self) -> JsonObject:
        return {
            "status": self.status,
            "approved_max_seq_length": self.approved_max_seq_length,
            "first_failure_boundary": self.first_failure_boundary,
            "lengths": list(self.lengths),
            "config_sha256": self.config_sha256,
            "host_profile_sha256": self.host_profile_sha256,
            "evidence_sha256": self.evidence_sha256,
            "preflight_hostname": self.preflight_hostname,
            "preflight_sha256": self.preflight_sha256,
        }


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_bytes(payload: JsonObject) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        _ = handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    _ = temporary.replace(path)


def write_json_atomic(path: Path, payload: JsonObject) -> None:
    write_bytes_atomic(path, json_bytes(payload))


def read_measurement(path: Path) -> Measurement:
    raw = JSON_DECODER.decode(path.read_text(encoding="utf-8"))
    return parse_measurement(raw)


def parse_measurement(raw: JsonValue) -> Measurement:
    if not isinstance(raw, dict):
        raise ProbeError("child measurement must be a JSON object")
    return Measurement(
        requested_length=_integer(raw, "requested_length"),
        status=_status(raw),
        failure_class=_text_or_none(raw, "failure_class"),
        input_shape=_shape(raw),
        unpadded_length=_integer_or_none(raw, "unpadded_length"),
        supervised_token_count=_integer_or_none(raw, "supervised_token_count"),
        loss_is_finite=_boolean_or_none(raw, "loss_is_finite"),
        global_grad_norm=_number_or_none(raw, "global_grad_norm"),
        in_proj_a_grad_norm=_number_or_none(raw, "in_proj_a_grad_norm"),
        in_proj_a_grad_is_finite=_boolean_or_none(raw, "in_proj_a_grad_is_finite"),
        in_proj_b_grad_norm=_number_or_none(raw, "in_proj_b_grad_norm"),
        in_proj_b_grad_is_finite=_boolean_or_none(raw, "in_proj_b_grad_is_finite"),
        optimizer_state_allocated=_boolean_or_none(raw, "optimizer_state_allocated"),
        optimizer_step_succeeded=_boolean_or_none(raw, "optimizer_step_succeeded"),
        packing_enabled=_boolean_or_none(raw, "packing_enabled"),
        max_memory_allocated_bytes=_integer_or_none(raw, "max_memory_allocated_bytes"),
        max_memory_reserved_bytes=_integer_or_none(raw, "max_memory_reserved_bytes"),
        pre_step_free_vram_bytes=_integer_or_none(raw, "pre_step_free_vram_bytes"),
        post_step_free_vram_bytes=_integer_or_none(raw, "post_step_free_vram_bytes"),
        host_peak_rss_bytes=_integer_or_none(raw, "host_peak_rss_bytes"),
        wall_time_seconds=_number_or_none(raw, "wall_time_seconds"),
    )


def read_measurements(path: Path) -> tuple[Measurement, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ProbeError(f"cannot read measurements: {error}") from error
    if not lines:
        raise ProbeError("measurements evidence is empty")
    measurements: list[Measurement] = []
    for line in lines:
        try:
            measurements.append(parse_measurement(JSON_DECODER.decode(line)))
        except json.JSONDecodeError as error:
            raise ProbeError(f"invalid measurement JSON: {error}") from error
    return tuple(measurements)


def measurement_qualifies(measurement: Measurement) -> bool:
    shape = measurement.input_shape
    required_flags = (
        measurement.loss_is_finite,
        measurement.in_proj_a_grad_is_finite,
        measurement.in_proj_b_grad_is_finite,
        measurement.optimizer_state_allocated,
        measurement.optimizer_step_succeeded,
    )
    numeric_evidence = (
        measurement.global_grad_norm,
        measurement.in_proj_a_grad_norm,
        measurement.in_proj_b_grad_norm,
    )
    if measurement.status != "completed" or shape != (1, measurement.requested_length):
        return False
    if measurement.unpadded_length is None or not 0 < measurement.unpadded_length <= measurement.requested_length:
        return False
    if measurement.supervised_token_count is None or measurement.supervised_token_count <= 0:
        return False
    if measurement.packing_enabled is not False or not all(flag is True for flag in required_flags):
        return False
    if any(value is None or not math.isfinite(value) for value in numeric_evidence):
        return False
    pre_step_free = measurement.pre_step_free_vram_bytes
    post_step_free = measurement.post_step_free_vram_bytes
    return pre_step_free is not None and post_step_free is not None and pre_step_free > 0 and post_step_free / pre_step_free >= 0.1


def has_full_qualification_ladder(measurements: tuple[Measurement, ...]) -> bool:
    return (
        tuple(measurement.requested_length for measurement in measurements) == DEFAULT_LENGTHS
        and len(measurements) == len(DEFAULT_LENGTHS)
        and all(measurement_qualifies(measurement) for measurement in measurements)
    )


def _value(raw: Mapping[str, JsonValue], key: str) -> JsonValue:
    if key not in raw:
        raise ProbeError(f"child measurement missing {key}")
    return raw[key]


def _integer(raw: Mapping[str, JsonValue], key: str) -> int:
    value = _value(raw, key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise ProbeError(f"child measurement {key} must be an integer")


def _integer_or_none(raw: Mapping[str, JsonValue], key: str) -> int | None:
    value = _value(raw, key)
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise ProbeError(f"child measurement {key} must be an integer or null")


def _number_or_none(raw: Mapping[str, JsonValue], key: str) -> float | None:
    value = _value(raw, key)
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    raise ProbeError(f"child measurement {key} must be a number or null")


def _boolean_or_none(raw: Mapping[str, JsonValue], key: str) -> bool | None:
    value = _value(raw, key)
    if value is None or isinstance(value, bool):
        return value
    raise ProbeError(f"child measurement {key} must be a boolean or null")


def _text_or_none(raw: Mapping[str, JsonValue], key: str) -> str | None:
    value = _value(raw, key)
    if value is None or isinstance(value, str):
        return value
    raise ProbeError(f"child measurement {key} must be text or null")


def _shape(raw: Mapping[str, JsonValue]) -> tuple[int, int] | None:
    value = _value(raw, "input_shape")
    if value is None:
        return None
    if isinstance(value, list) and len(value) == 2:
        first, second = value
        if (
            isinstance(first, int)
            and not isinstance(first, bool)
            and isinstance(second, int)
            and not isinstance(second, bool)
        ):
            return first, second
    raise ProbeError("child measurement input_shape must be a pair of integers or null")


def _status(raw: Mapping[str, JsonValue]) -> ProbeStatus:
    match _value(raw, "status"):
        case "completed":
            return "completed"
        case "terminal_failure":
            return "terminal_failure"
        case "not_attempted_after_boundary":
            return "not_attempted_after_boundary"
        case _:
            raise ProbeError("child measurement status is unknown")


@dataclass(frozen=True, slots=True)
class TrainingDataError(Exception):
    reason: str

    @override
    def __str__(self) -> str:
        return self.reason


class QualificationEvidence(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    status: str
    approved_max_seq_length: int
    first_failure_boundary: int | None
    lengths: tuple[int, ...]
    config_sha256: str
    host_profile_sha256: str
    evidence_sha256: str
    preflight_hostname: str | None
    preflight_sha256: str | None


@dataclass(frozen=True, slots=True)
class QualificationInputs:
    evidence_dir: Path | None
    config_path: Path
    host_profile_path: Path | None
    max_seq_length: int


def require_qualification_evidence(inputs: QualificationInputs) -> Path:
    """Return evidence only when it exactly approves the selected full run."""
    evidence_dir = inputs.evidence_dir
    if evidence_dir is None:
        raise TrainingDataError("--evidence-dir is required for a full run")
    host_profile_path = inputs.host_profile_path
    if host_profile_path is None:
        raise TrainingDataError("--host-profile is required for qualification verification")
    missing = [
        name for name in (
            "template-inspection.json", "overflow-rows.json", "measurements.jsonl",
            "qualification.json",
        ) if not (evidence_dir / name).is_file()
    ]
    if missing:
        raise TrainingDataError(f"missing qualification evidence: {', '.join(missing)}")
    try:
        qualification = QualificationEvidence.model_validate_json(
            (evidence_dir / "qualification.json").read_text(encoding="utf-8")
        )
    except (OSError, ValidationError) as error:
        raise TrainingDataError(f"invalid qualification evidence: {error}") from error
    if qualification.status != "approved_16384":
        raise TrainingDataError("qualification status is not approved")
    if qualification.approved_max_seq_length != inputs.max_seq_length:
        raise TrainingDataError("qualification approved_max_seq_length does not match configuration")
    if qualification.lengths != DEFAULT_LENGTHS:
        raise TrainingDataError("qualification was not produced by the canonical measurement ladder")
    _verify_source_hashes(qualification, inputs)
    _verify_measurements(evidence_dir)
    _verify_preflight(qualification, inputs, evidence_dir)
    return evidence_dir


def _verify_source_hashes(
    qualification: QualificationEvidence, inputs: QualificationInputs
) -> None:
    evidence_dir = inputs.evidence_dir
    host_profile_path = inputs.host_profile_path
    if evidence_dir is None or host_profile_path is None:
        raise TrainingDataError("qualification inputs are incomplete")
    sources = (
        ("config_sha256", qualification.config_sha256, inputs.config_path),
        ("host_profile_sha256", qualification.host_profile_sha256, host_profile_path),
        ("evidence_sha256", qualification.evidence_sha256, evidence_dir / "measurements.jsonl"),
    )
    for field, expected, path in sources:
        try:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as error:
            raise TrainingDataError(f"cannot verify qualification {field}: {error}") from error
        if actual != expected:
            raise TrainingDataError(f"qualification {field} does not match {path}")


def _verify_measurements(evidence_dir: Path) -> None:
    try:
        measurements = read_measurements(evidence_dir / "measurements.jsonl")
    except ProbeError as error:
        raise TrainingDataError(f"invalid measurement evidence: {error}") from error
    if not has_full_qualification_ladder(measurements):
        raise TrainingDataError("measurements do not satisfy the canonical qualification ladder")


def _verify_preflight(
    qualification: QualificationEvidence,
    inputs: QualificationInputs,
    evidence_dir: Path,
) -> None:
    host_profile_path = inputs.host_profile_path
    if host_profile_path is None:
        raise TrainingDataError("qualification inputs are incomplete")
    try:
        preflight = verify_preflight_evidence(
            evidence_dir / "preflight.json", inputs.config_path, host_profile_path
        )
    except PreflightError as error:
        raise TrainingDataError(f"invalid preflight evidence: {error}") from error
    if qualification.preflight_hostname != preflight.hostname:
        raise TrainingDataError("qualification preflight hostname does not match evidence")
    preflight_hash = hashlib.sha256((evidence_dir / "preflight.json").read_bytes()).hexdigest()
    if qualification.preflight_sha256 != preflight_hash:
        raise TrainingDataError("qualification preflight hash does not match evidence")
