"""Validated dataset adapters and response-only rendering for local training."""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import (
    Final,
    Literal,
    Never,
    Protocol,
    TypeAlias,
    TypedDict,
    runtime_checkable,
)

from utils.llms.constant import COMMIT_TYPES
from utils.prompt import get_system_prompt

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
SplitName: TypeAlias = Literal["train", "test"]
DatasetSource: TypeAlias = Literal["hub", "local"]

DATASET_COLUMNS: Final[tuple[str, ...]] = ("commit_message", "diff", "concern_count", "shas", "types", "repo")
HUB_DATASET_ID: Final[str] = "Berom0227/tangled-ccs-commits"
HUB_DATASET_REVISION: Final[str] = "65b09af76f3e9badf4a28bf7a641b1d2930a26b5"
SPLIT_ROW_COUNTS: Final[Mapping[str, int]] = MappingProxyType({"train": 1400, "test": 350})
LOCAL_DATA_DIR: Final[Path] = Path(__file__).parents[3] / "datasets" / "data"
COMMIT_TYPE_SET: Final[frozenset[str]] = frozenset(COMMIT_TYPES)
IGNORE_LABEL: Final[int] = -100


class _JsonDecoder(Protocol):
    def decode(self, s: str) -> JsonValue: ...


def _json_decoder() -> _JsonDecoder:
    return json.JSONDecoder()


JSON_DECODER: Final[_JsonDecoder] = _json_decoder()


class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatExample(TypedDict):
    messages: list[ChatMessage]


class TokenIds(TypedDict):
    input_ids: Sequence[int]


@dataclass(frozen=True, slots=True)
class DatasetValidationError(Exception):
    reason: str
    column: str | None = None
    row_index: int | None = None

    def __post_init__(self) -> None:
        location = "dataset" if self.row_index is None else f"row {self.row_index}"
        if self.column is not None:
            location = f"{location} column {self.column!r}"
        object.__setattr__(self, "args", (f"{location}: {self.reason}",))


@dataclass(frozen=True, slots=True)
class ResponseMaskingError(Exception):
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", (self.reason,))


@dataclass(frozen=True, slots=True)
class DatasetRow:
    commit_message: str
    diff: str
    concern_count: int
    shas: str
    types: str
    repo: str

    def as_mapping(self) -> Mapping[str, JsonScalar]:
        return {"commit_message": self.commit_message, "diff": self.diff, "concern_count": self.concern_count, "shas": self.shas, "types": self.types, "repo": self.repo}


RowInput: TypeAlias = DatasetRow | Mapping[str, JsonScalar]


@dataclass(frozen=True, slots=True)
class SourceComparison:
    matches: bool
    first_mismatch_index: int | None
    left_hash: str | None = None
    right_hash: str | None = None


@dataclass(frozen=True, slots=True)
class ResponseOnlyExample:
    text: str
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]


@runtime_checkable
class _DatasetsModule(Protocol):
    def load_dataset(self, dataset_id: str, *, revision: str, split: str) -> Sequence[Mapping[str, JsonScalar]]: ...


class ResponseOnlyTokenizer(Protocol):
    eos_token_id: int

    def apply_chat_template(self, conversation: Sequence[ChatMessage], *, tokenize: bool, add_generation_prompt: bool, enable_thinking: bool) -> str: ...

    def __call__(self, text: str, *, add_special_tokens: bool) -> TokenIds: ...


def _error(reason: str, column: str | None = None, row_index: int | None = None) -> Never:
    raise DatasetValidationError(reason=reason, column=column, row_index=row_index)


def _split_name(split: str) -> SplitName:
    if split == "train" or split == "test":
        return split
    _error(f"unknown split {split!r}")


def _required_text(row: Mapping[str, JsonScalar], column: str, row_index: int | None) -> str:
    value = row[column]
    if isinstance(value, str):
        return value
    _error("must be a string", column, row_index)


def _concern_count(value: JsonScalar, row_index: int | None) -> int:
    if isinstance(value, bool):
        _error("must be an integer", "concern_count", row_index)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as error:
            raise DatasetValidationError("must be an integer", "concern_count", row_index) from error
    _error("must be an integer", "concern_count", row_index)


def _string_array(raw: str, column: str, row_index: int | None) -> tuple[str, ...]:
    try:
        decoded = JSON_DECODER.decode(raw)
    except json.JSONDecodeError as error:
        raise DatasetValidationError("malformed JSON", column, row_index) from error
    if not isinstance(decoded, list):
        _error("JSON value must be an array of strings", column, row_index)
    parsed: list[str] = []
    for value in decoded:
        if not isinstance(value, str):
            _error("JSON value must be an array of strings", column, row_index)
        parsed.append(value)
    return tuple(parsed)


def validate_row(row: Mapping[str, JsonScalar], row_index: int | None = None) -> DatasetRow:
    """Parse and validate one raw six-column dataset row."""
    keys = frozenset(row)
    missing = tuple(column for column in DATASET_COLUMNS if column not in keys)
    extra = tuple(column for column in keys if column not in DATASET_COLUMNS)
    if missing:
        _error(f"missing column {missing[0]!r}", missing[0], row_index)
    if extra:
        _error(f"unexpected column {extra[0]!r}", extra[0], row_index)
    diff, shas, types = (_required_text(row, column, row_index) for column in ("diff", "shas", "types"))
    _ = _string_array(diff, "diff", row_index), _string_array(shas, "shas", row_index)
    parsed_types = _string_array(types, "types", row_index)
    concern_count = _concern_count(row["concern_count"], row_index)
    unknown = tuple(label for label in parsed_types if label not in COMMIT_TYPE_SET)
    if unknown:
        _error(f"unknown label {unknown[0]!r}", "types", row_index)
    if len(parsed_types) != len(frozenset(parsed_types)):
        _error("duplicate label", "types", row_index)
    if concern_count != len(parsed_types):
        _error("concern_count does not equal len(types)", "concern_count", row_index)
    return DatasetRow(_required_text(row, "commit_message", row_index), diff, concern_count, shas, types, _required_text(row, "repo", row_index))


def build_chat_messages(row: RowInput) -> ChatExample:
    """Build the exact system/user/assistant structure used by local training."""
    validated = row if isinstance(row, DatasetRow) else validate_row(row)
    assistant_content = json.dumps({"types": list(_string_array(validated.types, "types", None))}, ensure_ascii=False)
    return {"messages": [{"role": "system", "content": get_system_prompt()}, {"role": "user", "content": f"- given commit message:\n {validated.commit_message}\n Diff: {validated.diff}"}, {"role": "assistant", "content": assistant_content}]}


def canonical_row_hash(row: RowInput) -> str:
    """Hash fixed-order columns after normalizing JSON and integer encoding."""
    validated = row if isinstance(row, DatasetRow) else validate_row(row)
    payload = json.dumps((validated.commit_message, list(_string_array(validated.diff, "diff", None)), validated.concern_count, list(_string_array(validated.shas, "shas", None)), list(_string_array(validated.types, "types", None)), validated.repo), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compare_sources(left: Sequence[RowInput], right: Sequence[RowInput]) -> SourceComparison:
    """Compare source rows in order and return the first mismatch only."""
    for index, (left_row, right_row) in enumerate(zip(left, right, strict=False)):
        left_hash, right_hash = canonical_row_hash(left_row), canonical_row_hash(right_row)
        if left_hash != right_hash:
            return SourceComparison(False, index, left_hash, right_hash)
    return SourceComparison(False, min(len(left), len(right))) if len(left) != len(right) else SourceComparison(True, None)


def _checked_rows(rows: Sequence[RowInput], split: SplitName) -> tuple[DatasetRow, ...]:
    validated = tuple(row if isinstance(row, DatasetRow) else validate_row(row, index) for index, row in enumerate(rows))
    expected = SPLIT_ROW_COUNTS[split]
    if len(validated) != expected:
        _error(f"expected {expected} rows, found {len(validated)}")
    return validated


def load_local_split(split: str, data_dir: Path = LOCAL_DATA_DIR) -> tuple[DatasetRow, ...]:
    """Load and validate one canonical CSV split without third-party libraries."""
    split_name = _split_name(split)
    with (data_dir / f"tangled_ccs_dataset_{split_name}.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != DATASET_COLUMNS:
            _error("CSV columns do not match the six-column contract")
        rows = tuple(reader)
    return _checked_rows(rows, split_name)


def load_hub_split(split: str) -> tuple[DatasetRow, ...]:
    """Load the immutable canonical Hub revision using a lazy datasets import."""
    datasets_module = importlib.import_module("datasets")
    if not isinstance(datasets_module, _DatasetsModule):
        _error("datasets.load_dataset is unavailable")
    split_name = _split_name(split)
    return _checked_rows(tuple(datasets_module.load_dataset(HUB_DATASET_ID, revision=HUB_DATASET_REVISION, split=split_name)), split_name)


def load_split(source: DatasetSource, split: str, data_dir: Path = LOCAL_DATA_DIR) -> tuple[DatasetRow, ...]:
    """Select the Hub or local CSV adapter for one validated split."""
    return load_hub_split(split) if source == "hub" else load_local_split(split, data_dir)


def render_response_only(tokenizer: ResponseOnlyTokenizer, messages: Sequence[ChatMessage]) -> ResponseOnlyExample:
    """Mask the rendered assistant scaffold and supervise canonical JSON plus EOS."""
    # Invariant: thinking is always disabled; the response must be pure JSON+EOS.
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    prompt_ids = tuple(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    input_ids = tuple(tokenizer(text, add_special_tokens=False)["input_ids"])
    if input_ids[: len(prompt_ids)] != prompt_ids:
        raise ResponseMaskingError("rendered assistant prefix does not preserve the prompt tokens")
    json_ids = tuple(tokenizer(messages[-1]["content"], add_special_tokens=False)["input_ids"])
    supervised_ids = input_ids[len(prompt_ids) :]
    terminated = json_ids + (tokenizer.eos_token_id,)
    # Qwen closes every turn with "<|im_end|>\n", so the rendered assistant turn carries one
    # trailing separator token that belongs to a following turn a single-turn SFT example never
    # has. Demanding an exact json+EOS match rejected the real template on the first row of the
    # first rung. Accept that separator and nothing else: the supervised content itself must
    # still be exactly the canonical JSON, terminated by exactly one EOS.
    separator = tuple(tokenizer("\n", add_special_tokens=False)["input_ids"])
    if supervised_ids not in (terminated, terminated + separator):
        raise ResponseMaskingError("rendered response is not canonical JSON followed by one EOS token")
    return ResponseOnlyExample(text, input_ids, (IGNORE_LABEL,) * len(prompt_ids) + supervised_ids)
