import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Final, Protocol, TypeAlias, final

import pytest

from RQ.SLM.unsloth._memory_worker import select_supervised_json_row
from RQ.SLM.unsloth.train import render_training_rows
from RQ.SLM.unsloth.data import (
    DATASET_COLUMNS,
    HUB_DATASET_ID,
    HUB_DATASET_REVISION,
    IGNORE_LABEL,
    SPLIT_ROW_COUNTS,
    ChatMessage,
    DatasetRow,
    DatasetValidationError,
    ResponseMaskingError,
    TokenIds,
    build_chat_messages,
    canonical_row_hash,
    compare_sources,
    load_hub_split,
    load_local_split,
    render_response_only,
    validate_row,
)
from utils.llms.constant import COMMIT_TYPES
from utils.prompt import get_system_prompt

DATA_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "datasets" / "data"
RawDatasetRow: TypeAlias = dict[str, str | int]


class TangledSplitLike(Protocol):
    type_lists: tuple[tuple[str, ...], ...]
    sha_lists: tuple[tuple[str, ...], ...]

    @property
    def concern_counts(self) -> tuple[int, ...]: ...


def _valid_raw_row() -> RawDatasetRow:
    return {"commit_message": "repair parser", "diff": json.dumps(["diff --git a/a.py b/a.py\n-old\n+new"]), "concern_count": 2, "shas": json.dumps(["abc", "def"]), "types": json.dumps(["fix", "test"]), "repo": "owner/repo"}


@pytest.mark.parametrize(("split", "expected_count"), (("train", 1400), ("test", 350)))
def test_load_local_split_when_csv_is_canonical(split: str, expected_count: int) -> None:
    # Given: the repository's canonical local CSV files.
    # When: a split is loaded through the standard-library adapter.
    rows = load_local_split(split, DATA_DIR)

    # Then: every row is validated and the pinned split size is preserved.
    assert len(rows) == expected_count == SPLIT_ROW_COUNTS[split]
    assert tuple(rows[0].as_mapping()) == DATASET_COLUMNS


def test_load_local_split_when_using_real_fixture_matches_parsed_values(train_split: TangledSplitLike) -> None:
    # Given: the parsed fixture for the same canonical train CSV.
    # When: the first row is loaded through the new adapter.
    first = load_local_split("train", DATA_DIR)[0]

    # Then: its JSON-backed labels and SHAs retain the fixture's values.
    assert json.loads(first.types) == list(train_split.type_lists[0])
    assert json.loads(first.shas) == list(train_split.sha_lists[0])
    assert first.concern_count == train_split.concern_counts[0]


def test_build_chat_messages_when_row_is_valid_matches_legacy_structure() -> None:
    # Given: one validated training row.
    row = validate_row(_valid_raw_row())

    # When: the legacy chat example is built.
    result = build_chat_messages(row)

    # Then: roles, exact user framing, and constrained assistant JSON match training.
    assert result == {"messages": [{"role": "system", "content": get_system_prompt()}, {"role": "user", "content": f"- given commit message:\n repair parser\n Diff: {row.diff}"}, {"role": "assistant", "content": json.dumps({"types": ["fix", "test"]}, ensure_ascii=False)}]}


@pytest.mark.parametrize("column", ("diff", "shas", "types"))
def test_validate_row_when_json_column_is_malformed_raises_typed_error(column: str) -> None:
    # Given: malformed JSON in one JSON-backed column.
    row = _valid_raw_row()
    row[column] = "["

    # When/Then: boundary validation reports the typed dataset error.
    with pytest.raises(DatasetValidationError, match=column):
        _ = validate_row(row)


@pytest.mark.parametrize(("types", "concern_count", "reason"), ((["fix", "unknown"], 2, "unknown label"), (["fix", "fix"], 2, "duplicate label"), (["fix", "test"], 1, "concern_count")))
def test_validate_row_when_label_invariants_fail_raises_typed_error(types: list[str], concern_count: int, reason: str) -> None:
    # Given: a row violating one label invariant.
    row = _valid_raw_row()
    row["types"] = json.dumps(types)
    row["concern_count"] = concern_count

    # When/Then: boundary validation rejects it with a useful typed error.
    with pytest.raises(DatasetValidationError, match=reason):
        _ = validate_row(row)


def test_validate_row_when_column_is_missing_raises_typed_error() -> None:
    # Given: a row missing one of the six required columns.
    row = _valid_raw_row()
    del row["repo"]

    # When/Then: validation identifies the missing column.
    with pytest.raises(DatasetValidationError, match="repo"):
        _ = validate_row(row)


def test_canonical_row_hash_when_json_whitespace_differs_is_equal() -> None:
    # Given: equivalent Hub-like and CSV-like rows with different JSON whitespace/types.
    hub_row = _valid_raw_row()
    csv_row = _valid_raw_row()
    csv_row["concern_count"] = "2"
    csv_row["diff"] = '[  "diff --git a/a.py b/a.py\\n-old\\n+new"  ]'
    csv_row["shas"] = '[ "abc", "def" ]'
    csv_row["types"] = '[ "fix", "test" ]'

    # When: both rows are canonicalized and hashed.
    hub_hash = canonical_row_hash(hub_row)
    csv_hash = canonical_row_hash(csv_row)

    # Then: representation-only differences do not change identity.
    assert hub_hash == csv_hash


def test_compare_sources_when_second_row_differs_reports_first_index() -> None:
    # Given: two sources whose first rows match and second rows differ.
    matching = _valid_raw_row()
    different = _valid_raw_row()
    different["repo"] = "other/repo"

    # When: source parity is compared.
    comparison = compare_sources((matching, matching), (matching, different))

    # Then: only the first mismatch location is reported.
    assert comparison.matches is False
    assert comparison.first_mismatch_index == 1


def test_load_hub_split_when_selected_uses_canonical_pinned_source(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given: a lightweight stand-in for the lazily imported datasets package.
    calls: list[tuple[str, str, str]] = []
    fake_datasets = ModuleType("datasets")

    def fake_load_dataset(dataset_id: str, *, revision: str, split: str) -> list[RawDatasetRow]:
        calls.append((dataset_id, revision, split))
        return [_valid_raw_row() for _ in range(SPLIT_ROW_COUNTS[split])]

    loader_name = "load_dataset"
    setattr(fake_datasets, loader_name, fake_load_dataset)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    # When: the Hub train split is selected.
    rows = load_hub_split("train")

    # Then: the canonical id and immutable revision are passed to the adapter.
    assert len(rows) == 1400
    assert calls == [(HUB_DATASET_ID, HUB_DATASET_REVISION, "train")]


def test_module_import_when_dependencies_are_blocked_stays_lightweight() -> None:
    # Given: a fresh interpreter that fails if a forbidden heavy package is imported.
    code = """
import builtins
real_import = builtins.__import__
blocked = {"torch", "transformers", "unsloth", "peft", "trl", "outlines", "datasets"}
def guarded_import(name, *args, **kwargs):
    if name.split(".")[0] in blocked:
        raise RuntimeError(f"heavy import: {name}")
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import RQ.SLM.unsloth.data
"""

    # When: the module is imported without invoking a source adapter.
    completed = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, check=False)

    # Then: no blocked dependency is requested at import time.
    assert completed.returncode == 0, completed.stderr


def test_label_space_is_reused_from_shared_constant() -> None:
    # Given/When: the shared concern labels are exposed through row validation.
    _ = validate_row(_valid_raw_row())

    # Then: every accepted label belongs to the one shared seven-label space.
    assert len(COMMIT_TYPES) == 7
    assert {"fix", "test"} <= set(COMMIT_TYPES)


PROMPT_IDS: Final[tuple[int, ...]] = (10, 11, 12)
JSON_IDS: Final[tuple[int, ...]] = (20, 21, 22)
EOS_ID: Final[int] = 99
NEWLINE_IDS: Final[tuple[int, ...]] = (7,)
FULL_IDS: Final[tuple[int, ...]] = PROMPT_IDS + JSON_IDS + (EOS_ID,)
# Qwen renders the assistant turn as "<JSON><|im_end|>\n" — the separator the old fake omitted,
# which is why an invariant the real template never satisfied reached the GPU to be discovered.
SEPARATED_IDS: Final[tuple[int, ...]] = FULL_IDS + NEWLINE_IDS


@final
class _NonThinkingTokenizer:
    padding_side: str
    chat_template: str
    eos_token_id: int
    pad_token_id: int

    def __init__(self, response_ids: tuple[int, ...] = FULL_IDS) -> None:
        self.response_ids = response_ids
        self.padding_side = "right"
        self.chat_template = "qwen"
        self.eos_token_id = EOS_ID
        self.pad_token_id = 0
        self.thinking_values: list[bool] = []

    def apply_chat_template(self, conversation: Sequence[ChatMessage], *, tokenize: bool, add_generation_prompt: bool, enable_thinking: bool) -> str:
        _ = (conversation, tokenize)
        self.thinking_values.append(enable_thinking)
        return "prompt" if add_generation_prompt else "response"

    def __call__(self, text: str, *, add_special_tokens: bool) -> TokenIds:
        _ = add_special_tokens
        tokens = {"prompt": PROMPT_IDS, "response": self.response_ids, "\n": NEWLINE_IDS, '{"types": ["fix"]}': JSON_IDS}
        return {"input_ids": tokens[text]}

    def save_pretrained(self, save_directory: str) -> None:
        _ = save_directory


def _row() -> DatasetRow:
    return DatasetRow("fix", "[]", 1, "[]", '["fix"]', "repo")


def test_response_only_rendering_when_qwen_thinking_is_disabled_masks_prompt_and_supervises_json_eos() -> None:
    # Given: a Qwen-compatible template with prompt, thinking scaffold, JSON, and EOS tokens.
    tokenizer = _NonThinkingTokenizer()
    messages = build_chat_messages(_row())["messages"]

    # When: one response-only sample is rendered.
    rendered = render_response_only(tokenizer, messages)

    # Then: prompt/scaffold labels are ignored and only canonical JSON plus one EOS is supervised.
    assert tokenizer.thinking_values == [False, False]
    assert rendered.input_ids == FULL_IDS
    assert rendered.labels == (IGNORE_LABEL, IGNORE_LABEL, IGNORE_LABEL, *JSON_IDS, EOS_ID)


def test_training_and_memory_probe_when_rendering_one_row_share_response_only_label_evidence() -> None:
    # Given: one canonical row and the same non-thinking tokenizer for both paths.
    tokenizer = _NonThinkingTokenizer()
    row = _row()

    # When: training renders its dataset record and the probe selects its measured batch.
    prepared = render_training_rows((row,), tokenizer, max_seq_length=len(FULL_IDS))
    probe_batch = select_supervised_json_row((row,), tokenizer, len(FULL_IDS), build_chat_messages)

    # Then: both paths retain the rendered sample and the probe reuses the response-only labels.
    assert prepared.examples == ({"text": "response"},)
    assert probe_batch.input_ids == FULL_IDS
    assert probe_batch.labels == (IGNORE_LABEL, IGNORE_LABEL, IGNORE_LABEL, *JSON_IDS, EOS_ID)


def test_response_only_rendering_when_the_template_appends_a_turn_separator_still_supervises_json_eos() -> None:
    # Given: the real Qwen3.6 rendering — canonical JSON, one EOS, then the "\n" that closes
    # every turn. Requiring an exact json+EOS match failed here on the first row of the first
    # memory-ladder rung, after a full 27B load, for a template that was never going to match.
    tokenizer = _NonThinkingTokenizer(SEPARATED_IDS)
    messages = build_chat_messages(_row())["messages"]

    # When: one response-only sample is rendered.
    rendered = render_response_only(tokenizer, messages)

    # Then: the separator is tolerated and the supervised content is still exactly JSON + EOS.
    assert rendered.input_ids == SEPARATED_IDS
    assert rendered.labels == (IGNORE_LABEL, IGNORE_LABEL, IGNORE_LABEL, *JSON_IDS, EOS_ID, *NEWLINE_IDS)


def test_response_only_rendering_when_content_follows_the_eos_is_rejected() -> None:
    # Given: a template that emits a real token after EOS rather than the turn separator —
    # supervised content the canonical-JSON invariant exists to refuse.
    tokenizer = _NonThinkingTokenizer(FULL_IDS + (4242,))
    messages = build_chat_messages(_row())["messages"]

    # When/Then: rendering still fails closed.
    with pytest.raises(ResponseMaskingError):
        _ = render_response_only(tokenizer, messages)
