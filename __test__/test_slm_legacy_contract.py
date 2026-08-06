import ast
import hashlib
from pathlib import Path
from typing import Final

from utils.llms.constant import COMMIT_TYPES, DEFAULT_DF_COLUMNS


REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
INFER_PATH: Final[Path] = REPO_ROOT / "RQ" / "SLM" / "infer.py"
CONVERSION_PATH: Final[Path] = REPO_ROOT / "RQ" / "SLM" / "convert_to_gguf.py"
QWEN_CONFIG_PATH: Final[Path] = REPO_ROOT / "RQ" / "SLM" / "configs" / "qwen.yml"
PROTECTED_FILES_PATH: Final[Path] = (
    REPO_ROOT / "__test__" / "fixtures" / "slm" / "protected-files.sha256"
)

EXPECTED_QWEN_LORA: Final[dict[str, str]] = {
    "repo_id": "Berom0227/Semantic-Concern-SLM-Qwen-gguf",
    "filename": "Semantic-Concern-SLM-Qwen-f16.gguf",
    "model_name": "Qwen3-14B-LoRA",
    "chat_format": "chatml",
}
EXPECTED_CONTEXT_WINDOWS: Final[list[int]] = [12288, 8192, 4096, 2048, 1024]
EXPECTED_MAX_TOKENS: Final[int] = 16384
EXPECTED_SEED: Final[int] = 42
EXPECTED_TEMPERATURE: Final[float] = 0.3

EXPECTED_QWEN_BASE_MODEL_ID: Final[str] = "Qwen/Qwen3-14B"
EXPECTED_QWEN_MODEL_NAME: Final[str] = "Semantic-Concern-SLM-Qwen"
EXPECTED_QWEN_GGUF_REPO: Final[str] = "Berom0227/Semantic-Concern-SLM-Qwen-gguf"
EXPECTED_QWEN_ADAPTER_REPO: Final[str] = (
    "Berom0227/Semantic-Concern-SLM-Qwen-adapter"
)
EXPECTED_QUANTIZATION_TARGETS: Final[list[str]] = ["q4_K_M", "q8_0"]
EXPECTED_QWEN_LORA_CONFIG: Final[dict[str, str]] = {
    "rank": "32",
    "alpha": "48",
    "dropout": "0.05",
}
EXPECTED_QWEN_TRAINING_CONFIG: Final[dict[str, str]] = {
    "learning_rate": "5e-5",
    "num_train_epochs": "5",
    "per_device_train_batch_size": "1",
    "gradient_accumulation_steps": "8",
    "warmup_ratio": "0.1",
    "save_strategy": "no",
    "seed": "42",
    "max_seq_length": "16384",
    "packing": "true",
}

EXPECTED_DEFAULT_DF_COLUMNS: Final[list[str]] = [
    "predicted_types",
    "actual_types",
    "inference_time",
    "shas",
    "precision",
    "recall",
    "f1",
    "exact_match",
    "hamming_loss",
    "context_len",
    "with_message",
    "concern_count",
]
EXPECTED_COMMIT_TYPES: Final[list[str]] = [
    "docs",
    "test",
    "ci",
    "build",
    "refactor",
    "feat",
    "fix",
]


def _assignment_expression(path: Path, name: str) -> ast.expr:
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for statement in module.body:
        if not isinstance(statement, ast.Assign):
            continue
        for target in statement.targets:
            if isinstance(target, ast.Name) and target.id == name:
                return statement.value
    raise AssertionError(f"Missing assignment for {name} in {path}")


def _string_constant(expression: ast.expr) -> str:
    assert isinstance(expression, ast.Constant)
    assert isinstance(expression.value, str)
    return expression.value


def _scalar_constant(expression: ast.expr) -> str | int | float | bool:
    assert isinstance(expression, ast.Constant)
    value = expression.value
    assert isinstance(value, (str, int, float, bool))
    return value


def _string_mapping(expression: ast.expr) -> dict[str, str]:
    assert isinstance(expression, ast.Dict)
    values: dict[str, str] = {}
    for key_expression, value_expression in zip(expression.keys, expression.values):
        assert key_expression is not None
        values[_string_constant(key_expression)] = _string_constant(value_expression)
    return values


def _nested_string_mapping(expression: ast.expr) -> dict[str, dict[str, str]]:
    assert isinstance(expression, ast.Dict)
    values: dict[str, dict[str, str]] = {}
    for key_expression, value_expression in zip(expression.keys, expression.values):
        assert key_expression is not None
        values[_string_constant(key_expression)] = _string_mapping(value_expression)
    return values


def _integer_list(expression: ast.expr) -> list[int]:
    assert isinstance(expression, ast.List)
    values: list[int] = []
    for element in expression.elts:
        value = _scalar_constant(element)
        assert isinstance(value, int)
        values.append(value)
    return values


def _string_list(expression: ast.expr) -> list[str]:
    assert isinstance(expression, ast.List)
    return [_string_constant(element) for element in expression.elts]


def _yaml_section(path: Path, section_name: str) -> dict[str, str]:
    values: dict[str, str] = {}
    in_section = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped_line = raw_line.strip()
        if not in_section:
            in_section = stripped_line == f"{section_name}:" and not raw_line.startswith(" ")
            continue
        if stripped_line and not raw_line.startswith(" "):
            break
        if not stripped_line or stripped_line.startswith("#"):
            continue
        key, separator, raw_value = stripped_line.partition(":")
        assert separator
        values[key] = raw_value.partition("#")[0].strip().strip('"')
    assert values
    return values


def _render_model_name_template(expression: ast.expr, model_name: str) -> str:
    if isinstance(expression, ast.Constant):
        assert isinstance(expression.value, str)
        return expression.value

    assert isinstance(expression, ast.JoinedStr)
    assert len(expression.values) == 3
    prefix, placeholder, suffix = expression.values
    assert isinstance(prefix, ast.Constant)
    assert isinstance(prefix.value, str)
    assert isinstance(placeholder, ast.FormattedValue)
    assert isinstance(placeholder.value, ast.Name)
    assert placeholder.value.id == "MODEL_NAME"
    assert isinstance(suffix, ast.Constant)
    assert isinstance(suffix.value, str)
    return f"{prefix.value}{model_name}{suffix.value}"


def test_legacy_inference_qwen_lora_registry_entry_is_unchanged() -> None:
    model_configs = _nested_string_mapping(
        _assignment_expression(INFER_PATH, "MODEL_CONFIGS")
    )

    assert model_configs["qwen_lora"] == EXPECTED_QWEN_LORA


def test_legacy_inference_defaults_are_unchanged() -> None:
    assert _integer_list(
        _assignment_expression(INFER_PATH, "DEFAULT_CONTEXT_WINDOWS")
    ) == EXPECTED_CONTEXT_WINDOWS
    assert (
        _scalar_constant(_assignment_expression(INFER_PATH, "DEFAULT_MAX_TOKENS"))
        == EXPECTED_MAX_TOKENS
    )
    assert (
        _scalar_constant(_assignment_expression(INFER_PATH, "DEFAULT_SEED"))
        == EXPECTED_SEED
    )
    assert (
        _scalar_constant(_assignment_expression(INFER_PATH, "DEFAULT_TEMPERATURE"))
        == EXPECTED_TEMPERATURE
    )


def test_legacy_conversion_qwen_identity_and_repositories_are_unchanged() -> None:
    model_configs = _nested_string_mapping(
        _assignment_expression(CONVERSION_PATH, "MODEL_CONFIGS")
    )
    qwen_config = model_configs["qwen"]

    assert qwen_config["base_model_id"] == EXPECTED_QWEN_BASE_MODEL_ID
    assert qwen_config["model_name"] == EXPECTED_QWEN_MODEL_NAME
    assert _render_model_name_template(
        _assignment_expression(CONVERSION_PATH, "HF_REPO_NAME"),
        qwen_config["model_name"],
    ) == EXPECTED_QWEN_GGUF_REPO
    assert _render_model_name_template(
        _assignment_expression(CONVERSION_PATH, "HF_ADAPTER_REPO"),
        qwen_config["model_name"],
    ) == EXPECTED_QWEN_ADAPTER_REPO


def test_legacy_conversion_quantization_targets_are_unchanged() -> None:
    assert _string_list(
        _assignment_expression(CONVERSION_PATH, "QUANT_TYPES")
    ) == EXPECTED_QUANTIZATION_TARGETS


def test_legacy_qwen_model_and_lora_configuration_is_unchanged() -> None:
    model_config = _yaml_section(QWEN_CONFIG_PATH, "model")
    lora_config = _yaml_section(QWEN_CONFIG_PATH, "lora")

    assert model_config["id"] == EXPECTED_QWEN_BASE_MODEL_ID
    assert {
        key: lora_config[key] for key in EXPECTED_QWEN_LORA_CONFIG
    } == EXPECTED_QWEN_LORA_CONFIG


def test_legacy_qwen_training_configuration_is_unchanged() -> None:
    training_config = _yaml_section(QWEN_CONFIG_PATH, "training")

    assert {
        key: training_config[key] for key in EXPECTED_QWEN_TRAINING_CONFIG
    } == EXPECTED_QWEN_TRAINING_CONFIG


def test_legacy_default_dataframe_columns_are_unchanged() -> None:
    assert DEFAULT_DF_COLUMNS == EXPECTED_DEFAULT_DF_COLUMNS


def test_legacy_commit_type_labels_are_unchanged() -> None:
    assert COMMIT_TYPES == EXPECTED_COMMIT_TYPES


def test_unsloth_writer_encodes_the_label_columns_the_legacy_way() -> None:
    """Pin the label columns' encoding against the frozen 14B pipeline's.

    ``predicted_types`` and ``actual_types`` are the columns that carry structure, that
    ``RQ/analysis`` parses across both models, and that the writer and the finalization gate
    already once disagreed about. The 14B writer cannot drift — it is sha256-pinned above —
    so pinning the live writer here leaves the pair with no way to diverge.

    ``shas`` is deliberately not pinned. The two pipelines do encode it differently (the 14B
    path hands pandas a list, which stringifies to a Python repr; this one emits JSON), but
    nothing reads that column across pipelines: ``RQ/analysis`` never touches it, and the
    only reader is this package's own resume path, on files this package itself wrote.
    """
    from RQ.SLM.unsloth.results import InferenceResult

    row = InferenceResult(
        predicted_types=("fix", "test"),
        actual_types=("fix", "refactor"),
        inference_time=1.25,
        shas=("abc123", "def456"),
        context_len=4096,
        with_message=True,
    ).as_csv_row()

    # Byte-identical to what RQ/SLM/infer.py writes for the same observation.
    assert row["predicted_types"] == '["fix", "test"]'
    assert row["actual_types"] == '["fix", "refactor"]'
    assert list(row) == EXPECTED_DEFAULT_DF_COLUMNS


def test_legacy_protected_file_digests_still_verify() -> None:
    manifest_lines = PROTECTED_FILES_PATH.read_text(encoding="utf-8").splitlines()

    for manifest_line in manifest_lines:
        expected_digest, relative_path = manifest_line.split(maxsplit=1)
        protected_path = REPO_ROOT / relative_path
        actual_digest = hashlib.sha256(protected_path.read_bytes()).hexdigest()
        assert actual_digest == expected_digest, relative_path
