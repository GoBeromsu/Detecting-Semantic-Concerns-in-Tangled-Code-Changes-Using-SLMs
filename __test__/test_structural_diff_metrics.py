"""Path/hunk/line evidence contracts for committed unified diffs (Todo 3)."""

from __future__ import annotations

import csv
import math
import subprocess
import sys
from pathlib import Path
from typing import Final

import pytest

_ = csv.field_size_limit(sys.maxsize)

from __test__.conftest import DATASET_DIR, REPO_ROOT

SCRIPTS_DIR: Final[Path] = REPO_ROOT / "datasets" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from structural_validity.diff_metrics import (  # noqa: E402
    ChangeStatus,
    ReasonCode,
    dirname_of,
    pair_colocation,
    pair_diff_metrics,
    parse_atomic_diff,
    parse_committed_diff,
)
from structural_validity.colocation_cli import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    ExistingOutputError,
    resolve_output_paths,
)


def _modified(path: str, body: str, *, context: str = "def demo():") -> str:
    return (
        f"diff --git a/{path} b/{path}\n"
        f"--- a/{path}\n"
        f"+++ b/{path}\n"
        f"@@ -1,3 +1,3 @@ {context}\n"
        f"{body}"
    )


def test_ordinary_add_copy_rename_use_new_side_identity() -> None:
    # Given ordinary, added, copied, and renamed file blocks
    ordinary = (
        "diff --git a/src/a.py b/src/a.py\n"
        "--- a/src/a.py\n"
        "+++ b/src/a.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )
    added = (
        "diff --git a/src/new.py b/src/new.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/src/new.py\n"
        "@@ -0,0 +1,2 @@\n"
        "+one\n"
        "+two\n"
    )
    copied = (
        'diff --git a/src/old.py b/src/copy of old.py\n'
        "copy from src/old.py\n"
        "copy to src/copy of old.py\n"
        "--- a/src/old.py\n"
        '+++ b/src/copy of old.py\n'
        "@@ -1 +1,2 @@\n"
        " keep\n"
        "+extra\n"
    )
    renamed = (
        "diff --git a/src/before.py b/src/after.py\n"
        "similarity index 90%\n"
        "rename from src/before.py\n"
        "rename to src/after.py\n"
        "--- a/src/before.py\n"
        "+++ b/src/after.py\n"
        "@@ -1 +1 @@\n"
        "-before\n"
        "+after\n"
    )

    # When each committed diff is parsed
    ordinary_files = parse_committed_diff(ordinary).files
    added_files = parse_committed_diff(added).files
    copied_files = parse_committed_diff(copied).files
    renamed_files = parse_committed_diff(renamed).files

    # Then headline identity is the new-side path and rename keeps both audit paths
    assert ordinary_files[0].identity.path == "src/a.py"
    assert ordinary_files[0].identity.status is ChangeStatus.ORDINARY
    assert added_files[0].identity.path == "src/new.py"
    assert added_files[0].identity.status is ChangeStatus.ADD
    assert added_files[0].new_side_lines == frozenset({1, 2})
    assert copied_files[0].identity.path == "src/copy of old.py"
    assert copied_files[0].identity.status is ChangeStatus.COPY
    assert renamed_files[0].identity.path == "src/after.py"
    assert renamed_files[0].identity.old_path == "src/before.py"
    assert renamed_files[0].identity.new_path == "src/after.py"
    assert renamed_files[0].identity.status is ChangeStatus.RENAME
    assert parse_committed_diff(renamed).headline_paths() == frozenset({"src/after.py"})


def test_deletion_uses_old_side_identity_and_has_no_new_side_coordinates() -> None:
    # Given a deleted file with only removed lines
    deleted = (
        "diff --git a/gone.py b/gone.py\n"
        "deleted file mode 100644\n"
        "--- a/gone.py\n"
        "+++ /dev/null\n"
        "@@ -1,2 +0,0 @@\n"
        "-one\n"
        "-two\n"
    )

    # When the diff is parsed
    evidence = parse_committed_diff(deleted)
    file_evidence = evidence.files[0]

    # Then identity is the old path and no new-side line coordinates exist
    assert file_evidence.identity.path == "gone.py"
    assert file_evidence.identity.status is ChangeStatus.DELETE
    assert file_evidence.new_side_lines == frozenset()
    assert file_evidence.hunk_starts == (0,)


def test_quoted_paths_with_spaces_and_root_files_parse() -> None:
    # Given a quoted path containing spaces and a root-level file
    quoted = (
        'diff --git "a/src/my file.py" "b/src/my file.py"\n'
        '--- "a/src/my file.py"\n'
        '+++ "b/src/my file.py"\n'
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )
    root = (
        "diff --git a/README.md b/README.md\n"
        "--- a/README.md\n"
        "+++ b/README.md\n"
        "@@ -1 +1 @@\n"
        "-a\n"
        "+b\n"
    )

    # When parsed
    quoted_path = parse_committed_diff(quoted).files[0].identity.path
    root_path = parse_committed_diff(root).files[0].identity.path

    # Then path identity preserves spaces and root basenames
    assert quoted_path == "src/my file.py"
    assert root_path == "README.md"
    assert dirname_of(root_path) == ""


def test_git_octal_quoted_utf8_path_decodes_to_nfc_text() -> None:
    # Given git C-style quoting with octal UTF-8 bytes for café.txt
    quoted = (
        'diff --git "a/caf\\303\\251.txt" "b/caf\\303\\251.txt"\n'
        '--- "a/caf\\303\\251.txt"\n'
        '+++ "b/caf\\303\\251.txt"\n'
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )

    # When the committed diff is parsed
    path = parse_committed_diff(quoted).files[0].identity.path

    # Then octal bytes decode as UTF-8 text, not mojibake
    assert path == "café.txt"


def test_binary_and_malformed_headers_remain_reason_coded() -> None:
    # Given a binary block and a malformed header
    binary = (
        "diff --git a/a.wasm b/a.wasm\n"
        "new file mode 100644\n"
        "Binary files /dev/null and b/a.wasm differ\n"
    )
    malformed = "diff --git totally-broken\n@@ -1 +1 @@\n+x\n"

    # When parsed
    binary_file = parse_committed_diff(binary).files[0]
    malformed_file = parse_committed_diff(malformed).files[0]

    # Then binary is retained without line coords and malformed stays ambiguous
    assert binary_file.identity.status is ChangeStatus.BINARY
    assert binary_file.identity.path == "a.wasm"
    assert binary_file.new_side_lines == frozenset()
    assert binary_file.is_binary is True
    assert malformed_file.identity.status is ChangeStatus.AMBIGUOUS_PATH
    assert malformed_file.identity.path is None
    assert ReasonCode.AMBIGUOUS_PATH in malformed_file.identity.reason_codes


def test_multiple_hunks_and_function_contexts_accumulate() -> None:
    # Given one file with two hunks and one empty context
    multi = (
        "diff --git a/src/app.py b/src/app.py\n"
        "--- a/src/app.py\n"
        "+++ b/src/app.py\n"
        "@@ -10,2 +10,3 @@ def first():\n"
        " keep\n"
        "+added_a\n"
        " keep2\n"
        "@@ -40,1 +41,2 @@\n"
        " keep3\n"
        "+added_b\n"
    )

    # When parsed
    file_evidence = parse_committed_diff(multi).files[0]

    # Then contexts and new-side coordinates accumulate across hunks
    assert file_evidence.contexts == frozenset({"def first():"})
    assert file_evidence.hunk_starts == (10, 41)
    assert file_evidence.new_side_lines == frozenset({11, 42})
    assert file_evidence.max_new_line == 42


def test_same_path_disjoint_added_lines_are_same_file_with_finite_gap_not_same_line() -> None:
    # Given two historical diffs that touch one path on disjoint added lines
    left = (
        "diff --git a/src/demo.py b/src/demo.py\n"
        "--- a/src/demo.py\n"
        "+++ b/src/demo.py\n"
        "@@ -1,3 +1,4 @@ def demo():\n"
        " keep\n"
        "+left_only\n"
        " keep2\n"
        " keep3\n"
    )
    right = (
        "diff --git a/src/demo.py b/src/demo.py\n"
        "--- a/src/demo.py\n"
        "+++ b/src/demo.py\n"
        "@@ -20,3 +20,4 @@ def demo():\n"
        " keep\n"
        "+right_only\n"
        " keep2\n"
        " keep3\n"
    )

    # When pair metrics are computed
    metrics = pair_diff_metrics(parse_committed_diff(left), parse_committed_diff(right))

    # Then same_file is true with a finite gap and no same-line claim
    assert metrics.same_file is True
    assert metrics.same_function is True
    assert metrics.min_line_gap == 19
    assert metrics.diff_local_normalized_gap is not None
    assert math.isclose(metrics.diff_local_normalized_gap, 19 / 21)
    assert metrics.legacy_hunk_start_gap == 19
    assert metrics.same_line is False
    assert metrics.min_line_gap is not None and metrics.min_line_gap > 0


def test_overlapping_added_lines_set_same_line_and_zero_gap() -> None:
    # Given two diffs that add the same new-side coordinate
    left = _modified("x.py", " keep\n+shared\n keep2\n")
    right = _modified("x.py", " keep\n+also_shared\n keep2\n")

    # When pair metrics are computed
    metrics = pair_diff_metrics(parse_committed_diff(left), parse_committed_diff(right))

    # Then the shared coordinate is an explicit same-line observation
    assert metrics.same_file is True
    assert metrics.same_line is True
    assert metrics.min_line_gap == 0
    assert metrics.diff_local_normalized_gap == 0.0


def test_deletion_only_shared_path_has_no_measurable_line_gap() -> None:
    # Given two deletion-only edits on the same path
    left = (
        "diff --git a/x.py b/x.py\n"
        "--- a/x.py\n"
        "+++ b/x.py\n"
        "@@ -5,1 +5,0 @@\n"
        "-gone_left\n"
    )
    right = (
        "diff --git a/x.py b/x.py\n"
        "--- a/x.py\n"
        "+++ b/x.py\n"
        "@@ -9,1 +9,0 @@\n"
        "-gone_right\n"
    )

    # When pair metrics are computed
    metrics = pair_diff_metrics(parse_committed_diff(left), parse_committed_diff(right))

    # Then same_file holds but line proximity is not measurable
    assert metrics.same_file is True
    assert metrics.min_line_gap is None
    assert metrics.diff_local_normalized_gap is None
    assert metrics.same_line is False
    assert metrics.legacy_hunk_start_gap == 4


def test_legacy_parse_atomic_diff_schema_and_pair_colocation_keys() -> None:
    # Given a simple modified file
    diff_text = _modified("pkg/mod.py", " keep\n+added\n keep2\n", context="class Mod:")

    # When the compatibility wrappers run
    legacy = parse_atomic_diff(diff_text)
    pair = pair_colocation(legacy, legacy)

    # Then the historical dict schema and pair keys remain intact
    assert set(legacy) == {"pkg/mod.py"}
    assert set(legacy["pkg/mod.py"]) == {"contexts", "starts"}
    assert legacy["pkg/mod.py"]["contexts"] == {"class Mod:"}
    assert legacy["pkg/mod.py"]["starts"] == [1]
    assert set(pair) == {"same_dir", "same_file", "same_function", "min_line_gap"}
    assert pair["same_file"] is True
    assert pair["same_function"] is True
    assert pair["min_line_gap"] == 0


def test_compatibility_entrypoint_reexports_legacy_callables() -> None:
    # Given the thin analyze_colocation entrypoint
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(SCRIPTS_DIR)!r}); "
        "import analyze_colocation as module; "
        "assert callable(module.parse_atomic_diff); "
        "assert callable(module.pair_colocation); "
        "assert module.dirname_of('src/pkg/a.py') == 'src/pkg'"
    )

    # When imported by its historical module name
    completed = subprocess.run(
        (sys.executable, "-c", code),
        check=False,
        capture_output=True,
        text=True,
    )

    # Then the public compatibility names remain importable
    assert completed.returncode == 0, completed.stderr


def test_default_output_paths_refuse_existing_dirty_files(tmp_path: Path) -> None:
    # Given the compatibility entrypoint and pre-existing default outputs
    default_csv = DEFAULT_OUTPUT_DIR / "colocation_by_k.csv"
    default_summary = DEFAULT_OUTPUT_DIR / "colocation_summary.md"
    assert default_csv.exists()
    assert default_summary.exists()

    # When resolve_output_paths is called without an override
    # Then it refuses rather than overwriting dirty defaults
    with pytest.raises(ExistingOutputError):
        _ = resolve_output_paths(None, force_default_output=False)

    # When an explicit output directory is selected
    out_dir = tmp_path / "fresh"
    directory, csv_path, summary_path = resolve_output_paths(out_dir)

    # Then writes are redirected away from the dirty defaults
    assert directory == out_dir
    assert csv_path == out_dir / "colocation_by_k.csv"
    assert summary_path == out_dir / "colocation_summary.md"


def test_real_fixture_same_path_disjoint_hunks_from_ccs_dataset() -> None:
    # Given a committed CCS row known to edit one path across multiple hunks
    target_sha = "fd53cd188555f5c3dc8bc341c5d7eb04b761a70f"
    with (DATASET_DIR / "CCS Dataset.csv").open(encoding="utf-8", newline="") as handle:
        row = next(r for r in csv.DictReader(handle) if r["sha"] == target_sha)

    # When the committed git_diff is parsed twice as a synthetic pair
    evidence = parse_committed_diff(row["git_diff"])
    metrics = pair_diff_metrics(evidence, evidence)

    # Then the path is shared, line proximity is finite, and same-line is not over-claimed
    # beyond the actual self-overlap of identical new-side coordinates.
    assert "src/app.rs" in evidence.headline_paths()
    assert metrics.same_file is True
    assert metrics.min_line_gap == 0
    assert metrics.same_line is True
    file_evidence = next(f for f in evidence.files if f.identity.path == "src/app.rs")
    assert len(file_evidence.hunk_starts) >= 2
    assert len(file_evidence.new_side_lines) >= 2


def test_empty_diff_and_no_hunk_body_are_safe() -> None:
    # Given empty text and a header-only block
    empty = parse_committed_diff("")
    header_only = parse_committed_diff("diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n")

    # When converted to legacy maps / pair metrics
    # Then no exception is raised and values stay conservative
    assert empty.files == ()
    assert parse_atomic_diff("") == {}
    assert header_only.files[0].new_side_lines == frozenset()
    metrics = pair_diff_metrics(header_only, header_only)
    assert metrics.same_file is True
    assert metrics.min_line_gap is None
