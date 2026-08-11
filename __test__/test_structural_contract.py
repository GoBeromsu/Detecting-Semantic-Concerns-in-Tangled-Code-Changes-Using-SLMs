import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

SCRIPTS_DIRECTORY = Path(__file__).parents[1] / "datasets" / "scripts"
DATA_DIRECTORY = Path(__file__).parents[1] / "datasets" / "data"


def _run_driver(source: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(SCRIPTS_DIRECTORY)
    return subprocess.run(
        (sys.executable, "-c", dedent(source), *arguments),
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )


SEMANTICS_DRIVER = """
from structural_validity import *

def pair(a, b, file=False, directory=False, depth=0, reasons=()):
    return PairPathMetrics(a, b, file, directory, depth, reasons)

commit = SyntheticCommit(CommitId("train:7"), Split.TRAIN, "owner/repo", 3, ("a", "b", "c"))
assert common_parent_prefix_depth(("README.md",), ("README.md",)) == 0
assert common_parent_prefix_depth(("src/domain/a.py",), ("src/domain/b.py",)) == 2

one_edge = aggregate_commit_metrics(commit, (
    pair(0, 1, True, True, 2), pair(0, 2), pair(1, 2),
))
assert one_edge.same_file_any is True
assert one_edge.same_file_all_concerns is False
assert one_edge.same_leaf_directory_any is True
assert one_edge.same_leaf_directory_all_concerns is False

participation = aggregate_commit_metrics(commit, (
    pair(0, 1, True, True, 2), pair(0, 2), pair(1, 2, True, True, 3),
))
assert participation.same_file_all_concerns is True
assert participation.same_file_all_pairs is False
assert participation.same_leaf_directory_all_concerns is True
assert participation.same_leaf_directory_all_pairs is False
assert participation.common_parent_prefix_depth == 3
assert participation.strict_common_parent_prefix_depth == 2

clique = aggregate_commit_metrics(commit, (
    pair(0, 1, True, True), pair(0, 2, True, True), pair(1, 2, True, True),
))
assert clique.same_file_all_pairs is True
assert clique.same_leaf_directory_all_pairs is True

no_overlap = aggregate_commit_metrics(commit, (
    pair(0, 1, reasons=(ReasonCode.AMBIGUOUS_PATH,)), pair(0, 2), pair(1, 2),
))
assert no_overlap.same_file_any is False
assert no_overlap.same_leaf_directory_any is False
assert no_overlap.common_parent_prefix_depth == 0
assert no_overlap.strict_common_parent_prefix_depth == 0
assert no_overlap.reason_codes == (ReasonCode.AMBIGUOUS_PATH,)
print("any=true all-concerns=false; participation-all-concerns=true")
"""


def test_pair_and_commit_metric_semantics_are_deterministic() -> None:
    # Given root, overlap, participation, clique, and failure fixtures
    first = _run_driver(SEMANTICS_DRIVER)
    second = _run_driver(SEMANTICS_DRIVER)

    # When the real contract import is executed twice
    assert first.returncode == second.returncode == 0, first.stderr + second.stderr

    # Then every formula passes and output is byte-deterministic
    assert first.stdout == second.stdout == "any=true all-concerns=false; participation-all-concerns=true\n"


def test_boundary_rejects_k_one_sha_mismatch_and_missing_pairs() -> None:
    # Given malformed commit and pair inputs
    result = _run_driver(
        """
        from dataclasses import FrozenInstanceError
        from structural_validity import *

        cases = (
            ((CommitId("test:4"), Split.TEST, "owner/repo", 1, ("a",)), ReasonCode.INVALID_CONCERN_COUNT),
            ((CommitId("train:99"), Split.TRAIN, "owner/repo", 3, ("a", "b")), ReasonCode.SHA_COUNT_MISMATCH),
        )
        for arguments, reason in cases:
            try:
                SyntheticCommit(*arguments)
            except StructuralContractError as error:
                assert error.reason is reason
                assert str(arguments[0]) in str(error)
            else:
                raise AssertionError("malformed commit accepted")

        pair = PairPathMetrics(0, 1, False, False, 0)
        mutation_rejected = False
        try:
            setattr(pair, "same_file", True)
        except FrozenInstanceError:
            mutation_rejected = True
        assert mutation_rejected

        try:
            PairPathMetrics(0, 1, True, False, 0, (ReasonCode.UNRESOLVED_PATH,))
        except StructuralContractError as error:
            assert error.reason is ReasonCode.PAIR_CONSERVATION
        else:
            raise AssertionError("reason-coded failure reported overlap")

        commit = SyntheticCommit(CommitId("train:7"), Split.TRAIN, "owner/repo", 3, ("a", "b", "c"))
        try:
            aggregate_commit_metrics(commit, (pair,))
        except StructuralContractError as error:
            assert error.reason is ReasonCode.PAIR_CONSERVATION
        else:
            raise AssertionError("incomplete pair set accepted")
        print("rejected: test:4, train:99, incomplete-pairs")
        """
    )

    # When the typed boundary and conservation gate execute
    # Then no malformed input receives misleading success
    assert result.returncode == 0, result.stderr
    assert result.stdout == "rejected: test:4, train:99, incomplete-pairs\n"


def test_metadata_and_pair_record_serialize_with_exact_denominators() -> None:
    # Given the public metadata and one overlapping pair record
    result = _run_driver(
        """
        import json
        from structural_validity import *

        assert HEADLINE_CONTRACT.commit_denominator == PRIMARY_COMMIT_COUNT == 1400
        assert HEADLINE_CONTRACT.pair_denominator == SECONDARY_PAIR_COUNT == 7000
        assert HEADLINE_CONTRACT.concern_counts == (2, 3, 4, 5)
        assert DENOMINATOR_CONTRACTS[0].unit is ObservationUnit.COMMIT
        assert DENOMINATOR_CONTRACTS[0].failure_policy == "retain_false_or_zero"
        assert CoverageStatus.SOURCE_AST.value == "source_ast"
        json.dumps(HEADLINE_CONTRACT.as_json(), sort_keys=True)
        json.dumps([rule.as_json() for rule in DENOMINATOR_CONTRACTS], sort_keys=True)
        json.dumps([COMMIT_ARTIFACT_SCHEMA.as_json(), PAIR_ARTIFACT_SCHEMA.as_json()], sort_keys=True)

        commit = SyntheticCommit(CommitId("train:7"), Split.TRAIN, "owner/repo", 3, ("a", "b", "c"))
        record = PairRecord.from_commit(commit, PairPathMetrics(0, 1, True, False, 2))
        payload = json.loads(json.dumps(record.as_json(), sort_keys=True))
        assert payload["commit_id"] == "train:7"
        assert payload["same_file"] is True
        assert payload["same_leaf_directory"] is False
        assert payload["common_parent_prefix_depth"] == 2
        assert payload["reason_codes"] == []
        print(json.dumps(payload, sort_keys=True))
        """
    )

    # When serialized through the real imported module
    # Then typed values and primary/secondary semantics survive JSON
    assert result.returncode == 0, result.stderr
    assert '"commit_id": "train:7"' in result.stdout
    assert '"same_file": true' in result.stdout


def test_committed_data_conserves_1400_commits_and_7000_pairs() -> None:
    # Given both committed seed-43 split files
    result = _run_driver(
        """
        import csv
        import json
        import sys
        from pathlib import Path
        from structural_validity import *

        data_directory = Path(sys.argv[1])
        commits = []
        for split in Split:
            with (data_directory / f"tangled_ccs_dataset_{split.value}.csv").open(encoding="utf-8", newline="") as handle:
                for row_index, row in enumerate(csv.DictReader(handle)):
                    concern_count = int(row["concern_count"])
                    if concern_count >= 2:
                        commits.append(SyntheticCommit(
                            CommitId(f"{split.value}:{row_index}"), split, row["repo"],
                            concern_count, tuple(json.loads(row["shas"])),
                        ))
        pair_count = sum(len(commit.pair_indices()) for commit in commits)
        assert len(commits) == PRIMARY_COMMIT_COUNT == 1400
        assert pair_count == SECONDARY_PAIR_COUNT == 7000
        print(f"commits={len(commits)} pairs={pair_count}")
        """,
        str(DATA_DIRECTORY),
    )

    # When every primary model creates its unordered pair indices
    # Then both conservation laws match the frozen headline contract
    assert result.returncode == 0, result.stderr
    assert result.stdout == "commits=1400 pairs=7000\n"
