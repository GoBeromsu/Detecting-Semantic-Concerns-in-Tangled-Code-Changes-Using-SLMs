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


PROXIMITY_DRIVER = """
from structural_validity import *

a = ("src/domain/order.py", "src/domain/cart.py", "docs/readme.md")
b = ("src/domain/order.py", "src/domain/cart.py", "src/api/view.py")
assert shared_file_count(a, b) == 2
assert shared_directory_count(a, b) == 1
assert shared_path_depth(a, b) == 2

assert shared_file_count(a, ()) == 0

# A directory is keyed by its full path, so a shared basename is not a match.
assert shared_directory_count(("app/utils/x.py",), ("lib/utils/y.py",)) == 0
assert shared_path_depth(("app/utils/x.py",), ("lib/utils/y.py",)) == 0

# The repository root is excluded: every top-level file would share it.
assert shared_directory_count(("README.md",), ("CHANGELOG.md",)) == 0
assert shared_path_depth(("README.md",), ("CHANGELOG.md",)) == 0

# Sharing the same top-level file still registers, as a file rather than a
# directory -- which is the only case where files exceed directories at zero.
assert shared_file_count(("README.md",), ("README.md",)) == 1
assert shared_directory_count(("README.md",), ("README.md",)) == 0

# Below the root, sharing a file always implies sharing its directory.
for path in ("src/a.py", "src/domain/deep/a.py", "./pkg/mod/a.py"):
    assert shared_file_count((path,), (path,)) == 1
    assert shared_directory_count((path,), (path,)) == 1

assert shared_path_depth(("src/domain/a.py",), ("src/domain/b.py",)) == 2
assert shared_path_depth(("src/a.py",), ("lib/b.py",)) == 0
print("files=2 directories=1 depth=2")
"""


def test_path_proximity_counts_how_much_two_concerns_have_in_common() -> None:
    # Given two concerns sharing two of three changed files
    first = _run_driver(PROXIMITY_DRIVER)
    second = _run_driver(PROXIMITY_DRIVER)

    # When proximity is measured twice through the real import
    assert first.returncode == second.returncode == 0, first.stderr + second.stderr

    # Then counts, not flags, distinguish partial from total overlap
    assert first.stdout == second.stdout == "files=2 directories=1 depth=2\n"


SEMANTICS_DRIVER = """
from structural_validity import *

def pair(a, b, files=0, directories=0, depth=0, reasons=()):
    return PairPathMetrics(a, b, files, directories, depth, reasons)

commit = SyntheticCommit(CommitId("train:7"), Split.TRAIN, "owner/repo", 3, ("a", "b", "c"))

one_edge = aggregate_commit_metrics(commit, (
    pair(0, 1, 1, 1, 2), pair(0, 2), pair(1, 2),
))
assert one_edge.any_pair_shares_file is True
assert one_edge.any_pair_shares_directory is True
assert one_edge.concerns_sharing_file == 2
assert one_edge.file_share == 2 / 3
assert one_edge.all_concerns_share_file is False

participation = aggregate_commit_metrics(commit, (
    pair(0, 1, 1, 1, 2), pair(0, 2), pair(1, 2, 4, 1, 3),
))
assert participation.concerns_sharing_file == 3
assert participation.file_share == 1.0
assert participation.all_concerns_share_file is True
assert participation.all_pairs_share_file is False
assert participation.all_concerns_share_directory is True
assert participation.all_pairs_share_directory is False
assert participation.max_shared_path_depth == 3
assert participation.min_shared_path_depth == 2

clique = aggregate_commit_metrics(commit, (
    pair(0, 1, 1, 1), pair(0, 2, 1, 1), pair(1, 2, 1, 1),
))
assert clique.all_pairs_share_file is True
assert clique.all_pairs_share_directory is True

no_overlap = aggregate_commit_metrics(commit, (
    pair(0, 1, reasons=(ReasonCode.AMBIGUOUS_PATH,)), pair(0, 2), pair(1, 2),
))
assert no_overlap.any_pair_shares_file is False
assert no_overlap.any_pair_shares_directory is False
assert no_overlap.concerns_sharing_file == 0
assert no_overlap.file_share == 0.0
assert no_overlap.max_shared_path_depth == 0
assert no_overlap.min_shared_path_depth == 0
assert no_overlap.reason_codes == (ReasonCode.AMBIGUOUS_PATH,)
print("one-edge-share=2/3 participation-share=3/3")
"""


def test_pair_and_commit_metric_semantics_are_deterministic() -> None:
    # Given single-edge, participation, clique, and failure fixtures
    first = _run_driver(SEMANTICS_DRIVER)
    second = _run_driver(SEMANTICS_DRIVER)

    # When the real contract import is executed twice
    assert first.returncode == second.returncode == 0, first.stderr + second.stderr

    # Then every formula passes and output is byte-deterministic
    assert first.stdout == second.stdout == "one-edge-share=2/3 participation-share=3/3\n"


def test_concern_share_normalises_within_a_commit_before_averaging() -> None:
    # Given one k=2 and one k=5 commit, each with a single sharing pair
    result = _run_driver(
        """
        from structural_validity import *

        def metrics(k, edges):
            commit = SyntheticCommit(
                CommitId(f"train:{k}"), Split.TRAIN, "owner/repo", k,
                tuple(str(index) for index in range(k)),
            )
            pairs = tuple(
                PairPathMetrics(a, b, 1, 1, 1) if (a, b) in edges else PairPathMetrics(a, b, 0, 0, 0)
                for a, b in commit.pair_indices()
            )
            return aggregate_commit_metrics(commit, pairs)

        small, large = metrics(2, {(0, 1)}), metrics(5, {(0, 1)})

        # Pooling over pairs would weight the k=5 commit ten times the k=2 one.
        assert small.concerns_sharing_file == large.concerns_sharing_file == 2
        assert small.file_share == 1.0
        assert large.file_share == 0.4
        assert small.any_pair_shares_file is large.any_pair_shares_file is True

        # Sharing is symmetric, so exactly one concern can never be connected.
        for k in (2, 3, 4, 5):
            for share in (metrics(k, set()), metrics(k, {(0, 1)})):
                assert share.concerns_sharing_file != 1
        print(f"k2={small.file_share} k5={large.file_share}")
        """
    )

    # When both contribute exactly one commit-level observation
    # Then the "at least one pair" flag agrees while the share separates them
    assert result.returncode == 0, result.stderr
    assert result.stdout == "k2=1.0 k5=0.4\n"


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

        pair = PairPathMetrics(0, 1, 0, 0, 0)
        mutation_rejected = False
        try:
            setattr(pair, "shared_file_count", 1)
        except FrozenInstanceError:
            mutation_rejected = True
        assert mutation_rejected

        try:
            PairPathMetrics(0, 1, 1, 1, 0, (ReasonCode.UNRESOLVED_PATH,))
        except StructuralContractError as error:
            assert error.reason is ReasonCode.PAIR_CONSERVATION
        else:
            raise AssertionError("reason-coded failure reported overlap")

        # Files without directories is legal: a shared *top-level* file has no
        # non-root directory to report, so the two counts are not tied here.
        assert PairPathMetrics(0, 1, 1, 0, 0).shares_file is True

        # Many files can still sit inside one shared directory.
        assert PairPathMetrics(0, 1, 5, 1, 2).shares_file is True

        for negative in ((0, 1, -1, 0, 0), (0, 1, 0, -1, 0), (0, 1, 0, 0, -1)):
            try:
                PairPathMetrics(*negative)
            except StructuralContractError as error:
                assert error.reason is ReasonCode.PAIR_CONSERVATION
            else:
                raise AssertionError("negative count accepted")

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
        record = PairRecord.from_commit(commit, PairPathMetrics(0, 1, 3, 1, 2))
        payload = json.loads(json.dumps(record.as_json(), sort_keys=True))
        assert payload["commit_id"] == "train:7"
        assert payload["shared_file_count"] == 3
        assert payload["shares_file"] is True
        assert payload["shared_directory_count"] == 1
        assert payload["shares_directory"] is True
        assert payload["shared_path_depth"] == 2
        assert payload["reason_codes"] == []
        assert frozenset(payload) == frozenset(PAIR_ARTIFACT_SCHEMA.required_fields)
        print(json.dumps(payload, sort_keys=True))
        """
    )

    # When serialized through the real imported module
    # Then typed values and primary/secondary semantics survive JSON
    assert result.returncode == 0, result.stderr
    assert '"commit_id": "train:7"' in result.stdout
    assert '"shared_file_count": 3' in result.stdout


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
