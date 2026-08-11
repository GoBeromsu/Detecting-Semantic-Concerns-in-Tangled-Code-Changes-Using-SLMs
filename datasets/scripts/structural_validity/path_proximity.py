"""How close two concerns' changed paths are, in plain counts.

Three questions in ascending coarseness: how many of the same files, how
many of the same leaf directories, and -- when they share neither -- how far
down a common directory prefix they still agree. Nothing here reads file
contents; these are string comparisons over path names.
"""

from __future__ import annotations

from collections.abc import Sequence


def _parent_parts(path: str) -> tuple[str, ...]:
    return tuple(part for part in path.split("/")[:-1] if part and part != ".")


def shared_file_count(paths_a: Sequence[str], paths_b: Sequence[str]) -> int:
    """How many changed file paths the two concerns have in common."""
    return len(frozenset(paths_a) & frozenset(paths_b))


def shared_directory_count(paths_a: Sequence[str], paths_b: Sequence[str]) -> int:
    """How many leaf directories the two concerns both changed a file in."""
    return len({_parent_parts(path) for path in paths_a} & {_parent_parts(path) for path in paths_b})


def shared_path_depth(paths_a: Sequence[str], paths_b: Sequence[str]) -> int:
    """Longest shared run of leading directory components across the two sets.

    The maximum over the cross product, so it answers "how close do these two
    concerns ever get", not "how close are they typically".
    """
    maximum = 0
    for path_a in paths_a:
        parent_a = _parent_parts(path_a)
        for path_b in paths_b:
            depth = 0
            for left, right in zip(parent_a, _parent_parts(path_b)):
                if left != right:
                    break
                depth += 1
            maximum = max(maximum, depth)
    return maximum


__all__ = ["shared_directory_count", "shared_file_count", "shared_path_depth"]
