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
    """How many non-root leaf directories the two concerns both changed a file in.

    Directories are keyed by their full path from the repository root, so
    `app/utils` and `lib/utils` are different directories despite the shared
    name.

    The repository root is excluded. Every top-level file shares it by
    construction, so counting it made "one concern touched README.md, the other
    touched .gitignore" indistinguishable from two concerns working inside the
    same module. On the seed-43 population that case was 51% of all observed
    directory sharing -- the rate halves, from 9.1% to 4.4% of pairs, once the
    root is removed. Root-level co-touch is still visible as a shared file when
    the two concerns edit the *same* top-level file.
    """
    shared = {_parent_parts(path) for path in paths_a} & {_parent_parts(path) for path in paths_b}
    return len(shared - {()})


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
