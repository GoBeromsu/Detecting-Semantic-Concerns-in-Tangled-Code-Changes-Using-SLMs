from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Final, final

from structural_validity.provenance import (
    MetadataPaths,
    MetadataRecord,
    ResolutionRow,
    ResolutionStatus,
    load_metadata_manifest,
    resolve_cached_sources,
)


@final
class CliArguments(argparse.Namespace):
    def __init__(self) -> None:
        super().__init__()
        self.train_csv = Path()
        self.test_csv = Path()
        self.pool_csv = Path()
        self.raw_csv = Path()
        self.repo_split_json = Path()
        self.output = Path()
        self.cache: Path | None = None
        self.force = False


CSV_FIELDS: Final[tuple[str, ...]] = (
    "sha",
    "repo",
    "split",
    "remote_url",
    "reference_count",
    "metadata_reason",
    "resolution_status",
    "parent_count",
    "parent_sha",
    "parent_policy",
    "change_kinds",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write deterministic source-resolution metadata for seed-43 SHAs."
    )
    _ = parser.add_argument("--train-csv", type=Path, required=True)
    _ = parser.add_argument("--test-csv", type=Path, required=True)
    _ = parser.add_argument("--pool-csv", type=Path, required=True)
    _ = parser.add_argument("--raw-csv", type=Path, required=True)
    _ = parser.add_argument("--repo-split-json", type=Path, required=True)
    _ = parser.add_argument("--output", type=Path, required=True)
    _ = parser.add_argument("--cache", type=Path)
    _ = parser.add_argument("--force", action="store_true")
    return parser


def _metadata_only_rows(records: Sequence[MetadataRecord]) -> tuple[ResolutionRow, ...]:
    return tuple(
        ResolutionRow(
            record.sha,
            record.repo,
            record.remote_url,
            ResolutionStatus.METADATA_ONLY,
        )
        for record in records
    )


def _csv_rows(
    records: Sequence[MetadataRecord],
    resolutions: Sequence[ResolutionRow],
) -> tuple[tuple[str, ...], ...]:
    resolution_by_sha = {row.sha: row for row in resolutions}
    rows: list[tuple[str, ...]] = []
    for record in records:
        resolution = resolution_by_sha[record.sha]
        rows.append(
            (
                record.sha,
                record.repo,
                record.split,
                record.remote_url,
                str(record.reference_count),
                record.reason.value,
                resolution.status.value,
                ""
                if resolution.parent_count is None
                else str(resolution.parent_count),
                resolution.parent_sha or "",
                resolution.parent_policy.value,
                json.dumps(
                    [kind.value for kind in resolution.change_kinds],
                    separators=(",", ":"),
                ),
            )
        )
    return tuple(rows)


def _write_atomic(
    output: Path,
    rows: Sequence[tuple[str, ...]],
) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            _ = handle.write(_render_csv(rows))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _render_csv(rows: Sequence[tuple[str, ...]]) -> str:
    lines = [",".join(CSV_FIELDS)]
    for row in rows:
        lines.append(
            ",".join(
                f'"{value.replace(chr(34), chr(34) * 2)}"'
                if any(character in value for character in (",", '"', "\n", "\r"))
                else value
                for value in row
            )
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv, namespace=CliArguments())
    if arguments.output.exists() and not arguments.force:
        parser.error(f"output already exists: {arguments.output}; pass --force to replace it")
    if not arguments.output.parent.is_dir():
        parser.error(f"output parent does not exist: {arguments.output.parent}")
    paths = MetadataPaths(
        arguments.train_csv,
        arguments.test_csv,
        arguments.pool_csv,
        arguments.raw_csv,
        arguments.repo_split_json,
    )
    manifest = load_metadata_manifest(paths)
    resolutions = (
        _metadata_only_rows(manifest.records)
        if arguments.cache is None
        else resolve_cached_sources(manifest.records, arguments.cache)
    )
    _write_atomic(arguments.output, _csv_rows(manifest.records, resolutions))
    mode = "metadata-only" if arguments.cache is None else "cache-resolution"
    print(f"wrote {len(manifest.records)} rows to {arguments.output} (mode={mode})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
