#!/usr/bin/env python3
"""Build the release inventory and SHA-256 checksum list deterministically."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


MANIFEST_PATH = "manifests/release_manifest.csv"
SHA_PATH = "MANIFEST_SHA256.txt"
TREE_PATH = "docs/directory_tree.txt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def category(path: str) -> str:
    first = path.split("/", 1)[0]
    return {
        "src": "scientific_code",
        "scripts": "entry_or_release_utility",
        "configs": "configuration_template",
        "docs": "documentation_and_audit",
        "results": "public_aggregate_result",
        "figure_source_data": "public_figure_source_data",
        "data": "data_documentation_only",
        "tests": "synthetic_test",
        "metadata": "publication_metadata",
        "manifests": "provenance_manifest",
    }.get(first, "release_documentation")


def included_files(root: Path, excluded: set[str]) -> list[Path]:
    return sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.relative_to(root).as_posix() not in excluded
            and "__pycache__" not in path.parts
            and not path.name.endswith((".pyc", ".pyo"))
        ),
        key=lambda path: path.relative_to(root).as_posix().lower(),
    )


def write_directory_tree(root: Path) -> None:
    lines = [root.name + "/"]

    def visit(directory: Path, prefix: str) -> None:
        entries = sorted(
            (
                entry
                for entry in directory.iterdir()
                if entry.name != "__pycache__" and not entry.name.endswith((".pyc", ".pyo"))
            ),
            key=lambda entry: (not entry.is_dir(), entry.name.lower()),
        )
        for index, entry in enumerate(entries):
            last = index == len(entries) - 1
            lines.append(prefix + ("`-- " if last else "|-- ") + entry.name + ("/" if entry.is_dir() else ""))
            if entry.is_dir():
                visit(entry, prefix + ("    " if last else "|   "))

    visit(root, "")
    path = root / TREE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.release_root.resolve()
    manifest_path = root / MANIFEST_PATH
    sha_path = root / SHA_PATH
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_directory_tree(root)

    files = included_files(root, {MANIFEST_PATH, SHA_PATH})
    rows = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        rows.append(
            {
                "release_path": relative,
                "size_bytes": str(path.stat().st_size),
                "sha256": sha256(path),
                "category": category(relative),
                "public_release_status": "INCLUDED",
            }
        )
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("release_path", "size_bytes", "sha256", "category", "public_release_status"),
        )
        writer.writeheader()
        writer.writerows(rows)

    checksum_files = included_files(root, {SHA_PATH})
    lines = [f"{sha256(path)}  {path.relative_to(root).as_posix()}" for path in checksum_files]
    sha_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"RELEASE_MANIFEST_ROWS={len(rows)}")
    print(f"SHA256_ENTRIES={len(lines)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
