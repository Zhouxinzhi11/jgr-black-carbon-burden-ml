#!/usr/bin/env python3
"""Copy allowlisted Python sources into staging and sanitize local paths.

This script never edits the source project. Scientific code is copied byte-for-
byte except for documented absolute-path substitutions and a release note.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path


PATH_REPLACEMENTS = (
    (re.compile(r"(?i)F:[\\/]+EDGAR_Earth-2[\\/]+data"), "data/RESTRICTED_INPUT_NOT_INCLUDED"),
    (re.compile(r"(?i)F:[\\/]+EDGAR_Earth-2[\\/]+code[\\/]+2021-2022"), "."),
    (re.compile(r"(?i)F:[\\/]+EDGAR_Earth-2[\\/]+code"), "."),
    (re.compile(r"(?i)F:[\\/]+EDGAR_Earth-2"), "."),
)
WINDOWS_ABSOLUTE_RE = re.compile(r"(?i)\b[A-Z]:[\\/]")
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
)
RELEASE_NOTE = (
    "# RELEASE COPY: local absolute path defaults were replaced with relative\n"
    "# placeholders. Scientific logic, seeds, splits, controls, and metrics were\n"
    "# not intentionally changed. Configure paths before restricted-data runs.\n"
)


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sanitize(text: str) -> tuple[str, list[str]]:
    changes: list[str] = []
    for pattern, replacement in PATH_REPLACEMENTS:
        text, count = pattern.subn(lambda _: replacement, text)
        if count:
            changes.append(f"{count} path substitution(s): {pattern.pattern} -> {replacement}")
    if text.startswith("#!"):
        first, rest = text.split("\n", 1)
        text = first + "\n" + RELEASE_NOTE + rest
    else:
        text = RELEASE_NOTE + text
    return text, changes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-csv", type=Path, required=True)
    parser.add_argument("--source-code-root", type=Path, required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    args = parser.parse_args()
    staging = args.staging_root.resolve()
    source_code_root = args.source_code_root.resolve()

    with args.audit_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))

    provenance: list[dict[str, str]] = []
    for row in rows:
        if row["classification"] not in {"CORE", "SUPPORTING"}:
            continue
        if row["file_type"] != ".py" or not row["proposed_release_path"]:
            continue
        source_identifier = row["original_path"].replace("\\", "/")
        if not source_identifier.startswith("code/"):
            raise RuntimeError(f"Unexpected source identifier: {source_identifier}")
        source = source_code_root / source_identifier.removeprefix("code/")
        destination = staging / row["proposed_release_path"]
        original = source.read_bytes()
        text = original.decode("utf-8-sig")
        sanitized, changes = sanitize(text)
        remaining_paths = WINDOWS_ABSOLUTE_RE.findall(sanitized)
        if remaining_paths:
            raise RuntimeError(f"Absolute Windows path remains in {destination}")
        if any(pattern.search(sanitized) for pattern in SECRET_PATTERNS):
            raise RuntimeError(f"Credential-like content remains in {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        encoded = sanitized.encode("utf-8")
        destination.write_bytes(encoded)
        provenance.append(
            {
                "original_path": source_identifier,
                "original_sha256": digest_bytes(original),
                "release_path": destination.relative_to(staging).as_posix(),
                "release_sha256": digest_bytes(encoded),
                "classification": row["classification"],
                "release_transformations": "; ".join(changes) if changes else "Release-note header only.",
            }
        )

    out = staging / "manifests" / "source_provenance.csv"
    with out.open("w", encoding="utf-8-sig", newline="") as stream:
        fields = [
            "original_path", "original_sha256", "release_path", "release_sha256",
            "classification", "release_transformations",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(provenance)
    print(f"COPIED_SOURCES={len(provenance)}")
    print(f"PROVENANCE={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
