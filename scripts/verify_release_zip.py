#!/usr/bin/env python3
"""Verify archive checksum, ZIP integrity/safety, and the extracted release."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", dest="zip_path", type=Path, required=True)
    parser.add_argument("--checksum", type=Path, required=True)
    args = parser.parse_args()
    zip_path = args.zip_path.resolve()
    checksum_path = args.checksum.resolve()
    expected, filename = checksum_path.read_text(encoding="utf-8-sig").strip().split("  ", 1)
    if filename != zip_path.name or sha256(zip_path) != expected:
        raise RuntimeError("ZIP sidecar checksum verification failed")

    with zipfile.ZipFile(zip_path, "r") as archive:
        corrupt = archive.testzip()
        if corrupt:
            raise RuntimeError(f"Corrupt ZIP member: {corrupt}")
        members = archive.namelist()
        for member in members:
            path = PurePosixPath(member)
            if path.is_absolute() or ".." in path.parts or len(path.parts) < 2:
                raise RuntimeError(f"Unsafe or unrooted ZIP member: {member}")
        roots = {PurePosixPath(member).parts[0] for member in members}
        if len(roots) != 1:
            raise RuntimeError(f"Expected one archive root, found: {sorted(roots)}")
        with tempfile.TemporaryDirectory(prefix="jgr_release_zip_verify_") as temporary:
            archive.extractall(temporary)
            extracted_root = Path(temporary) / next(iter(roots))
            environment = dict(os.environ)
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            commands = [
                (
                    "release validation",
                    [sys.executable, str(extracted_root / "scripts" / "validate_release.py"), "--release-root", str(extracted_root)],
                ),
                (
                    "formal-output verification",
                    [sys.executable, str(extracted_root / "scripts" / "verify_formal_outputs.py"), "--release-root", str(extracted_root)],
                ),
                (
                    "synthetic smoke test",
                    [sys.executable, str(extracted_root / "tests" / "smoke_test.py")],
                ),
            ]
            for label, command in commands:
                result = subprocess.run(command, check=False, capture_output=True, text=True, env=environment, cwd=extracted_root)
                if result.returncode:
                    raise RuntimeError(f"Extracted {label} failed:\n" + result.stdout + result.stderr)

            reproduced = Path(temporary) / "reproduced"
            result = subprocess.run(
                [sys.executable, str(extracted_root / "scripts" / "reproduce_tables_figures.py"), "--release-root", str(extracted_root), "--output-dir", str(reproduced)],
                check=False, capture_output=True, text=True, env=environment, cwd=extracted_root,
            )
            if result.returncode:
                raise RuntimeError("Extracted table/figure reproduction failed:\n" + result.stdout + result.stderr)
            if len(list((reproduced / "tables").glob("*.csv"))) != 17:
                raise RuntimeError("Extracted reproduction did not create 17 physical table CSVs")
            if len(list((reproduced / "figures").glob("*.png"))) != 8:
                raise RuntimeError("Extracted reproduction did not create eight aggregate figure PNGs")
    print("ZIP_VERIFICATION_STATUS=PASS")
    print(f"ZIP_SHA256={expected}")
    print(f"ZIP_MEMBERS={len(members)}")
    print("EXTRACTED_LIGHTWEIGHT_CHECKS=validation; formal-output verification; synthetic smoke; 17-table/8-figure reproduction")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
