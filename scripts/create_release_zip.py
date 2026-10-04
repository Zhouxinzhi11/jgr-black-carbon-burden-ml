#!/usr/bin/env python3
"""Create the deterministic local release ZIP and its SHA-256 sidecar."""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive-name", default="Zhou_et_al_JGR_Code_v1.0.0")
    args = parser.parse_args()
    root = args.release_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / f"{args.archive_name}.zip"
    checksum_path = output_dir / f"{args.archive_name}_SHA256.txt"
    if archive_path.exists() or checksum_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing archive or sidecar: {archive_path.name}"
        )

    files = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and not path.name.endswith((".pyc", ".pyo"))
        ),
        key=lambda path: path.relative_to(root).as_posix().lower(),
    )
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            relative = path.relative_to(root).as_posix()
            info = zipfile.ZipInfo(f"{args.archive_name}/{relative}")
            info.date_time = (2026, 10, 4, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)

    digest = sha256(archive_path)
    checksum_path.write_text(f"{digest}  {archive_path.name}\n", encoding="utf-8")
    print(f"ZIP={archive_path}")
    print(f"ZIP_FILES={len(files)}")
    print(f"ZIP_SHA256={digest}")
    print(f"ZIP_CHECKSUM_FILE={checksum_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
