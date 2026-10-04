#!/usr/bin/env python3
"""Validate completeness, portability, sensitivity, references, and hashes."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import re
import shlex
import sys
from pathlib import Path


REQUIRED = (
    "README.md",
    "CITATION.cff",
    "CHANGELOG.md",
    "LICENSE",
    "LICENSE_PUBLIC_RESULTS_CC_BY_4.0.txt",
    "LICENSES.md",
    ".zenodo.json",
    "requirements.txt",
    "environment.yml",
    ".gitignore",
    "configs/paths.example.yaml",
    "data/README.md",
    "docs/code_audit.csv",
    "docs/excluded_files.csv",
    "docs/review_required.csv",
    "docs/manuscript_code_map.csv",
    "docs/review_required_count_reconciliation.md",
    "docs/review_required_priority.csv",
    "docs/P0_P1_manual_review_summary.md",
    "docs/release_inclusion_changes.csv",
    "docs/legacy_models_resolution.md",
    "docs/final_figure_inventory.csv",
    "docs/table_figure_reproduction_map.csv",
    "docs/formal_output_coverage_summary.md",
    "docs/formal_output_numeric_verification.md",
    "docs/table_s7_split_verification.md",
    "docs/final_release_report.md",
    "scripts/reproduce_tables_figures.py",
    "scripts/verify_formal_outputs.py",
    "scripts/validate_release.py",
    "tests/smoke_test.py",
    "manifests/source_provenance.csv",
    "manifests/review_required_manifest.csv",
    "manifests/withheld_derived_data_manifest.csv",
    "manifests/release_manifest.csv",
    "src/checkpoint_compatibility/meanstate_ablation_reference.py",
    "MANIFEST_SHA256.txt",
)
TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".csv", ".json", ".yaml", ".yml", ".toml",
    ".cff", ".sh", ".bat", ".ini", ".cfg", ".svg", ".xml",
}
ABSOLUTE_PATH = re.compile(r"(?i)(?:\b[A-Z]:[\\/]|/(?:Users|home)/[^/\s]+/|/mnt/[a-z]/)")
EMAIL = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
SECRET_PATTERNS = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "aws_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    "secret_assignment": re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|password|passwd)\b\s*[:=]\s*['\"][^'\"]{8,}['\"]"
    ),
}
FORBIDDEN_PUBLIC_FILENAMES = {
    "posp_bccmass_daily_coverage.csv",
    "posp_bccmass_source_row_audit.csv",
    "posp_bccmass_sampling_maps.npz",
    "gate87_preplot_core_snapshot.json",
    "merra2_bccmass_fingerprint_samples.csv",
    "merra2_bccmass_fingerprint_report.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_files(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and (path.suffix.lower() in TEXT_SUFFIXES or path.name == ".gitignore"):
            if path.stat().st_size <= 15_000_000:
                yield path


def check_python(root: Path, errors: list[str]) -> None:
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            source = path.read_text(encoding="utf-8-sig")
            tree = ast.parse(source, filename=str(path))
            compile(tree, str(path), "exec")
        except Exception as exc:
            errors.append(f"Python syntax: {path.relative_to(root)}: {exc}")


def check_internal_imports(root: Path, errors: list[str]) -> None:
    """Resolve explicit release-local imports without executing modules."""
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        except Exception:
            continue
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if not name.startswith(("src.", "scripts.", "tests.")):
                    continue
                relative = Path(*name.split("."))
                if not ((root / relative).with_suffix(".py").is_file() or (root / relative / "__init__.py").is_file()):
                    errors.append(
                        f"Unresolved internal import in {path.relative_to(root)}: {name}"
                    )


def check_csvs(root: Path, errors: list[str]) -> None:
    for path in sorted(root.rglob("*.csv")):
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.reader(stream))
            if not rows or not rows[0]:
                errors.append(f"Empty CSV: {path.relative_to(root)}")
            elif any(len(row) != len(rows[0]) for row in rows[1:]):
                errors.append(f"Non-rectangular CSV: {path.relative_to(root)}")
        except Exception as exc:
            errors.append(f"CSV parse: {path.relative_to(root)}: {exc}")


def check_sensitive_text(root: Path, errors: list[str]) -> None:
    for path in text_files(root):
        relative = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        if ABSOLUTE_PATH.search(text):
            errors.append(f"Absolute personal/system path: {relative}")
        if EMAIL.search(text):
            errors.append(f"Email/account-like identifier: {relative}")
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                errors.append(f"Credential-like {label}: {relative}")


def check_references(root: Path, errors: list[str]) -> None:
    audit_path = root / "docs" / "code_audit.csv"
    if audit_path.is_file():
        with audit_path.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if row["classification"] in {"CORE", "SUPPORTING"} and row["proposed_release_path"]:
                    if row["file_type"] == ".py" and not (root / row["proposed_release_path"]).is_file():
                        errors.append(f"Missing allowlisted code: {row['proposed_release_path']}")
    map_path = root / "docs" / "manuscript_code_map.csv"
    if map_path.is_file():
        with map_path.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                for field in ("entry_script", "supporting_modules"):
                    for item in row[field].split(";"):
                        candidate = item.strip()
                        if candidate and "/" in candidate and Path(candidate).suffix:
                            if not (root / candidate).is_file():
                                errors.append(
                                    f"Code-map reference missing ({row['manuscript_element']}): {candidate}"
                                )
    readme = root / "README.md"
    if readme.is_file():
        for line in readme.read_text(encoding="utf-8-sig").splitlines():
            stripped = line.strip()
            if not (stripped.startswith("python ") or stripped.startswith("py ")):
                continue
            try:
                tokens = shlex.split(stripped, posix=False)
            except ValueError:
                continue
            script_token = next((token.strip("'\"") for token in tokens if token.endswith(".py") or token.strip("'\"").endswith(".py")), None)
            if not script_token:
                continue
            script = root / script_token.replace("\\", "/")
            if not script.is_file():
                errors.append(f"README command references missing script: {script_token}")
                continue
            script_text = script.read_text(encoding="utf-8-sig", errors="replace")
            for token in tokens:
                option = token.strip("'\"")
                if option.startswith("--") and option not in script_text:
                    errors.append(f"README option not found in {script_token}: {option}")


def check_rc2_scope_and_outputs(root: Path, errors: list[str]) -> None:
    if (root / "src" / "legacy_models").exists():
        errors.append("Obsolete src/legacy_models directory remains after rc2 rename")

    audit = root / "docs" / "code_audit.csv"
    if audit.is_file():
        rows = list(csv.DictReader(audit.open("r", encoding="utf-8-sig", newline="")))
        counts: dict[str, int] = {}
        for row in rows:
            counts[row["classification"]] = counts.get(row["classification"], 0) + 1
        expected = {"CORE": 33, "SUPPORTING": 61, "LEGACY_EXCLUDED": 723, "REVIEW_REQUIRED": 405}
        if len(rows) != 1222 or counts != expected:
            errors.append(f"Authoritative code-audit count mismatch: rows={len(rows)}, counts={counts}")

    priority = root / "docs" / "review_required_priority.csv"
    if priority.is_file():
        rows = list(csv.DictReader(priority.open("r", encoding="utf-8-sig", newline="")))
        if len(rows) != 405:
            errors.append(f"Priority review count mismatch: {len(rows)}")
        unresolved = [row for row in rows if row["priority"] == "P0" and not row["final_release_decision"].startswith("RESOLVED_")]
        if unresolved:
            errors.append(f"Unresolved P0 review entries: {len(unresolved)}")

    for relative, expected_count in (
        ("manifests/review_required_manifest.csv", 405),
        ("manifests/withheld_derived_data_manifest.csv", 10),
    ):
        path = root / relative
        if path.is_file():
            count = len(list(csv.DictReader(path.open("r", encoding="utf-8-sig", newline=""))))
            if count != expected_count:
                errors.append(f"Manifest count mismatch for {relative}: {count}")

    map_path = root / "docs" / "table_figure_reproduction_map.csv"
    if map_path.is_file():
        rows = list(csv.DictReader(map_path.open("r", encoding="utf-8-sig", newline="")))
        elements = [row["formal_element"] for row in rows]
        expected_elements = {
            *(f"Table {number}" for number in range(1, 5)),
            *(f"Figure {number}" for number in range(1, 6)),
            *(f"Table S{number}" for number in range(1, 13)),
            *(f"Figure S{number}" for number in range(1, 4)),
        }
        if len(elements) != 24 or len(set(elements)) != 24 or set(elements) != expected_elements:
            errors.append("Formal output map does not contain 24 unique expected elements")
        for row in rows:
            for item in row["physical_source_files"].split(";"):
                candidate = item.strip()
                if candidate and not (root / candidate).is_file():
                    errors.append(f"Formal output source missing ({row['formal_element']}): {candidate}")


def check_manifest(root: Path, errors: list[str], check_sha: bool) -> None:
    manifest_path = root / "manifests" / "release_manifest.csv"
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        expected = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.relative_to(root).as_posix() not in {"manifests/release_manifest.csv", "MANIFEST_SHA256.txt"}
            and not path.name.endswith((".pyc", ".pyo"))
        }
        listed = {row["release_path"] for row in rows}
        if listed != expected:
            missing = sorted(expected - listed)
            stale = sorted(listed - expected)
            errors.append(f"Release manifest set mismatch: missing={missing[:5]}, stale={stale[:5]}")
        for row in rows:
            path = root / row["release_path"]
            if path.is_file() and (str(path.stat().st_size) != row["size_bytes"] or sha256(path) != row["sha256"]):
                errors.append(f"Release manifest checksum/size mismatch: {row['release_path']}")
    if check_sha:
        checksum_path = root / "MANIFEST_SHA256.txt"
        if checksum_path.is_file():
            listed = {}
            for line in checksum_path.read_text(encoding="utf-8").splitlines():
                digest, relative = line.split("  ", 1)
                listed[relative] = digest
            expected = {
                path.relative_to(root).as_posix()
                for path in root.rglob("*")
                if path.is_file() and path != checksum_path and "__pycache__" not in path.parts
                and not path.name.endswith((".pyc", ".pyo"))
            }
            if set(listed) != expected:
                errors.append("MANIFEST_SHA256.txt file set mismatch")
            for relative, digest in listed.items():
                path = root / relative
                if not path.is_file() or sha256(path) != digest:
                    errors.append(f"SHA-256 mismatch: {relative}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--skip-sha", action="store_true", help="Use only before the final checksum manifest is built.")
    args = parser.parse_args()
    root = args.release_root.resolve()
    errors: list[str] = []
    for relative in REQUIRED:
        if not (root / relative).is_file():
            errors.append(f"Missing required file: {relative}")
    for path in root.rglob("*"):
        if path.is_file() and path.name.lower() in FORBIDDEN_PUBLIC_FILENAMES:
            errors.append(f"Restricted/review-only file present: {path.relative_to(root)}")
    check_python(root, errors)
    check_internal_imports(root, errors)
    check_csvs(root, errors)
    check_sensitive_text(root, errors)
    check_references(root, errors)
    check_rc2_scope_and_outputs(root, errors)
    check_manifest(root, errors, not args.skip_sha)
    if errors:
        print(f"VALIDATION_STATUS=FAIL ({len(errors)} issue(s))")
        for issue in errors:
            print(f"- {issue}")
        return 1
    print("VALIDATION_STATUS=PASS")
    print("CHECKS=required files; Python syntax; internal imports; CSV structure; absolute paths; credentials/privacy; restricted filenames; code-map references; README command options; authoritative audit-count reconciliation; P0 resolution; formal-output coverage; manifest; SHA-256")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
