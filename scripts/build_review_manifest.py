#!/usr/bin/env python3
"""Create a manifest of files withheld from the public staging archive."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-review-csv", type=Path, required=True)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--additional", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    project_root = args.project_root.resolve()

    rows: list[dict[str, str]] = []
    with args.code_review_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            rows.append(
                {
                    "scope": "code-tree-review-required",
                    "classification": "REVIEW_REQUIRED",
                    "original_path": row["original_path"],
                    "filename": row["filename"],
                    "size_bytes": row["size_bytes"],
                    "sha256": row["sha256"],
                    "withheld_reason": row["evidence"],
                    "required_manual_action": row["required_manual_action"],
                    "copied_to_review_directory": "No - manifest entry only",
                }
            )

    reasons = {
        "posp_bccmass_daily_coverage.csv": "Daily timestamps and daily support summaries require data-custodian review.",
        "posp_bccmass_source_row_audit.csv": "Contains restricted source-file lineage and row-level inventory counts.",
        "posp_bccmass_sampling_maps.npz": "Contains cellwise monthly support arrays on the analysis grid.",
        "gate87_preplot_core_snapshot.json": "Contains daily rows, source paths, and rich restricted-data lineage.",
        "evaluation_mask_difference_2022.png": "pointwise/cellwise source arrays are available only through the restricted-access record, subject to the approved access conditions.",
        "evaluation_mask_difference_2023.png": "pointwise/cellwise source arrays are available only through the restricted-access record, subject to the approved access conditions.",
        "valid_day_count_difference_2022.png": "pointwise/cellwise source arrays are available only through the restricted-access record, subject to the approved access conditions.",
        "valid_day_count_difference_2023.png": "pointwise/cellwise source arrays are available only through the restricted-access record, subject to the approved access conditions.",
        "merra2_bccmass_fingerprint_samples.csv": "Contains sampled coordinates, times, and paired raw/fused BCCMASS values.",
        "merra2_bccmass_fingerprint_report.json": "Rich provenance report may embed sample-level values and local paths.",
    }
    derived_rows: list[dict[str, str]] = []
    for path in args.additional:
        if not path.exists() or not path.is_file():
            continue
        try:
            source_identifier = path.resolve().relative_to(project_root).as_posix()
        except ValueError:
            source_identifier = f"EXTERNAL_SOURCE/{path.name}"
        derived_rows.append(
            {
                "scope": "withheld-derived-data",
                "classification": "NOT_A_CODE_CLASSIFICATION",
                "original_path": source_identifier,
                "filename": path.name,
                "size_bytes": str(path.stat().st_size),
                "sha256": digest(path),
                "withheld_reason": reasons.get(path.name, "Derived result requires author/data-custodian disclosure review."),
                "required_manual_action": "Confirm that the file cannot reveal restricted records before any public release.",
                "copied_to_review_directory": "No - manifest entry only",
            }
        )

    unique: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        unique[(row["original_path"].lower(), row["sha256"])] = row
    rows = sorted(unique.values(), key=lambda item: item["original_path"].lower())
    derived_unique: dict[tuple[str, str], dict[str, str]] = {}
    for row in derived_rows:
        derived_unique[(row["original_path"].lower(), row["sha256"])] = row
    derived_rows = sorted(derived_unique.values(), key=lambda item: item["original_path"].lower())

    out = args.review_root / "manifests" / "review_required_manifest.csv"
    derived_out = args.review_root / "manifests" / "withheld_derived_data_manifest.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "scope", "classification", "original_path", "filename", "size_bytes", "sha256", "withheld_reason",
        "required_manual_action", "copied_to_review_directory",
    ]
    with out.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with derived_out.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(derived_rows)
    print(f"REVIEW_MANIFEST_ROWS={len(rows)}")
    print(f"WITHHELD_DERIVED_DATA_ROWS={len(derived_rows)}")
    print(f"REVIEW_MANIFEST={out}")
    print(f"WITHHELD_DERIVED_DATA_MANIFEST={derived_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
