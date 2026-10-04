#!/usr/bin/env python3
"""Verify formal table/figure coverage and packaged aggregate numeric closure."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


EXPECTED_ELEMENTS = {
    *(f"Table {number}" for number in range(1, 5)),
    *(f"Figure {number}" for number in range(1, 6)),
    *(f"Table S{number}" for number in range(1, 13)),
    *(f"Figure S{number}" for number in range(1, 4)),
}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.release_root.resolve()
    errors: list[str] = []

    result_files = sorted((root / "results").glob("*.csv"))
    figure_files = sorted((root / "figure_source_data").glob("*.csv"))
    if len(result_files) != 17:
        errors.append(f"Expected 17 physical table CSVs, found {len(result_files)}")
    if len(figure_files) != 8:
        errors.append(f"Expected 8 figure-source CSVs, found {len(figure_files)}")

    mapping = rows(root / "docs" / "table_figure_reproduction_map.csv")
    elements = [row["formal_element"] for row in mapping]
    if len(elements) != 24 or len(set(elements)) != 24 or set(elements) != EXPECTED_ELEMENTS:
        errors.append("Formal output map is not a one-to-one map of the expected 24 logical outputs")
    if any(row["verification_result"] != "PASS" for row in mapping):
        errors.append("Formal output map contains a non-PASS verification result")

    exact_pairs = (
        ("results/main_table_2_meanstate_performance.csv", "figure_source_data/figure_2_meanstate_performance.csv"),
        ("results/main_table_3_daily_anomaly_performance.csv", "figure_source_data/figure_4_daily_anomaly_performance.csv"),
        ("results/main_table_4_bounded_adjustment.csv", "figure_source_data/figure_5_bounded_adjustment.csv"),
        ("results/si_table_s9_monthly_sampling_support.csv", "figure_source_data/figure_s1_monthly_sampling_support.csv"),
    )
    for left, right in exact_pairs:
        if (root / left).read_bytes() != (root / right).read_bytes():
            errors.append(f"Numeric source mismatch: {left} versus {right}")

    s7a = rows(root / "results" / "si_table_s7a_area_weighted_performance.csv")
    s7b = rows(root / "results" / "si_table_s7b_area_weighted_contrasts.csv")
    expected_a = [
        ["2022 held-out model-selection validation", "0.5767 ± 0.0384", "0.5034 ± 0.0409", "0.5403 ± 0.0186", "0.4943 ± 0.0011", "0.5407 ± 0.0253"],
        ["2023 external evaluation", "0.3437 ± 0.0328", "0.4327 ± 0.0908", "0.4420 ± 0.0467", "0.3673 ± 0.0780", "0.4208 ± 0.0769"],
    ]
    expected_b = [
        ["2022 held-out model-selection validation", "+0.0460 ± 0.0188", "−0.0004 ± 0.0124", "+0.0369 ± 0.0538"],
        ["2023 external/proxy evaluation", "+0.0747 ± 0.0492", "+0.0212 ± 0.0901", "+0.0093 ± 0.0913"],
    ]
    if [list(row.values()) for row in s7a] != expected_a or [list(row.values()) for row in s7b] != expected_b:
        errors.append("Table S7 Part A/Part B values do not match the verified final SI values")

    priority = rows(root / "docs" / "review_required_priority.csv")
    if len(priority) != 405:
        errors.append(f"Expected 405 prioritized review rows, found {len(priority)}")
    unresolved_p0 = [row for row in priority if row["priority"] == "P0" and not row["final_release_decision"].startswith("RESOLVED_")]
    if unresolved_p0:
        errors.append(f"Unresolved P0 rows: {len(unresolved_p0)}")
    if len(rows(root / "manifests" / "review_required_manifest.csv")) != 405:
        errors.append("Code-tree review manifest does not contain 405 rows")
    if len(rows(root / "manifests" / "withheld_derived_data_manifest.csv")) != 10:
        errors.append("Withheld-derived-data manifest does not contain 10 rows")

    if errors:
        print(f"FORMAL_OUTPUT_VERIFICATION=FAIL ({len(errors)} issue(s))")
        for error in errors:
            print(f"- {error}")
        return 1
    print("FORMAL_OUTPUT_VERIFICATION=PASS")
    print("LOGICAL_TABLES=16")
    print("PHYSICAL_TABLE_CSVS=17")
    print("LOGICAL_FIGURES=8")
    print("TABLE_S7=ONE_LOGICAL_TABLE_TWO_VERIFIED_PART_FILES")
    print("REVIEW_REQUIRED=405")
    print("WITHHELD_DERIVED_DATA=10")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
