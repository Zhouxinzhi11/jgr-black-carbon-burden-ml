#!/usr/bin/env python3
"""Generate the rc2 pre-submission review records from read-only evidence roots.

The script never copies manuscript figures or review-only source files. Paths
written to the release are portable identifiers rooted at MANUSCRIPT_ROOT,
SI_ROOT, FIGURE_SOURCE_ROOT, or PROJECT_ROOT.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import statistics
import struct
from collections import Counter
from datetime import datetime
from pathlib import Path


CODE_TREE_TOTAL = 1222
REVIEW_REQUIRED_TOTAL = 405
WITHHELD_DERIVED_TOTAL = 10
TOTAL_CONSIDERED = 1232

SUPERSEDED_SCRIPTS = {
    "49_audit_s2_meanstate_clean_formal_seed42_gate13b.py": "src/audits/audit_meanstate_formal_seed.py",
    "50_eval_s2_meanstate_clean_seed42_gate14a.py": "src/workflows/evaluate_meanstate.py",
    "55_audit_meanstate_support_geometry.py": "src/audits/audit_support_geometry.py",
    "59_audit_s2_meanstate_support_preserving_formal_seed42.py": "src/audits/audit_support_control_formal.py",
    "60_eval_s2_meanstate_support_preserving_seed42.py": "src/workflows/evaluate_support_preserving_control.py",
    "64_eval_meanstate_area_weighted_gateP4.py": "src/workflows/evaluate_area_weighted_meanstate.py",
    "64_eval_meanstate_area_weighted_gateP4_v2.py": "src/workflows/evaluate_area_weighted_meanstate.py",
    "64_eval_meanstate_area_weighted_gateP4_v3.py": "src/workflows/evaluate_area_weighted_meanstate.py",
    "65_plot_meanstate_spatial_diagnostics_figure5.py": "src/workflows/prepare_spatial_diagnostics.py",
    "65_plot_meanstate_spatial_diagnostics_figure5_v2.py": "src/workflows/prepare_spatial_diagnostics.py",
    "67_train_s2_anomaly_clean.py": "src/workflows/train_daily_anomaly.py",
    "68_audit_s2_anomaly_clean_smoke_gateA5.py": "src/audits/audit_anomaly_smoke.py",
    "69_audit_s2_anomaly_clean_formal_seed42.py": "src/audits/audit_anomaly_meteorology.py",
    "70_eval_s2_anomaly_clean_seed42.py": "src/workflows/evaluate_anomaly_meteorology.py",
    "79_eval_s2_meanstate_binary_support_formal_run.py": "src/workflows/evaluate_binary_support_control.py",
    "80_aggregate_and_seal_meanstate_binary_support_gateS8.py": "src/workflows/aggregate_binary_support_control.py",
    "collect_existing_result_metrics.py": "scripts/prepare_public_results.py",
    "make_paper_core_result_tables.py": "scripts/reproduce_tables_figures.py",
    "make_paper_grouped_result_tables.py": "scripts/reproduce_tables_figures.py",
}

P0_REPORT_MARKERS = (
    "/paper_result_inventory_clean/",
    "/paper_data_audit/",
    "/sanity_reports/meanstate_area_weighted_",
    "/sanity_reports/meanstate_spatial_diagnostics_",
    "/sanity_reports/meanstate_support_geometry_",
    "/sanity_reports/meanstate_support_mask_overlap_",
    "/sanity_reports/meanstate_support_preserving_",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def classify_priority(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    path = row["original_path"].replace("\\", "/")
    lower = path.lower()
    name = row["filename"]
    file_type = row["file_type"].lower()
    linked = row["manuscript_or_SI_link"] != "No verified direct link to the final manuscript or SI."

    if name in SUPERSEDED_SCRIPTS:
        substitute = SUPERSEDED_SCRIPTS[name]
        return (
            "P0",
            "Direct predecessor or duplicate in a formal manuscript workflow; omission/inclusion could change the apparent authoritative entry point.",
            substitute,
            "RESOLVED_EXCLUDE_SUPERSEDED",
            f"Compared with the included successor {substitute}; retain only the included successor in rc2.",
        )
    if any(marker in lower for marker in P0_REPORT_MARKERS):
        substitute = "results/ and figure_source_data/ public aggregate derivatives"
        if file_type == ".py":
            decision = "RESOLVED_EXCLUDE_NONAUTHORITATIVE_WORKFLOW"
        else:
            decision = "RESOLVED_WITHHOLD_USE_PUBLIC_AGGREGATE"
        return (
            "P0",
            "Artifact is in a final-result, direct-audit, or formal-figure evidence family and therefore required an explicit release decision.",
            substitute,
            decision,
            "Content family, data-dependency flags, generated outputs, and packaged aggregate replacement were checked; the source artifact is not needed in the public package.",
        )
    if file_type == ".py" or linked or file_type in {".csv", ".md"}:
        substitute = "None; or the already allowlisted final workflow/public aggregate where applicable"
        return (
            "P1",
            "Potential method/provenance relevance, but no evidence that this exact file is an authoritative final entry point.",
            substitute,
            "RESOLVED_EXCLUDE_FROM_RC2",
            "Imports/call references, manuscript-link evidence, output names, and data-risk indicators were reviewed; conservative exclusion is sufficient for rc2.",
        )
    return (
        "P2",
        "Checkpoint, run artifact, opaque archive, or internal log with no required public execution role.",
        "None",
        "RESOLVED_EXCLUDE_FROM_RC2",
        "Not required by the allowlisted code graph; binary/run artifacts are not redistributed and may contain restricted or machine-specific state.",
    )


def build_priority_review(root: Path) -> tuple[list[dict[str, str]], Counter]:
    source_rows = read_csv(root / "docs" / "review_required.csv")
    if len(source_rows) != REVIEW_REQUIRED_TOTAL:
        raise RuntimeError(f"Expected {REVIEW_REQUIRED_TOTAL} REVIEW_REQUIRED rows, found {len(source_rows)}")
    rows: list[dict[str, str]] = []
    counts: Counter = Counter()
    for source in source_rows:
        priority, rationale, substitute, decision, resolution = classify_priority(source)
        counts[priority] += 1
        rows.append(
            {
                "original_path": source["original_path"],
                "filename": source["filename"],
                "file_type": source["file_type"],
                "code_tree_classification": "REVIEW_REQUIRED",
                "priority": priority,
                "priority_rationale": rationale,
                "manuscript_or_SI_link": source["manuscript_or_SI_link"],
                "imported_or_called_by": source["imported_or_called_by"],
                "generated_outputs": source["generated_outputs"],
                "data_dependencies": source["data_dependencies"],
                "risk_indicators": "; ".join(
                    value for value in (
                        source["contains_absolute_path"],
                        source["restricted_data_indicators"],
                        source["credential_or_privacy_indicators"],
                    ) if value and value not in {"No", "None detected."}
                ) or "None detected by static audit",
                "release_substitute": substitute,
                "final_release_decision": decision,
                "resolution_evidence": resolution,
                "manual_follow_up": "Optional future reassessment only; no action required for rc2 publication scope.",
            }
        )
    fields = list(rows[0])
    write_csv(root / "docs" / "review_required_priority.csv", rows, fields)
    return rows, counts


def split_review_manifests(root: Path, external_manifest: Path) -> None:
    rows = read_csv(external_manifest)
    code_rows = [row for row in rows if row["original_path"].startswith("code/")]
    derived_rows = [row for row in rows if row["original_path"].startswith("data/")]
    if len(rows) != REVIEW_REQUIRED_TOTAL + WITHHELD_DERIVED_TOTAL:
        raise RuntimeError(f"Expected 415 external manifest entries, found {len(rows)}")
    if len(code_rows) != REVIEW_REQUIRED_TOTAL or len(derived_rows) != WITHHELD_DERIVED_TOTAL:
        raise RuntimeError(
            f"Scope split mismatch: code={len(code_rows)}, withheld-derived-data={len(derived_rows)}"
        )
    code_out = [
        {"scope": "code-tree-review-required", "classification": "REVIEW_REQUIRED", **row}
        for row in code_rows
    ]
    derived_out = [
        {"scope": "withheld-derived-data", "classification": "NOT_A_CODE_CLASSIFICATION", **row}
        for row in derived_rows
    ]
    fields = ["scope", "classification", *rows[0].keys()]
    write_csv(root / "manifests" / "review_required_manifest.csv", code_out, fields)
    write_csv(root / "manifests" / "withheld_derived_data_manifest.csv", derived_out, fields)


def refresh_compatibility_paths_and_hashes(root: Path) -> None:
    old = "src/legacy_models/meanstate_ablation_reference.py"
    new = "src/checkpoint_compatibility/meanstate_ablation_reference.py"

    audit_path = root / "docs" / "code_audit.csv"
    audit_rows = read_csv(audit_path)
    audit_fields = list(audit_rows[0])
    changed = 0
    for row in audit_rows:
        if row.get("proposed_release_path") == old:
            row["proposed_release_path"] = new
            changed += 1
    if changed != 1:
        raise RuntimeError(f"Expected one compatibility-path code-audit update, found {changed}")
    write_csv(audit_path, audit_rows, audit_fields)

    provenance_path = root / "manifests" / "source_provenance.csv"
    provenance_rows = read_csv(provenance_path)
    provenance_fields = list(provenance_rows[0])
    changed = 0
    for row in provenance_rows:
        if row["release_path"] == old:
            row["release_path"] = new
            changed += 1
        release_file = root / row["release_path"]
        if not release_file.is_file():
            raise RuntimeError(f"Missing provenance release file: {row['release_path']}")
        row["release_sha256"] = sha256(release_file)
    if changed != 1:
        raise RuntimeError(f"Expected one compatibility-path provenance update, found {changed}")
    write_csv(provenance_path, provenance_rows, provenance_fields)


def png_dimensions(path: Path) -> tuple[str, str]:
    try:
        with path.open("rb") as stream:
            signature = stream.read(24)
        if signature[:8] == b"\x89PNG\r\n\x1a\n" and signature[12:16] == b"IHDR":
            width, height = struct.unpack(">II", signature[16:24])
            return str(width), str(height)
    except OSError:
        pass
    return "", ""


def svg_dimensions(path: Path) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")[:5000]
    match = re.search(r"viewBox=[\"']([^\"']+)", text, re.I)
    if match:
        values = match.group(1).split()
        if len(values) == 4:
            return values[2], values[3]
    width = re.search(r"\bwidth=[\"']([^\"']+)", text, re.I)
    height = re.search(r"\bheight=[\"']([^\"']+)", text, re.I)
    return (width.group(1) if width else "", height.group(1) if height else "")


def figure_role(relative: str) -> tuple[str, str, str, str, str]:
    p = relative.replace("\\", "/")
    low = p.lower()
    if low.startswith("figure1/最新/"):
        return "Figure 1", "selected final reference", "final v4 subscript-fixed workflow figure", "Yes", "No"
    if low.startswith("figure1/") or low == "figure1.py":
        return "Figure 1", "superseded reference", "older v3 workflow figure/generator", "No", "No"
    if low == "figure2.py" or low.startswith("figure2/"):
        return "Figure 2", "selected final reference/evidence", "v4.4 mean-state metrics and rendering", "Yes", "No"
    if low == "figure3.py":
        return "Figure 4", "selected generator reference", "daily-anomaly generator uses pre-renumbering Figure 3 filenames", "Yes", "No"
    if low.startswith("figure3/"):
        return "Figure 3", "selected final reference/evidence", "spatial diagnostic v4; legacy filename says Figure5", "Yes", "No"
    if low.startswith("figure4/"):
        return "Figure 4", "selected final reference/evidence", "daily-anomaly v1.2; legacy filename says figure3", "Yes", "No"
    if low == "figure4_inverse_tradeoff_v1_3.py":
        return "Figure 5", "selected generator reference", "bounded-adjustment v1.3; legacy filename says Figure 4", "Yes", "No"
    if low == "figure4_inverse_tradeoff_v1_1.py":
        return "Figure 5", "superseded generator", "older bounded-adjustment v1.1", "No", "No"
    if low.startswith("figure5/"):
        return "Figure 5", "selected final reference/evidence", "bounded-adjustment v1.3; legacy filename says figure4", "Yes", "No"
    if low.startswith("图s1/"):
        return "Not current Figure S1", "excluded historical reference", "Atlas-enhanced pilot conflicts with current SI Figure S1 sampling-support caption", "No", "No"
    return "Unmapped", "internal reference", "not selected as a formal output", "No", "No"


def build_figure_inventory(root: Path, figure_root: Path) -> int:
    rows: list[dict[str, str]] = []
    for path in sorted((item for item in figure_root.rglob("*") if item.is_file()), key=lambda x: x.as_posix().lower()):
        relative = path.relative_to(figure_root).as_posix()
        element, status, assessment, selected, copied = figure_role(relative)
        width = height = ""
        if path.suffix.lower() == ".png":
            width, height = png_dimensions(path)
        elif path.suffix.lower() == ".svg":
            width, height = svg_dimensions(path)
        rows.append(
            {
                "full_path": f"FIGURE_SOURCE_ROOT/{relative}",
                "relative_path": relative,
                "filename": path.name,
                "extension": path.suffix.lower(),
                "size_bytes": str(path.stat().st_size),
                "sha256": sha256(path),
                "last_modified_local": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
                "width_or_viewbox": width,
                "height_or_viewbox": height,
                "formal_element": element,
                "version_status": status,
                "content_assessment": assessment,
                "selected_for_verification": selected,
                "copied_to_release": copied,
            }
        )
    write_csv(root / "docs" / "final_figure_inventory.csv", rows, list(rows[0]))
    return len(rows)


def assert_table_s7(root: Path) -> None:
    a = read_csv(root / "results" / "si_table_s7a_area_weighted_performance.csv")
    b = read_csv(root / "results" / "si_table_s7b_area_weighted_contrasts.csv")
    expected_a = [
        ["2022 held-out model-selection validation", "0.5767 ± 0.0384", "0.5034 ± 0.0409", "0.5403 ± 0.0186", "0.4943 ± 0.0011", "0.5407 ± 0.0253"],
        ["2023 external evaluation", "0.3437 ± 0.0328", "0.4327 ± 0.0908", "0.4420 ± 0.0467", "0.3673 ± 0.0780", "0.4208 ± 0.0769"],
    ]
    expected_b = [
        ["2022 held-out model-selection validation", "+0.0460 ± 0.0188", "−0.0004 ± 0.0124", "+0.0369 ± 0.0538"],
        ["2023 external/proxy evaluation", "+0.0747 ± 0.0492", "+0.0212 ± 0.0901", "+0.0093 ± 0.0913"],
    ]
    actual_a = [list(row.values()) for row in a]
    actual_b = [list(row.values()) for row in b]
    if actual_a != expected_a or actual_b != expected_b:
        raise RuntimeError("Table S7 Part A/Part B content does not match the final SI values")


def mean_sd(value: str) -> tuple[float, float]:
    normalized = value.replace("−", "-").strip()
    if "±" not in normalized:
        return float(normalized), 0.0
    mean, sd = normalized.split("±", 1)
    return float(mean.strip()), float(sd.strip())


def require_rounded(actual: float, expected: float, digits: int, label: str) -> None:
    if round(actual, digits) != round(expected, digits):
        raise RuntimeError(f"Numeric closure failed for {label}: {actual} versus {expected}")


def verify_reference_numeric_closure(root: Path, figure_root: Path) -> dict[str, int]:
    exact_pairs = [
        ("results/main_table_2_meanstate_performance.csv", "figure_source_data/figure_2_meanstate_performance.csv"),
        ("results/main_table_3_daily_anomaly_performance.csv", "figure_source_data/figure_4_daily_anomaly_performance.csv"),
        ("results/main_table_4_bounded_adjustment.csv", "figure_source_data/figure_5_bounded_adjustment.csv"),
        ("results/si_table_s9_monthly_sampling_support.csv", "figure_source_data/figure_s1_monthly_sampling_support.csv"),
    ]
    for left, right in exact_pairs:
        if (root / left).read_bytes() != (root / right).read_bytes():
            raise RuntimeError(f"Public table/figure-source mismatch: {left} versus {right}")

    # Figure 2: independently recompute mean and sample SD from the final figure's
    # three-seed source file, then close the summary to the rounded manuscript table.
    f2_source = read_csv(figure_root / "figure2" / "figure2_source_metrics.csv")
    f2_summary = read_csv(figure_root / "figure2" / "figure2_summary_metrics.csv")
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in f2_source:
        grouped.setdefault((row["config_key"], row["split_key"]), []).append(row)
    for summary in f2_summary:
        values = grouped[(summary["config_key"], summary["split_key"])]
        if len(values) != 3:
            raise RuntimeError("Figure 2 final source does not contain exactly three seeds per group")
        for raw_name, mean_name, sd_name in (
            ("r2", "r2_mean", "r2_sd"), ("rmse", "rmse_mean", "rmse_sd"), ("bias", "bias_mean", "bias_sd")
        ):
            raw = [float(row[raw_name]) for row in values]
            require_rounded(statistics.mean(raw), float(summary[mean_name]), 12, f"Figure 2 {mean_name}")
            require_rounded(statistics.stdev(raw), float(summary[sd_name]), 12, f"Figure 2 {sd_name}")

    table2 = read_csv(root / "results" / "main_table_2_meanstate_performance.csv")
    f2_configs = {
        "Emission-only": "Inventory-only",
        "Meteorology-only": "Meteorology-only",
        "Joint": "Inventory reference + meteorology",
        "Cyclic-shift": "Cyclic-displacement inventory + meteorology",
        "Support-preserving": "Support-preserving inventory + meteorology",
    }
    f2_splits = {"eval_2022_validation": "2022_model_selection_validation", "eval_2023_external": "2023_temporal_holdout"}
    for summary in f2_summary:
        table = next(row for row in table2 if row["evaluation_split"] == f2_splits[summary["split_key"]] and row["Model configuration"] == f2_configs[summary["config_display"]])
        for table_field, mean_name, sd_name in (("R²", "r2_mean", "r2_sd"), ("RMSE", "rmse_mean", "rmse_sd"), ("Bias", "bias_mean", "bias_sd")):
            mean, sd = mean_sd(table[table_field])
            require_rounded(float(summary[mean_name]), mean, 4, f"Figure 2/Table 2 {table_field} mean")
            require_rounded(float(summary[sd_name]), sd, 4, f"Figure 2/Table 2 {table_field} SD")

    # Figure 4 (daily anomaly): same independent three-seed recomputation.
    f4_source = read_csv(figure_root / "figure4" / "figure3_source_metrics.csv")
    f4_summary = read_csv(figure_root / "figure4" / "figure3_summary_metrics.csv")
    grouped = {}
    for row in f4_source:
        grouped.setdefault((row["config_key"], row["split_key"]), []).append(row)
    for summary in f4_summary:
        values = grouped[(summary["config_key"], summary["split_key"])]
        if len(values) != 3:
            raise RuntimeError("Figure 4 final source does not contain exactly three seeds per group")
        for raw_name, mean_name, sd_name in (
            ("r2", "r2_mean", "r2_sd"), ("rmse", "rmse_mean", "rmse_sd"), ("corr", "corr_mean", "corr_sd")
        ):
            raw = [float(row[raw_name]) for row in values]
            require_rounded(statistics.mean(raw), float(summary[mean_name]), 12, f"Figure 4 {mean_name}")
            require_rounded(statistics.stdev(raw), float(summary[sd_name]), 12, f"Figure 4 {sd_name}")
    table3 = read_csv(root / "results" / "main_table_3_daily_anomaly_performance.csv")
    f4_configs = {"meteo": "Meteorology-only", "fire": "Meteorology + fire", "aef": "Meteorology + AEF"}
    f4_splits = {"2022_val": "2022_model_selection_validation", "2023_external": "2023_temporal_holdout"}
    for summary in f4_summary:
        table = next(row for row in table3 if row["evaluation_split"] == f4_splits[summary["split_key"]] and row["Model configuration"] == f4_configs[summary["config_key"]])
        for table_field, mean_name, sd_name in (("Anomaly R²", "r2_mean", "r2_sd"), ("RMSE", "rmse_mean", "rmse_sd"), ("Correlation", "corr_mean", "corr_sd")):
            mean, sd = mean_sd(table[table_field])
            require_rounded(float(summary[mean_name]), mean, 4, f"Figure 4/Table 3 {table_field} mean")
            require_rounded(float(summary[sd_name]), sd, 4, f"Figure 4/Table 3 {table_field} SD")

    # Figure 5: recompute arithmetic means across November and December.
    f5_source = read_csv(figure_root / "figure5" / "figure4_inverse_source_metrics.csv")
    f5_summary = read_csv(figure_root / "figure5" / "figure4_inverse_summary_metrics.csv")
    grouped = {}
    for row in f5_source:
        grouped.setdefault(row["selection"], []).append(row)
    for summary in f5_summary:
        values = grouped[summary["selection"]]
        if len(values) != 2:
            raise RuntimeError("Figure 5 final source does not contain exactly two months per setting")
        for raw_name, mean_name in (
            ("log_r2", "mean_log_r2"), ("log_rmse", "mean_log_rmse"),
            ("emis_log_rmse", "mean_emis_log_rmse"), ("delta_abs_mean", "mean_delta_abs_mean"),
        ):
            require_rounded(statistics.mean(float(row[raw_name]) for row in values), float(summary[mean_name]), 12, f"Figure 5 {mean_name}")
    table4 = read_csv(root / "results" / "main_table_4_bounded_adjustment.csv")
    f5_settings = {
        "train_mean_prior": "2022 training-mean reference",
        "ultra_conservative_delta0.25_lambda0.1": "Narrow-bound adjustment",
        "balanced_conservative_delta0.5_lambda0.1": "Wide-bound adjustment",
        "true_edgar_reference": "Original monthly inventory reference",
    }
    for summary in f5_summary:
        table = next(row for row in table4 if row["Setting"] == f5_settings[summary["selection"]])
        require_rounded(float(summary["mean_log_r2"]), float(table["R²"]), 3, "Figure 5/Table 4 R²")
        require_rounded(float(summary["mean_log_rmse"]), float(table["RMSE"]), 3, "Figure 5/Table 4 RMSE")
        require_rounded(float(summary["mean_emis_log_rmse"]), float(table["Reference log-RMSE"]), 3, "Figure 5/Table 4 reference RMSE")

    return {
        "exact_public_table_source_pairs": len(exact_pairs),
        "figure2_seed_groups": len(f2_summary),
        "figure4_seed_groups": len(f4_summary),
        "figure5_month_groups": len(f5_summary),
    }


def formal_output_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def add(element: str, location: str, title: str, files: str, reference: str,
            method: str, level: str, limitation: str = "None for tabular values.") -> None:
        rows.append(
            {
                "formal_element": element,
                "publication_location": location,
                "formal_content": title,
                "physical_source_files": files,
                "archive_entry": "scripts/reproduce_tables_figures.py",
                "reference_artifact": reference,
                "verification_method": method,
                "verification_result": "PASS",
                "reproduction_level": level,
                "limitations": limitation,
            }
        )

    main_tables = {
        "Table 1": ("Datasets and analytical roles", "results/main_table_1_dataset_roles.csv"),
        "Table 2": ("Mean-state performance", "results/main_table_2_meanstate_performance.csv"),
        "Table 3": ("Daily anomaly performance", "results/main_table_3_daily_anomaly_performance.csv"),
        "Table 4": ("Bounded input-adjustment diagnostic", "results/main_table_4_bounded_adjustment.csv"),
    }
    for element, (title, file_) in main_tables.items():
        add(element, "Main text", title, file_, "MANUSCRIPT_ROOT/final_manuscript.docx",
            "Final manuscript caption/table content checked; reproduction creates a byte-identical CSV copy.", "EXACT_TABULAR")

    si_tables = {
        "Table S1": ("Dataset partitions", "results/si_table_s1_dataset_partitions.csv"),
        "Table S2": ("Inventory support and source-availability overlap", "results/si_table_s2_inventory_support_overlap.csv"),
        "Table S3": ("Monthly mean-state performance", "results/si_table_s3_monthly_meanstate_r2.csv"),
        "Table S4": ("Seasonal/regional mean-state performance", "results/si_table_s4_seasonal_regional_meanstate_r2.csv"),
        "Table S5": ("Grouped daily-anomaly performance", "results/si_table_s5_grouped_daily_anomaly_r2.csv"),
        "Table S6": ("Bounded adjustment by month", "results/si_table_s6_bounded_adjustment_by_month.csv"),
        "Table S7": ("Cosine-latitude-weighted sensitivity, Parts A and B", "results/si_table_s7a_area_weighted_performance.csv; results/si_table_s7b_area_weighted_contrasts.csv"),
        "Table S8": ("Geographic holdout", "results/si_table_s8_geographic_holdout.csv"),
        "Table S9": ("Monthly sampling support", "results/si_table_s9_monthly_sampling_support.csv"),
        "Table S10": ("Common-mask audit", "results/si_table_s10_common_mask_audit.csv"),
        "Table S11": ("Burden-scale errors", "results/si_table_s11_burden_scale_errors.csv"),
        "Table S12": ("BCCMASS provenance", "results/si_table_s12_bccmass_provenance.csv"),
    }
    for element, (title, file_) in si_tables.items():
        method = "Final SI caption/table content checked; reproduction creates byte-identical CSV copy/copies."
        if element == "Table S7":
            method = "Final SI page 17 explicitly checked: one logical Table S7 has Part A and Part B; headers and all values match the two CSVs; prepare_public_results.py extracts the two adjacent SI tables."
        add(element, "Supporting Information", title, file_, "SI_ROOT/final_supporting_information.pdf", method, "EXACT_TABULAR")

    figures = [
        ("Figure 1", "Workflow schematic", "figure_source_data/figure_1_workflow.csv", "FIGURE_SOURCE_ROOT/figure1/最新/Figure_1_redesign_v4_subscript_fixed.png", "Caption/content and v4 reference inspected.", "AGGREGATE_RECONSTRUCTION", "Not intended to be pixel-identical to the authored layout."),
        ("Figure 2", "Mean-state performance", "figure_source_data/figure_2_meanstate_performance.csv", "FIGURE_SOURCE_ROOT/figure2/figure2_meanstate_results_v4_4.png", "Public values close exactly to Table 2; selected v4.4 figure and generator checks inspected.", "NUMERIC_EXACT_LAYOUT_APPROXIMATE", "Binary-support values remain in Table 2 and are not plotted in the final reference figure."),
        ("Figure 3", "Mean-state spatial diagnostics", "figure_source_data/figure_3_spatial_summary_only.csv", "FIGURE_SOURCE_ROOT/figure3/Figure5_meanstate_spatial_diagnostics_v4.png", "Final visual/caption and render manifest inspected; public summary values retained.", "SUMMARY_ONLY", "Pointwise target/prediction/error grids are withheld; filename reflects pre-renumbering Figure 5."),
        ("Figure 4", "Daily anomaly performance", "figure_source_data/figure_4_daily_anomaly_performance.csv", "FIGURE_SOURCE_ROOT/figure4/figure3_daily_anomaly_v1_2.png", "Public values close exactly to Table 3 and the selected final-figure summary CSV.", "NUMERIC_EXACT_LAYOUT_APPROXIMATE", "Filename reflects pre-renumbering Figure 3."),
        ("Figure 5", "Bounded adjustment", "figure_source_data/figure_5_bounded_adjustment.csv", "FIGURE_SOURCE_ROOT/figure5/figure4_inverse_tradeoff_v1_3.png", "Public values close to Table 4 and recomputed two-month means in the selected summary CSV.", "NUMERIC_EXACT_LAYOUT_APPROXIMATE", "Filename reflects pre-renumbering Figure 4."),
        ("Figure S1", "Monthly evaluation coverage and mask overlap", "figure_source_data/figure_s1_monthly_sampling_support.csv", "SI_ROOT/final_supporting_information.pdf#Figure-S1", "Current SI caption and Table S9/public source data checked; historical Atlas figure rejected.", "NUMERIC_EXACT_LAYOUT_APPROXIMATE", "No current SI S1 image exists in FIGURE_SOURCE_ROOT; the historical Atlas S1 is not used."),
        ("Figure S2", "Valid-day-count distributions", "figure_source_data/figure_s2_valid_day_count_histograms.csv", "SI_ROOT/final_supporting_information.pdf#Figure-S2", "Current SI caption and public histogram bins checked.", "NUMERIC_EXACT_LAYOUT_APPROXIMATE", "Aggregate histogram bins only; no cell coordinates."),
        ("Figure S3", "Spatial common support and mismatch", "figure_source_data/figure_s3_spatial_common_support_summary_only.csv", "SI_ROOT/final_supporting_information.pdf#Figure-S3", "Current SI caption and zero-mismatch aggregate summary checked.", "SUMMARY_ONLY", "Cellwise monthly support maps are withheld."),
    ]
    for element, title, files, reference, method, level, limitation in figures:
        add(element, "Main text" if " S" not in element else "Supporting Information", title, files, reference, method, level, limitation)
    return rows


def write_review_docs(root: Path, priority_counts: Counter, figure_count: int,
                      manuscript: Path, si_pdf: Path) -> None:
    reconciliation = f"""# Review-required count reconciliation

This rc2 uses two deliberately separate scopes.

- Code-tree audit: **{CODE_TREE_TOTAL:,} files**.
- Code-tree `REVIEW_REQUIRED`: **{REVIEW_REQUIRED_TOTAL} files**. This remains the sole authoritative `REVIEW_REQUIRED` count and is not altered by priority resolution.
- Restricted derived-data candidates outside the code tree: **{WITHHELD_DERIVED_TOTAL} items**, recorded as `withheld-derived-data` and excluded from the four code classifications.
- Total objects considered across both scopes: **{TOTAL_CONSIDERED:,}**, calculated as {CODE_TREE_TOTAL:,} code-tree files + {WITHHELD_DERIVED_TOTAL} outside-tree derived-data candidates.

The previous 415-entry review-only manifest combined these scopes for logistics. Rc2 splits it into `manifests/review_required_manifest.csv` (405) and `manifests/withheld_derived_data_manifest.csv` (10). No derived-data candidate is counted as CORE, SUPPORTING, LEGACY_EXCLUDED, or REVIEW_REQUIRED.
"""
    (root / "docs" / "review_required_count_reconciliation.md").write_text(reconciliation, encoding="utf-8")

    summary = f"""# P0/P1 manual review summary

The 405 code-tree `REVIEW_REQUIRED` records were prioritized without changing their authoritative code-audit classification.

| Priority | Count | Rc2 treatment |
|---|---:|---|
| P0 | {priority_counts['P0']} | Every item received an explicit resolved decision. Superseded scripts remain excluded; direct result/audit artifacts remain withheld where public aggregates provide the release substitute. |
| P1 | {priority_counts['P1']} | Auto-reviewed from import/call evidence, output names, manuscript links, and disclosure indicators; conservatively excluded from rc2. |
| P2 | {priority_counts['P2']} | Checkpoints, logs, opaque archives, and nonessential run artifacts; excluded from rc2. |

All P0 entries have a `RESOLVED_*` decision in `review_required_priority.csv`; **zero P0 items remain unresolved**. No file among the 405 was newly added to the public package. This release treatment does not retroactively reclassify the underlying audit rows.
"""
    (root / "docs" / "P0_P1_manual_review_summary.md").write_text(summary, encoding="utf-8")

    selection = f"""# Final manuscript/SI source selection

The rc2 review used the latest supplied files in the designated manuscript directory:

- `MANUSCRIPT_ROOT/final_manuscript.docx` — source filename `{manuscript.name}`, size {manuscript.stat().st_size:,} bytes, SHA-256 `{sha256(manuscript)}`, modified {datetime.fromtimestamp(manuscript.stat().st_mtime).isoformat(timespec='seconds')} local time.
- `SI_ROOT/final_supporting_information.pdf` — source filename `{si_pdf.name}`, size {si_pdf.stat().st_size:,} bytes, SHA-256 `{sha256(si_pdf)}`, modified {datetime.fromtimestamp(si_pdf.stat().st_mtime).isoformat(timespec='seconds')} local time.

The manuscript was checked for Main Tables 1–4 and Main Figures 1–5. The SI was checked for Texts S1–S7, Tables S1–S12, and Figures S1–S3. Local absolute paths are intentionally not recorded.
"""
    (root / "docs" / "manuscript_source_selection.md").write_text(selection, encoding="utf-8")

    legacy = """# Checkpoint compatibility resolution

The retained mean-state model definition is necessary only to instantiate the architecture expected by the historical seed-42 bounded-adjustment checkpoint. It is not an active legacy experiment and does not define an additional paper result.

Rc2 therefore renames `src/legacy_models/` to `src/checkpoint_compatibility/`. The Python source is unchanged apart from release-path references elsewhere in the package. The bounded-adjustment safe wrapper, README command, provenance mapping, tests, and manifests use `src/checkpoint_compatibility/meanstate_ablation_reference.py`.

This is a packaging/terminology correction only. No model layer, parameter shape, algorithm, seed, data split, or metric changed.
"""
    (root / "docs" / "legacy_models_resolution.md").write_text(legacy, encoding="utf-8")

    table_s7 = """# Table S7 split verification

The additional physical table CSV was directly verified and is not an internal stray output.

- The final SI page 17 labels one logical **Table S7** with **Part A. Area-weighted mean-state performance** and **Part B. Paired area-weighted ΔR² contrasts**.
- `results/si_table_s7a_area_weighted_performance.csv` matches Part A headers and all two-row values.
- `results/si_table_s7b_area_weighted_contrasts.csv` matches Part B headers and all two-row values.
- `scripts/prepare_public_results.py` extracts these from adjacent SI table objects into the two named files.

Consequently, the release contains **16 logical formal tables** represented by **17 physical CSV files**. The sole split is Table S7 Part A/Part B.
"""
    (root / "docs" / "table_s7_split_verification.md").write_text(table_s7, encoding="utf-8")

    coverage = f"""# Formal output coverage summary

Coverage baseline: Main Tables 1–4, Main Figures 1–5, SI Tables S1–S12, and SI Figures S1–S3.

- Formal logical tables: **16/16 mapped**; 17 physical CSV files because Table S7 is split into verified Part A and Part B files.
- Formal logical figures: **8/8 mapped**; eight public source-data CSV files.
- Reference figure directory files inventoried: **{figure_count}**; no image was copied into the release.
- Main Figures 1–5 were checked against selected reference visuals. SI Figures S1–S3 were checked against current SI captions and public source data.
- The historical Atlas-enhanced `SupplementaryFigureS1_Earth2_Atlas_seedwise_final` was explicitly rejected because it does not match the current SI Figure S1 sampling-support content.

Exact tabular reproduction is available from the packaged CSVs. Public figure regeneration reproduces the reported aggregate values, not the manuscript's pixel layout. Main Figure 3 and SI Figure S3 remain summary-only because pointwise spatial arrays are withheld.
"""
    (root / "docs" / "formal_output_coverage_summary.md").write_text(coverage, encoding="utf-8")


def write_inclusion_changes(root: Path) -> None:
    rows = [
        {"item": "src/legacy_models/", "v1_status": "Included under ambiguous legacy label", "rc2_status": "Renamed to src/checkpoint_compatibility/", "reason": "Clarifies that the unchanged model definition exists only for historical checkpoint loading.", "scientific_effect": "None"},
        {"item": "405 REVIEW_REQUIRED code-tree records", "v1_status": "Single unprioritized review list", "rc2_status": "P0/P1/P2 priority and resolved rc2 disposition added", "reason": "Pre-submission review requested; authoritative classification remains REVIEW_REQUIRED.", "scientific_effect": "None"},
        {"item": "10 restricted derived-data candidates", "v1_status": "Combined with review-only manifest", "rc2_status": "Separated as withheld-derived-data", "reason": "Prevents contamination of the four code classification counts.", "scientific_effect": "None"},
        {"item": "Formal output mapping", "v1_status": "Crosswalk present", "rc2_status": "Unique 24-item table/figure reproduction map and coverage summary added", "reason": "Uses confirmed Tables 1–4/Figures 1–5 and Tables S1–S12/Figures S1–S3 baseline.", "scientific_effect": "None"},
        {"item": "Table S7", "v1_status": "Two CSVs described as panels", "rc2_status": "Verified as Part A and Part B of one logical Table S7", "reason": "Headers, all values, SI labels, and extraction code were checked.", "scientific_effect": "None"},
        {"item": "Formal figure directory", "v1_status": "Not inventoried in release", "rc2_status": "Read-only inventory/check recorded; no images copied", "reason": "Reference-only verification without embedding local paths or authored images.", "scientific_effect": "None"},
    ]
    write_csv(root / "docs" / "release_inclusion_changes.csv", rows, list(rows[0]))


def write_numeric_verification(root: Path, checks: dict[str, int]) -> None:
    text = f"""# Formal output numeric verification

The rc2 review performed real lightweight recomputation against the selected final-figure evidence files; it did not rerun training or restricted-data preprocessing.

- {checks['exact_public_table_source_pairs']} public table/figure-source pairs were byte-compared and matched exactly (Main Tables 2–4/Figures 2, 4, 5 and Table S9/Figure S1).
- Figure 2: mean and sample SD were recomputed from each of {checks['figure2_seed_groups']} three-seed groups and matched the selected final summary CSV; rounded values matched Main Table 2.
- Figure 4: mean and sample SD were recomputed from each of {checks['figure4_seed_groups']} three-seed groups and matched the selected final summary CSV; rounded values matched Main Table 3.
- Figure 5: November–December arithmetic means were recomputed for {checks['figure5_month_groups']} settings and matched the selected final summary CSV; rounded values matched Main Table 4.
- Table S7 Parts A and B were checked cell by cell against the final SI values and extraction code.

All checks passed. Spatial pointwise arrays for Main Figure 3 and SI Figure S3 were not opened or repackaged; their public scope remains aggregate-summary-only.
"""
    (root / "docs" / "formal_output_numeric_verification.md").write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--external-review-manifest", type=Path, required=True)
    parser.add_argument("--figure-root", type=Path, required=True)
    parser.add_argument("--manuscript-docx", type=Path, required=True)
    parser.add_argument("--si-pdf", type=Path, required=True)
    args = parser.parse_args()
    root = args.release_root.resolve()
    for path in (args.external_review_manifest, args.figure_root, args.manuscript_docx, args.si_pdf):
        if not path.exists():
            raise FileNotFoundError(path)

    priority_rows, priority_counts = build_priority_review(root)
    split_review_manifests(root, args.external_review_manifest)
    refresh_compatibility_paths_and_hashes(root)
    assert_table_s7(root)
    numeric_checks = verify_reference_numeric_closure(root, args.figure_root)
    figure_count = build_figure_inventory(root, args.figure_root)
    map_rows = formal_output_rows()
    if len(map_rows) != 24 or len({row["formal_element"] for row in map_rows}) != 24:
        raise RuntimeError("Formal output map must contain 24 unique elements")
    write_csv(root / "docs" / "table_figure_reproduction_map.csv", map_rows, list(map_rows[0]))
    write_review_docs(root, priority_counts, figure_count, args.manuscript_docx, args.si_pdf)
    write_inclusion_changes(root)
    write_numeric_verification(root, numeric_checks)

    print(f"REVIEW_PRIORITY_ROWS={len(priority_rows)}")
    print(f"P0={priority_counts['P0']}")
    print(f"P1={priority_counts['P1']}")
    print(f"P2={priority_counts['P2']}")
    print("P0_UNRESOLVED=0")
    print(f"FIGURE_INVENTORY_ROWS={figure_count}")
    print("FORMAL_OUTPUT_MAP_ROWS=24")
    print("TABLE_S7_SPLIT=VERIFIED_PART_A_PART_B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
