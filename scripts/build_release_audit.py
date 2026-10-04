#!/usr/bin/env python3
"""Build a conservative, evidence-oriented inventory of the source project.

The source tree is opened read-only. Output CSV files are written only below
the release staging directory supplied with ``--staging-root``.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable


CORE_RELEASE_PATHS = {
    "figure1  825.py": "scripts/generate_figure1_schematic.py",
    "2021-2022/45_build_phase2_meanstate_fullmonth_clean.py": "src/workflows/build_meanstate_cache.py",
    "2021-2022/47_train_s2_meanstate_clean.py": "src/workflows/train_meanstate.py",
    "2021-2022/52_eval_s2_meanstate_clean_formal_run.py": "src/workflows/evaluate_meanstate.py",
    "2021-2022/53_aggregate_s2_meanstate_clean_three_seed.py": "src/workflows/aggregate_meanstate.py",
    "2021-2022/57_train_s2_meanstate_support_preserving.py": "src/workflows/train_support_preserving_control.py",
    "2021-2022/62_eval_s2_meanstate_support_preserving_formal_run.py": "src/workflows/evaluate_support_preserving_control.py",
    "2021-2022/63_aggregate_and_seal_meanstate_support_control.py": "src/workflows/aggregate_support_preserving_control.py",
    "2021-2022/64_eval_meanstate_area_weighted_gateP4_v5.py": "src/workflows/evaluate_area_weighted_meanstate.py",
    "2021-2022/65_plot_meanstate_spatial_diagnostics_figure5_v3.py": "src/workflows/prepare_spatial_diagnostics.py",
    "2021-2022/66_render_figure5_spatial_diagnostics_v4.py": "scripts/render_spatial_diagnostics.py",
    "2021-2022/65_build_clean_anomaly_cache.py": "src/workflows/build_anomaly_cache.py",
    "2021-2022/67b_train_s2_anomaly_clean.py": "src/workflows/train_daily_anomaly.py",
    "2021-2022/70b_eval_s2_anomaly_clean_formal_run.py": "src/workflows/evaluate_anomaly_meteorology.py",
    "2021-2022/70c_eval_s2_anomaly_fire_formal_run.py": "src/workflows/evaluate_anomaly_fire.py",
    "2021-2022/70d_eval_s2_anomaly_aef_formal_run.py": "src/workflows/evaluate_anomaly_aef.py",
    "2021-2022/71_seal_anomaly_meteo_three_seed.py": "src/workflows/aggregate_anomaly_meteorology.py",
    "2021-2022/73_seal_anomaly_fire_three_seed.py": "src/workflows/aggregate_anomaly_fire.py",
    "2021-2022/75_seal_anomaly_aef_three_seed.py": "src/workflows/aggregate_anomaly_aef.py",
    "2021-2022/77_train_s2_meanstate_binary_support_gateS8.py": "src/workflows/train_binary_support_control.py",
    "2021-2022/79b_eval_s2_meanstate_binary_support_formal_run_v2.py": "src/workflows/evaluate_binary_support_control.py",
    "2021-2022/80b_aggregate_and_seal_meanstate_binary_support_gateS8_v2.py": "src/workflows/aggregate_binary_support_control.py",
    "2021-2022/83_train_s2_meanstate_spatial_holdout_gateS9C.py": "src/workflows/train_geographic_holdout.py",
    "2021-2022/86_open_meanstate_spatial_holdout_core_gateS9F.py": "src/workflows/evaluate_geographic_holdout.py",
    "2021-2022/87_audit_posp_bccmass_sampling_support.py": "src/audits/audit_sampling_support.py",
    "2021-2022/22_b0_inverse_meanstate_sector_offset_sweep.py": "src/workflows/run_bounded_adjustment_november.py",
    "2021-2022/23_b0_inverse_meanstate_sector_offset_sweep_2022_12.py": "src/workflows/run_bounded_adjustment_december.py",
    "2021-2022/24_b0_two_month_inverse_summary.py": "src/workflows/summarize_bounded_adjustment_sweep.py",
    "2021-2022/25_b0_fixed_setting_inverse_summary.py": "src/workflows/summarize_bounded_adjustment_fixed.py",
    "2021-2022/28_extract_table3_inverse_fixed_setting.py": "src/workflows/export_bounded_adjustment_table.py",
    "audit_meanstate_physical_scale_sensitivity.py": "src/audits/audit_burden_scale.py",
    "audit_merra2_bccmass_fingerprint.py": "src/audits/audit_bccmass_provenance.py",
    "summarize_bccmass_fingerprint_for_si.py": "src/audits/summarize_bccmass_provenance.py",
}

SUPPORTING_RELEASE_PATHS = {
    "2021-2022/13_train_s2_meanstate_ablation.py": "src/checkpoint_compatibility/meanstate_ablation_reference.py",
    "2021-2022/01_generate_grid_reference.py": "src/preprocessing/generate_grid_reference.py",
    "2021-2022/02_fetch_aef_grid.py": "src/preprocessing/fetch_aef_grid.py",
    "2021-2022/03_process_aef_grid.py": "src/preprocessing/process_aef_grid.py",
    "2021-2022/30_audit_bccmass_nc_metadata.py": "src/audits/audit_bccmass_metadata.py",
    "2021-2022/31_audit_emission_coordinate_stability.py": "src/audits/audit_inventory_coordinate_stability.py",
    "2021-2022/32a_audit_transport_same_day_grid_unordered.py": "src/audits/audit_inventory_daily_grid_consistency.py",
    "2021-2022/33a_audit_meanstate_monthly_emission_inputs_blas_free.py": "src/audits/audit_monthly_inventory_inputs.py",
    "2021-2022/35_audit_bccmass_unit_and_scale_trace.py": "src/audits/audit_bccmass_scale_trace.py",
    "2021-2022/36_compute_train_only_meteo_stats.py": "src/preprocessing/compute_training_meteorology_statistics.py",
    "2021-2022/39_rebuild_daily_tensors_meteo_trainonly_from_raw.py": "src/preprocessing/build_daily_tensors.py",
    "2021-2022/40_audit_meteo_trainonly_direct_tensors_gate9b.py": "src/audits/audit_daily_tensors.py",
    "2021-2022/43_rebuild_2023_tensors_union_trainonly.py": "src/preprocessing/build_2023_tensors.py",
    "2021-2022/44_audit_2023_union_trainonly_tensors_gate10c.py": "src/audits/audit_2023_tensors.py",
    "2021-2022/46_audit_phase2_meanstate_fullmonth_clean_gate11b.py": "src/audits/audit_meanstate_cache.py",
    "2021-2022/48_audit_s2_meanstate_clean_smoke_gate12b.py": "src/audits/audit_meanstate_smoke.py",
    "2021-2022/49b_audit_s2_meanstate_clean_formal_seed42_gate13b.py": "src/audits/audit_meanstate_formal_seed.py",
    "2021-2022/51_audit_s2_meanstate_clean_formal_run.py": "src/audits/audit_meanstate_formal.py",
    "2021-2022/54_validate_and_seal_clean_meanstate_inventory.py": "src/audits/seal_meanstate_inventory.py",
    "2021-2022/55_audit_meanstate_support_geometry_v02.py": "src/audits/audit_support_geometry.py",
    "2021-2022/56_validate_meanstate_support_preserving_permutation.py": "src/audits/validate_support_permutation.py",
    "2021-2022/58_audit_s2_meanstate_support_preserving_smoke.py": "src/audits/audit_support_control_smoke.py",
    "2021-2022/61_audit_s2_meanstate_support_preserving_formal_run.py": "src/audits/audit_support_control_formal.py",
    "2021-2022/63_audit_meanstate_mask_overlap_gateP3.py": "src/audits/audit_inventory_mask_overlap.py",
    "2021-2022/64_preflight_clean_anomaly_cache.py": "src/audits/preflight_anomaly_cache.py",
    "2021-2022/66_audit_clean_anomaly_cache_gateA3.py": "src/audits/audit_anomaly_cache.py",
    "2021-2022/68b_audit_s2_anomaly_clean_smoke_gateA5.py": "src/audits/audit_anomaly_smoke.py",
    "2021-2022/69b_audit_s2_anomaly_clean_formal_run.py": "src/audits/audit_anomaly_meteorology.py",
    "2021-2022/69c_audit_s2_anomaly_fire_formal_run.py": "src/audits/audit_anomaly_fire.py",
    "2021-2022/69d_audit_s2_anomaly_aef_formal_run.py": "src/audits/audit_anomaly_aef.py",
    "2021-2022/76_preflight_s2_meanstate_binary_support_gateS8A.py": "src/audits/preflight_binary_support.py",
    "2021-2022/78_audit_s2_meanstate_binary_support_formal_run.py": "src/audits/audit_binary_support.py",
    "2021-2022/81_preflight_meanstate_spatial_holdout_gateS9A.py": "src/audits/preflight_geographic_holdout.py",
    "2021-2022/82_compute_spatial_train_only_meteo_stats_gateS9B.py": "src/preprocessing/compute_holdout_meteorology_statistics.py",
    "2021-2022/84_audit_s2_meanstate_spatial_holdout_formal_run_gateS9D.py": "src/audits/audit_geographic_holdout_runs.py",
    "2021-2022/85_seal_meanstate_spatial_holdout_preopening_gateS9E.py": "src/audits/seal_geographic_holdout.py",
    "audit_firms_csv_metadata.py": "src/audits/audit_firms_metadata.py",
}

FINAL_AGGREGATE_PREFIXES = (
    "2021-2022/paper_result_inventory_clean/",
)

LEGACY_PATH_MARKERS = (
    "/rubbish/",
    "/_handoff/",
    "/_paper_workspace/",
    "/__pycache__/",
    "/runs_a2_ablation/",
    "/runs_s2_meanstate_multiseed/",
    "/runs_s2_meanstate/",
    "/runs_s2_anomaly/",
    "/paper_tables/",
    "/paper_result_inventory/",
    "/gpt",
    "/3.6",
)

_DRIVE_PATH_PATTERN = r"[A-Z]" + r":[\\/]"
_UNIX_USER_PATH_PATTERN = r"/(?:home|Users)/[^/\s]+/|/" + r"mnt/[a-z]/"
ABSOLUTE_PATH_RE = re.compile(rf"(?i)(?:{_DRIVE_PATH_PATTERN}|{_UNIX_USER_PATH_PATTERN})")
EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
SECRET_PATTERNS = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "aws_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "generic_secret_assignment": re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|secret|password|passwd)\b\s*[:=]\s*['\"][^'\"]{8,}['\"]"
    ),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
}
RESTRICTED_TERMS = (
    "posp",
    "i_380",
    "i380",
    "fused",
    "bccmass_train",
    "source_nc",
    "row-level",
    "per-record",
)
TEXT_EXTENSIONS = {
    ".py", ".ipynb", ".sh", ".bat", ".yaml", ".yml", ".json",
    ".toml", ".txt", ".cfg", ".ini", ".csv", ".md", ".svg", ".xml",
}


def read_text(path: Path) -> str:
    if path.suffix.lower() not in TEXT_EXTENSIONS or path.stat().st_size > 12_000_000:
        return ""
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return ""


def relpath(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def python_imports(text: str) -> list[str]:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return []
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return sorted(set(imports))


def purpose_for(rel: str, text: str) -> str:
    name = Path(rel).stem.lower()
    joined = f"{name} {text[:5000].lower()}"
    if "train" in name:
        return "Model training workflow."
    if "eval" in name or "evaluate" in name or "rescore" in name:
        return "Model evaluation or fixed-checkpoint rescoring workflow."
    if "audit" in name or "validate" in name or "preflight" in name or "seal" in name:
        return "Audit, validation, preflight, or provenance-sealing artifact."
    if "aggregate" in name or "summary" in name or "summarize" in name or "extract_table" in name:
        return "Aggregation or manuscript table preparation artifact."
    if "plot" in name or "figure" in name or "render" in name:
        return "Figure generation or rendering artifact."
    if "build" in name or "construct" in name or "rebuild" in name or "process" in name:
        return "Data or tensor preprocessing workflow."
    if Path(rel).suffix.lower() == ".pt":
        return "Serialized model checkpoint or tensor artifact."
    if Path(rel).suffix.lower() == ".pyc":
        return "Compiled Python cache; not source code."
    if Path(rel).suffix.lower() in {".json", ".csv", ".txt", ".md"}:
        return "Configuration, manifest, log, audit report, or derived result artifact."
    if any(word in joined for word in ("checkpoint", "manifest", "metric")):
        return "Run configuration, checkpoint, or metric artifact."
    return "Project artifact; specific final-manuscript role not established."


def manuscript_link(rel: str, text: str) -> str:
    low = f"{rel} {text[:12000]}".lower()
    links: list[str] = []
    rules = [
        (("meanstate", "mean_state", "mean-state"), "Main Table 2; Figure 2; Text S4; Tables S3-S4"),
        (("spatial_holdout", "geographic holdout", "eastasia"), "Text S3; Table S8"),
        (("anomaly",), "Main Table 3; Figure 4; Table S5"),
        (("support_preserving", "supportperm"), "Text S2; Main Table 2; Figure 2; Table S7"),
        (("binary_support", "binary support"), "Text S2; Main Table 2"),
        (("area_weighted", "cos(latitude)", "cosine"), "Table S7"),
        (("physical_scale", "back-transform", "burden-scale"), "Text S6; Table S11"),
        (("bccmass_fingerprint", "bccmass provenance"), "Text S7; Table S12"),
        (("sampling_support", "common_mask", "mask_overlap"), "Text S5; Tables S9-S10; Figures S1-S3"),
        (("inverse", "bounded", "sector_offset"), "Main Table 4; Figure 5; Table S6"),
        (("figure1", "figure_1"), "Main Figure 1"),
        (("spatial_diagnostics",), "Main Figure 3"),
    ]
    for needles, label in rules:
        if any(needle in low for needle in needles):
            links.append(label)
    return "; ".join(dict.fromkeys(links)) or "No verified direct link to the final manuscript or SI."


def generated_outputs(text: str) -> str:
    names = re.findall(
        r"(?i)[A-Za-z0-9_.-]+\.(?:csv|json|txt|md|png|svg|pdf|pt|npz|yaml|yml)",
        text,
    )
    return "; ".join(sorted(set(names))[:24])


def data_dependencies(text: str) -> str:
    low = text.lower()
    deps: list[str] = []
    for suffix, label in (
        (".nc", "NetCDF inputs"), (".pt", "PyTorch tensors/checkpoints"),
        (".npz", "NumPy archives"), (".csv", "CSV inputs"), (".json", "JSON manifests/reports"),
    ):
        if suffix in low:
            deps.append(label)
    if any(term in low for term in RESTRICTED_TERMS):
        deps.append("restricted POSP-centered fused inputs or derivatives")
    if "merra-2" in low or "merra2" in low:
        deps.append("MERRA-2")
    if "era5" in low:
        deps.append("ERA5")
    if "firms" in low or "fire" in low:
        deps.append("MODIS/FIRMS fire data")
    if "aef" in low or "alphaearth" in low:
        deps.append("AlphaEarth Foundations embeddings")
    return "; ".join(dict.fromkeys(deps)) or "No explicit data dependency detected."


def classify(rel: str, path: Path, text: str) -> tuple[str, str, str]:
    rel_low = "/" + rel.lower()
    ext = path.suffix.lower()
    has_abs = bool(ABSOLUTE_PATH_RE.search(text))
    has_restricted = any(term in text.lower() for term in RESTRICTED_TERMS)
    secret_hits = [name for name, pattern in SECRET_PATTERNS.items() if pattern.search(text)]

    if rel in CORE_RELEASE_PATHS:
        evidence = "Explicitly linked to a final manuscript/SI method, result, table, figure, or control and corroborated by current result provenance."
        action = "Sanitize path defaults and retain scientific logic unchanged."
        if secret_hits:
            action += " Resolve detected credential-like text before release."
        return "CORE", evidence, action
    if rel in SUPPORTING_RELEASE_PATHS:
        evidence = "Required preprocessing, preflight, audit, or sealing step used by an evidence-confirmed final workflow."
        action = "Sanitize path defaults; document that restricted inputs are required."
        return "SUPPORTING", evidence, action

    if any(rel.startswith(prefix) for prefix in FINAL_AGGREGATE_PREFIXES):
        if ext == ".csv" and "predictions" not in rel_low:
            if has_abs:
                return (
                    "SUPPORTING",
                    "Current clean-result aggregate used to verify final manuscript/SI summaries.",
                    "Create a sanitized public derivative with absolute-path columns removed or relativized.",
                )
            return (
                "SUPPORTING",
                "Current clean-result aggregate used to verify final manuscript/SI summaries.",
                "Confirm aggregate-only disclosure and copy through the result sanitization step.",
            )
        if ext in {".npz", ".pt"} or "prediction" in rel_low:
            return (
                "REVIEW_REQUIRED",
                "Current result artifact may contain grid-cell predictions or other recoverable restricted derivatives.",
                "Author/data-custodian review required; do not copy into the public package.",
            )
        return (
            "REVIEW_REQUIRED",
            "Current result/provenance file contains rich nested metadata and may include absolute paths or sensitive lineage.",
            "Release only a purpose-built aggregate derivative after manual review.",
        )

    if any(marker in rel_low for marker in LEGACY_PATH_MARKERS):
        return (
            "LEGACY_EXCLUDED",
            "Located in an explicitly historical, superseded, cache, handoff, or earlier-result area and not required by the final evidence chain.",
            "None; retain unchanged only in the original project.",
        )

    if ext in {".pyc", ".pyo"}:
        return (
            "LEGACY_EXCLUDED",
            "Compiled cache is not maintainable source and is unnecessary for reproducibility.",
            "None; do not release.",
        )
    if ext in {".pt", ".pth", ".npz", ".npy", ".pkl", ".pickle", ".zip"}:
        return (
            "REVIEW_REQUIRED",
            "Binary artifact may contain trained weights, tensors, spatial arrays, caches, or an opaque archive.",
            "Inspect contents and redistribution rights manually; do not copy by default.",
        )
    if secret_hits:
        return (
            "REVIEW_REQUIRED",
            "Credential-like content detected: " + ", ".join(secret_hits),
            "Manually determine whether this is a real secret or a placeholder; redact before any release.",
        )
    if ext == ".py":
        if rel.startswith("2021-2022/"):
            return (
                "REVIEW_REQUIRED",
                "Executable source exists, but a direct final-manuscript role or supersession relationship was not proven by the conservative allowlist.",
                "Author confirmation required before inclusion; do not copy into the public package.",
            )
        return (
            "LEGACY_EXCLUDED",
            "Earlier or parallel root-level workflow not used by the verified final clean-analysis chain.",
            "None unless the author identifies a missing final dependency.",
        )
    if ext in {".json", ".csv", ".txt", ".md", ".yaml", ".yml", ".toml", ".cfg", ".ini"}:
        if has_abs or has_restricted or EMAIL_RE.search(text):
            return (
                "REVIEW_REQUIRED",
                "Text artifact contains absolute paths, restricted-data lineage, identifiers, or potentially sensitive metadata.",
                "Manual content review and sanitization required; do not copy by default.",
            )
        return (
            "LEGACY_EXCLUDED",
            "No verified need in the final release and no final evidence-chain link was established.",
            "None; retain unchanged only in the original project.",
        )
    return (
        "LEGACY_EXCLUDED",
        "No verified final-manuscript role was established for this artifact type.",
        "None; retain unchanged only in the original project.",
    )


def write_csv(path: Path, rows: Iterable[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-code-root", type=Path, required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_code_root.resolve()
    staging = args.staging_root.resolve()
    docs = staging / "docs"

    files = sorted((path for path in source.rglob("*") if path.is_file()), key=lambda p: relpath(p, source).lower())
    texts: dict[str, str] = {}
    imports_by_rel: dict[str, list[str]] = {}
    basenames: dict[str, list[str]] = defaultdict(list)
    for path in files:
        rel = relpath(path, source)
        basenames[path.name.lower()].append(rel)
        text = read_text(path)
        texts[rel] = text
        if path.suffix.lower() == ".py" and text:
            imports_by_rel[rel] = python_imports(text)

    references: dict[str, set[str]] = defaultdict(set)
    module_to_rel: dict[str, list[str]] = defaultdict(list)
    for rel in imports_by_rel:
        module_to_rel[Path(rel).stem].append(rel)
    for caller, text in texts.items():
        if not text:
            continue
        imports = imports_by_rel.get(caller, [])
        for imported in imports:
            stem = imported.rsplit(".", 1)[-1]
            for target in module_to_rel.get(stem, []):
                if target != caller:
                    references[target].add(caller)
        low = text.lower()
        for filename, targets in basenames.items():
            if filename in low:
                for target in targets:
                    if target != caller:
                        references[target].add(caller)

    rows: list[dict[str, str]] = []
    for path in files:
        rel = relpath(path, source)
        text = texts[rel]
        classification, evidence, action = classify(rel, path, text)
        abs_hits = len(ABSOLUTE_PATH_RE.findall(text))
        restricted_hits = sorted(term for term in RESTRICTED_TERMS if term in text.lower())
        secret_hits = sorted(name for name, pattern in SECRET_PATTERNS.items() if pattern.search(text))
        proposed = CORE_RELEASE_PATHS.get(rel) or SUPPORTING_RELEASE_PATHS.get(rel, "")
        if classification == "SUPPORTING" and rel.startswith(FINAL_AGGREGATE_PREFIXES):
            proposed = "results/sanitized_derivatives/" + path.name
        runnable = "No"
        if classification in {"CORE", "SUPPORTING"} and not restricted_hits:
            runnable = "Yes, subject to listed software dependencies."
        elif classification in {"CORE", "SUPPORTING"}:
            runnable = "Code path only; full execution requires authorized restricted inputs."
        row = {
            # Use a portable project-relative identifier in public audit tables.
            # The local source root is intentionally not disclosed in the archive.
            "original_path": f"code/{rel}",
            "filename": path.name,
            "file_type": path.suffix.lower() or "[no extension]",
            "size_bytes": str(path.stat().st_size),
            "sha256": sha256(path),
            "purpose": purpose_for(rel, text),
            "manuscript_or_SI_link": manuscript_link(rel, text),
            "imported_or_called_by": "; ".join(sorted(references.get(rel, set()))[:30]) or "No static caller identified.",
            "generated_outputs": generated_outputs(text) or "No explicit generated filename identified.",
            "data_dependencies": data_dependencies(text),
            "contains_absolute_path": "Yes" if abs_hits else "No",
            "restricted_data_indicators": "; ".join(restricted_hits) or "None detected.",
            "credential_or_privacy_indicators": "; ".join(secret_hits) or ("Email-like identifier detected." if EMAIL_RE.search(text) else "None detected."),
            "runnable_without_restricted_data": runnable,
            "classification": classification,
            "evidence": evidence,
            "proposed_release_path": proposed,
            "required_manual_action": action,
        }
        rows.append(row)

    fields = [
        "original_path", "filename", "file_type", "size_bytes", "sha256", "purpose",
        "manuscript_or_SI_link", "imported_or_called_by", "generated_outputs", "data_dependencies",
        "contains_absolute_path", "restricted_data_indicators", "credential_or_privacy_indicators",
        "runnable_without_restricted_data", "classification", "evidence", "proposed_release_path",
        "required_manual_action",
    ]
    write_csv(docs / "code_audit.csv", rows, fields)
    write_csv(docs / "excluded_files.csv", (row for row in rows if row["classification"] == "LEGACY_EXCLUDED"), fields)
    write_csv(docs / "review_required.csv", (row for row in rows if row["classification"] == "REVIEW_REQUIRED"), fields)

    counts = defaultdict(int)
    for row in rows:
        counts[row["classification"]] += 1
    print(f"AUDITED_FILES={len(rows)}")
    for label in ("CORE", "SUPPORTING", "LEGACY_EXCLUDED", "REVIEW_REQUIRED"):
        print(f"{label}={counts[label]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
