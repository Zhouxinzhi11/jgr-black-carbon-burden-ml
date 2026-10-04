# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
# -*- coding: utf-8 -*-
"""
55_audit_meanstate_support_geometry.py

Read-only diagnostic for the current clean mean-state emission-geometry control.

Scientific question
-------------------
Does the deterministic cyclic roll used by the shuffled-emission control
materially change the overlap between:
  1. the monthly target valid mask, and
  2. the non-zero support of the processed emission-prior tensor?

Important limitation
--------------------
The cached processed emission tensor uses zero both for cells without mapped
source rows and potentially for genuinely zero processed emissions. Therefore
this script audits "non-zero processed-emission support", not a separately
observed source-row support mask.

The script:
- never modifies cache, training, evaluation, or manuscript artifacts;
- imports the exact deterministic_roll_2d implementation from the recorded
  training script;
- audits all real-emission 2022 monthly groups;
- builds the exact fixed proxy from the six 2022 training-month emission tensors
  and audits all 12 external 2023 monthly target masks;
- writes JSON, CSV, and TXT diagnostics to a new output directory only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import torch


EXPECTED_TRAIN_MONTHS = [2, 3, 5, 6, 8, 9]
EXPECTED_VALIDATION_MONTHS = [11, 12]
EXPECTED_2023_MONTHS = list(range(1, 13))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    require(isinstance(obj, dict), f"JSON_ROOT_NOT_OBJECT: {path}")
    return obj


def load_pt(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_training_module(script_path: Path):
    module_name = "meanstate_clean_training_for_support_audit"
    spec = importlib.util.spec_from_file_location(module_name, str(script_path))
    require(spec is not None and spec.loader is not None, "TRAINING_SCRIPT_IMPORT_SPEC_FAILED")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
        raise
    return module


def safe_fraction(num: int, den: int) -> float | None:
    return float(num / den) if den > 0 else None


def jaccard(
    a: torch.Tensor,
    b: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> float | None:
    aa = a.bool()
    bb = b.bool()
    if mask is not None:
        m = mask.bool()
        aa = aa & m
        bb = bb & m
    inter = int(torch.count_nonzero(aa & bb).item())
    union = int(torch.count_nonzero(aa | bb).item())
    return safe_fraction(inter, union)


def finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value) if math.isfinite(value) else None


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    require(rows, "EMPTY_SUMMARY_GROUP")

    defined = [
        r for r in rows
        if r["target_valid_coverage_unshifted"] is not None
        and r["target_valid_coverage_rolled"] is not None
        and r["coverage_delta_rolled_minus_unshifted"] is not None
    ]
    require(defined, "SUMMARY_GROUP_HAS_NO_DEFINED_COVERAGE")

    delta = [
        float(r["coverage_delta_rolled_minus_unshifted"])
        for r in defined
    ]
    abs_delta = [abs(x) for x in delta]
    j_all = [
        float(r["support_jaccard_global"])
        for r in rows
        if r["support_jaccard_global"] is not None
    ]
    j_valid = [
        float(r["support_jaccard_within_target_valid"])
        for r in rows
        if r["support_jaccard_within_target_valid"] is not None
    ]

    return {
        "n_months_total": len(rows),
        "n_months_with_defined_target_coverage": len(defined),
        "undefined_target_coverage_months": [
            f"{int(r['year'])}-{int(r['month']):02d}"
            for r in rows
            if r not in defined
        ],
        "mean_coverage_unshifted": statistics.mean(
            float(r["target_valid_coverage_unshifted"]) for r in defined
        ),
        "mean_coverage_rolled": statistics.mean(
            float(r["target_valid_coverage_rolled"]) for r in defined
        ),
        "mean_coverage_delta_rolled_minus_unshifted": statistics.mean(delta),
        "mean_absolute_coverage_delta": statistics.mean(abs_delta),
        "max_absolute_coverage_delta": max(abs_delta),
        "months_abs_delta_ge_0p02": sum(x >= 0.02 for x in abs_delta),
        "months_abs_delta_ge_0p05": sum(x >= 0.05 for x in abs_delta),
        "months_rolled_coverage_lower": sum(x < 0.0 for x in delta),
        "months_rolled_coverage_higher": sum(x > 0.0 for x in delta),
        "months_equal_coverage": sum(x == 0.0 for x in delta),
        "mean_support_jaccard_global": (
            statistics.mean(j_all) if j_all else None
        ),
        "mean_support_jaccard_within_target_valid": (
            statistics.mean(j_valid) if j_valid else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument(
        "--training-script",
        default="./47_train_s2_meanstate_clean.py",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--support-epsilon",
        type=float,
        default=0.0,
        help="A grid cell is in processed-emission support if any sector abs(value) > epsilon.",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    training_script = Path(args.training_script).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    manifest_path = cache_dir / "manifests" / "monthly_manifest.json"

    require(cache_dir.is_dir(), f"CACHE_DIR_NOT_FOUND: {cache_dir}")
    require(training_script.is_file(), f"TRAINING_SCRIPT_NOT_FOUND: {training_script}")
    require(manifest_path.is_file(), f"MONTHLY_MANIFEST_NOT_FOUND: {manifest_path}")
    require(args.support_epsilon >= 0.0, "SUPPORT_EPSILON_MUST_BE_NONNEGATIVE")
    require(not output_dir.exists(), f"STOP_OUTPUT_DIR_ALREADY_EXISTS: {output_dir}")

    module = load_training_module(training_script)
    require(hasattr(module, "deterministic_roll_2d"), "TRAINING_ROLL_FUNCTION_MISSING")
    H = int(module.H)
    W = int(module.W)

    manifest = load_json(manifest_path)
    entries = list(manifest.get("groups", []))
    require(len(entries) == 21, f"EXPECTED_21_MONTHLY_GROUPS_GOT_{len(entries)}")

    entries = sorted(entries, key=lambda e: (int(e["year"]), int(e["month"])))
    entries_2022 = [e for e in entries if int(e["year"]) == 2022]
    entries_2023 = [e for e in entries if int(e["year"]) == 2023]

    require(
        sorted(int(e["month"]) for e in entries_2023) == EXPECTED_2023_MONTHS,
        "2023_MONTH_COVERAGE_MISMATCH",
    )

    by_2022_month = {int(e["month"]): e for e in entries_2022}
    for month in EXPECTED_TRAIN_MONTHS + EXPECTED_VALIDATION_MONTHS:
        require(month in by_2022_month, f"MISSING_REQUIRED_2022_MONTH_{month:02d}")

    # Exact fixed proxy used in formal evaluation:
    # unweighted arithmetic mean of the six 2022 training-month tensors.
    proxy_acc = None
    proxy_sources: list[dict[str, Any]] = []
    for month in EXPECTED_TRAIN_MONTHS:
        entry = by_2022_month[month]
        path = Path(entry["file"]).expanduser().resolve()
        payload = load_pt(path)
        emission = payload.get("emission_log_mean")
        require(isinstance(emission, torch.Tensor), f"MISSING_2022_EMISSION_{month:02d}")
        emission = torch.nan_to_num(
            emission.detach().cpu().float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        require(tuple(emission.shape) == (8, H, W), f"BAD_EMISSION_SHAPE_{month:02d}")
        if proxy_acc is None:
            proxy_acc = torch.zeros_like(emission)
        proxy_acc += emission
        proxy_sources.append(
            {
                "month": month,
                "file": str(path),
                "sha256": sha256_file(path),
            }
        )
    require(proxy_acc is not None, "PROXY_BUILD_FAILED")
    fixed_proxy = proxy_acc / float(len(EXPECTED_TRAIN_MONTHS))

    rows: list[dict[str, Any]] = []

    for entry in entries:
        year = int(entry["year"])
        month = int(entry["month"])
        path = Path(entry["file"]).expanduser().resolve()
        require(path.is_file(), f"CACHE_FILE_NOT_FOUND: {path}")
        payload = load_pt(path)

        valid = payload["valid_mask"].detach().cpu().float() > 0.5
        require(tuple(valid.shape) == (H, W), f"BAD_VALID_MASK_SHAPE: {path}")

        if year == 2022:
            emission = payload.get("emission_log_mean")
            require(isinstance(emission, torch.Tensor), f"2022_EMISSION_MISSING: {path}")
            emission_mode = "actual_month_specific"
        elif year == 2023:
            emission = fixed_proxy
            emission_mode = "fixed_2022_train_mean_proxy"
        else:
            raise RuntimeError(f"UNEXPECTED_YEAR: {year}")

        emission = torch.nan_to_num(
            emission.detach().cpu().float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        require(tuple(emission.shape) == (8, H, W), f"BAD_EMISSION_SHAPE: {path}")

        key = path.name
        rolled = module.deterministic_roll_2d(emission, key)

        support = torch.any(torch.abs(emission) > float(args.support_epsilon), dim=0)
        rolled_support = torch.any(
            torch.abs(rolled) > float(args.support_epsilon),
            dim=0,
        )

        valid_n = int(torch.count_nonzero(valid).item())
        support_n = int(torch.count_nonzero(support).item())
        rolled_support_n = int(torch.count_nonzero(rolled_support).item())
        require(
            support_n == rolled_support_n,
            f"ROLL_DID_NOT_PRESERVE_SUPPORT_COUNT: {year}-{month:02d}",
        )

        overlap_unshifted = int(torch.count_nonzero(valid & support).item())
        overlap_rolled = int(torch.count_nonzero(valid & rolled_support).item())

        coverage_unshifted = safe_fraction(overlap_unshifted, valid_n)
        coverage_rolled = safe_fraction(overlap_rolled, valid_n)
        delta = (
            coverage_rolled - coverage_unshifted
            if coverage_unshifted is not None and coverage_rolled is not None
            else None
        )

        row = {
            "year": year,
            "month": month,
            "season": str(entry.get("season")),
            "split": str(entry.get("split")),
            "analysis_group": (
                "2022_train"
                if year == 2022 and month in EXPECTED_TRAIN_MONTHS
                else "2022_validation"
                if year == 2022 and month in EXPECTED_VALIDATION_MONTHS
                else "2022_other"
                if year == 2022
                else "2023_proxy_external"
            ),
            "emission_mode": emission_mode,
            "cache_file": str(path),
            "cache_file_sha256": sha256_file(path),
            "roll_key": key,
            "target_valid_pixels": valid_n,
            "processed_emission_support_pixels": support_n,
            "processed_emission_support_fraction_global": safe_fraction(
                support_n, H * W
            ),
            "target_valid_overlap_unshifted_pixels": overlap_unshifted,
            "target_valid_overlap_rolled_pixels": overlap_rolled,
            "target_valid_coverage_unshifted": coverage_unshifted,
            "target_valid_coverage_rolled": coverage_rolled,
            "coverage_delta_rolled_minus_unshifted": delta,
            "absolute_coverage_delta": abs(delta) if delta is not None else None,
            "support_jaccard_global": jaccard(support, rolled_support),
            "support_jaccard_within_target_valid": finite_or_none(
                jaccard(support, rolled_support, mask=valid)
            ),
            "target_coverage_defined": coverage_unshifted is not None,
        }
        rows.append(row)

    groups = {
        "2022_train": [r for r in rows if r["analysis_group"] == "2022_train"],
        "2022_validation": [
            r for r in rows if r["analysis_group"] == "2022_validation"
        ],
        "2022_all_real_emission": [r for r in rows if int(r["year"]) == 2022],
        "2023_proxy_external": [r for r in rows if int(r["year"]) == 2023],
    }
    summaries = {name: summarize_rows(group) for name, group in groups.items()}

    require(
        summaries["2022_validation"]["n_months_with_defined_target_coverage"]
        == len(groups["2022_validation"]),
        "2022_VALIDATION_HAS_UNDEFINED_TARGET_COVERAGE",
    )
    require(
        summaries["2023_proxy_external"]["n_months_with_defined_target_coverage"]
        == len(groups["2023_proxy_external"]),
        "2023_PROXY_HAS_UNDEFINED_TARGET_COVERAGE",
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    json_path = output_dir / "support_geometry_audit.json"
    csv_path = output_dir / "support_geometry_monthly.csv"
    txt_path = output_dir / "support_geometry_summary.txt"

    report = {
        "artifact_name": "clean_meanstate_support_geometry_audit",
        "artifact_version": 1,
        "inputs": {
            "cache_dir": str(cache_dir),
            "monthly_manifest": str(manifest_path),
            "monthly_manifest_sha256": sha256_file(manifest_path),
            "training_script": str(training_script),
            "training_script_sha256": sha256_file(training_script),
            "support_epsilon": float(args.support_epsilon),
            "fixed_proxy_sources": proxy_sources,
        },
        "support_definition": (
            "A cell is supported when at least one of the eight processed emission-prior "
            "channels has abs(value) > support_epsilon."
        ),
        "interpretation_boundary": (
            "This is a non-zero processed-emission support audit. Because zero can represent "
            "either an unmapped source-row cell or a genuinely zero processed emission, this "
            "report cannot reconstruct an independent source-row support mask."
        ),
        "monthly": rows,
        "summaries": summaries,
        "verification": {
            "all_21_monthly_groups_audited": len(rows) == 21,
            "all_2022_required_train_validation_months_present": True,
            "all_12_2023_proxy_months_present": True,
            "exact_training_roll_function_imported": True,
            "roll_preserved_global_support_pixel_count_every_month": True,
            "undefined_coverage_serialized_as_null": True,
            "all_metrics_finite_where_defined": True,
            "overall_pass": True,
        },
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=True, indent=2, allow_nan=False)

    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    val = summaries["2022_validation"]
    ext = summaries["2023_proxy_external"]
    lines = [
        "CLEAN_MEANSTATE_SUPPORT_GEOMETRY_AUDIT_COMPLETE",
        f"OUTPUT_JSON={json_path}",
        f"OUTPUT_CSV={csv_path}",
        f"OUTPUT_TXT={txt_path}",
        f"TRAINING_SCRIPT_SHA256={sha256_file(training_script)}",
        f"MONTHLY_MANIFEST_SHA256={sha256_file(manifest_path)}",
        (
            "2022_VALIDATION_MEAN_COVERAGE_UNSHIFTED="
            f"{val['mean_coverage_unshifted']:.8f}"
        ),
        (
            "2022_VALIDATION_MEAN_COVERAGE_ROLLED="
            f"{val['mean_coverage_rolled']:.8f}"
        ),
        (
            "2022_VALIDATION_MEAN_ABS_COVERAGE_DELTA="
            f"{val['mean_absolute_coverage_delta']:.8f}"
        ),
        (
            "2022_VALIDATION_MAX_ABS_COVERAGE_DELTA="
            f"{val['max_absolute_coverage_delta']:.8f}"
        ),
        (
            "2023_PROXY_MEAN_COVERAGE_UNSHIFTED="
            f"{ext['mean_coverage_unshifted']:.8f}"
        ),
        (
            "2023_PROXY_MEAN_COVERAGE_ROLLED="
            f"{ext['mean_coverage_rolled']:.8f}"
        ),
        (
            "2023_PROXY_MEAN_ABS_COVERAGE_DELTA="
            f"{ext['mean_absolute_coverage_delta']:.8f}"
        ),
        (
            "2023_PROXY_MAX_ABS_COVERAGE_DELTA="
            f"{ext['max_absolute_coverage_delta']:.8f}"
        ),
        (
            "2022_VALIDATION_MONTHS_ABS_DELTA_GE_0P05="
            f"{val['months_abs_delta_ge_0p05']}"
        ),
        (
            "2023_PROXY_MONTHS_ABS_DELTA_GE_0P05="
            f"{ext['months_abs_delta_ge_0p05']}"
        ),
        "SCIENTIFIC_DECISION_AUTOMATICALLY_ESTABLISHED=False",
        "SUPPORT_PRESERVING_CONTROL_REQUIRED=TO_BE_DECIDED_AFTER_REVIEW",
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
