# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
# -*- coding: utf-8 -*-
"""
54_validate_and_seal_clean_meanstate_inventory.py

Read-only validation and sealing of the four clean mean-state
three-seed aggregate reports.

This script:
1. reads every aggregate_report.json explicitly as UTF-8;
2. validates JSON structure and required provenance flags;
3. verifies the four expected experiment names and seeds [42, 43, 44];
4. computes SHA256 hashes;
5. computes paired contrasts from per-seed metrics;
6. writes a compact closure JSON/TXT without modifying source reports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


EXPECTED = [
    ("mean_meteo_clim_only", "ordinary_external", False),
    ("mean_emis_only", "proxy_external", True),
    ("mean_emis_meteo_clim", "proxy_external", True),
    ("mean_emis_meteo_clim_shuffled", "proxy_external", True),
]
EXPECTED_SEEDS = [42, 43, 44]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json_utf8(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"UTF8_DECODE_FAILED: {path}: {exc}") from exc
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"JSON_PARSE_FAILED: {path}: line={exc.lineno} col={exc.colno} msg={exc.msg}"
        ) from exc
    require(isinstance(obj, dict), f"JSON_ROOT_NOT_OBJECT: {path}")
    return obj


def sample_summary(values: list[float]) -> dict[str, Any]:
    require(len(values) == 3, f"EXPECTED_THREE_VALUES_GOT_{len(values)}")
    require(all(math.isfinite(v) for v in values), "NONFINITE_PAIRED_CONTRAST")
    return {
        "per_seed": {str(seed): value for seed, value in zip(EXPECTED_SEEDS, values)},
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values),
        "minimum": min(values),
        "maximum": max(values),
        "all_positive": all(v > 0 for v in values),
        "all_negative": all(v < 0 for v in values),
    }


def metric_by_seed(report: dict[str, Any], field: str) -> dict[int, float]:
    rows = report.get("per_seed")
    require(isinstance(rows, list) and len(rows) == 3, "INVALID_PER_SEED_BLOCK")
    out: dict[int, float] = {}
    for row in rows:
        seed = int(row["seed"])
        value = float(row[field])
        require(seed not in out, f"DUPLICATE_SEED_{seed}")
        require(math.isfinite(value), f"NONFINITE_{field}_SEED_{seed}")
        out[seed] = value
    require(sorted(out) == EXPECTED_SEEDS, f"SEED_SET_MISMATCH_{sorted(out)}")
    return out


def paired_difference(
    left: dict[str, Any],
    right: dict[str, Any],
    field: str,
) -> dict[str, Any]:
    l = metric_by_seed(left, field)
    r = metric_by_seed(right, field)
    values = [l[s] - r[s] for s in EXPECTED_SEEDS]
    return sample_summary(values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root containing paper_result_inventory_clean.",
    )
    parser.add_argument(
        "--output-stem",
        default="meanstate_four_config_closure_v01",
    )
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    inventory = root / "paper_result_inventory_clean"

    reports: dict[str, dict[str, Any]] = {}
    source_records: list[dict[str, Any]] = []
    shared_manifest_hashes: set[str] = set()

    for experiment, external_type, proxy_used in EXPECTED:
        path = inventory / f"{experiment}_three_seed" / "aggregate_report.json"
        require(path.is_file(), f"MISSING_REPORT: {path}")

        report = read_json_utf8(path)
        actual_name = report.get("experiment", {}).get("name")
        require(actual_name == experiment, f"EXPERIMENT_MISMATCH: {path}: {actual_name}")
        require(report.get("seeds") == EXPECTED_SEEDS, f"SEED_MISMATCH: {path}")
        require(report.get("n_seeds") == 3, f"N_SEEDS_MISMATCH: {path}")

        provenance = report.get("shared_provenance", {})
        require(
            provenance.get("external_2023_type") == external_type,
            f"EXTERNAL_TYPE_MISMATCH: {path}",
        )
        require(
            provenance.get("emission_proxy_used") is proxy_used,
            f"PROXY_FLAG_MISMATCH: {path}",
        )
        require(
            provenance.get("same_monthly_payload_hashes_across_seeds") is True,
            f"PAYLOAD_HASH_FLAG_FAILED: {path}",
        )

        verification = report.get("verification", {})
        require(verification.get("overall_pass") is True, f"OVERALL_PASS_FALSE: {path}")
        require(
            verification.get("all_input_evaluations_pass") is True,
            f"INPUT_EVALUATION_FLAG_FAILED: {path}",
        )
        require(
            verification.get("all_aggregate_metrics_finite") is True,
            f"FINITE_FLAG_FAILED: {path}",
        )

        manifest_hash = provenance.get("monthly_manifest_sha256")
        require(
            isinstance(manifest_hash, str) and len(manifest_hash) == 64,
            f"INVALID_MANIFEST_HASH: {path}",
        )
        shared_manifest_hashes.add(manifest_hash)

        report_hash = sha256_file(path)
        top_keys = list(report.keys())

        reports[experiment] = report
        source_records.append(
            {
                "experiment": experiment,
                "path": str(path),
                "sha256": report_hash,
                "top_level_keys": top_keys,
                "external_2023_type": external_type,
                "emission_proxy_used": proxy_used,
            }
        )

        print(
            f"REPORT_OK experiment={experiment} "
            f"sha256={report_hash} top_level_key_count={len(top_keys)}"
        )

    require(
        len(shared_manifest_hashes) == 1,
        f"MONTHLY_MANIFEST_HASH_MISMATCH: {sorted(shared_manifest_hashes)}",
    )

    meteo = reports["mean_meteo_clim_only"]
    emis = reports["mean_emis_only"]
    joint = reports["mean_emis_meteo_clim"]
    shuffled = reports["mean_emis_meteo_clim_shuffled"]

    contrasts = {
        "joint_minus_shuffled_val_2022_log_r2": paired_difference(
            joint, shuffled, "val_2022_log_r2"
        ),
        "joint_minus_shuffled_external_2023_log_r2": paired_difference(
            joint, shuffled, "external_2023_log_r2"
        ),
        "joint_minus_shuffled_val_2022_log_rmse": paired_difference(
            joint, shuffled, "val_2022_log_rmse"
        ),
        "joint_minus_shuffled_external_2023_log_rmse": paired_difference(
            joint, shuffled, "external_2023_log_rmse"
        ),
        "joint_minus_meteo_val_2022_log_r2": paired_difference(
            joint, meteo, "val_2022_log_r2"
        ),
        "joint_minus_meteo_external_2023_log_r2": paired_difference(
            joint, meteo, "external_2023_log_r2"
        ),
        "emission_minus_meteo_val_2022_log_r2": paired_difference(
            emis, meteo, "val_2022_log_r2"
        ),
        "joint_minus_emission_external_2023_log_r2": paired_difference(
            joint, emis, "external_2023_log_r2"
        ),
    }

    closure = {
        "artifact_name": "clean_meanstate_four_config_closure",
        "artifact_version": 1,
        "source_reports": source_records,
        "shared_monthly_manifest_sha256": next(iter(shared_manifest_hashes)),
        "experiments": {
            name: report["aggregate_global"]
            for name, report in reports.items()
        },
        "paired_contrasts": contrasts,
        "verification": {
            "four_expected_reports_present": True,
            "all_reports_valid_utf8_json": True,
            "exact_seed_set_pass": True,
            "all_source_overall_pass": True,
            "shared_monthly_manifest_hash_pass": True,
            "overall_pass": True,
        },
        "reporting_boundary": (
            "Three pre-specified seeds provide descriptive stability only. "
            "The 2022 split is held-out model-selection validation. "
            "The 2023 meteorology-only result is ordinary external, while "
            "emission-input results are proxy external tests."
        ),
    }

    out_json = inventory / f"{args.output_stem}.json"
    out_txt = inventory / f"{args.output_stem}.txt"
    require(not out_json.exists(), f"STOP_OUTPUT_ALREADY_EXISTS: {out_json}")
    require(not out_txt.exists(), f"STOP_OUTPUT_ALREADY_EXISTS: {out_txt}")

    out_json.write_text(
        json.dumps(closure, ensure_ascii=True, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    c1 = contrasts["joint_minus_shuffled_val_2022_log_r2"]
    c2 = contrasts["joint_minus_shuffled_external_2023_log_r2"]
    lines = [
        "CLEAN_MEANSTATE_FOUR_CONFIG_CLOSURE_PASS",
        f"OUTPUT_JSON={out_json}",
        f"OUTPUT_TXT={out_txt}",
        f"SHARED_MONTHLY_MANIFEST_SHA256={closure['shared_monthly_manifest_sha256']}",
        (
            "JOINT_MINUS_SHUFFLED_VAL_2022_LOG_R2_MEAN_SD="
            f"{c1['mean']:.8f}+/-{c1['sample_sd']:.8f}"
        ),
        (
            "JOINT_MINUS_SHUFFLED_EXTERNAL_2023_LOG_R2_MEAN_SD="
            f"{c2['mean']:.8f}+/-{c2['sample_sd']:.8f}"
        ),
        f"JOINT_MINUS_SHUFFLED_VAL_2022_ALL_POSITIVE={c1['all_positive']}",
        f"JOINT_MINUS_SHUFFLED_EXTERNAL_2023_ALL_POSITIVE={c2['all_positive']}",
        "DESCRIPTIVE_STABILITY_SUMMARY_ONLY=True",
        "STATISTICAL_SIGNIFICANCE_ESTABLISHED=False",
    ]
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
