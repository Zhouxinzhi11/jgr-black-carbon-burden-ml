# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
# -*- coding: utf-8 -*-
r"""
53_aggregate_s2_meanstate_clean_three_seed.py

Read-only three-seed aggregation for clean mean-state formal evaluations.

Expected inputs:
- exactly three eval_report.json files;
- seeds exactly 42, 43, 44;
- same experiment, cache, monthly manifest, split and evaluated payload hashes;
- all reports produced by 52_eval_s2_meanstate_clean_formal_run.py;
- all verification flags pass.

Outputs:
- aggregate_report.json
- global_seed_metrics.csv
- manuscript_row.csv
- monthly_2023_summary.csv
- seasonal_2023_summary.csv
- regional_2023_summary.csv
- aggregate_summary.txt

All means are arithmetic means across the three pre-specified seeds.
All SD values are sample standard deviations across seeds (ddof=1).
These are descriptive stability summaries, not significance tests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any, Iterable, Sequence


EXPECTED_SEEDS = [42, 43, 44]
ALLOWED_EXPERIMENTS = {
    "mean_emis_only",
    "mean_meteo_clim_only",
    "mean_emis_meteo_clim",
    "mean_emis_meteo_clim_shuffled",
}
EXPECTED_VALIDATION_MONTHS = [11, 12]
EXPECTED_EXTERNAL_MONTHS = list(range(1, 13))
EXPECTED_SEASONS = ["DJF", "MAM", "JJA", "SON"]
EXPECTED_REGIONS = [
    "North_America",
    "Europe",
    "East_Asia",
    "South_Asia",
    "Africa",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def normalized_path(path: str | Path) -> str:
    return os.path.normcase(
        os.path.abspath(
            os.path.normpath(str(path))
        )
    )


def finite_float(value: Any, name: str) -> float:
    require(value is not None, f"{name} is None")
    result = float(value)
    require(math.isfinite(result), f"{name} is non-finite: {value}")
    return result


def sample_summary(values: Sequence[float]) -> dict[str, float]:
    require(len(values) == 3, f"Expected three values, got {len(values)}")
    finite = [finite_float(v, "sample value") for v in values]
    return {
        "mean": statistics.mean(finite),
        "sample_sd": statistics.stdev(finite),
        "minimum": min(finite),
        "maximum": max(finite),
    }


def metric_block(
    reports: Sequence[dict[str, Any]],
    section: str,
    subgroup: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric in ["log_r2", "log_rmse", "log_bias"]:
        values: list[float] = []
        for report in reports:
            node = report[section]
            if subgroup is None:
                node = node["global"]
            else:
                node = node[subgroup]
            values.append(
                finite_float(
                    node[metric],
                    f"{section}.{subgroup}.{metric}",
                )
            )
        result[metric] = {
            "per_seed": {
                str(int(report["seed"])): value
                for report, value in zip(reports, values)
            },
            **sample_summary(values),
        }
    return result


def per_group_lookup(
    report: dict[str, Any],
    section: str,
    month: int,
) -> dict[str, Any]:
    matches = [
        row
        for row in report[section]["per_group"]
        if int(row["month"]) == month
    ]
    require(
        len(matches) == 1,
        f"Expected one month={month} row in {section}, found {len(matches)}",
    )
    return matches[0]


def write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate three audited clean mean-state formal evaluation reports."
        )
    )
    parser.add_argument(
        "--eval-report",
        action="append",
        required=True,
        help="Pass exactly three times, for seeds 42, 43 and 44.",
    )
    parser.add_argument("--expected-experiment", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    require(
        args.expected_experiment in ALLOWED_EXPERIMENTS,
        f"Unsupported experiment: {args.expected_experiment}",
    )
    require(
        len(args.eval_report) == 3,
        f"Exactly three --eval-report arguments are required; got {len(args.eval_report)}",
    )

    input_paths = [
        Path(value).expanduser().resolve()
        for value in args.eval_report
    ]
    output_dir = Path(args.output_dir).expanduser().resolve()

    require(
        not output_dir.exists(),
        "Aggregation output directory already exists. This script never overwrites.",
    )
    for path in input_paths:
        require(
            path.is_file() and path.stat().st_size > 0,
            f"Missing or empty evaluation report: {path}",
        )

    reports = [load_json(path) for path in input_paths]
    paired = sorted(
        zip(input_paths, reports),
        key=lambda pair: int(pair[1].get("seed", -1)),
    )
    input_paths = [pair[0] for pair in paired]
    reports = [pair[1] for pair in paired]

    seeds = [int(report.get("seed", -1)) for report in reports]
    require(seeds == EXPECTED_SEEDS, f"Seeds must be {EXPECTED_SEEDS}; got {seeds}")

    for path, report in zip(input_paths, reports):
        require(
            report.get("artifact_name")
            == "s2_meanstate_clean_formal_evaluation_generic",
            f"Report was not produced by the generic evaluator: {path}",
        )
        require(
            int(report.get("artifact_version", -1)) == 1,
            f"Unexpected artifact version: {path}",
        )
        require(
            report.get("experiment", {}).get("name")
            == args.expected_experiment,
            f"Experiment mismatch: {path}",
        )
        verification = report.get("verification", {})
        require(
            verification.get("overall_pass") is True,
            f"Evaluation overall_pass is not true: {path}",
        )
        require(
            verification.get("training_audit_provenance_pass") is True,
            f"Training audit provenance failed: {path}",
        )
        require(
            verification.get("checkpoint_strict_load_pass") is True,
            f"Checkpoint strict load failed: {path}",
        )
        require(
            verification.get("2022_best_metric_reproduction_pass") is True,
            f"2022 metric reproduction failed: {path}",
        )
        require(
            verification.get("2023_all_12_months_evaluated") is True,
            f"Not all 2023 months were evaluated: {path}",
        )
        require(
            verification.get("all_reported_global_metrics_finite") is True,
            f"Global metrics are not all finite: {path}",
        )

    reference = reports[0]
    reference_experiment = reference["experiment"]
    reference_cache = normalized_path(reference["provenance"]["cache_dir"])
    reference_manifest_hash = reference["provenance"]["monthly_manifest_sha256"]
    reference_regions = reference["evaluation_design"]["regions"]

    uses_emission = bool(reference_experiment["uses_emission"])
    expected_external_type = "proxy_external" if uses_emission else "ordinary_external"
    expected_proxy_used = uses_emission

    for path, report in zip(input_paths, reports):
        require(
            report["experiment"] == reference_experiment,
            f"Experiment specification differs across seeds: {path}",
        )
        require(
            normalized_path(report["provenance"]["cache_dir"]) == reference_cache,
            f"Cache directory differs across seeds: {path}",
        )
        require(
            report["provenance"]["monthly_manifest_sha256"]
            == reference_manifest_hash,
            f"Monthly-manifest hash differs across seeds: {path}",
        )
        require(
            report["evaluation_design"]["regions"] == reference_regions,
            f"Region definitions differ across seeds: {path}",
        )
        require(
            report["evaluation_design"]["2022_validation_months"]
            == EXPECTED_VALIDATION_MONTHS,
            f"Unexpected 2022 validation months: {path}",
        )
        require(
            report["evaluation_design"]["2023_external_months"]
            == EXPECTED_EXTERNAL_MONTHS,
            f"Unexpected 2023 external months: {path}",
        )
        require(
            report["evaluation_design"]["2023_status"] == expected_external_type,
            f"Unexpected 2023 evaluation type: {path}",
        )
        require(
            bool(report["evaluation_design"]["emission_proxy_used"])
            == expected_proxy_used,
            f"Emission proxy flag mismatch: {path}",
        )

    # Verify that every seed evaluated the exact same cached monthly payloads.
    for section, months in [
        ("eval_2022_validation", EXPECTED_VALIDATION_MONTHS),
        ("eval_2023_external", EXPECTED_EXTERNAL_MONTHS),
    ]:
        ref_section = reference[section]
        for report in reports[1:]:
            require(
                int(report[section]["n_groups"]) == int(ref_section["n_groups"]),
                f"{section} n_groups differs across seeds.",
            )
            require(
                int(report[section]["n_total_valid_pixels"])
                == int(ref_section["n_total_valid_pixels"]),
                f"{section} n_total_valid_pixels differs across seeds.",
            )

        for month in months:
            ref_row = per_group_lookup(reference, section, month)
            for report in reports[1:]:
                row = per_group_lookup(report, section, month)
                require(
                    row["file_sha256"] == ref_row["file_sha256"],
                    f"{section} month {month} payload hash differs across seeds.",
                )
                require(
                    int(row["n_valid_pixels"]) == int(ref_row["n_valid_pixels"]),
                    f"{section} month {month} valid count differs across seeds.",
                )
                require(
                    int(row["n_days"]) == int(ref_row["n_days"]),
                    f"{section} month {month} day count differs across seeds.",
                )

    global_2022 = metric_block(reports, "eval_2022_validation")
    global_2023 = metric_block(reports, "eval_2023_external")

    validation_to_external_r2_change = [
        finite_float(
            report["eval_2023_external"]["global"]["log_r2"],
            "external log_r2",
        )
        - finite_float(
            report["eval_2022_validation"]["global"]["log_r2"],
            "validation log_r2",
        )
        for report in reports
    ]

    monthly_2022: dict[str, Any] = {}
    for month in EXPECTED_VALIDATION_MONTHS:
        rows = [
            per_group_lookup(report, "eval_2022_validation", month)
            for report in reports
        ]
        monthly_2022[str(month)] = {
            metric: {
                "per_seed": {
                    str(seed): finite_float(row[metric], f"2022 month {month} {metric}")
                    for seed, row in zip(seeds, rows)
                },
                **sample_summary(
                    [
                        finite_float(row[metric], f"2022 month {month} {metric}")
                        for row in rows
                    ]
                ),
            }
            for metric in ["log_r2", "log_rmse", "log_bias"]
        }
        monthly_2022[str(month)]["n_days"] = int(rows[0]["n_days"])
        monthly_2022[str(month)]["n_valid_pixels"] = int(rows[0]["n_valid_pixels"])

    monthly_2023: dict[str, Any] = {}
    for month in EXPECTED_EXTERNAL_MONTHS:
        rows = [
            per_group_lookup(report, "eval_2023_external", month)
            for report in reports
        ]
        monthly_2023[str(month)] = {
            metric: {
                "per_seed": {
                    str(seed): finite_float(row[metric], f"2023 month {month} {metric}")
                    for seed, row in zip(seeds, rows)
                },
                **sample_summary(
                    [
                        finite_float(row[metric], f"2023 month {month} {metric}")
                        for row in rows
                    ]
                ),
            }
            for metric in ["log_r2", "log_rmse", "log_bias"]
        }
        monthly_2023[str(month)]["season"] = str(rows[0]["season"])
        monthly_2023[str(month)]["n_days"] = int(rows[0]["n_days"])
        monthly_2023[str(month)]["n_valid_pixels"] = int(rows[0]["n_valid_pixels"])

    seasonal_2023: dict[str, Any] = {}
    for season in EXPECTED_SEASONS:
        seasonal_2023[season] = {}
        for metric in ["log_r2", "log_rmse", "log_bias"]:
            values = [
                finite_float(
                    report["eval_2023_external"]["season_summary"][season][metric],
                    f"2023 season {season} {metric}",
                )
                for report in reports
            ]
            seasonal_2023[season][metric] = {
                "per_seed": {
                    str(seed): value
                    for seed, value in zip(seeds, values)
                },
                **sample_summary(values),
            }
        seasonal_2023[season]["n_groups"] = int(
            reference["eval_2023_external"]["season_summary"][season]["n_groups"]
        )

    regional_2023: dict[str, Any] = {}
    for region in EXPECTED_REGIONS:
        regional_2023[region] = {}
        for metric in ["log_r2", "log_rmse", "log_bias"]:
            values = [
                finite_float(
                    report["eval_2023_external"]["region_summary"][region][metric],
                    f"2023 region {region} {metric}",
                )
                for report in reports
            ]
            regional_2023[region][metric] = {
                "per_seed": {
                    str(seed): value
                    for seed, value in zip(seeds, values)
                },
                **sample_summary(values),
            }
        regional_2023[region]["n_groups_with_valid_pixels"] = int(
            reference["eval_2023_external"]["region_summary"][region][
                "n_groups_with_valid_pixels"
            ]
        )

    per_seed_records = []
    for path, report in zip(input_paths, reports):
        seed = int(report["seed"])
        per_seed_records.append(
            {
                "seed": seed,
                "best_epoch": int(report["checkpoint"]["selected_epoch"]),
                "checkpoint_sha256": report["checkpoint"]["sha256"],
                "val_2022_log_r2": finite_float(
                    report["eval_2022_validation"]["global"]["log_r2"],
                    "val_2022_log_r2",
                ),
                "val_2022_log_rmse": finite_float(
                    report["eval_2022_validation"]["global"]["log_rmse"],
                    "val_2022_log_rmse",
                ),
                "val_2022_log_bias": finite_float(
                    report["eval_2022_validation"]["global"]["log_bias"],
                    "val_2022_log_bias",
                ),
                "external_2023_log_r2": finite_float(
                    report["eval_2023_external"]["global"]["log_r2"],
                    "external_2023_log_r2",
                ),
                "external_2023_log_rmse": finite_float(
                    report["eval_2023_external"]["global"]["log_rmse"],
                    "external_2023_log_rmse",
                ),
                "external_2023_log_bias": finite_float(
                    report["eval_2023_external"]["global"]["log_bias"],
                    "external_2023_log_bias",
                ),
                "external_minus_validation_log_r2": (
                    finite_float(
                        report["eval_2023_external"]["global"]["log_r2"],
                        "external_2023_log_r2",
                    )
                    - finite_float(
                        report["eval_2022_validation"]["global"]["log_r2"],
                        "val_2022_log_r2",
                    )
                ),
                "eval_report": str(path),
                "eval_report_sha256": sha256_file(path),
            }
        )

    report = {
        "artifact_name": "s2_meanstate_clean_three_seed_aggregate",
        "artifact_version": 1,
        "experiment": reference_experiment,
        "seeds": seeds,
        "n_seeds": 3,
        "input_reports": [
            {
                "seed": seed,
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for seed, path in zip(seeds, input_paths)
        ],
        "shared_provenance": {
            "cache_dir": reference["provenance"]["cache_dir"],
            "monthly_manifest": reference["provenance"]["monthly_manifest"],
            "monthly_manifest_sha256": reference_manifest_hash,
            "validation_months_2022": EXPECTED_VALIDATION_MONTHS,
            "external_months_2023": EXPECTED_EXTERNAL_MONTHS,
            "external_2023_type": expected_external_type,
            "emission_proxy_used": expected_proxy_used,
            "same_monthly_payload_hashes_across_seeds": True,
        },
        "per_seed": per_seed_records,
        "aggregate_global": {
            "eval_2022_validation": global_2022,
            "eval_2023_external": global_2023,
            "external_minus_validation_log_r2": {
                "per_seed": {
                    str(seed): value
                    for seed, value in zip(seeds, validation_to_external_r2_change)
                },
                **sample_summary(validation_to_external_r2_change),
            },
        },
        "monthly_2022_validation": monthly_2022,
        "monthly_2023_external": monthly_2023,
        "seasonal_2023_external": seasonal_2023,
        "regional_2023_external": regional_2023,
        "verification": {
            "exact_seed_set_pass": True,
            "all_input_evaluations_pass": True,
            "shared_experiment_and_provenance_pass": True,
            "same_monthly_payload_hashes_across_seeds": True,
            "all_aggregate_metrics_finite": True,
            "overall_pass": True,
        },
        "reporting_boundary": (
            "Values are arithmetic mean ± sample SD across the three pre-specified "
            "random seeds 42, 43 and 44. They are descriptive stability summaries, "
            "not statistical-significance estimates. The 2022 split is held-out "
            "model-selection validation. The 2023 result is ordinary external only "
            "for meteorology-only and proxy external for emission-input experiments."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "aggregate_report.json"
    seed_csv_path = output_dir / "global_seed_metrics.csv"
    manuscript_csv_path = output_dir / "manuscript_row.csv"
    monthly_csv_path = output_dir / "monthly_2023_summary.csv"
    seasonal_csv_path = output_dir / "seasonal_2023_summary.csv"
    regional_csv_path = output_dir / "regional_2023_summary.csv"
    summary_path = output_dir / "aggregate_summary.txt"

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)

    write_csv(
        seed_csv_path,
        [
            "seed",
            "best_epoch",
            "val_2022_log_r2",
            "val_2022_log_rmse",
            "val_2022_log_bias",
            "external_2023_log_r2",
            "external_2023_log_rmse",
            "external_2023_log_bias",
            "external_minus_validation_log_r2",
            "checkpoint_sha256",
            "eval_report",
            "eval_report_sha256",
        ],
        per_seed_records,
    )

    manuscript_row = {
        "experiment": args.expected_experiment,
        "input_design": reference_experiment["mapping"],
        "n_seeds": 3,
        "seed_set": "42;43;44",
        "2022_status": "held-out model-selection validation",
        "2022_log_r2_mean": global_2022["log_r2"]["mean"],
        "2022_log_r2_sample_sd": global_2022["log_r2"]["sample_sd"],
        "2022_log_rmse_mean": global_2022["log_rmse"]["mean"],
        "2022_log_rmse_sample_sd": global_2022["log_rmse"]["sample_sd"],
        "2022_log_bias_mean": global_2022["log_bias"]["mean"],
        "2022_log_bias_sample_sd": global_2022["log_bias"]["sample_sd"],
        "2023_status": expected_external_type,
        "2023_emission_proxy_used": expected_proxy_used,
        "2023_log_r2_mean": global_2023["log_r2"]["mean"],
        "2023_log_r2_sample_sd": global_2023["log_r2"]["sample_sd"],
        "2023_log_rmse_mean": global_2023["log_rmse"]["mean"],
        "2023_log_rmse_sample_sd": global_2023["log_rmse"]["sample_sd"],
        "2023_log_bias_mean": global_2023["log_bias"]["mean"],
        "2023_log_bias_sample_sd": global_2023["log_bias"]["sample_sd"],
        "interpretation": (
            "Three-seed descriptive mean ± sample SD; not a significance test."
        ),
    }
    write_csv(
        manuscript_csv_path,
        list(manuscript_row.keys()),
        [manuscript_row],
    )

    monthly_rows = []
    for month in EXPECTED_EXTERNAL_MONTHS:
        node = monthly_2023[str(month)]
        monthly_rows.append(
            {
                "month": month,
                "season": node["season"],
                "n_days": node["n_days"],
                "n_valid_pixels": node["n_valid_pixels"],
                "log_r2_mean": node["log_r2"]["mean"],
                "log_r2_sample_sd": node["log_r2"]["sample_sd"],
                "log_rmse_mean": node["log_rmse"]["mean"],
                "log_rmse_sample_sd": node["log_rmse"]["sample_sd"],
                "log_bias_mean": node["log_bias"]["mean"],
                "log_bias_sample_sd": node["log_bias"]["sample_sd"],
            }
        )
    write_csv(
        monthly_csv_path,
        list(monthly_rows[0].keys()),
        monthly_rows,
    )

    seasonal_rows = []
    for season in EXPECTED_SEASONS:
        node = seasonal_2023[season]
        seasonal_rows.append(
            {
                "season": season,
                "n_groups": node["n_groups"],
                "log_r2_mean": node["log_r2"]["mean"],
                "log_r2_sample_sd": node["log_r2"]["sample_sd"],
                "log_rmse_mean": node["log_rmse"]["mean"],
                "log_rmse_sample_sd": node["log_rmse"]["sample_sd"],
                "log_bias_mean": node["log_bias"]["mean"],
                "log_bias_sample_sd": node["log_bias"]["sample_sd"],
            }
        )
    write_csv(
        seasonal_csv_path,
        list(seasonal_rows[0].keys()),
        seasonal_rows,
    )

    regional_rows = []
    for region in EXPECTED_REGIONS:
        node = regional_2023[region]
        regional_rows.append(
            {
                "region": region,
                "n_groups_with_valid_pixels": node[
                    "n_groups_with_valid_pixels"
                ],
                "log_r2_mean": node["log_r2"]["mean"],
                "log_r2_sample_sd": node["log_r2"]["sample_sd"],
                "log_rmse_mean": node["log_rmse"]["mean"],
                "log_rmse_sample_sd": node["log_rmse"]["sample_sd"],
                "log_bias_mean": node["log_bias"]["mean"],
                "log_bias_sample_sd": node["log_bias"]["sample_sd"],
            }
        )
    write_csv(
        regional_csv_path,
        list(regional_rows[0].keys()),
        regional_rows,
    )

    summary_lines = [
        "Clean mean-state three-seed aggregation",
        f"Experiment: {args.expected_experiment}",
        "Seeds: 42, 43, 44",
        (
            "2022 held-out validation log R2: "
            f"{global_2022['log_r2']['mean']:.8f} ± "
            f"{global_2022['log_r2']['sample_sd']:.8f}"
        ),
        (
            "2022 held-out validation log RMSE: "
            f"{global_2022['log_rmse']['mean']:.8f} ± "
            f"{global_2022['log_rmse']['sample_sd']:.8f}"
        ),
        (
            "2022 held-out validation log bias: "
            f"{global_2022['log_bias']['mean']:.8f} ± "
            f"{global_2022['log_bias']['sample_sd']:.8f}"
        ),
        f"2023 status: {expected_external_type}",
        f"2023 emission proxy used: {expected_proxy_used}",
        (
            "2023 external log R2: "
            f"{global_2023['log_r2']['mean']:.8f} ± "
            f"{global_2023['log_r2']['sample_sd']:.8f}"
        ),
        (
            "2023 external log RMSE: "
            f"{global_2023['log_rmse']['mean']:.8f} ± "
            f"{global_2023['log_rmse']['sample_sd']:.8f}"
        ),
        (
            "2023 external log bias: "
            f"{global_2023['log_bias']['mean']:.8f} ± "
            f"{global_2023['log_bias']['sample_sd']:.8f}"
        ),
        (
            "External minus validation log R2: "
            f"{statistics.mean(validation_to_external_r2_change):.8f} ± "
            f"{statistics.stdev(validation_to_external_r2_change):.8f}"
        ),
        "Descriptive stability summary only: True",
        "Statistical significance established: False",
        "Overall pass: True",
    ]
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    reloaded = load_json(report_path)
    require(
        reloaded.get("verification", {}).get("overall_pass") is True,
        "Written aggregate report failed strict reload check.",
    )

    print("GATE_THREE_SEED_CLEAN_MEANSTATE_AGGREGATION_PASS")
    print(f"EXPERIMENT={args.expected_experiment}")
    print("SEEDS=[42, 43, 44]")
    print(f"AGGREGATE_REPORT={report_path}")
    print(f"MANUSCRIPT_ROW={manuscript_csv_path}")
    print("SAME_MONTHLY_PAYLOAD_HASHES_ACROSS_SEEDS=True")
    print(
        "VAL_2022_LOG_R2_MEAN_SD="
        f"{global_2022['log_r2']['mean']:.8f}±"
        f"{global_2022['log_r2']['sample_sd']:.8f}"
    )
    print(
        "VAL_2022_LOG_RMSE_MEAN_SD="
        f"{global_2022['log_rmse']['mean']:.8f}±"
        f"{global_2022['log_rmse']['sample_sd']:.8f}"
    )
    print(
        "VAL_2022_LOG_BIAS_MEAN_SD="
        f"{global_2022['log_bias']['mean']:.8f}±"
        f"{global_2022['log_bias']['sample_sd']:.8f}"
    )
    print(f"EXTERNAL_2023_TYPE={expected_external_type}")
    print(f"EXTERNAL_2023_EMISSION_PROXY_USED={expected_proxy_used}")
    print(
        "EXTERNAL_2023_LOG_R2_MEAN_SD="
        f"{global_2023['log_r2']['mean']:.8f}±"
        f"{global_2023['log_r2']['sample_sd']:.8f}"
    )
    print(
        "EXTERNAL_2023_LOG_RMSE_MEAN_SD="
        f"{global_2023['log_rmse']['mean']:.8f}±"
        f"{global_2023['log_rmse']['sample_sd']:.8f}"
    )
    print(
        "EXTERNAL_2023_LOG_BIAS_MEAN_SD="
        f"{global_2023['log_bias']['mean']:.8f}±"
        f"{global_2023['log_bias']['sample_sd']:.8f}"
    )
    print(
        "DESCRIPTIVE_STABILITY_SUMMARY_ONLY=True"
    )
    print("STATISTICAL_SIGNIFICANCE_ESTABLISHED=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
