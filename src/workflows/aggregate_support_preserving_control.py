# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
# -*- coding: utf-8 -*-
"""
63_aggregate_and_seal_meanstate_support_control.py

Aggregate the three formal support-preserving mean-state evaluations and compare
them, seed by seed, against the previously sealed correct-joint and cyclic-roll
controls.

This script is read-only with respect to all input reports. It creates a new,
versioned closure directory and never overwrites existing output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


EXPECTED_SEEDS = [42, 43, 44]
EXPECTED_SUPPORT_EXPERIMENT = "mean_emis_meteo_clim_support_preserving"
EXPECTED_CLOSURE_ARTIFACT = "clean_meanstate_four_config_closure"
EXPECTED_EVAL_ARTIFACT = "s2_meanstate_support_preserving_formal_evaluation"
EXPECTED_MANIFEST_SHA256 = "f397c74f4bcdcef477a813524e9f9399989e519a8407affbb9e4ce3d491d2393"
METRICS = ["log_r2", "log_rmse", "log_bias"]
SPLITS = {
    "val_2022": "eval_2022_validation",
    "external_2023": "eval_2023_external",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"INPUT_NOT_FOUND: {path}")
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    require(isinstance(obj, dict), f"JSON_ROOT_NOT_OBJECT: {path}")
    return obj


def finite_float(value: Any, label: str) -> float:
    x = float(value)
    require(math.isfinite(x), f"NONFINITE_{label}: {value}")
    return x


def summary(values_by_seed: dict[str, float]) -> dict[str, Any]:
    require(sorted(int(k) for k in values_by_seed) == EXPECTED_SEEDS, "SEED_SET_MISMATCH")
    values = [finite_float(values_by_seed[str(seed)], f"seed{seed}") for seed in EXPECTED_SEEDS]
    return {
        "per_seed": {str(seed): value for seed, value in zip(EXPECTED_SEEDS, values)},
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values),
        "minimum": min(values),
        "maximum": max(values),
        "all_positive": all(v > 0.0 for v in values),
        "all_negative": all(v < 0.0 for v in values),
        "mixed_sign": not all(v > 0.0 for v in values) and not all(v < 0.0 for v in values),
    }


def metric_block_from_reports(
    reports_by_seed: dict[int, dict[str, Any]],
    split_key: str,
) -> dict[str, Any]:
    block: dict[str, Any] = {}
    for metric in METRICS:
        per_seed = {
            str(seed): finite_float(
                reports_by_seed[seed][split_key]["global"][metric],
                f"{split_key}_{metric}_seed{seed}",
            )
            for seed in EXPECTED_SEEDS
        }
        block[metric] = summary(per_seed)
    return block


def contrast(
    left_per_seed: dict[str, Any],
    right_per_seed: dict[str, Any],
) -> dict[str, Any]:
    values = {
        str(seed): finite_float(left_per_seed[str(seed)], f"left_seed{seed}")
        - finite_float(right_per_seed[str(seed)], f"right_seed{seed}")
        for seed in EXPECTED_SEEDS
    }
    return summary(values)


def metric_per_seed(
    experiment_block: dict[str, Any],
    split_key: str,
    metric: str,
) -> dict[str, Any]:
    return dict(experiment_block[split_key][metric]["per_seed"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--closure-json", required=True)
    parser.add_argument(
        "--support-eval",
        required=True,
        action="append",
        help="Repeat exactly three times for seeds 42, 43 and 44.",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    closure_path = Path(args.closure_json).expanduser().resolve()
    support_paths = [Path(p).expanduser().resolve() for p in args.support_eval]
    output_dir = Path(args.output_dir).expanduser().resolve()

    require(len(support_paths) == 3, f"EXPECTED_THREE_SUPPORT_EVALS_GOT={len(support_paths)}")
    require(not output_dir.exists(), f"STOP_OUTPUT_DIR_ALREADY_EXISTS: {output_dir}")

    closure = load_json(closure_path)
    require(closure.get("artifact_name") == EXPECTED_CLOSURE_ARTIFACT, "CLOSURE_ARTIFACT_MISMATCH")
    require(int(closure.get("artifact_version", -1)) == 1, "CLOSURE_VERSION_MISMATCH")
    require(closure.get("verification", {}).get("overall_pass") is True, "SOURCE_CLOSURE_NOT_PASS")
    require(
        closure.get("shared_monthly_manifest_sha256") == EXPECTED_MANIFEST_SHA256,
        "SOURCE_CLOSURE_MANIFEST_HASH_MISMATCH",
    )

    experiments = closure.get("experiments", {})
    require("mean_emis_meteo_clim" in experiments, "CORRECT_JOINT_MISSING")
    require("mean_emis_meteo_clim_shuffled" in experiments, "CYCLIC_CONTROL_MISSING")
    joint = experiments["mean_emis_meteo_clim"]
    cyclic = experiments["mean_emis_meteo_clim_shuffled"]

    reports_by_seed: dict[int, dict[str, Any]] = {}
    support_input_records: list[dict[str, Any]] = []

    shared_gatep2a_hashes: set[str] = set()
    shared_training_script_hashes: set[str] = set()
    shared_proxy_tensor_hashes: set[str] = set()
    shared_manifest_hashes: set[str] = set()

    for path in support_paths:
        report = load_json(path)
        require(report.get("artifact_name") == EXPECTED_EVAL_ARTIFACT, f"EVAL_ARTIFACT_MISMATCH: {path}")
        require(int(report.get("artifact_version", -1)) == 1, f"EVAL_VERSION_MISMATCH: {path}")

        seed = int(report.get("seed", -1))
        require(seed in EXPECTED_SEEDS, f"UNEXPECTED_SEED={seed}: {path}")
        require(seed not in reports_by_seed, f"DUPLICATE_SEED={seed}")

        exp = report.get("experiment", {})
        require(exp.get("name") == EXPECTED_SUPPORT_EXPERIMENT, f"EXPERIMENT_MISMATCH_SEED={seed}")
        require(exp.get("uses_emission") is True, f"USES_EMISSION_FALSE_SEED={seed}")
        require(exp.get("uses_meteo") is True, f"USES_METEO_FALSE_SEED={seed}")
        require(
            exp.get("support_preserving_permutation") is True,
            f"SUPPORT_PRESERVING_FALSE_SEED={seed}",
        )
        require(int(exp.get("emission_channels", -1)) == 8, f"EMISSION_CHANNELS_MISMATCH_SEED={seed}")
        require(int(exp.get("meteo_channels", -1)) == 5, f"METEO_CHANNELS_MISMATCH_SEED={seed}")

        verification = report.get("verification", {})
        require(verification.get("overall_pass") is True, f"EVAL_OVERALL_NOT_PASS_SEED={seed}")
        require(
            verification.get("training_audit_provenance_pass") is True,
            f"TRAINING_AUDIT_PROVENANCE_FAIL_SEED={seed}",
        )
        require(
            verification.get("support_preserving_contract_pass") is True,
            f"SUPPORT_CONTRACT_FAIL_SEED={seed}",
        )
        require(
            verification.get("checkpoint_strict_load_pass") is True,
            f"CHECKPOINT_LOAD_FAIL_SEED={seed}",
        )
        require(
            verification.get("2022_best_metric_reproduction_pass") is True,
            f"BEST_METRIC_REPRODUCTION_FAIL_SEED={seed}",
        )
        require(
            verification.get("2023_all_12_months_evaluated") is True,
            f"2023_MONTH_COUNT_FAIL_SEED={seed}",
        )
        require(
            verification.get("2023_emission_proxy_used") is True,
            f"2023_PROXY_FLAG_FAIL_SEED={seed}",
        )

        design = report.get("evaluation_design", {})
        require(
            design.get("2022_status") == "held-out model-selection validation",
            f"2022_STATUS_MISMATCH_SEED={seed}",
        )
        require(design.get("2023_status") == "proxy_external", f"2023_STATUS_MISMATCH_SEED={seed}")
        require(design.get("emission_proxy_used") is True, f"PROXY_USED_FALSE_SEED={seed}")
        require(
            design.get("support_preserving_permutation") is True,
            f"DESIGN_SUPPORT_FLAG_FALSE_SEED={seed}",
        )

        provenance = report.get("provenance", {})
        manifest_hash = str(provenance.get("monthly_manifest_sha256", ""))
        gatep2a_hash = str(provenance.get("gatep2a_report_sha256", ""))
        training_hash = str(provenance.get("training_script_sha256", ""))
        proxy_hash = str(report.get("proxy_emission", {}).get("proxy_tensor_sha256", ""))

        require(manifest_hash == EXPECTED_MANIFEST_SHA256, f"MANIFEST_HASH_MISMATCH_SEED={seed}")
        require(len(gatep2a_hash) == 64, f"GATEP2A_HASH_INVALID_SEED={seed}")
        require(len(training_hash) == 64, f"TRAINING_HASH_INVALID_SEED={seed}")
        require(len(proxy_hash) == 64, f"PROXY_HASH_INVALID_SEED={seed}")

        shared_manifest_hashes.add(manifest_hash)
        shared_gatep2a_hashes.add(gatep2a_hash)
        shared_training_script_hashes.add(training_hash)
        shared_proxy_tensor_hashes.add(proxy_hash)

        for split_key in SPLITS.values():
            global_block = report.get(split_key, {}).get("global", {})
            for metric in METRICS:
                finite_float(global_block.get(metric), f"{split_key}_{metric}_seed{seed}")

        reports_by_seed[seed] = report
        support_input_records.append(
            {
                "seed": seed,
                "path": str(path),
                "sha256": sha256_file(path),
                "checkpoint_sha256": report.get("checkpoint", {}).get("sha256"),
                "selected_epoch": report.get("checkpoint", {}).get("selected_epoch"),
            }
        )

    require(sorted(reports_by_seed) == EXPECTED_SEEDS, "EXACT_SEED_SET_FAIL")
    require(len(shared_manifest_hashes) == 1, "SUPPORT_REPORT_MANIFEST_HASHES_DIFFER")
    require(len(shared_gatep2a_hashes) == 1, "GATEP2A_HASHES_DIFFER")
    require(len(shared_training_script_hashes) == 1, "TRAINING_SCRIPT_HASHES_DIFFER")
    require(len(shared_proxy_tensor_hashes) == 1, "PROXY_TENSOR_HASHES_DIFFER")

    support_aggregate = {
        split_label: metric_block_from_reports(reports_by_seed, split_key)
        for split_label, split_key in SPLITS.items()
    }

    contrasts: dict[str, Any] = {}
    split_map = {
        "val_2022": "eval_2022_validation",
        "external_2023": "eval_2023_external",
    }
    for split_label, closure_split in split_map.items():
        for metric in METRICS:
            support_per_seed = support_aggregate[split_label][metric]["per_seed"]
            joint_per_seed = metric_per_seed(joint, closure_split, metric)
            cyclic_per_seed = metric_per_seed(cyclic, closure_split, metric)

            contrasts[f"joint_minus_support_{split_label}_{metric}"] = contrast(
                joint_per_seed,
                support_per_seed,
            )
            contrasts[f"support_minus_cyclic_{split_label}_{metric}"] = contrast(
                support_per_seed,
                cyclic_per_seed,
            )

    jms_2022 = contrasts["joint_minus_support_val_2022_log_r2"]
    jms_2023 = contrasts["joint_minus_support_external_2023_log_r2"]
    smc_2022_r2 = contrasts["support_minus_cyclic_val_2022_log_r2"]
    smc_2022_rmse = contrasts["support_minus_cyclic_val_2022_log_rmse"]
    smc_2023_r2 = contrasts["support_minus_cyclic_external_2023_log_r2"]
    smc_2023_rmse = contrasts["support_minus_cyclic_external_2023_log_rmse"]

    scientific_decision = {
        "within_support_value_order_advantage_stable_2022": jms_2022["all_positive"],
        "within_support_value_order_advantage_stable_2023": jms_2023["all_positive"],
        "within_support_value_order_advantage_stable_across_both_splits": (
            jms_2022["all_positive"] and jms_2023["all_positive"]
        ),
        "support_preserving_better_than_cyclic_2022_all_seeds_r2": smc_2022_r2["all_positive"],
        "support_preserving_better_than_cyclic_2022_all_seeds_rmse": smc_2022_rmse["all_negative"],
        "support_preserving_better_than_cyclic_2023_all_seeds_r2": smc_2023_r2["all_positive"],
        "support_preserving_better_than_cyclic_2023_all_seeds_rmse": smc_2023_rmse["all_negative"],
        "primary_conclusion": (
            "The correct-joint model did not show a seed-consistent advantage over the "
            "support-preserving permutation control in either 2022 validation or the "
            "2023 proxy external test. Therefore, the current experiments do not support "
            "a robust independent contribution from the exact within-support spatial "
            "ordering of the processed 8-D emission vectors."
        ),
        "cyclic_control_conclusion": (
            "The support-preserving control outperformed the cyclic-roll control across "
            "all three seeds in 2022 validation, but this contrast was not directionally "
            "stable across all three seeds in the 2023 proxy external test. The cyclic-roll "
            "control must therefore be reported as a support-disrupting strong negative "
            "control rather than as a clean test of emission-value spatial ordering."
        ),
        "meanstate_scope_conclusion": (
            "These findings narrow the interpretation of the shuffled-control contrast; "
            "they do not invalidate the completed mean-state framework, the meteorology-only "
            "results, the emission-only results, or the correct-joint predictive estimates."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    json_path = output_dir / "meanstate_support_control_closure_v01.json"
    txt_path = output_dir / "meanstate_support_control_closure_v01.txt"
    csv_path = output_dir / "meanstate_support_control_per_seed_v01.csv"

    closure_report = {
        "artifact_name": "clean_meanstate_support_control_closure",
        "artifact_version": 1,
        "source_four_config_closure": {
            "path": str(closure_path),
            "sha256": sha256_file(closure_path),
            "artifact_name": closure.get("artifact_name"),
            "artifact_version": closure.get("artifact_version"),
        },
        "support_preserving_input_reports": sorted(
            support_input_records,
            key=lambda item: item["seed"],
        ),
        "shared_provenance": {
            "monthly_manifest_sha256": next(iter(shared_manifest_hashes)),
            "gatep2a_report_sha256": next(iter(shared_gatep2a_hashes)),
            "training_script_sha256": next(iter(shared_training_script_hashes)),
            "proxy_tensor_sha256": next(iter(shared_proxy_tensor_hashes)),
        },
        "support_preserving_aggregate_global": support_aggregate,
        "paired_contrasts": contrasts,
        "scientific_decision": scientific_decision,
        "verification": {
            "source_four_config_closure_pass": True,
            "exact_support_seed_set_pass": True,
            "all_support_reports_overall_pass": True,
            "shared_monthly_manifest_hash_pass": True,
            "shared_gatep2a_hash_pass": True,
            "shared_training_script_hash_pass": True,
            "shared_proxy_tensor_hash_pass": True,
            "all_metrics_finite": True,
            "overall_pass": True,
        },
        "reporting_boundary": (
            "Seeds 42, 43 and 44 provide descriptive stability only; no statistical "
            "significance is established. The 2022 split is held-out model-selection "
            "validation. All 2023 emission-input results are proxy external consistency "
            "tests because real 2023 emission-prior channels are unavailable. The "
            "support-preserving control conditions on the processed non-zero support mask "
            "and tests only the within-support ordering of complete 8-D vectors. It does "
            "not validate the native inventory or independently reconstruct source-row support."
        ),
        "meanstate_support_control_line_closed": True,
        "additional_meanstate_shuffle_training_required": False,
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(closure_report, f, ensure_ascii=True, indent=2, allow_nan=False)

    rows: list[dict[str, Any]] = []
    for seed in EXPECTED_SEEDS:
        for split_label, closure_split in split_map.items():
            row = {
                "seed": seed,
                "split": split_label,
            }
            for metric in METRICS:
                support_value = support_aggregate[split_label][metric]["per_seed"][str(seed)]
                joint_value = joint[closure_split][metric]["per_seed"][str(seed)]
                cyclic_value = cyclic[closure_split][metric]["per_seed"][str(seed)]
                row[f"joint_{metric}"] = joint_value
                row[f"support_preserving_{metric}"] = support_value
                row[f"cyclic_{metric}"] = cyclic_value
                row[f"joint_minus_support_{metric}"] = joint_value - support_value
                row[f"support_minus_cyclic_{metric}"] = support_value - cyclic_value
            rows.append(row)

    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    def fmt(block: dict[str, Any]) -> str:
        return f"{block['mean']:.8f}+/-{block['sample_sd']:.8f}"

    lines = [
        "MEANSTATE_SUPPORT_CONTROL_CLOSURE_PASS",
        f"SEEDS={EXPECTED_SEEDS}",
        f"SHARED_MONTHLY_MANIFEST_SHA256={next(iter(shared_manifest_hashes))}",
        f"SUPPORT_VAL_2022_LOG_R2_MEAN_SD={fmt(support_aggregate['val_2022']['log_r2'])}",
        f"SUPPORT_VAL_2022_LOG_RMSE_MEAN_SD={fmt(support_aggregate['val_2022']['log_rmse'])}",
        f"SUPPORT_VAL_2022_LOG_BIAS_MEAN_SD={fmt(support_aggregate['val_2022']['log_bias'])}",
        f"SUPPORT_EXTERNAL_2023_LOG_R2_MEAN_SD={fmt(support_aggregate['external_2023']['log_r2'])}",
        f"SUPPORT_EXTERNAL_2023_LOG_RMSE_MEAN_SD={fmt(support_aggregate['external_2023']['log_rmse'])}",
        f"SUPPORT_EXTERNAL_2023_LOG_BIAS_MEAN_SD={fmt(support_aggregate['external_2023']['log_bias'])}",
        f"JOINT_MINUS_SUPPORT_VAL_2022_LOG_R2_MEAN_SD={fmt(jms_2022)}",
        f"JOINT_MINUS_SUPPORT_EXTERNAL_2023_LOG_R2_MEAN_SD={fmt(jms_2023)}",
        f"JOINT_MINUS_SUPPORT_VAL_2022_ALL_POSITIVE={jms_2022['all_positive']}",
        f"JOINT_MINUS_SUPPORT_EXTERNAL_2023_ALL_POSITIVE={jms_2023['all_positive']}",
        f"SUPPORT_MINUS_CYCLIC_VAL_2022_LOG_R2_MEAN_SD={fmt(smc_2022_r2)}",
        f"SUPPORT_MINUS_CYCLIC_EXTERNAL_2023_LOG_R2_MEAN_SD={fmt(smc_2023_r2)}",
        f"SUPPORT_MINUS_CYCLIC_VAL_2022_ALL_POSITIVE={smc_2022_r2['all_positive']}",
        f"SUPPORT_MINUS_CYCLIC_EXTERNAL_2023_ALL_POSITIVE={smc_2023_r2['all_positive']}",
        "WITHIN_SUPPORT_VALUE_ORDER_ADVANTAGE_STABLE_ACROSS_BOTH_SPLITS=False",
        "DESCRIPTIVE_STABILITY_SUMMARY_ONLY=True",
        "STATISTICAL_SIGNIFICANCE_ESTABLISHED=False",
        "ADDITIONAL_MEANSTATE_SHUFFLE_TRAINING_REQUIRED=False",
        "MEANSTATE_SUPPORT_CONTROL_LINE_CLOSED=True",
        f"OUTPUT_JSON={json_path}",
        f"OUTPUT_CSV={csv_path}",
        f"OUTPUT_TXT={txt_path}",
    ]

    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
