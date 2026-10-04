# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
# -*- coding: utf-8 -*-
"""
80_aggregate_and_seal_meanstate_binary_support_gateS8.py

Gate S8F v2 three-seed aggregation and paired comparison for the binary-support
 meteorology mean-state control.

Inputs:
- three Gate-S8E eval_report.json files (seeds 42, 43, 44);
- existing clean four-configuration closure;
- existing support-preserving control closure.

v2 keeps the Gate-P3/S8A raw-byte proxy hash separate from the existing
support-control closure canonical (dtype+shape+bytes) proxy hash.

The script computes descriptive three-seed mean/sample-SD and predeclared
paired contrasts only. It does NOT define any post-hoc equivalence threshold,
does NOT perform significance testing, and does NOT overwrite outputs.
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
EXPECTED_BINARY_EXPERIMENT = "mean_binary_support_meteo_clim"
EXPECTED_BINARY_ARTIFACT = "s2_meanstate_binary_support_formal_evaluation_gateS8E"
EXPECTED_FOUR_CLOSURE = "clean_meanstate_four_config_closure"
EXPECTED_SUPPORT_CLOSURE = "clean_meanstate_support_control_closure"
EXPECTED_MANIFEST_SHA256 = "f397c74f4bcdcef477a813524e9f9399989e519a8407affbb9e4ce3d491d2393"
EXPECTED_PROXY_SHA256 = "283867292ed0b51ebce073db35940c4b091d23cacba8ca929f5248982bb0d749"
EXPECTED_SUPPORT_CONTROL_PROXY_CANONICAL_SHA256 = "5ebcb5ff868dcca5fe02bed6a6b4ea5a1a11f8ae60a71c36b3acea82f0052fe2"
EXPECTED_PROXY_SUPPORT_CELLS = 42949
METRICS = ["log_r2", "log_rmse", "log_bias"]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    require(isinstance(obj, dict), f"JSON_ROOT_NOT_OBJECT: {path}")
    return obj


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def finite_float(value: Any, name: str) -> float:
    require(value is not None, f"{name}_IS_NONE")
    number = float(value)
    require(math.isfinite(number), f"NONFINITE_{name}: {value}")
    return number


def summarize(per_seed: dict[int, float]) -> dict[str, Any]:
    require(sorted(per_seed) == EXPECTED_SEEDS, f"SEED_SET_MISMATCH: {sorted(per_seed)}")
    values = [float(per_seed[s]) for s in EXPECTED_SEEDS]
    for seed, value in zip(EXPECTED_SEEDS, values):
        finite_float(value, f"metric_seed{seed}")
    return {
        "per_seed": {str(seed): per_seed[seed] for seed in EXPECTED_SEEDS},
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values),
        "minimum": min(values),
        "maximum": max(values),
        "all_positive": all(v > 0 for v in values),
        "all_negative": all(v < 0 for v in values),
        "mixed_sign": any(v > 0 for v in values) and any(v < 0 for v in values),
    }


def paired_difference(
    lhs: dict[int, float],
    rhs: dict[int, float],
) -> dict[str, Any]:
    require(sorted(lhs) == EXPECTED_SEEDS, "LHS_SEED_SET_MISMATCH")
    require(sorted(rhs) == EXPECTED_SEEDS, "RHS_SEED_SET_MISMATCH")
    return summarize({seed: float(lhs[seed]) - float(rhs[seed]) for seed in EXPECTED_SEEDS})


def closure_metric_per_seed(
    closure: dict[str, Any],
    experiment: str,
    split: str,
    metric: str,
) -> dict[int, float]:
    block = closure["experiments"][experiment][split][metric]["per_seed"]
    return {int(seed): finite_float(value, f"{experiment}_{split}_{metric}_seed{seed}")
            for seed, value in block.items()}


def support_metric_per_seed(
    support_closure: dict[str, Any],
    split: str,
    metric: str,
) -> dict[int, float]:
    block = support_closure["support_preserving_aggregate_global"][split][metric]["per_seed"]
    return {int(seed): finite_float(value, f"support_{split}_{metric}_seed{seed}")
            for seed, value in block.items()}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gate S8F aggregate and seal three binary-support formal evaluations."
    )
    parser.add_argument("--seed42-report", required=True)
    parser.add_argument("--seed43-report", required=True)
    parser.add_argument("--seed44-report", required=True)
    parser.add_argument("--four-config-closure", required=True)
    parser.add_argument("--support-control-closure", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    report_paths = {
        42: Path(args.seed42_report).expanduser().resolve(),
        43: Path(args.seed43_report).expanduser().resolve(),
        44: Path(args.seed44_report).expanduser().resolve(),
    }
    four_path = Path(args.four_config_closure).expanduser().resolve()
    support_path = Path(args.support_control_closure).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    for path in [*report_paths.values(), four_path, support_path]:
        require(path.is_file(), f"INPUT_NOT_FOUND: {path}")
        require(path.stat().st_size > 0, f"INPUT_EMPTY: {path}")
    require(not output_dir.exists(), f"STOP_OUTPUT_DIR_ALREADY_EXISTS: {output_dir}")

    four = load_json(four_path)
    support = load_json(support_path)
    require(four.get("artifact_name") == EXPECTED_FOUR_CLOSURE, "FOUR_CLOSURE_ARTIFACT_MISMATCH")
    require(four.get("verification", {}).get("overall_pass") is True, "FOUR_CLOSURE_NOT_PASS")
    require(support.get("artifact_name") == EXPECTED_SUPPORT_CLOSURE, "SUPPORT_CLOSURE_ARTIFACT_MISMATCH")
    require(support.get("verification", {}).get("overall_pass") is True, "SUPPORT_CLOSURE_NOT_PASS")
    require(four.get("shared_monthly_manifest_sha256") == EXPECTED_MANIFEST_SHA256, "FOUR_CLOSURE_MANIFEST_HASH_MISMATCH")
    require(
        support.get("shared_provenance", {}).get("monthly_manifest_sha256")
        == EXPECTED_MANIFEST_SHA256,
        "SUPPORT_CLOSURE_MANIFEST_HASH_MISMATCH",
    )
    # The existing support-preserving closure was produced by evaluator 62,
    # whose tensor hash semantics are dtype + shape + raw bytes. Gate P3/S8A
    # locked the same proxy with raw bytes only. These digests are therefore
    # intentionally different and must not be compared directly.
    require(
        support.get("shared_provenance", {}).get("proxy_tensor_sha256")
        == EXPECTED_SUPPORT_CONTROL_PROXY_CANONICAL_SHA256,
        "SUPPORT_CLOSURE_PROXY_CANONICAL_HASH_MISMATCH",
    )

    reports: dict[int, dict[str, Any]] = {}
    input_records: list[dict[str, Any]] = []
    binary_manifest_hashes: set[str] = set()
    binary_training_hashes: set[str] = set()
    binary_proxy_hashes: set[str] = set()

    for seed in EXPECTED_SEEDS:
        path = report_paths[seed]
        report = load_json(path)
        require(report.get("artifact_name") == EXPECTED_BINARY_ARTIFACT, f"BINARY_ARTIFACT_MISMATCH_SEED={seed}")
        require(int(report.get("artifact_version", -1)) == 1, f"BINARY_VERSION_MISMATCH_SEED={seed}")
        require(int(report.get("seed", -1)) == seed, f"BINARY_SEED_MISMATCH_SEED={seed}")
        exp = report.get("experiment", {})
        require(exp.get("name") == EXPECTED_BINARY_EXPERIMENT, f"BINARY_EXPERIMENT_MISMATCH_SEED={seed}")
        require(exp.get("binary_support_emission") is True, f"BINARY_FLAG_FALSE_SEED={seed}")
        require(int(exp.get("emission_channels", -1)) == 1, f"BINARY_CHANNELS_MISMATCH_SEED={seed}")
        require(int(exp.get("meteo_channels", -1)) == 5, f"METEO_CHANNELS_MISMATCH_SEED={seed}")

        verification = report.get("verification", {})
        require(verification.get("overall_pass") is True, f"BINARY_EVAL_NOT_PASS_SEED={seed}")
        for key in [
            "training_audit_provenance_pass",
            "binary_support_contract_pass",
            "checkpoint_strict_load_pass",
            "2022_best_metric_reproduction_pass",
            "2023_all_12_months_evaluated",
            "2023_emission_proxy_used",
            "proxy_hash_exact",
            "proxy_support_count_exact",
        ]:
            require(verification.get(key) is True, f"{key.upper()}_FAIL_SEED={seed}")

        design = report.get("evaluation_design", {})
        require(design.get("2022_status") == "held-out model-selection validation",
                f"2022_STATUS_MISMATCH_SEED={seed}")
        require(design.get("2023_status") == "proxy_external",
                f"2023_STATUS_MISMATCH_SEED={seed}")
        require(design.get("emission_proxy_used") is True,
                f"2023_PROXY_USED_FALSE_SEED={seed}")
        require(design.get("support_epsilon") == 0.0,
                f"SUPPORT_EPSILON_MISMATCH_SEED={seed}")

        provenance = report.get("provenance", {})
        manifest_hash = str(provenance.get("monthly_manifest_sha256", ""))
        training_hash = str(provenance.get("training_script_sha256", ""))
        proxy_hash = str(report.get("proxy_emission", {}).get("proxy_tensor_sha256", ""))
        proxy_cells = int(report.get("proxy_emission", {}).get("proxy_support_cells", -1))
        require(manifest_hash == EXPECTED_MANIFEST_SHA256, f"BINARY_MANIFEST_HASH_MISMATCH_SEED={seed}")
        require(len(training_hash) == 64, f"BINARY_TRAINING_HASH_INVALID_SEED={seed}")
        require(proxy_hash == EXPECTED_PROXY_SHA256, f"BINARY_PROXY_HASH_MISMATCH_SEED={seed}")
        require(proxy_cells == EXPECTED_PROXY_SUPPORT_CELLS, f"BINARY_PROXY_CELLS_MISMATCH_SEED={seed}")
        binary_manifest_hashes.add(manifest_hash)
        binary_training_hashes.add(training_hash)
        binary_proxy_hashes.add(proxy_hash)

        for split_key in ["eval_2022_validation", "eval_2023_external"]:
            global_block = report.get(split_key, {}).get("global", {})
            for metric in METRICS:
                finite_float(global_block.get(metric), f"{split_key}_{metric}_seed{seed}")

        reports[seed] = report
        input_records.append(
            {
                "seed": seed,
                "path": str(path),
                "sha256": sha256_file(path),
                "checkpoint_sha256": report.get("checkpoint", {}).get("sha256"),
                "selected_epoch": report.get("checkpoint", {}).get("selected_epoch"),
            }
        )

    require(len(binary_manifest_hashes) == 1, "BINARY_MANIFEST_HASHES_DIFFER")
    require(len(binary_training_hashes) == 1, "BINARY_TRAINING_SCRIPT_HASHES_DIFFER")
    require(len(binary_proxy_hashes) == 1, "BINARY_PROXY_HASHES_DIFFER")

    binary_aggregate: dict[str, Any] = {}
    for split_label, split_key in [
        ("val_2022", "eval_2022_validation"),
        ("external_2023", "eval_2023_external"),
    ]:
        binary_aggregate[split_label] = {}
        for metric in METRICS:
            per_seed = {
                seed: finite_float(
                    reports[seed][split_key]["global"][metric],
                    f"binary_{split_label}_{metric}_seed{seed}",
                )
                for seed in EXPECTED_SEEDS
            }
            binary_aggregate[split_label][metric] = summarize(per_seed)

    contrasts: dict[str, Any] = {}
    for split_label, closure_split, support_split in [
        ("val_2022", "eval_2022_validation", "val_2022"),
        ("external_2023", "eval_2023_external", "external_2023"),
    ]:
        for metric in METRICS:
            binary = {int(k): float(v) for k, v in binary_aggregate[split_label][metric]["per_seed"].items()}
            meteo = closure_metric_per_seed(four, "mean_meteo_clim_only", closure_split, metric)
            joint = closure_metric_per_seed(four, "mean_emis_meteo_clim", closure_split, metric)
            cyclic = closure_metric_per_seed(four, "mean_emis_meteo_clim_shuffled", closure_split, metric)
            support_preserving = support_metric_per_seed(support, support_split, metric)

            contrasts[f"binary_minus_meteo_{split_label}_{metric}"] = paired_difference(binary, meteo)
            contrasts[f"joint_minus_binary_{split_label}_{metric}"] = paired_difference(joint, binary)
            contrasts[f"support_preserving_minus_binary_{split_label}_{metric}"] = paired_difference(support_preserving, binary)
            contrasts[f"binary_minus_cyclic_{split_label}_{metric}"] = paired_difference(binary, cyclic)

    output_dir.mkdir(parents=True, exist_ok=False)
    json_path = output_dir / "meanstate_binary_support_gateS8_closure_v01.json"
    txt_path = output_dir / "meanstate_binary_support_gateS8_closure_v01.txt"
    csv_path = output_dir / "meanstate_binary_support_gateS8_per_seed_v01.csv"

    report = {
        "artifact_name": "clean_meanstate_binary_support_gateS8_closure",
        "artifact_version": 1,
        "binary_support_input_reports": input_records,
        "source_four_config_closure": {
            "path": str(four_path),
            "sha256": sha256_file(four_path),
            "artifact_name": four.get("artifact_name"),
            "artifact_version": four.get("artifact_version"),
        },
        "source_support_control_closure": {
            "path": str(support_path),
            "sha256": sha256_file(support_path),
            "artifact_name": support.get("artifact_name"),
            "artifact_version": support.get("artifact_version"),
        },
        "shared_provenance": {
            "monthly_manifest_sha256": next(iter(binary_manifest_hashes)),
            "binary_training_script_sha256": next(iter(binary_training_hashes)),
            "binary_proxy_tensor_sha256": next(iter(binary_proxy_hashes)),
            "binary_proxy_hash_semantics": "raw contiguous tensor bytes (Gate P3 / Gate S8A)",
            "support_control_proxy_tensor_sha256": support.get("shared_provenance", {}).get("proxy_tensor_sha256"),
            "support_control_proxy_hash_semantics": "dtype + shape + raw bytes (evaluator 62)",
            "proxy_support_cells": EXPECTED_PROXY_SUPPORT_CELLS,
        },
        "binary_support_aggregate_global": binary_aggregate,
        "paired_contrasts": contrasts,
        "predeclared_interpretation_contract": {
            "no_posthoc_equivalence_threshold": True,
            "no_significance_test": True,
            "comparison_order": [
                "meteorology-only",
                "binary-support + meteorology",
                "support-preserving vectors + meteorology",
                "original processed vectors + meteorology",
            ],
            "interpretation_rule": (
                "Interpret magnitude, seed-wise direction and uncertainty descriptively. "
                "No numerical threshold for 'approximately equal' is introduced after seeing results."
            ),
        },
        "verification": {
            "exact_binary_seed_set_pass": True,
            "all_binary_eval_reports_pass": True,
            "shared_monthly_manifest_hash_pass": True,
            "shared_binary_training_script_hash_pass": True,
            "shared_proxy_tensor_hash_pass": True,
            "source_four_config_closure_pass": True,
            "source_support_control_closure_pass": True,
            "all_global_metrics_finite": True,
            "all_predeclared_paired_contrasts_computed": True,
            "overall_pass": True,
        },
        "reporting_boundary": (
            "Seeds 42, 43 and 44 provide descriptive stability only; no statistical significance "
            "or equivalence is established. November-December 2022 is held-out model-selection "
            "validation. The 2023 binary-support result is a proxy external consistency test because "
            "the support mask is derived from the fixed 2022 train-mean processed-prior proxy. "
            "Binary processed support must not be interpreted as verified physical emission geography."
        ),
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=True, indent=2, allow_nan=False)

    rows = []
    for seed in EXPECTED_SEEDS:
        for split_label, split_key in [
            ("val_2022", "eval_2022_validation"),
            ("external_2023", "eval_2023_external"),
        ]:
            row = {
                "seed": seed,
                "split": split_label,
            }
            for metric in METRICS:
                row[metric] = reports[seed][split_key]["global"][metric]
            row["meteo_log_r2"] = closure_metric_per_seed(
                four, "mean_meteo_clim_only", split_key, "log_r2"
            )[seed]
            row["joint_log_r2"] = closure_metric_per_seed(
                four, "mean_emis_meteo_clim", split_key, "log_r2"
            )[seed]
            row["cyclic_log_r2"] = closure_metric_per_seed(
                four, "mean_emis_meteo_clim_shuffled", split_key, "log_r2"
            )[seed]
            row["support_preserving_log_r2"] = support_metric_per_seed(
                support,
                "val_2022" if split_label == "val_2022" else "external_2023",
                "log_r2",
            )[seed]
            rows.append(row)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed", "split", "log_r2", "log_rmse", "log_bias",
                "meteo_log_r2", "joint_log_r2", "cyclic_log_r2",
                "support_preserving_log_r2",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    b22 = binary_aggregate["val_2022"]["log_r2"]
    b23 = binary_aggregate["external_2023"]["log_r2"]
    jm22 = contrasts["joint_minus_binary_val_2022_log_r2"]
    jm23 = contrasts["joint_minus_binary_external_2023_log_r2"]
    bm22 = contrasts["binary_minus_meteo_val_2022_log_r2"]
    bm23 = contrasts["binary_minus_meteo_external_2023_log_r2"]
    sb22 = contrasts["support_preserving_minus_binary_val_2022_log_r2"]
    sb23 = contrasts["support_preserving_minus_binary_external_2023_log_r2"]

    lines = [
        "GATE_S8F_BINARY_SUPPORT_THREE_SEED_CLOSURE_PASS",
        f"BINARY_2022_LOG_R2_MEAN={b22['mean']:.8f}",
        f"BINARY_2022_LOG_R2_SD={b22['sample_sd']:.8f}",
        f"BINARY_2023_LOG_R2_MEAN={b23['mean']:.8f}",
        f"BINARY_2023_LOG_R2_SD={b23['sample_sd']:.8f}",
        f"BINARY_MINUS_METEO_2022_LOG_R2_MEAN={bm22['mean']:.8f}",
        f"BINARY_MINUS_METEO_2022_LOG_R2_SD={bm22['sample_sd']:.8f}",
        f"BINARY_MINUS_METEO_2022_ALL_POSITIVE={bm22['all_positive']}",
        f"BINARY_MINUS_METEO_2023_LOG_R2_MEAN={bm23['mean']:.8f}",
        f"BINARY_MINUS_METEO_2023_LOG_R2_SD={bm23['sample_sd']:.8f}",
        f"BINARY_MINUS_METEO_2023_ALL_POSITIVE={bm23['all_positive']}",
        f"JOINT_MINUS_BINARY_2022_LOG_R2_MEAN={jm22['mean']:.8f}",
        f"JOINT_MINUS_BINARY_2022_LOG_R2_SD={jm22['sample_sd']:.8f}",
        f"JOINT_MINUS_BINARY_2022_ALL_POSITIVE={jm22['all_positive']}",
        f"JOINT_MINUS_BINARY_2023_LOG_R2_MEAN={jm23['mean']:.8f}",
        f"JOINT_MINUS_BINARY_2023_LOG_R2_SD={jm23['sample_sd']:.8f}",
        f"JOINT_MINUS_BINARY_2023_ALL_POSITIVE={jm23['all_positive']}",
        f"SUPPORT_PRESERVING_MINUS_BINARY_2022_LOG_R2_MEAN={sb22['mean']:.8f}",
        f"SUPPORT_PRESERVING_MINUS_BINARY_2022_LOG_R2_SD={sb22['sample_sd']:.8f}",
        f"SUPPORT_PRESERVING_MINUS_BINARY_2023_LOG_R2_MEAN={sb23['mean']:.8f}",
        f"SUPPORT_PRESERVING_MINUS_BINARY_2023_LOG_R2_SD={sb23['sample_sd']:.8f}",
        "POSTHOC_EQUIVALENCE_THRESHOLD_USED=False",
        "STATISTICAL_SIGNIFICANCE_CLAIMED=False",
        f"OUTPUT_JSON={json_path}",
        f"OUTPUT_CSV={csv_path}",
        f"OUTPUT_TXT={txt_path}",
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
