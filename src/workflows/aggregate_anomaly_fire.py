# RELEASE COPY: local absolute path defaults were replaced with relative
# placeholders. Scientific logic, seeds, splits, controls, and metrics were
# not intentionally changed. Configure paths before restricted-data runs.
# -*- coding: utf-8 -*-
"""
73_seal_anomaly_fire_three_seed.py

One-command closure for the clean meteorology+fire anomaly contrast.

This script performs no training. It:
1. audits/evaluates missing fire runs for seeds 43 and 44 using the frozen
   audit/evaluation scripts;
2. verifies all three fire runs and their provenance;
3. loads the already sealed clean meteo-only evaluations for the same seeds;
4. computes paired fire-minus-meteo diagnostics for 2022 held-out validation
   and 2023 ordinary external evaluation;
5. compares the clean three-seed fire mean with the earlier paper Table 3
   single-run values;
6. seals the fire line or pauses if the old conclusion would be overturned.

No existing artifact is overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


SEEDS = [42, 43, 44]
FIRE_EXPERIMENT = "anom_meteo_fire"
METEO_EXPERIMENT = "anom_meteo_only"
EXPECTED_TRAIN_FILES = 178
EXPECTED_VAL_FILES = 61
EXPECTED_EXTERNAL_FILES = 361
EXPECTED_FIRE_PARAMETERS = 477313
EXPECTED_METEO_PARAMETERS = 477025
ABS_OLD_REFERENCE_R2_TRIGGER = 0.01

OLD_REFERENCE = {
    "2022_val": {
        "meteo_r2": 0.002671,
        "fire_r2": 0.010529,
        "fire_minus_meteo_r2": 0.010529 - 0.002671,
    },
    "2023_external": {
        "meteo_r2": 0.000072,
        "fire_r2": 0.013052,
        "fire_minus_meteo_r2": 0.013052 - 0.000072,
    },
}

EXPECTED_CONTRACT = {
    "batch_size": 4,
    "num_workers": 0,
    "patch_size": 96,
    "patches_per_file": 4,
    "base_features": 32,
    "max_epochs": 140,
    "patience": 25,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "val_every": 5,
    "train_limit": 0,
    "val_limit": 0,
}

METRIC_PATHS = {
    "anom_log_r2": ("model_anomaly", "r2"),
    "anom_log_rmse": ("model_anomaly", "rmse"),
    "anom_log_bias": ("model_anomaly", "bias"),
    "anom_log_corr": ("model_anomaly", "corr"),
    "anom_mse_skill_vs_zero": ("anomaly_mse_skill_vs_zero",),
    "recon_r2_delta_vs_background": (
        "reconstruction_r2_delta_vs_month_background",
    ),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical(path: str | Path) -> str:
    return os.path.normcase(os.path.abspath(os.path.normpath(str(path))))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"JSON_NOT_FOUND: {path}")
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    require(isinstance(obj, dict), f"JSON_ROOT_NOT_OBJECT: {path}")
    return obj


def save_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=True, indent=2, allow_nan=False)


def finite_float(value: Any, label: str) -> float:
    x = float(value)
    require(math.isfinite(x), f"NONFINITE_{label}: {value}")
    return x


def nested_get(obj: dict[str, Any], path: tuple[str, ...], label: str) -> Any:
    value: Any = obj
    for key in path:
        require(isinstance(value, dict) and key in value,
                f"MISSING_{label}_{'_'.join(path)}")
        value = value[key]
    return value


def summarize(values: list[float], label: str) -> dict[str, Any]:
    require(len(values) == len(SEEDS),
            f"SUMMARY_SEED_COUNT_MISMATCH_{label}")
    for seed, value in zip(SEEDS, values):
        finite_float(value, f"{label}_SEED{seed}")
    return {
        "seeds": SEEDS,
        "values": values,
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
        "all_positive": all(value > 0.0 for value in values),
        "all_nonnegative": all(value >= 0.0 for value in values),
    }


def invoke(command: list[str], label: str) -> None:
    print("=" * 96, flush=True)
    print(f"INVOKE_{label}=True", flush=True)
    print("COMMAND=" + subprocess.list2cmdline(command), flush=True)
    completed = subprocess.run(command, check=False)
    require(completed.returncode == 0,
            f"{label}_FAILED_EXIT_CODE={completed.returncode}")


def verify_contract(actual: dict[str, Any], seed: int) -> None:
    require(set(actual) == set(EXPECTED_CONTRACT),
            f"CONTRACT_KEY_SET_MISMATCH_SEED{seed}")
    for key, expected in EXPECTED_CONTRACT.items():
        value = actual[key]
        if isinstance(expected, float):
            require(
                math.isclose(float(value), expected,
                             rel_tol=0.0, abs_tol=1e-12),
                f"CONTRACT_MISMATCH_SEED{seed}_{key}: {value}",
            )
        else:
            require(value == expected,
                    f"CONTRACT_MISMATCH_SEED{seed}_{key}: {value}")


def global_metric(report: dict[str, Any], split: str,
                  metric_path: tuple[str, ...], label: str) -> float:
    return finite_float(
        nested_get(report["results"][split]["global"], metric_path, label),
        label,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Seal clean meteo+fire anomaly contrast across three seeds."
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--training-script", required=True)
    parser.add_argument("--audit-script", required=True)
    parser.add_argument("--eval-script", required=True)
    parser.add_argument("--cache-audit", required=True)
    parser.add_argument("--sanity-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metric-tolerance", type=float, default=1e-6)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    require(
        math.isclose(args.metric_tolerance, 1e-6,
                     rel_tol=0.0, abs_tol=0.0),
        "METRIC_TOLERANCE_LOCKED_TO_1E-6",
    )

    run_root = Path(args.run_root).expanduser().resolve()
    training_script = Path(args.training_script).expanduser().resolve()
    audit_script = Path(args.audit_script).expanduser().resolve()
    eval_script = Path(args.eval_script).expanduser().resolve()
    cache_audit_path = Path(args.cache_audit).expanduser().resolve()
    sanity_root = Path(args.sanity_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    require(run_root.is_dir(), f"RUN_ROOT_NOT_FOUND: {run_root}")
    require(training_script.is_file(),
            f"TRAINING_SCRIPT_NOT_FOUND: {training_script}")
    require(audit_script.is_file(),
            f"AUDIT_SCRIPT_NOT_FOUND: {audit_script}")
    require(eval_script.is_file(),
            f"EVAL_SCRIPT_NOT_FOUND: {eval_script}")
    require(cache_audit_path.is_file(),
            f"CACHE_AUDIT_NOT_FOUND: {cache_audit_path}")
    require(sanity_root.is_dir(), f"SANITY_ROOT_NOT_FOUND: {sanity_root}")
    require(not output_dir.exists(),
            f"STOP_OUTPUT_DIR_EXISTS: {output_dir}")

    fire_run_dirs: dict[int, Path] = {}
    fire_audit_paths: dict[int, Path] = {}
    fire_eval_paths: dict[int, Path] = {}
    meteo_eval_paths: dict[int, Path] = {}

    for seed in SEEDS:
        fire_run = (
            run_root / f"seed{seed}" /
            f"{FIRE_EXPERIMENT}_seed{seed}"
        )
        meteo_run = (
            run_root / f"seed{seed}" /
            f"{METEO_EXPERIMENT}_seed{seed}"
        )
        audit_dir = (
            sanity_root /
            f"anomaly_clean_formal_fire_seed{seed}_audit_v01"
        )
        audit_path = (
            audit_dir /
            f"anomaly_clean_formal_fire_seed{seed}_audit_v01.json"
        )
        fire_eval_dir = fire_run / "eval_v01"
        fire_eval_path = fire_eval_dir / "eval_report.json"
        meteo_eval_path = meteo_run / "eval_v01" / "eval_report.json"

        require(fire_run.is_dir(),
                f"FIRE_RUN_DIR_NOT_FOUND_SEED{seed}: {fire_run}")
        require(meteo_eval_path.is_file(),
                f"METEO_EVAL_NOT_FOUND_SEED{seed}: {meteo_eval_path}")

        if not audit_path.is_file():
            require(not audit_dir.exists(),
                    f"PARTIAL_FIRE_AUDIT_DIR_EXISTS_SEED{seed}: {audit_dir}")
            command = [
                sys.executable, "-u", str(audit_script),
                "--run-dir", str(fire_run),
                "--expected-seed", str(seed),
                "--training-script", str(training_script),
                "--cache-audit", str(cache_audit_path),
                "--output-dir", str(audit_dir),
                "--metric-tolerance", f"{args.metric_tolerance:.12g}",
            ]
            if args.require_cuda:
                command.append("--require-cuda")
            invoke(command, f"FIRE_SEED{seed}_FORMAL_AUDIT")

        require(audit_path.is_file(),
                f"FIRE_AUDIT_NOT_FOUND_SEED{seed}: {audit_path}")

        if not fire_eval_path.is_file():
            require(not fire_eval_dir.exists(),
                    f"PARTIAL_FIRE_EVAL_DIR_EXISTS_SEED{seed}: {fire_eval_dir}")
            command = [
                sys.executable, "-u", str(eval_script),
                "--run-dir", str(fire_run),
                "--expected-seed", str(seed),
                "--training-script", str(training_script),
                "--formal-audit", str(audit_path),
                "--output-dir", str(fire_eval_dir),
                "--metric-tolerance", f"{args.metric_tolerance:.12g}",
            ]
            if args.require_cuda:
                command.append("--require-cuda")
            invoke(command, f"FIRE_SEED{seed}_EXTERNAL_EVALUATION")

        require(fire_eval_path.is_file(),
                f"FIRE_EVAL_NOT_FOUND_SEED{seed}: {fire_eval_path}")

        fire_run_dirs[seed] = fire_run
        fire_audit_paths[seed] = audit_path
        fire_eval_paths[seed] = fire_eval_path
        meteo_eval_paths[seed] = meteo_eval_path

    fire_reports: dict[int, dict[str, Any]] = {}
    meteo_reports: dict[int, dict[str, Any]] = {}
    provenance: dict[str, Any] = {}
    shared_cache_dir: str | None = None
    shared_cache_hashes: dict[str, str] | None = None

    for seed in SEEDS:
        fire_run = fire_run_dirs[seed]
        manifest_path = fire_run / "logs" / "run_manifest.json"
        history_path = fire_run / "logs" / "training_history.json"
        checkpoint_path = fire_run / "checkpoints" / "best_model.pt"

        manifest = load_json(manifest_path)
        audit = load_json(fire_audit_paths[seed])
        fire_eval = load_json(fire_eval_paths[seed])
        meteo_eval = load_json(meteo_eval_paths[seed])

        require(manifest.get("artifact_name") ==
                "clean_anomaly_training_run",
                f"MANIFEST_ARTIFACT_MISMATCH_SEED{seed}")
        require(manifest.get("experiment") == FIRE_EXPERIMENT,
                f"FIRE_EXPERIMENT_MISMATCH_SEED{seed}")
        require(manifest.get("run_mode") == "formal",
                f"RUN_MODE_MISMATCH_SEED{seed}")
        require(int(manifest.get("seed", -1)) == seed,
                f"MANIFEST_SEED_MISMATCH_SEED{seed}")
        require(int(manifest.get("trainable_parameters", -1)) ==
                EXPECTED_FIRE_PARAMETERS,
                f"FIRE_PARAMETER_COUNT_MISMATCH_SEED{seed}")
        verify_contract(manifest.get("contract", {}), seed)

        effective = manifest.get("effective_counts", {})
        require(int(effective.get("train_files", -1)) ==
                EXPECTED_TRAIN_FILES,
                f"TRAIN_COUNT_MISMATCH_SEED{seed}")
        require(int(effective.get("val_files", -1)) ==
                EXPECTED_VAL_FILES,
                f"VAL_COUNT_MISMATCH_SEED{seed}")
        require(int(effective.get("external_manifest_files", -1)) ==
                EXPECTED_EXTERNAL_FILES,
                f"EXTERNAL_COUNT_MISMATCH_SEED{seed}")

        require(audit.get("artifact_name") ==
                "clean_anomaly_formal_run_audit",
                f"AUDIT_ARTIFACT_MISMATCH_SEED{seed}")
        require(audit.get("verification", {}).get("overall_pass") is True,
                f"AUDIT_NOT_PASS_SEED{seed}")
        require(
            audit.get("decision") ==
            f"PASS_PROCEED_TO_SEED{seed}_EXTERNAL_EVALUATION",
            f"AUDIT_DECISION_MISMATCH_SEED{seed}",
        )

        require(fire_eval.get("artifact_name") ==
                "clean_anomaly_formal_run_evaluation",
                f"FIRE_EVAL_ARTIFACT_MISMATCH_SEED{seed}")
        require(fire_eval.get("experiment") == FIRE_EXPERIMENT,
                f"FIRE_EVAL_EXPERIMENT_MISMATCH_SEED{seed}")
        require(int(fire_eval.get("seed", -1)) == seed,
                f"FIRE_EVAL_SEED_MISMATCH_SEED{seed}")
        require(fire_eval.get("verification", {}).get("overall_pass") is True,
                f"FIRE_EVAL_NOT_PASS_SEED{seed}")
        require(
            fire_eval.get("verification", {}).get(
                "validation_metric_matches_formal_audit"
            ) is True,
            f"FIRE_EVAL_REPRODUCTION_FAIL_SEED{seed}",
        )

        require(meteo_eval.get("experiment") == METEO_EXPERIMENT,
                f"METEO_EVAL_EXPERIMENT_MISMATCH_SEED{seed}")
        require(int(meteo_eval.get("seed", -1)) == seed,
                f"METEO_EVAL_SEED_MISMATCH_SEED{seed}")
        require(meteo_eval.get("verification", {}).get("overall_pass") is True,
                f"METEO_EVAL_NOT_PASS_SEED{seed}")

        cache_dir = canonical(manifest.get("cache_dir", ""))
        cache_hashes = manifest.get("manifest_sha256", {})
        require(isinstance(cache_hashes, dict) and cache_hashes,
                f"CACHE_HASHES_EMPTY_SEED{seed}")
        if shared_cache_dir is None:
            shared_cache_dir = cache_dir
            shared_cache_hashes = cache_hashes
        else:
            require(cache_dir == shared_cache_dir,
                    f"CACHE_DIR_DIFFERS_SEED{seed}")
            require(cache_hashes == shared_cache_hashes,
                    f"CACHE_HASHES_DIFFER_SEED{seed}")

        require(
            fire_eval["best_checkpoint"]["sha256"] ==
            sha256_file(checkpoint_path),
            f"CHECKPOINT_HASH_MISMATCH_SEED{seed}",
        )
        require(
            fire_eval["input_provenance"]["formal_audit"]["sha256"] ==
            sha256_file(fire_audit_paths[seed]),
            f"AUDIT_HASH_MISMATCH_SEED{seed}",
        )

        fire_reports[seed] = fire_eval
        meteo_reports[seed] = meteo_eval
        provenance[str(seed)] = {
            "fire_run_dir": str(fire_run),
            "fire_run_manifest": {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
            },
            "fire_training_history": {
                "path": str(history_path),
                "sha256": sha256_file(history_path),
            },
            "fire_checkpoint": {
                "path": str(checkpoint_path),
                "sha256": sha256_file(checkpoint_path),
            },
            "fire_formal_audit": {
                "path": str(fire_audit_paths[seed]),
                "sha256": sha256_file(fire_audit_paths[seed]),
            },
            "fire_evaluation": {
                "path": str(fire_eval_paths[seed]),
                "sha256": sha256_file(fire_eval_paths[seed]),
            },
            "meteo_evaluation": {
                "path": str(meteo_eval_paths[seed]),
                "sha256": sha256_file(meteo_eval_paths[seed]),
            },
        }

    fire_aggregate: dict[str, Any] = {}
    meteo_aggregate: dict[str, Any] = {}
    paired_aggregate: dict[str, Any] = {}

    for split in ("2022_val", "2023_external"):
        fire_aggregate[split] = {}
        meteo_aggregate[split] = {}
        paired_aggregate[split] = {}

        for metric_name, metric_path in METRIC_PATHS.items():
            fire_values = [
                global_metric(fire_reports[seed], split, metric_path,
                              f"FIRE_{split}_{metric_name}_SEED{seed}")
                for seed in SEEDS
            ]
            meteo_values = [
                global_metric(meteo_reports[seed], split, metric_path,
                              f"METEO_{split}_{metric_name}_SEED{seed}")
                for seed in SEEDS
            ]
            deltas = [
                fire_value - meteo_value
                for fire_value, meteo_value in zip(fire_values, meteo_values)
            ]
            fire_aggregate[split][metric_name] = summarize(
                fire_values, f"FIRE_{split}_{metric_name}"
            )
            meteo_aggregate[split][metric_name] = summarize(
                meteo_values, f"METEO_{split}_{metric_name}"
            )
            paired_aggregate[split][f"fire_minus_meteo_{metric_name}"] = (
                summarize(deltas, f"PAIRED_{split}_{metric_name}")
            )

    val_r2_delta = paired_aggregate["2022_val"][
        "fire_minus_meteo_anom_log_r2"
    ]
    ext_r2_delta = paired_aggregate["2023_external"][
        "fire_minus_meteo_anom_log_r2"
    ]

    clean_fire_val_mean = fire_aggregate["2022_val"][
        "anom_log_r2"
    ]["mean"]
    clean_fire_ext_mean = fire_aggregate["2023_external"][
        "anom_log_r2"
    ]["mean"]
    old_val_abs_diff = abs(
        clean_fire_val_mean - OLD_REFERENCE["2022_val"]["fire_r2"]
    )
    old_ext_abs_diff = abs(
        clean_fire_ext_mean - OLD_REFERENCE["2023_external"]["fire_r2"]
    )

    paired_positive_both_splits = (
        val_r2_delta["all_positive"] and ext_r2_delta["all_positive"]
    )
    old_reference_close_both_splits = (
        old_val_abs_diff <= ABS_OLD_REFERENCE_R2_TRIGGER
        and old_ext_abs_diff <= ABS_OLD_REFERENCE_R2_TRIGGER
    )

    if paired_positive_both_splits and old_reference_close_both_splits:
        decision = "CLOSE_FIRE_LINE_PRESERVE_LIMITED_STABLE_INCREMENT"
        overall_pass = True
        paper_fire_interpretation = (
            "small_positive_fire_increment_stable_across_three_seeds_and_"
            "both_evaluation_splits"
        )
    else:
        decision = "PAUSE_AUDIT_OLD_VS_CLEAN_FIRE_COMPARABILITY"
        overall_pass = False
        paper_fire_interpretation = "frozen_pending_discrepancy_audit"

    output_dir.mkdir(parents=True, exist_ok=False)
    json_path = output_dir / "anomaly_fire_three_seed_closure_v01.json"
    txt_path = output_dir / "anomaly_fire_three_seed_closure_v01.txt"

    report = {
        "artifact_name": "clean_anomaly_fire_three_seed_closure",
        "artifact_version": 1,
        "fire_experiment": FIRE_EXPERIMENT,
        "baseline_experiment": METEO_EXPERIMENT,
        "seeds": SEEDS,
        "input_provenance": {
            "per_seed": provenance,
            "training_script": {
                "path": str(training_script),
                "sha256": sha256_file(training_script),
            },
            "audit_script": {
                "path": str(audit_script),
                "sha256": sha256_file(audit_script),
            },
            "evaluation_script": {
                "path": str(eval_script),
                "sha256": sha256_file(eval_script),
            },
            "cache_audit": {
                "path": str(cache_audit_path),
                "sha256": sha256_file(cache_audit_path),
            },
            "closure_script": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
        "shared_cache_dir": shared_cache_dir,
        "shared_cache_manifest_sha256": shared_cache_hashes,
        "fire_aggregate": fire_aggregate,
        "meteo_aggregate": meteo_aggregate,
        "paired_fire_minus_meteo_aggregate": paired_aggregate,
        "earlier_paper_reference": OLD_REFERENCE,
        "old_vs_clean_fire_mean": {
            "2022_val_abs_r2_difference": old_val_abs_diff,
            "2023_external_abs_r2_difference": old_ext_abs_diff,
            "diagnostic_trigger": ABS_OLD_REFERENCE_R2_TRIGGER,
            "trigger_is_not_significance_threshold": True,
        },
        "verification": {
            "all_three_fire_training_runs_present": True,
            "all_three_fire_formal_audits_pass": True,
            "all_three_fire_external_evaluations_pass": True,
            "shared_cache_provenance_exact": True,
            "paired_fire_r2_increment_positive_all_seeds_2022":
                val_r2_delta["all_positive"],
            "paired_fire_r2_increment_positive_all_seeds_2023":
                ext_r2_delta["all_positive"],
            "old_reference_close_both_splits":
                old_reference_close_both_splits,
            "overall_pass": overall_pass,
        },
        "scientific_interpretation_boundary": {
            "2022_validation_type": "held-out_model-selection_validation",
            "2023_evaluation_type":
                "ordinary_external_anomaly_evaluation",
            "three_seed_summary_type":
                "descriptive_mean_and_sample_sd",
            "statistical_significance_established": False,
            "causal_fire_effect_established": False,
            "reconstruction_requires_month_background_baseline": True,
            "paper_fire_interpretation": paper_fire_interpretation,
            "paper_revision_allowed_only_after_closure_pass": overall_pass,
        },
        "atlas_direction_closed": True,
        "new_atlas_experiments_allowed": False,
        "decision": decision,
    }
    save_json(report, json_path)

    fv = fire_aggregate["2022_val"]
    fe = fire_aggregate["2023_external"]
    pv = paired_aggregate["2022_val"]
    pe = paired_aggregate["2023_external"]

    lines = [
        (
            "CLEAN_ANOMALY_FIRE_THREE_SEED_CLOSURE_"
            + ("PASS" if overall_pass else "PAUSE")
        ),
        f"SEEDS={SEEDS}",
        f"FIRE_VAL_2022_ANOM_LOG_R2_MEAN_SD="
        f"{fv['anom_log_r2']['mean']:.8f}+/-"
        f"{fv['anom_log_r2']['sample_sd']:.8f}",
        f"FIRE_VAL_2022_ANOM_LOG_RMSE_MEAN_SD="
        f"{fv['anom_log_rmse']['mean']:.8f}+/-"
        f"{fv['anom_log_rmse']['sample_sd']:.8f}",
        f"FIRE_VAL_2022_ANOM_LOG_CORR_MEAN_SD="
        f"{fv['anom_log_corr']['mean']:.8f}+/-"
        f"{fv['anom_log_corr']['sample_sd']:.8f}",
        f"FIRE_MINUS_METEO_VAL_2022_ANOM_LOG_R2_MEAN_SD="
        f"{pv['fire_minus_meteo_anom_log_r2']['mean']:.8f}+/-"
        f"{pv['fire_minus_meteo_anom_log_r2']['sample_sd']:.8f}",
        f"FIRE_MINUS_METEO_VAL_2022_ALL_POSITIVE="
        f"{pv['fire_minus_meteo_anom_log_r2']['all_positive']}",
        f"FIRE_EXTERNAL_2023_ANOM_LOG_R2_MEAN_SD="
        f"{fe['anom_log_r2']['mean']:.8f}+/-"
        f"{fe['anom_log_r2']['sample_sd']:.8f}",
        f"FIRE_EXTERNAL_2023_ANOM_LOG_RMSE_MEAN_SD="
        f"{fe['anom_log_rmse']['mean']:.8f}+/-"
        f"{fe['anom_log_rmse']['sample_sd']:.8f}",
        f"FIRE_EXTERNAL_2023_ANOM_LOG_CORR_MEAN_SD="
        f"{fe['anom_log_corr']['mean']:.8f}+/-"
        f"{fe['anom_log_corr']['sample_sd']:.8f}",
        f"FIRE_MINUS_METEO_EXTERNAL_2023_ANOM_LOG_R2_MEAN_SD="
        f"{pe['fire_minus_meteo_anom_log_r2']['mean']:.8f}+/-"
        f"{pe['fire_minus_meteo_anom_log_r2']['sample_sd']:.8f}",
        f"FIRE_MINUS_METEO_EXTERNAL_2023_ALL_POSITIVE="
        f"{pe['fire_minus_meteo_anom_log_r2']['all_positive']}",
        f"OLD_FIRE_VAL_2022_R2={OLD_REFERENCE['2022_val']['fire_r2']:.8f}",
        f"CLEAN_FIRE_VAL_MEAN_ABS_DIFFERENCE_VS_OLD="
        f"{old_val_abs_diff:.8f}",
        f"OLD_FIRE_EXTERNAL_2023_R2="
        f"{OLD_REFERENCE['2023_external']['fire_r2']:.8f}",
        f"CLEAN_FIRE_EXTERNAL_MEAN_ABS_DIFFERENCE_VS_OLD="
        f"{old_ext_abs_diff:.8f}",
        f"OLD_REFERENCE_CLOSE_BOTH_SPLITS="
        f"{old_reference_close_both_splits}",
        "THREE_SEED_SUMMARY_TYPE=descriptive_mean_and_sample_sd",
        "STATISTICAL_SIGNIFICANCE_ESTABLISHED=False",
        "CAUSAL_FIRE_EFFECT_ESTABLISHED=False",
        f"FIRE_LINE_CLOSED={overall_pass}",
        "ATLAS_DIRECTION_CLOSED=True",
        "NEW_ATLAS_EXPERIMENTS_ALLOWED=False",
        f"DECISION={decision}",
        f"OUTPUT_JSON={json_path}",
        f"OUTPUT_TXT={txt_path}",
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print()
    print("\n".join(lines), flush=True)
    return 0 if overall_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
